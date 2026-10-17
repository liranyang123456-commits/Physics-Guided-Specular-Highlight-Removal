"""Re-render benchmark scenes with an independent renderer (Mitsuba 3) for the renderer-mismatch test.

The analytic surface of each scene (sphere, ellipsoid, bumped ellipsoid, or torus, with the parameters of
scripts/generate_world_benchmark.py) is triangulated and ray traced; the materials are carried to the mesh vertices from the
Gaussians of the scene (a Gaussian-weighted blend of their diffuse weight, specular weight, roughness, and metalness, with the
analytic base colour of the generator), the cameras and the per-frame lights are those of transforms.json, and the ambient term
is a constant environment of the same irradiance.  Differences to the renderer that the method fits: ray-traced triangles
instead of splatted Gaussians, Mitsuba's GGX conductor (exact Smith shadowing, no Schlick Fresnel) instead of our Cook-Torrance
term, ambient occlusion and cast shadows, and box-filtered antialiasing.  The diffuse ground truth is the diffuse pass; the input
is the diffuse plus the specular pass; both are truncated to 8 bits like the benchmark.  The Gaussian prior, cameras, and lights
of the original scene are copied, so the method receives the same given geometry and lighting as on the benchmark.

    python scripts/render_mitsuba_scenes.py --scenes 70 71 72 73 --spp 256
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_world_benchmark import make_geometry  # noqa: E402

AMBIENT = 0.886227          # SH irradiance of the DC coefficient in the benchmark renderer (mvbrdf_shr/models/envlight.py)


def surface(n: int, shape: str, count: int, device):
    """Analytic surface and base-colour parameters of a scene, drawn from the generator's random stream."""
    seed = 2026 + n * 101
    g = torch.Generator(device=device).manual_seed(seed)
    first = torch.rand(3, generator=g, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    make_geometry(shape, count, g, device)
    phase = (torch.rand(3, generator=g, device=device) * math.pi * 2).cpu().numpy()
    frequency = 2 + seed % 5
    if shape == "torus":
        major, minor = 0.55 + 0.1 * float(first[0]), 0.18 + 0.08 * float(first[1])
        nu, nv = 360, 120
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, nu, endpoint=False), np.linspace(0, 2 * np.pi, nv, endpoint=False), indexing="ij")
        pos = np.stack([(major + minor * np.cos(v)) * np.cos(u), minor * np.sin(v), (major + minor * np.cos(v)) * np.sin(u)], -1)
        nrm = np.stack([np.cos(v) * np.cos(u), np.sin(v), np.cos(v) * np.sin(u)], -1)
        faces = []
        for i in range(nu):
            for j in range(nv):
                a, b, c, d = i * nv + j, ((i + 1) % nu) * nv + j, ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv
                faces += [(a, c, b), (a, d, c)]
        pos, nrm = pos.reshape(-1, 3), nrm.reshape(-1, 3)
    else:
        axes = (0.55 + 0.45 * first).cpu().numpy()
        if shape == "sphere":
            axes[:] = axes.mean()
        npol, naz = 180, 360
        polar = np.linspace(0, np.pi, npol)
        az = np.linspace(0, 2 * np.pi, naz, endpoint=False)
        P, A = np.meshgrid(polar, az, indexing="ij")
        z = np.cos(P)
        r = np.sin(P)
        unit = np.stack([r * np.cos(A), z, r * np.sin(A)], -1)
        bump = 1 + 0.12 * np.sin(A * 3 + z * 5) if shape == "bumpy" else np.ones_like(z)
        pos = unit * axes * bump[..., None]
        nrm = unit / axes
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        faces = []
        for i in range(npol - 1):
            for j in range(naz):
                a, b, c, d = i * naz + j, (i + 1) * naz + j, (i + 1) * naz + (j + 1) % naz, i * naz + (j + 1) % naz
                faces += [(a, b, c), (a, c, d)]
        pos, nrm = pos.reshape(-1, 3), nrm.reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.uint32)
    base = np.clip(0.18 + 0.72 * (0.5 + 0.5 * np.sin(pos[:, :1] * frequency + pos[:, 1:2] * (frequency + 1) + phase.reshape(1, 3))), 0.03, 0.97)
    return pos.astype(np.float32), nrm.astype(np.float32), faces, base.astype(np.float32), (frequency, phase)


def vertex_materials(pos, base, prior, colour_params, scale: float):
    """Gaussian-weighted blend of the per-Gaussian material over the Gaussians near each vertex."""
    frequency, phase = colour_params
    mu = prior["position"]
    base_mu = np.clip(0.18 + 0.72 * (0.5 + 0.5 * np.sin(mu[:, :1] * frequency + mu[:, 1:2] * (frequency + 1) + phase.reshape(1, 3))), 0.03, 0.97)
    w_d = (prior["albedo"] / base_mu).mean(1, keepdims=True)        # per-Gaussian diffuse weight
    attrs = np.concatenate([w_d, prior["specular"], prior["roughness"], prior["metalness"]], 1)
    out = np.zeros((len(pos), attrs.shape[1]), np.float32)
    P = torch.from_numpy(pos).cuda()
    M = torch.from_numpy(mu).float().cuda()
    A = torch.from_numpy(attrs).float().cuda()
    for k in range(0, len(pos), 20000):
        d2 = torch.cdist(P[k:k + 20000], M).square()
        w = torch.exp(-d2 / (2 * scale ** 2)) + 1e-12
        out[k:k + 20000] = ((w @ A) / w.sum(1, keepdim=True)).cpu().numpy()
    w_d, rho_s, rough, metal = out[:, :1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
    albedo = base * w_d
    f0 = 0.04 * (1 - metal) + albedo * metal
    return albedo, rho_s * f0, np.square(rough)


def make_mesh(mi, pos, nrm, faces, bsdf: dict, attrs: dict):
    props = mi.Properties()
    props["bsdf"] = mi.load_dict(bsdf)
    mesh = mi.Mesh("surface", len(pos), len(faces), props=props, has_vertex_normals=True)
    params = mi.traverse(mesh)
    params["vertex_positions"] = pos.ravel()
    params["vertex_normals"] = nrm.ravel()
    params["faces"] = faces.ravel()
    params.update()
    for name, (channels, values) in attrs.items():
        mesh.add_attribute(name, channels, values.ravel())
    return mesh


def save8(img: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)     # truncation, as the benchmark generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, nargs="+", default=[70, 71, 72, 73])
    parser.add_argument("--spp", type=int, default=256)
    parser.add_argument("--out", default="data/world_benchmark_mitsuba")
    parser.add_argument("--views", type=int, nargs="*", default=None, help="only these views (test renders)")
    args = parser.parse_args()
    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    T = mi.ScalarTransform4f
    for n in args.scenes:
        src = ROOT / f"data/world_benchmark_v2/scenes/test/scene_00{n}"
        dst = ROOT / args.out / "scenes" / "test" / f"scene_00{n}"
        (dst / "images").mkdir(parents=True, exist_ok=True)
        (dst / "diffuse").mkdir(parents=True, exist_ok=True)
        for f in ("transforms.json", "gaussian_brdf_gt.npz", "points.npy", "point_colors.npy"):
            shutil.copy2(src / f, dst / f)
        tr = json.loads((src / "transforms.json").read_text(encoding="utf-8"))
        prior = dict(np.load(src / "gaussian_brdf_gt.npz"))
        shape = tr["shape"]
        pos, nrm, faces, base, cparams = surface(n, shape, len(prior["position"]), torch.device("cuda"))
        albedo, specref, alpha = vertex_materials(pos, base, prior, cparams, 0.065 if shape == "torus" else 0.055)
        diffuse_mesh = make_mesh(mi, pos, nrm, faces, {"type": "diffuse", "reflectance": {"type": "mesh_attribute", "name": "vertex_albedo"}},
                                 {"vertex_albedo": (3, albedo)})
        specular_mesh = make_mesh(mi, pos, nrm, faces, {"type": "roughconductor", "distribution": "ggx", "material": "none",
                                                         "alpha": {"type": "mesh_attribute", "name": "vertex_alpha"},
                                                         "specular_reflectance": {"type": "mesh_attribute", "name": "vertex_spec"}},
                                  {"vertex_alpha": (1, alpha), "vertex_spec": (3, specref)})
        fx, w, h = tr["fl_x"], int(tr["w"]), int(tr["h"])
        fov = math.degrees(2 * math.atan(w / (2 * fx)))
        for k, frame in enumerate(tr["frames"]):
            if args.views is not None and k not in args.views:
                continue
            c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)          # NeRF / OpenGL convention: +y up, -z forward
            centre, up, forward = c2w[:3, 3], c2w[:3, 1], -c2w[:3, 2]
            sensor = {"type": "perspective", "fov": fov, "fov_axis": "x",
                      "to_world": T().look_at(origin=centre.tolist(), target=(centre + forward).tolist(), up=up.tolist()),
                      "film": {"type": "hdrfilm", "width": w, "height": h, "rfilter": {"type": "box"}, "pixel_format": "rgb"},
                      "sampler": {"type": "independent", "sample_count": args.spp}}
            irradiance = (np.asarray(frame["light_color"]) * frame["light_intensity"]).tolist()
            sun = {"type": "directional", "direction": (-np.asarray(frame["light_direction"])).tolist(),
                   "irradiance": {"type": "rgb", "value": irradiance}}
            ambient = {"type": "constant", "radiance": {"type": "rgb", "value": (AMBIENT * np.asarray(frame["env_sh"][0]) / math.pi).tolist()}}
            integrator = {"type": "direct", "hide_emitters": True}
            d = np.asarray(mi.render(mi.load_dict({"type": "scene", "integrator": integrator, "sensor": sensor, "sun": sun, "ambient": ambient,
                                                   "shape": diffuse_mesh})))[..., :3]
            s = np.asarray(mi.render(mi.load_dict({"type": "scene", "integrator": integrator, "sensor": sensor, "sun": sun,
                                                   "shape": specular_mesh})))[..., :3]
            save8(d + s, dst / frame["file_path"])
            save8(d, dst / frame["diffuse_path"])
            print(f"scene_00{n} view {k}: diffuse max {d.max():.3f}, specular max {s.max():.3f}", flush=True)


if __name__ == "__main__":
    main()
