"""Renderer-mismatch scenes: one scene rendered by the renderer that the method fits and by an independent renderer (Mitsuba 3).

For each benchmark scene, the analytic surface of scripts/generate_world_benchmark.py is triangulated
(scripts/render_mitsuba_scenes.py) and the per-Gaussian materials of the scene are carried to the mesh vertices.  The given
geometry is then a set of thin disc Gaussians sampled on the mesh (6000 by default, oriented by the surface normal), so that it
describes the surface that Mitsuba ray traces.  The materials of the discs are interpolated from the vertices; Mitsuba uses the
same vertex materials.  Two datasets share cameras, lights (transforms.json of the original scene), and the disc prior:
  data/world_benchmark_mismatch/ours/...     rendered by our Gaussian renderer (control: no renderer mismatch)
  data/world_benchmark_mismatch/mitsuba/...  ray traced by Mitsuba (diffuse pass and specular pass, see render_mitsuba_scenes.py)
Both are truncated to 8 bits like the benchmark.

    python scripts/make_mismatch_scenes.py --scenes 70 71 72 73 --spp 256
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import render_mitsuba_scenes as R  # noqa: E402

OUT = ROOT / "data" / "world_benchmark_mismatch"


def vertex_budget(pos, base, prior, colour_params, scale):
    """Gaussian-weighted blend of the per-Gaussian diffuse weight, specular weight, absorption, roughness, metalness."""
    frequency, phase = colour_params
    mu = prior["position"]
    base_mu = np.clip(0.18 + 0.72 * (0.5 + 0.5 * np.sin(mu[:, :1] * frequency + mu[:, 1:2] * (frequency + 1) + phase.reshape(1, 3))), 0.03, 0.97)
    w_d = (prior["albedo"] / base_mu).mean(1, keepdims=True)
    attrs = np.concatenate([w_d, prior["specular"], prior["absorption"], prior["roughness"], prior["metalness"]], 1)
    P, M, A = torch.from_numpy(pos).cuda(), torch.from_numpy(mu).float().cuda(), torch.from_numpy(attrs).float().cuda()
    out = np.zeros((len(pos), attrs.shape[1]), np.float32)
    for k in range(0, len(pos), 20000):
        w = torch.exp(-torch.cdist(P[k:k + 20000], M).square() / (2 * scale ** 2)) + 1e-12
        out[k:k + 20000] = ((w @ A) / w.sum(1, keepdim=True)).cpu().numpy()
    budget = out[:, :3] / out[:, :3].sum(1, keepdims=True)
    return {"base": base, "w_d": budget[:, :1], "rho_s": budget[:, 1:2], "absorption": budget[:, 2:3],
            "roughness": out[:, 3:4], "metalness": out[:, 4:5]}


def sample_discs(pos, nrm, faces, attrs: dict, count: int, rng: np.random.Generator):
    a, b, c = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    f = rng.choice(len(faces), size=count, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random((count, 1))), rng.random((count, 1))
    wa, wb, wc = 1 - r1, r1 * (1 - r2), r1 * r2
    fa, fb, fc = faces[f, 0], faces[f, 1], faces[f, 2]
    mix = lambda x: wa * x[fa] + wb * x[fb] + wc * x[fc]
    p, n = mix(pos), mix(nrm)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    ref = np.where(np.abs(n[:, 1:2]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    Rm = np.stack([t1, t2, n], axis=2)                                    # columns: tangent, bitangent, normal
    q = torch.from_numpy(matrix_to_quaternion(Rm)).float()
    s = 1.0 * math.sqrt(area.sum() / count)
    return {"position": p.astype(np.float32), "normal": n.astype(np.float32), "quaternion": q.numpy(),
            "scale": np.tile(np.array([[s, s, 0.1 * s]], np.float32), (count, 1)),
            "opacity": np.full((count, 1), 0.95, np.float32), **{k: mix(v).astype(np.float32) for k, v in attrs.items()}}


def matrix_to_quaternion(Rm: np.ndarray) -> np.ndarray:
    """(N,3,3) rotation matrices to (w,x,y,z) quaternions (the convention of mvbrdf_shr/world/scene.py)."""
    m = Rm
    w = np.sqrt(np.clip(1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2], 1e-12, None)) / 2
    x = np.sqrt(np.clip(1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], 0, None)) / 2
    y = np.sqrt(np.clip(1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], 0, None)) / 2
    z = np.sqrt(np.clip(1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2], 0, None)) / 2
    x = np.copysign(x, m[:, 2, 1] - m[:, 1, 2])
    y = np.copysign(y, m[:, 0, 2] - m[:, 2, 0])
    z = np.copysign(z, m[:, 1, 0] - m[:, 0, 1])
    q = np.stack([w, x, y, z], 1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def write_scene(dst: Path, src: Path, prior: dict) -> None:
    for d in ("images", "diffuse"):
        (dst / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "transforms.json", dst / "transforms.json")
    albedo = prior["base"] * prior["w_d"]
    arrays = {"position": prior["position"], "scale": prior["scale"], "quaternion": prior["quaternion"], "opacity": prior["opacity"],
              "normal": prior["normal"], "albedo": albedo, "specular": prior["rho_s"], "roughness": prior["roughness"],
              "metalness": prior["metalness"], "absorption": prior["absorption"]}
    np.savez(dst / "gaussian_brdf_gt.npz", **arrays)
    np.save(dst / "points.npy", prior["position"])
    np.save(dst / "point_colors.npy", albedo)


@torch.no_grad()
def render_ours(n: int, dst: Path, device) -> None:
    """Our renderer on the disc scene with its exact materials (mode 'oracle'), cameras and lights of transforms.json."""
    from mvbrdf_shr.world.train import build_pipeline, load_dataset
    cfg = yaml.safe_load((ROOT / f"configs/world/v2/benchmark_scene_00{n}_v2.yaml").read_text(encoding="utf-8"))
    cfg.update(data_root=str(dst.relative_to(ROOT)).replace("\\", "/"), gaussian_init="oracle", raster_backend="gsplat",
               n_gaussians=int(len(np.load(dst / "gaussian_brdf_gt.npz")["position"])))
    dataset = load_dataset(cfg)
    ids = list(range(len(dataset)))
    pipeline = build_pipeline(dataset, cfg, device, ids)
    pipeline.eval()
    tr = json.loads((dst / "transforms.json").read_text(encoding="utf-8"))
    for fid in ids:
        frame = dataset[fid]
        out = pipeline(frame.camera.to(device), fid, image=frame.image.to(device), refine=False)["render"]
        R.save8(out.full.permute(1, 2, 0).cpu().numpy(), dst / tr["frames"][fid]["file_path"])
        R.save8(out.diffuse.permute(1, 2, 0).cpu().numpy(), dst / tr["frames"][fid]["diffuse_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, nargs="+", default=[70, 71, 72, 73])
    parser.add_argument("--discs", type=int, default=6000)
    parser.add_argument("--spp", type=int, default=256)
    parser.add_argument("--alpha-power", type=float, default=2.0,
                        help="GGX alpha = roughness ** power in Mitsuba (2: as our renderer; 1: broader highlights, diagnostic)")
    parser.add_argument("--variant", default="mitsuba", help="output folder of the Mitsuba renders")
    parser.add_argument("--skip-ours", action="store_true", help="do not render the control with our renderer")
    parser.add_argument("--smoothing", type=float, default=0.0,
                        help="blend radius of the per-Gaussian materials on the mesh (0: the Gaussian scale of the scene); a larger radius "
                             "gives material fields that the discs can represent")
    parser.add_argument("--ours-variant", default="ours", help="output folder of the control rendered by our renderer")
    args = parser.parse_args()
    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    T = mi.ScalarTransform4f
    device = torch.device("cuda")
    for n in args.scenes:
        src = ROOT / f"data/world_benchmark_v2/scenes/test/scene_00{n}"
        tr = json.loads((src / "transforms.json").read_text(encoding="utf-8"))
        prior0 = dict(np.load(src / "gaussian_brdf_gt.npz"))
        shape = tr["shape"]
        pos, nrm, faces, base, cparams = R.surface(n, shape, len(prior0["position"]), device)
        attrs = vertex_budget(pos, base, prior0, cparams, args.smoothing or (0.065 if shape == "torus" else 0.055))
        discs = sample_discs(pos, nrm, faces, attrs, args.discs, np.random.default_rng(n))
        dst_m, dst_o = OUT / args.variant / "scenes" / "test" / f"scene_00{n}", OUT / args.ours_variant / "scenes" / "test" / f"scene_00{n}"
        write_scene(dst_m, src, discs)
        if not args.skip_ours:
            write_scene(dst_o, src, discs)
        # Mitsuba: vertex materials of the mesh
        albedo = attrs["base"] * attrs["w_d"]
        f0 = 0.04 * (1 - attrs["metalness"]) + albedo * attrs["metalness"]
        diffuse_mesh = R.make_mesh(mi, pos, nrm, faces, {"type": "diffuse", "reflectance": {"type": "mesh_attribute", "name": "vertex_albedo"}},
                                   {"vertex_albedo": (3, albedo)})
        specular_mesh = R.make_mesh(mi, pos, nrm, faces, {"type": "roughconductor", "distribution": "ggx", "material": "none",
                                                           "alpha": {"type": "mesh_attribute", "name": "vertex_alpha"},
                                                           "specular_reflectance": {"type": "mesh_attribute", "name": "vertex_spec"}},
                                    {"vertex_alpha": (1, np.power(attrs["roughness"], args.alpha_power)), "vertex_spec": (3, attrs["rho_s"] * f0)})
        fov = math.degrees(2 * math.atan(tr["w"] / (2 * tr["fl_x"])))
        for k, frame in enumerate(tr["frames"]):
            c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)          # NeRF / OpenGL convention: +y up, -z forward
            centre, up, forward = c2w[:3, 3], c2w[:3, 1], -c2w[:3, 2]
            sensor = {"type": "perspective", "fov": fov, "fov_axis": "x",
                      "to_world": T().look_at(origin=centre.tolist(), target=(centre + forward).tolist(), up=up.tolist()),
                      "film": {"type": "hdrfilm", "width": int(tr["w"]), "height": int(tr["h"]), "rfilter": {"type": "box"}, "pixel_format": "rgb"},
                      "sampler": {"type": "independent", "sample_count": args.spp}}
            sun = {"type": "directional", "direction": (-np.asarray(frame["light_direction"])).tolist(),
                   "irradiance": {"type": "rgb", "value": (np.asarray(frame["light_color"]) * frame["light_intensity"]).tolist()}}
            ambient = {"type": "constant", "radiance": {"type": "rgb", "value": (R.AMBIENT * np.asarray(frame["env_sh"][0]) / math.pi).tolist()}}
            integ = {"type": "direct", "hide_emitters": True}
            # one light per pass: with two lights the direct integrator samples one of them at random, which leaves the
            # directional light out of half of the samples
            d = (np.asarray(mi.render(mi.load_dict({"type": "scene", "integrator": integ, "sensor": sensor, "sun": sun, "shape": diffuse_mesh})))[..., :3]
                 + np.asarray(mi.render(mi.load_dict({"type": "scene", "integrator": integ, "sensor": sensor, "ambient": ambient,
                                                      "shape": diffuse_mesh})))[..., :3])
            s = np.asarray(mi.render(mi.load_dict({"type": "scene", "integrator": integ, "sensor": sensor, "sun": sun,
                                                   "shape": specular_mesh})))[..., :3]
            R.save8(d + s, dst_m / frame["file_path"])
            R.save8(d, dst_m / frame["diffuse_path"])
            if not args.skip_ours:
                shutil.copy2(dst_m / frame["file_path"], dst_o / frame["file_path"])      # placeholders, overwritten below
                shutil.copy2(dst_m / frame["diffuse_path"], dst_o / frame["diffuse_path"])
        if not args.skip_ours:
            render_ours(n, dst_o, device)
        print(f"scene_00{n} ({shape}): {args.discs} discs, Mitsuba and our renderer written", flush=True)


if __name__ == "__main__":
    main()
