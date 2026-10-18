"""Benchmark scenes rendered with reflectance models outside the model of the method (material-model mismatch).

The geometry, the per-Gaussian materials, the cameras, and the lights are those of world_benchmark_v2; only the per-Gaussian
shading of the renderer is replaced, so the test isolates the reflectance model from the renderer (the renderer mismatch is
tested separately with Mitsuba 3, scripts/make_mismatch_scenes.py).  Variants:

  control    Lambertian diffuse and one isotropic GGX lobe, as in the method (checks the generation path)
  aniso      anisotropic GGX lobe, alpha_x = 2 alpha, alpha_y = alpha / 2, tangent along the projected x axis
  clearcoat  a second, sharp GGX lobe (roughness 0.15, F0 = 0.04, weight 1) on top of the base layer
  sheen      a Charlie sheen lobe (roughness 0.5, white, weight 0.06) at grazing angles, as on fabric
  wrap       wrapped diffuse lighting, max(0, (n.l + 0.5) / 1.5), a common stand-in for subsurface scattering
  oren       Oren-Nayar diffuse with roughness sigma = 0.5 rad, as on matte rough surfaces

The diffuse target of every frame is the diffuse lobe of the variant (for wrap and oren, the non-Lambertian diffuse itself);
images are stored as 8-bit truncations, as in the benchmark.  Output: data/world_benchmark_materials/<variant>/scenes/test/.

    python scripts/make_material_mismatch_scenes.py --variants control aniso clearcoat sheen wrap oren --scenes 70 71 72 73
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import render_mitsuba_scenes as R  # noqa: E402
from mvbrdf_shr.models.envlight import sh_to_irradiance  # noqa: E402
from mvbrdf_shr.world import renderer as wr  # noqa: E402

OUT = ROOT / "data" / "world_benchmark_materials"
VARIANTS = ("control", "aniso", "clearcoat", "sheen", "wrap", "oren")


def make_shader(variant: str):
    """Per-Gaussian shading with the signature of GaussianBRDFRenderer._shade_sh."""

    def shade(self, scene, materials, camera, light):
        n = scene.normals
        view = F.normalize(camera.center.view(1, 3) - scene.means, dim=-1)
        directional = F.normalize(light.direction, dim=-1).view(1, 3).expand_as(n)
        point_vector = light.position.view(1, 3) - scene.means
        point_distance = point_vector.norm(dim=-1, keepdim=True).clamp(min=0.1)
        point_weight = light.point_weight.view(1, 1)
        direction = F.normalize(directional * (1 - point_weight) + (point_vector / point_distance) * point_weight, dim=-1)
        cos_l = (n * direction).sum(-1, keepdim=True)
        n_dot_l = cos_l.clamp(min=0)
        n_dot_v = (n * view).sum(-1, keepdim=True).clamp(min=1e-4)
        irradiance = sh_to_irradiance(n.T.unsqueeze(0).unsqueeze(-1), light.env_sh.unsqueeze(0))[0, :, :, 0].T
        attenuation = (1 - point_weight) + point_weight / point_distance.square()
        radiance = light.color.view(1, 3) * light.intensity.view(1, 1) * attenuation
        direct = radiance * n_dot_l

        # diffuse lobe
        if variant == "wrap":
            diffuse = materials.albedo / wr._PI * (irradiance + radiance * ((cos_l + 0.5) / 1.5).clamp(min=0))
        elif variant == "oren":
            s2 = 0.5 ** 2
            a, b = 1 - 0.5 * s2 / (s2 + 0.33), 0.45 * s2 / (s2 + 0.09)
            theta_i, theta_o = torch.acos(n_dot_l.clamp(0, 1)), torch.acos(n_dot_v.clamp(0, 1))
            lt, vt = direction - cos_l * n, view - (n * view).sum(-1, keepdim=True) * n
            cos_phi = ((lt * vt).sum(-1, keepdim=True) / (lt.norm(dim=-1, keepdim=True) * vt.norm(dim=-1, keepdim=True) + 1e-6)).clamp(min=0)
            alpha_, beta_ = torch.maximum(theta_i, theta_o), torch.minimum(theta_i, theta_o)
            on = a + b * cos_phi * torch.sin(alpha_) * torch.tan(beta_).clamp(max=10.0)
            diffuse = materials.albedo / wr._PI * (irradiance + direct * on)
        else:
            diffuse = materials.albedo / wr._PI * (irradiance + direct)

        # base specular lobe
        half = F.normalize(direction + view, dim=-1)
        n_dot_h = (n * half).sum(-1, keepdim=True).clamp(min=0)
        h_dot_v = (half * view).sum(-1, keepdim=True).clamp(min=0)
        alpha = materials.roughness.square()
        if variant == "aniso":
            axis = torch.zeros_like(n)
            axis[:, 0] = 1
            alt = torch.zeros_like(n)
            alt[:, 1] = 1
            axis = torch.where((n * axis).sum(-1, keepdim=True).abs() > 0.95, alt, axis)
            t = F.normalize(axis - (n * axis).sum(-1, keepdim=True) * n, dim=-1)
            bt = torch.cross(n, t, dim=-1)
            ax, ay = alpha * 2.0, alpha / 2.0
            d_term = 1.0 / (wr._PI * ax * ay * (((half * t).sum(-1, keepdim=True) / ax).square()
                                                  + ((half * bt).sum(-1, keepdim=True) / ay).square() + n_dot_h.square()).square() + wr._EPS)
        else:
            d_term = wr._ggx_d(n_dot_h, alpha)
        g_term = wr._smith_g1(n_dot_l, alpha) * wr._smith_g1(n_dot_v, alpha)
        f0 = 0.04 * (1 - materials.metalness) + materials.albedo * materials.metalness
        fresnel = wr._schlick_fresnel(h_dot_v, f0)
        specular = materials.specular * d_term * g_term * fresnel / (4 * n_dot_l * n_dot_v + wr._EPS) * direct

        if variant == "clearcoat":
            a_cc = torch.full_like(alpha, 0.15 ** 2)
            f_cc = wr._schlick_fresnel(h_dot_v, torch.full_like(alpha, 0.04))
            g_cc = wr._smith_g1(n_dot_l, a_cc) * wr._smith_g1(n_dot_v, a_cc)
            specular = specular + 1.0 * wr._ggx_d(n_dot_h, a_cc) * g_cc * f_cc / (4 * n_dot_l * n_dot_v + wr._EPS) * direct
        elif variant == "sheen":
            a_s = 0.5
            sin2 = (1 - n_dot_h.square()).clamp(min=0)
            d_sheen = (2 + 1 / a_s) * sin2.pow(0.5 / a_s) / (2 * wr._PI)
            v_sheen = 1.0 / (4 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v) + wr._EPS)
            specular = specular + 0.06 * d_sheen * v_sheen * direct
        return diffuse, specular

    return shade


@torch.no_grad()
def render_variant(n: int, variant: str, device) -> Path:
    from mvbrdf_shr.world.train import build_pipeline, load_dataset
    src = ROOT / f"data/world_benchmark_v2/scenes/test/scene_00{n}"
    dst = OUT / variant / "scenes" / "test" / f"scene_00{n}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.pt"))     # placeholders, cameras, lights, and the exact field
    cfg = yaml.safe_load((ROOT / f"configs/world/v2/benchmark_scene_00{n}_v2.yaml").read_text(encoding="utf-8"))
    cfg.update(data_root=str(dst.relative_to(ROOT)).replace("\\", "/"), gaussian_init="oracle", raster_backend="gsplat",
               n_gaussians=int(len(np.load(dst / "gaussian_brdf_gt.npz")["position"])))
    dataset = load_dataset(cfg)
    ids = list(range(len(dataset)))
    pipeline = build_pipeline(dataset, cfg, device, ids)
    pipeline.eval()
    pipeline.renderer._shade_sh = types.MethodType(make_shader(variant), pipeline.renderer)
    tr = json.loads((dst / "transforms.json").read_text(encoding="utf-8"))
    for fid in ids:
        frame = dataset[fid]
        out = pipeline(frame.camera.to(device), fid, image=frame.image.to(device), refine=False)["render"]
        R.save8(out.full.permute(1, 2, 0).cpu().numpy(), dst / tr["frames"][fid]["file_path"])
        R.save8(out.diffuse.permute(1, 2, 0).cpu().numpy(), dst / tr["frames"][fid]["diffuse_path"])
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--scenes", type=int, nargs="+", default=list(range(70, 80)))
    args = parser.parse_args()
    device = torch.device("cuda")
    for variant in args.variants:
        for n in args.scenes:
            dst = render_variant(n, variant, device)
            print(f"{variant} scene_00{n} -> {dst.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
