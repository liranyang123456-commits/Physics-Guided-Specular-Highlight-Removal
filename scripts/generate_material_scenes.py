"""Material-controlled scenes for the strict-holdout protocol.

Each base scene (shape, Gaussian geometry, albedo pattern, cameras, lights) is
rendered once per material family.  The same uniform draws are mapped to four
families inside the material ranges of world_benchmark_v2 (roughness
[0.04, 0.28], metalness [0.15, 0.90]), so only roughness and metalness differ
between the renders of a base scene; specular weight and absorption keep the
benchmark distributions.

    python scripts/generate_material_scenes.py            # write the scenes
    python scripts/generate_material_scenes.py --verify   # re-render scene_0070
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_world_benchmark import (  # noqa: E402
    make_scene,
    save_gray,
    save_rgb,
    scene_arrays,
    view_camera_and_light,
)
from mvbrdf_shr.world.renderer import GaussianBRDFRenderer  # noqa: E402

# (low, span) ranges, all inside the world_benchmark_v2 ranges.  Roughness
# below ~0.10 produces almost no visible highlight with 384 per-primitive-shaded
# Gaussians (input-to-diffuse PSNR > 50 dB), so the low-roughness family starts
# at 0.10.
FAMILIES = {
    "lowrough_lowmetal": ((0.10, 0.06), (0.15, 0.20)),
    "lowrough_highmetal": ((0.10, 0.06), (0.70, 0.20)),
    "highrough_lowmetal": ((0.20, 0.08), (0.15, 0.20)),
    "highrough_highmetal": ((0.20, 0.08), (0.70, 0.20)),
}
# Seeds follow the benchmark rule seed = 2026 + 101 * index for unused indices
# 90 and 93; shape = seed % 4 gives a sphere and a torus.
BASES = {"sphere": 2026 + 101 * 90, "torus": 2026 + 101 * 93}
VIEWS, SIZE, GAUSSIANS = 12, 128, 384


def render_scene(scene, seed: int, scene_dir: Path, shape: str, renderer, device) -> list[float]:
    focal = SIZE * 0.9
    K = torch.tensor([[focal, 0, SIZE / 2], [0, focal, SIZE / 2], [0, 0, 1]], device=device)
    convert = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=device))
    for kind in ("images", "diffuse", "specular", "mask", "normal", "depth", "albedo"):
        (scene_dir / kind).mkdir(parents=True, exist_ok=True)
    arrays = scene_arrays(scene)
    np.savez(scene_dir / "gaussian_brdf_gt.npz", **arrays)
    np.save(scene_dir / "points.npy", arrays["position"])
    np.save(scene_dir / "point_colors.npy", arrays["albedo"])
    transforms = {"fl_x": focal, "fl_y": focal, "cx": SIZE / 2, "cy": SIZE / 2, "w": SIZE, "h": SIZE,
                  "shape": shape, "frames": []}
    input_psnr = []
    for view in range(VIEWS):
        camera, light, c2w = view_camera_and_light(seed, view, VIEWS, SIZE, K, device)
        with torch.no_grad():
            rendered = renderer(scene, camera, light=light)
        save_rgb(rendered.full, scene_dir / "images" / f"{view:03d}.png")
        save_rgb(rendered.diffuse, scene_dir / "diffuse" / f"{view:03d}.png")
        save_rgb(rendered.specular, scene_dir / "specular" / f"{view:03d}.png")
        save_gray(rendered.mask, scene_dir / "mask" / f"{view:03d}.png")
        save_rgb((rendered.normal + 1) * 0.5, scene_dir / "normal" / f"{view:03d}.png")
        valid = rendered.depth[rendered.alpha > 1e-3]
        depth = ((rendered.depth - valid.min()) / (valid.max() - valid.min() + 1e-6)
                 if len(valid) else torch.zeros_like(rendered.depth))
        save_gray(depth * (rendered.alpha > 1e-3), scene_dir / "depth" / f"{view:03d}.png")
        save_rgb(rendered.albedo, scene_dir / "albedo" / f"{view:03d}.png")
        mse = (rendered.full - rendered.diffuse).square().mean().clamp(min=1e-12)
        input_psnr.append(float(20 * torch.log10(1 / mse.sqrt())))
        transforms["frames"].append({
            "file_path": f"images/{view:03d}.png",
            "diffuse_path": f"diffuse/{view:03d}.png",
            "transform_matrix": (c2w @ convert).detach().cpu().tolist(),
            "light_position": light.position.detach().cpu().tolist(),
            "light_direction": light.direction.detach().cpu().tolist(),
            "light_color": light.color.detach().cpu().tolist(),
            "point_light_weight": float(light.point_weight),
            "light_intensity": float(light.intensity),
            "env_sh": light.env_sh.detach().cpu().tolist(),
        })
    (scene_dir / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    return input_psnr


def verify(renderer, device) -> None:
    seed = 2026 + 101 * 70
    scene, shape = make_scene(seed, GAUSSIANS, device)
    out = ROOT / "outputs" / "material_study" / "_verify_scene_0070"
    render_scene(scene, seed, out, shape, renderer, device)
    ref = ROOT / "data" / "world_benchmark_v2" / "scenes" / "test" / "scene_0070"
    worst = 0
    for kind in ("images", "diffuse"):
        for view in range(VIEWS):
            a = np.asarray(Image.open(out / kind / f"{view:03d}.png"), dtype=int)
            b = np.asarray(Image.open(ref / kind / f"{view:03d}.png"), dtype=int)
            worst = max(worst, int(np.abs(a - b).max()))
    print(f"scene_0070 re-render ({shape}): max abs pixel difference {worst} (8-bit levels)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/world_benchmark_material")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    renderer = GaussianBRDFRenderer(backend="gsplat")
    if args.verify:
        verify(renderer, device)
        return
    manifest = {"families": {k: {"roughness": [lo, lo + sp], "metalness": [mlo, mlo + msp]}
                             for k, ((lo, sp), (mlo, msp)) in FAMILIES.items()},
                "bases": BASES, "scenes": []}
    for base, seed in BASES.items():
        for family, (rough, metal) in FAMILIES.items():
            scene, shape = make_scene(seed, GAUSSIANS, device, roughness_range=rough, metalness_range=metal)
            assert shape == base, (shape, base)
            name = f"{base}_{family}"
            psnr = render_scene(scene, seed, ROOT / args.out / "scenes" / "test" / name, shape, renderer, device)
            manifest["scenes"].append({"name": name, "shape": shape, "family": family, "seed": seed,
                                       "input_psnr_mean": float(np.mean(psnr))})
            print(f"{name}: mean input-to-diffuse PSNR {np.mean(psnr):.2f} dB", flush=True)
    (ROOT / args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
