"""Write the configurations of the corrected physical-branch recipe ("v2") for the ten benchmark scenes.

Recipe (the only differences to benchmark_scene_0070_inverse.yaml are the loss weights and the optimizer settings):
  * oracle geometry and calibrated lighting, albedo and specular BRDF unknown (flat albedo 0.25),
  * RGB-only: photometric L1 (+ the small energy / smoothness / texture regularizers); no diffuse, specular, or refiner
    supervision, so no ground-truth diffuse image is used in training,
  * no SSIM photometric term: with weight 0.2 it stops the optimization (see outputs/inverse_diag/),
  * Adam, learning rate 0.03, 3 views per step.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def v2_config(scene: int, steps: int, lr: float, out_root: str) -> dict:
    cfg = yaml.safe_load((ROOT / "configs/world/benchmark_scene_0070_inverse.yaml").read_text(encoding="utf-8"))
    data_root = f"data/world_benchmark_v2/scenes/test/scene_00{scene}"
    prior = ROOT / data_root / cfg.get("gaussian_prior", "gaussian_brdf_gt.npz")
    cfg["data_root"] = data_root
    cfg["n_gaussians"] = int(len(np.load(prior)["position"]))
    cfg.update(steps=steps, scene_lr=lr, views_per_step=3, use_refiner=False, raster_backend="gsplat",
               output_dir=f"{out_root}/world_scene_00{scene}")
    cfg["loss"] = dict(photo_l1=1.0, photo_ssim=0.0, diffuse_gt=0.0, refine_gt=0.0, specular_gt=0.0, energy=0.1,
                       material_smooth=0.01, normal_axis=0.0, scale=0.0, opacity_sparse=0.0, texture_consistency=0.01)
    cfg["log_every"] = 100
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--out-root", default="outputs/v2")
    parser.add_argument("--config-dir", default="configs/world/v2")
    parser.add_argument("--scenes", type=int, nargs="*", default=list(range(70, 80)))
    args = parser.parse_args()
    out = ROOT / args.config_dir
    out.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        path = out / f"benchmark_scene_00{scene}_v2.yaml"
        path.write_text(yaml.safe_dump(v2_config(scene, args.steps, args.lr, args.out_root), sort_keys=False), encoding="utf-8")
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
