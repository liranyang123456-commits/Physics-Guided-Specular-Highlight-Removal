"""Re-run the DTU and BlendedMVS diagnostic scenes (Section 4.7, Table tab:nvs) with the two loader/scale bugs fixed:

1. DTU (configs/world/dtu_scan1_nossim.yaml): world coordinates are in millimetres (initial_scale = 3.72 mm), so
   GaussianBRDFField.scales now clamps at max(1.0, initial_scale * 15.0) instead of 1.0 mm, loss.scale is 0.0 (not penalizing
   millimetre scales), and background_color is None instead of forced white [1, 1, 1] on dark views.
2. BlendedMVS (configs/world/blendedmvs_57f8d9bb_nossim.yaml): MultiViewScene.from_mvsnet now scales camera.K by
   width_d / camera.width instead of width_d / original_width = 1.0 when unprojecting depth maps.

All other settings (10,000 Gaussians, 1,500 steps, learning rates, train/holdout split, and mvbrdf_shr.world.train.train) are
unchanged.  Writes outputs/dtu_scan1_corrected and outputs/blendedmvs_57f8d9bb_corrected.

    python scripts/run_mvs_corrected.py --scene both
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mvbrdf_shr.world.evaluate import evaluate  # noqa: E402
from mvbrdf_shr.world.train import train  # noqa: E402


def run_scene(cfg_rel: str, out_rel: str, is_dtu: bool, steps: int = 0, per_frame_lighting: bool = False, points: int = 0) -> None:
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    cfg = yaml.safe_load((ROOT / cfg_rel).read_text(encoding="utf-8"))
    if per_frame_lighting:
        cfg["shared_lighting"] = False   # real captures vary per frame; the benchmark also uses per-frame lighting
        out_rel += "_pf"
    if points:
        cfg["max_init_points"] = points
        cfg["n_gaussians"] = points
        out_rel += f"_{points // 1000}k"
    if steps:
        cfg["steps"] = steps
        out_rel += f"_s{steps}"
    cfg["output_dir"] = out_rel
    if is_dtu:
        cfg["background_color"] = None
        cfg["loss"] = dict(cfg.get("loss", {}), scale=0.0)
    ckpt = train(cfg)
    for split in ("train", "holdout", "all"):
        evaluate(str(ckpt), split=split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=("dtu", "blended", "both"), default="both")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--per-frame-lighting", action="store_true")
    parser.add_argument("--points", type=int, default=0)
    args = parser.parse_args()
    if args.scene in ("dtu", "both"):
        run_scene("configs/world/dtu_scan1_nossim.yaml", "outputs/dtu_scan1_corrected", True, args.steps, args.per_frame_lighting, args.points)
    if args.scene in ("blended", "both"):
        run_scene("configs/world/blendedmvs_57f8d9bb_nossim.yaml", "outputs/blendedmvs_57f8d9bb_corrected", False, args.steps, args.per_frame_lighting, args.points)


if __name__ == "__main__":
    main()
