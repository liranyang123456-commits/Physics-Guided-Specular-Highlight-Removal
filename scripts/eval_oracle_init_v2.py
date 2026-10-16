"""Held-out accuracy of the scene_0070 exact initialization before any optimization, with the v2 fusion gate.

Counterpart of eval_oracle_init.py (revision, step A): the scene state (oracle geometry/albedo from
gaussian_brdf_gt.npz, calibrated lighting) is identical, but the fused output uses the gate trained on the v2 world field
(outputs/v2/fusion_scene_0070/fusion.pt).  The physical and single-image numbers do not depend on the optimization recipe.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_sensitivity_v2 import evaluate  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402


def main() -> None:
    os.chdir(ROOT)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = torch.load("outputs/v2/world_scene_0070/last.pt", map_location="cpu", weights_only=False)
    cfg = dict(reference["config"])
    cfg["raster_backend"] = "gsplat"
    cfg["gaussian_init"] = "geometry_albedo"
    dataset = load_dataset(cfg)
    pipeline = build_pipeline(dataset, cfg, device, list(reference["train_ids"]))
    summary = evaluate(pipeline, dataset, list(reference["holdout_ids"]), device)
    summary["note"] = "exact initialization, 0 optimization steps, v2 fusion gate"
    out = ROOT / "outputs" / "v2" / "sensitivity" / "oracle_init.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key in ("phys", "fuse", "single"):
        print(key, round(summary[key]["psnr"], 2), round(summary[key]["ssim"], 4),
              [round(v["psnr"], 2) for v in summary[key]["per_view"]])
    print("gate", round(summary["gate_mean"], 3))


if __name__ == "__main__":
    main()
