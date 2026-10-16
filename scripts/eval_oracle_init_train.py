"""Training-view accuracy of the scene_0070 oracle initialization before any optimization.

Companion of eval_oracle_init.py (which reports the held-out views).  It builds the
same strict-holdout pipeline (oracle geometry/albedo from gaussian_brdf_gt.npz,
calibrated lighting), skips training, and evaluates the nine training views with
the metrics of mvbrdf_shr/world/evaluate.py: full-radiance PSNR/SSIM against the
input image and diffuse PSNR/SSIM against the diffuse ground truth.  The values
before optimization are compared in the text with outputs/.../world_metrics_train.json
(after optimization).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mvbrdf_shr.metrics import psnr_torch, ssim_torch  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402


def main() -> None:
    os.chdir(ROOT)
    torch.manual_seed(0)
    device = torch.device("cuda")
    reference = torch.load("outputs/world_benchmark_v2_scene_0070_strict/last.pt", map_location="cpu", weights_only=False)
    cfg = dict(reference["config"])
    cfg["raster_backend"] = "auto"
    dataset = load_dataset(cfg)
    train_ids = list(reference["train_ids"])
    pipeline = build_pipeline(dataset, cfg, device, train_ids)
    pipeline.eval()
    rows = []
    with torch.no_grad():
        for frame_id in train_ids:
            frame = dataset[frame_id]
            image = frame.image.to(device)
            target = frame.diffuse_gt.to(device)
            render = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)["render"]
            rows.append({
                "frame": frame_id,
                "full_psnr": float(psnr_torch(render.full, image)),
                "full_ssim": float(ssim_torch(render.full, image)),
                "diffuse_psnr": float(psnr_torch(render.diffuse, target)),
                "diffuse_ssim": float(ssim_torch(render.diffuse, target)),
            })
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "frame"}
    result = {
        "note": "oracle initialization, 0 optimization steps, training views",
        "train_ids": train_ids,
        "n": len(rows),
        "summary": summary,
        "frames": rows,
    }
    out = ROOT / "outputs" / "sensitivity_reopt" / "oracle_init_train.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
