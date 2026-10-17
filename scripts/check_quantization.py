"""Why the exact generating state of scene_0070 reproduces its held-out diffuse targets at only ~53.9 dB.

The benchmark generator stores images with save_rgb(), which truncates to uint8 and so lowers every stored value by half a
quantization step on average.  This script renders the exact state on the held-out views and compares it with the stored
targets as rendered, after the same truncation, after rounding, and after subtracting the mean truncation offset (0.5/255).
Writes outputs/v2/quantization_check.json.

    python scripts/check_quantization.py
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

from eval_world_ablation import psnr  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402


def main() -> None:
    os.chdir(ROOT)
    device = torch.device("cuda")
    reference = torch.load("outputs/v2/world_scene_0070/last.pt", map_location="cpu", weights_only=False)
    cfg = dict(reference["config"])
    cfg["raster_backend"] = "gsplat"
    cfg["gaussian_init"] = "geometry_albedo"          # the exact state: geometry, albedo, and lighting of the generator
    dataset = load_dataset(cfg)
    pipeline = build_pipeline(dataset, cfg, device, list(reference["train_ids"]))
    pipeline.eval()
    rows = []
    with torch.no_grad():
        for fid in reference["holdout_ids"]:
            frame = dataset[fid]
            gt = frame.diffuse_gt.to(device)
            d = pipeline(frame.camera.to(device), fid, image=frame.image.to(device), refine=False)["render"].diffuse.clamp(0, 1)
            rows.append({
                "frame": int(fid),
                "as_rendered": psnr(d, gt),
                "floor_quantized": psnr(torch.floor(d * 255) / 255, gt),
                "round_quantized": psnr(torch.round(d * 255) / 255, gt),
                "minus_half_step": psnr((d - 0.5 / 255).clamp(0, 1), gt),
                "share_identical_after_floor": float((torch.floor(d * 255) == torch.round(gt * 255)).float().mean()),
            })
    mean = {k: float(sum(r[k] for r in rows) / len(rows)) for k in rows[0] if k != "frame"}
    out = {"scene": "scene_0070", "views": "held-out", "per_view": rows, "mean": mean}
    dest = ROOT / "outputs" / "v2" / "quantization_check.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
