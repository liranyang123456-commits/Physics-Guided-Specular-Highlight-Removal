"""Held-out accuracy of the scene_0070 oracle initialization before any optimization.

Builds the strict-holdout pipeline exactly as run_sensitivity_reopt.py does
(oracle geometry/albedo from gaussian_brdf_gt.npz, calibrated lighting), skips
training, and evaluates the physical, fused, and single-image outputs.
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

from eval_world_ablation import psnr, ssim  # noqa: E402
from run_sensitivity_reopt import evaluate  # noqa: E402
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402


def main() -> None:
    os.chdir(ROOT)
    torch.manual_seed(0)
    device = torch.device("cuda")
    reference = torch.load("outputs/world_benchmark_v2_scene_0070_strict/last.pt", map_location="cpu", weights_only=False)
    cfg = dict(reference["config"])
    cfg["raster_backend"] = "auto"
    dataset = load_dataset(cfg)
    pipeline = build_pipeline(dataset, cfg, device, list(reference["train_ids"]))
    holdout = list(reference["holdout_ids"])
    summary = evaluate(pipeline, dataset, holdout, device)
    summary["note"] = "oracle initialization, 0 optimization steps"

    # Fixed blend and confidence/diffuse-error correlation in the same state.
    single_by_frame = {v["frame"]: v for v in summary["single"]["per_view"]}
    pro = MVBRDFSHRPro(128, 48, 96).to(device)
    pro.load_state_dict(torch.load("outputs/world_benchmark_single/best.pt", map_location=device, weights_only=False)["model"])
    pro.eval()
    fixed, corr = [], []
    with torch.no_grad():
        for frame_id in holdout:
            frame = dataset[frame_id]
            image, target = frame.image.to(device), frame.diffuse_gt.to(device)
            render = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)["render"]
            single = pro(image.unsqueeze(0), use_refine=True, detach_physics=True)["pred"][0]
            blend = (0.5 * render.diffuse + 0.5 * single).clamp(0, 1)
            fixed.append((psnr(blend, target), ssim(blend, target)))
            err = (image - (render.diffuse + render.specular).clamp(0, 1)).abs().mean(0)
            covered = render.alpha[0] > 0.1
            diffuse_err = (render.diffuse - target).abs().mean(0)
            pair = torch.stack([err[covered], diffuse_err[covered]])
            corr.append({"frame": frame_id, "pearson": float(torch.corrcoef(pair)[0, 1]),
                         "mean_residual": float(err[covered].mean()), "mean_diffuse_error": float(diffuse_err[covered].mean())})
    summary["fixed_blend"] = {"psnr": float(np.mean([f[0] for f in fixed])), "ssim": float(np.mean([f[1] for f in fixed]))}
    summary["confidence_correlation"] = corr
    assert set(single_by_frame) == set(holdout)
    out = ROOT / "outputs" / "sensitivity_reopt" / "oracle_init.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key in ("phys", "fuse", "single"):
        views = [round(v["psnr"], 2) for v in summary[key]["per_view"]]
        print(key, round(summary[key]["psnr"], 2), round(summary[key]["ssim"], 4), views)
    print("gate", round(summary["gate_mean"], 3))
    print("fixed blend", round(summary["fixed_blend"]["psnr"], 2), round(summary["fixed_blend"]["ssim"], 4))
    for row in corr:
        print(row)


if __name__ == "__main__":
    main()
