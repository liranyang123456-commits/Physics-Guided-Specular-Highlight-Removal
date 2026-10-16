"""Exploration: a confidence for the physical diffuse render that is not contaminated by specular misfit.

The analytic confidence of the fusion module uses the full-radiance residual |I - (I_d + I_s)|.  On a held-out view the
diffuse render of a correctly optimized field is almost exact (56.5 dB) while its full render is not (27.1 dB), because the
view-dependent specular lobe is not identified from nine views, so that residual is dominated by specular misfit.  Here the
residual is instead measured against the *diffuse* render only, on pixels the single-image branch does not flag as
highlight (there I ~ I_d, so |I - I_d| measures the diffuse error), and spread over a neighbourhood by normalized
convolution so that highlight pixels inherit the confidence of the surrounding non-highlight pixels:

    e_nh(u) = sum_v w(v) |I(v) - I_d(v)| / sum_v w(v)  over a k x k window,   w = (1 - m_single)^p,
    c_nh(u) = A(u) exp(-e_nh(u) / T)

Calibration data (cross-fitted renders of clean and corrupted fold fields at their left-out views) select T; the
re-optimized perturbed fields of the sensitivity study and the ten benchmark scenes are only used for evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_world_ablation import psnr, ssim  # noqa: E402
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402
from mvbrdf_shr.world.train_fusion_robust import corrupted  # noqa: E402

TEMPS = (0.01, 0.02, 0.04, 0.08)
KERNELS = (9, 17)


def confidence_nh(image, render, mask, kernel: int, temperature: float, power: float = 1.0) -> torch.Tensor:
    """image (3,H,W), render with .diffuse/.alpha, mask (1,H,W) in [0,1] -> confidence (1,H,W)."""
    residual = (image - render.diffuse).abs().mean(0, keepdim=True)
    w = (1 - mask).clamp(0, 1) ** power
    pad = kernel // 2
    num = F.avg_pool2d((w * residual).unsqueeze(0), kernel, stride=1, padding=pad, count_include_pad=False)
    den = F.avg_pool2d(w.unsqueeze(0), kernel, stride=1, padding=pad, count_include_pad=False)
    e = (num / den.clamp_min(1e-4))[0]
    return (render.alpha * torch.exp(-e / temperature)).clamp(1e-4, 1 - 1e-4)


@torch.no_grad()
def single_branch(pro, image, device):
    x = F.interpolate(image.unsqueeze(0), (128, 128), mode="bilinear", align_corners=False)
    out = pro(x, use_refine=True, detach_physics=True)
    size = image.shape[-2:]
    pred = F.interpolate(out["pred"], size=size, mode="bilinear", align_corners=False)[0]
    mask = F.interpolate(out["mask_pred"], size=size, mode="bilinear", align_corners=False)[0]
    return pred, mask


def evaluate_views(pipeline, dataset, ids, pro, device, corrupt: str = "clean") -> dict:
    """PSNR of single / physical / analytic-full / analytic-nh variants, mean over the views."""
    rows = {"phys": [], "single": [], "an_full": []}
    for k in KERNELS:
        for t in TEMPS:
            rows[f"nh_k{k}_T{t}"] = []
    for frame_id in ids:
        frame = dataset[frame_id]
        image = frame.image.to(device)
        target = frame.diffuse_gt.to(device)
        with torch.no_grad(), corrupted(pipeline, corrupt):
            render = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)["render"]
        phys = render.diffuse
        single, mask = single_branch(pro, image, device)
        err = (image - (render.diffuse + render.specular).clamp(0, 1)).abs().mean(0, keepdim=True)
        c_full = (render.alpha * torch.exp(-err / 0.08)).clamp(1e-4, 1 - 1e-4)
        rows["phys"].append(psnr(phys, target))
        rows["single"].append(psnr(single, target))
        rows["an_full"].append(psnr(c_full * phys + (1 - c_full) * single, target))
        for k in KERNELS:
            for t in TEMPS:
                c = confidence_nh(image, render, mask, k, t)
                rows[f"nh_k{k}_T{t}"].append(psnr(c * phys + (1 - c) * single, target))
    return {k: float(np.mean(v)) for k, v in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/v2/confidence_exploration.json")
    args = parser.parse_args()
    os.chdir(ROOT)
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pro = MVBRDFSHRPro(128, 48, 96).to(device)
    state = torch.load("outputs/world_benchmark_single/best.pt", map_location=device, weights_only=False)
    pro.load_state_dict(state.get("model", state), strict=False)
    pro.eval()
    result = {"calibration": {}, "evaluation": {}}

    # calibration: cross-fitted fold fields at their left-out views, clean and corrupted (geometry, lighting)
    for kind in ("clean", "geo", "light"):
        agg: dict = {}
        for fold in range(3):
            st = torch.load(f"outputs/v2/crossfit/fold{fold}.pt", map_location="cpu", weights_only=False)
            cfg = dict(st["config"])
            cfg["raster_backend"] = "gsplat"
            dataset = load_dataset(cfg)
            pipeline = build_pipeline(dataset, cfg, device, list(st["train_ids"]))
            pipeline.load_state_dict(st["model"], strict=False)
            pipeline.eval().requires_grad_(False)
            for rep in range(3 if kind != "clean" else 1):
                res = evaluate_views(pipeline, dataset, list(st["left_out_ids"]), pro, device, kind)
                for key, val in res.items():
                    agg.setdefault(key, []).append(val)
        result["calibration"][kind] = {k: float(np.mean(v)) for k, v in agg.items()}
        print("calibration", kind, {k: round(v, 2) for k, v in result["calibration"][kind].items() if k in ("phys", "single", "an_full")}, flush=True)

    # evaluation: re-optimized sensitivity fields (held-out views)
    for path in sorted((ROOT / "outputs/v2/sensitivity").glob("*_seed0.pt")) + sorted((ROOT / "outputs/v2/sensitivity").glob("clean_seed[12].pt")):
        st = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(st["config"])
        cfg["raster_backend"] = "gsplat"
        dataset = load_dataset(cfg)
        pipeline = build_pipeline(dataset, cfg, device, list(st["train_ids"]))
        pipeline.load_state_dict(st["model"], strict=False)
        pipeline.eval().requires_grad_(False)
        result["evaluation"][path.stem] = evaluate_views(pipeline, dataset, list(st["holdout_ids"]), pro, device)
        print("evaluation", path.stem, {k: round(v, 2) for k, v in result["evaluation"][path.stem].items() if k in ("phys", "single", "an_full")}, flush=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
