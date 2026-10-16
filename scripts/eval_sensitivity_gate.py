"""Evaluate several fusion gates on the re-optimised perturbed world fields saved by run_sensitivity_v2.py.

Every field (``outputs/v2/sensitivity/<axis>_seed<k>.pt``) is rendered once at the held-out views and combined with the
single-image branch by each gate; the physical render, the single-image branch, the fixed 0.5/0.5 blend, and the purely
analytic gate (no learned residual) are reported next to it.

    python scripts/eval_sensitivity_gate.py --gate clean=outputs/v2/fusion_scene_0070/fusion.pt \
        --gate robust=outputs/v2/fusion_robust_scene_0070/fusion.pt --tag gates
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
from mvbrdf_shr.world.fusion import HybridMVBRDFSHR  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402

AXES = ("clean", "geo_mild", "geo_severe", "albedo15", "albedo30", "light", "drop25", "drop50", "sparse6", "sparse3")


def load_hybrid(path: str, device: torch.device) -> HybridMVBRDFSHR:
    hybrid = HybridMVBRDFSHR(resolution=128).to(device)
    hybrid.load_single_checkpoint(ROOT / "outputs/world_benchmark_single/best.pt")
    ckpt = torch.load(ROOT / path, map_location=device, weights_only=False)
    hybrid.gate.load_state_dict(ckpt["gate"])
    hybrid.correction.load_state_dict(ckpt["correction"])
    return hybrid.eval()


@torch.no_grad()
def evaluate_field(pipeline, dataset, holdout, device, hybrids: dict) -> dict:
    pipeline.eval()
    pro = MVBRDFSHRPro(128, 48, 96).to(device)
    state = torch.load(ROOT / "outputs/world_benchmark_single/best.pt", map_location=device, weights_only=False)
    pro.load_state_dict(state.get("model", state), strict=False)
    pro.eval()
    names = ["phys", "single", "fixed_blend", "analytic"] + list(hybrids)
    rows = {name: [] for name in names}
    for frame_id in holdout:
        frame = dataset[frame_id]
        image = frame.image.to(device)
        target = frame.diffuse_gt.to(device)
        out = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)
        render = out["render"]
        phys = render.diffuse
        x = F.interpolate(image.unsqueeze(0), (128, 128), mode="bilinear", align_corners=False)
        single = pro(x, use_refine=True, detach_physics=True)["pred"]
        single = F.interpolate(single, size=image.shape[-2:], mode="bilinear", align_corners=False)[0]
        err = (image.unsqueeze(0) - (render.diffuse + render.specular).unsqueeze(0).clamp(0, 1)).abs().mean(1, keepdim=True)
        confidence = (render.alpha.unsqueeze(0) * torch.exp(-err / 0.08)).clamp(1e-4, 1 - 1e-4)
        preds = {
            "phys": phys,
            "single": single,
            "fixed_blend": (0.5 * phys + 0.5 * single).clamp(0, 1),
            "analytic": (confidence * phys.unsqueeze(0) + (1 - confidence) * single.unsqueeze(0))[0],
        }
        for name, hybrid in hybrids.items():
            preds[name] = hybrid(image.unsqueeze(0), out, detach_backbones=True)["pred"][0]
        for name in names:
            rows[name].append((psnr(preds[name], target), ssim(preds[name], target)))
    return {name: {"psnr": float(np.mean([v[0] for v in rows[name]])), "ssim": float(np.mean([v[1] for v in rows[name]])),
                   "per_view": [{"frame": holdout[i], "psnr": rows[name][i][0], "ssim": rows[name][i][1]}
                                for i in range(len(holdout))]} for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--tag", default="gates")
    parser.add_argument("--axes", nargs="*", default=list(AXES))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--dir", default="outputs/v2/sensitivity")
    args = parser.parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gates = dict(item.split("=", 1) for item in args.gate)
    hybrids = {name: load_hybrid(path, device) for name, path in gates.items()}
    results = {}
    for axis in args.axes:
        for seed in args.seeds:
            path = ROOT / args.dir / f"{axis}_seed{seed}.pt"
            if not path.exists():
                continue
            state = torch.load(path, map_location="cpu", weights_only=False)
            cfg = dict(state["config"])
            cfg["raster_backend"] = "gsplat"
            dataset = load_dataset(cfg)
            pipeline = build_pipeline(dataset, cfg, device, list(state["train_ids"]))
            pipeline.load_state_dict(state["model"], strict=False)
            results[f"{axis}_seed{seed}"] = evaluate_field(pipeline, dataset, list(state["holdout_ids"]), device, hybrids)
            row = results[f"{axis}_seed{seed}"]
            print(f"{axis:11s} seed{seed} " + " ".join(f"{k}={row[k]['psnr']:.2f}" for k in row), flush=True)
    out = ROOT / args.dir / f"{args.tag}.json"
    out.write_text(json.dumps({"gates": gates, "results": results}, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
