"""Scores the three fine-tuned Table 3 models on one footing.

The matched fine-tuning scripts of DHAN-SHR and TSHRNet scored their floating-point outputs, whereas the row of our
single-image branch was scored on its saved 8-bit predictions (scripts/_collect_baseline_metrics.py); viz.save_image
truncates to uint8, which lowers every value by half a quantization step on average.  This script runs the three
fine-tuned checkpoints again and scores each output (a) in floating point, (b) truncated to 8 bits as saved, and
(c) rounded to 8 bits, on all 240 test frames, the 121 hard frames, and the other 119.  Writes
outputs/v2/table3_consistent.json.

    python scripts/eval_table3_consistent.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mvbrdf_shr.metrics import psnr_torch, ssim_torch  # noqa: E402

BENCH = ROOT / "data" / "world_benchmark_v2"


def rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def build(device):
    from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro
    import finetune_dhan_world_v2 as D

    ours = MVBRDFSHRPro(128, 48, 96).to(device)
    ours.load_state_dict(torch.load(ROOT / "outputs/world_benchmark_single/best.pt", map_location=device, weights_only=False)["model"])
    ours.eval()
    dhan = D.load_dhan(device)
    dhan.load_state_dict(torch.load(ROOT / "outputs/dhan_world_v2_finetuned/best.pt", map_location=device, weights_only=False)["model"])
    dhan.eval()
    for name in [k for k in sys.modules if k == "models" or k.startswith("models.")]:   # DHAN-SHR and TSHRNet both ship "models"
        del sys.modules[name]
    sys.path[:] = [p for p in sys.path if "DHAN-SHR" not in p]
    import finetune_tshrnet_world_v2 as T

    nets = T.load_tshrnet(device)
    state = torch.load(ROOT / "outputs/tshrnet_world_v2_finetuned/best.pt", map_location=device, weights_only=False)
    for net, sd in zip(nets, state["networks"]):
        net.load_state_dict(sd)
        net.eval()
    return {
        "ours_ft": lambda x: ours(x, use_refine=True, detach_physics=True)["pred"].clamp(0, 1),
        "dhan_ft": lambda x: dhan(x).clamp(0, 1),
        "tshr_ft": lambda x: T.to_output(T.forward(nets, x)[3], tuple(x.shape[-2:])),
    }


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = sorted(p.name for p in (BENCH / "flat" / "test" / "input").glob("*.png"))
    hard = set(p.name for p in (BENCH / "flat_hard" / "test" / "input").glob("*.png"))
    models = build(device)
    variants = {"float": lambda p: p, "uint8_truncated": lambda p: torch.floor(p * 255) / 255,
                "uint8_rounded": lambda p: torch.round(p * 255) / 255}
    per = {m: {v: {} for v in variants} for m in models}
    for n in names:
        x = rgb(BENCH / "flat" / "test" / "input" / n).to(device)
        y = rgb(BENCH / "flat" / "test" / "gt" / n).to(device)
        for m, f in models.items():
            p = f(x)
            for v, q in variants.items():
                pq = q(p)
                per[m][v][n] = (psnr_torch(pq, y), ssim_torch(pq, y))
    subsets = {"test": names, "hard_test": [n for n in names if n in hard], "non_hard": [n for n in names if n not in hard]}
    out = {m: {v: {s: {"n": len(ns), "psnr": float(np.mean([per[m][v][n][0] for n in ns])),
                       "ssim": float(np.mean([per[m][v][n][1] for n in ns]))} for s, ns in subsets.items()}
               for v in variants} for m in models}
    (ROOT / "outputs" / "v2" / "table3_consistent.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for m in models:
        for v in variants:
            r = out[m][v]
            print(f"{m:8s} {v:16s} all {r['test']['psnr']:.2f}/{r['test']['ssim']:.4f}  hard {r['hard_test']['psnr']:.2f}/"
                  f"{r['hard_test']['ssim']:.4f}  non-hard {r['non_hard']['psnr']:.2f}")


if __name__ == "__main__":
    main()
