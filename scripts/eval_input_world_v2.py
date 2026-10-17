"""PSNR / SSIM of the raw highlighted input against the diffuse ground truth on world_benchmark_v2 (240 test and 121 hard
frames), with the metric functions of the other rows of Table 3.  Writes data/world_benchmark_v2/predictions/input/metrics.json.

    python scripts/eval_input_world_v2.py
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

from mvbrdf_shr.metrics import psnr_torch, ssim_torch  # noqa: E402


def rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
    bench = ROOT / "data" / "world_benchmark_v2"
    out = {"method": "input (no removal)"}
    for key, flat in (("test", "flat"), ("hard_test", "flat_hard")):
        ps, ss = [], []
        for p in sorted((bench / flat / "test" / "input").glob("*.png")):
            x, y = rgb(p), rgb(bench / flat / "test" / "gt" / p.name)
            ps.append(float(psnr_torch(x, y)))
            ss.append(float(ssim_torch(x, y)))
        out[key] = {"n": len(ps), "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))}
    dest = bench / "predictions" / "input"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
