"""PSNR/SSIM of saved zero-shot baseline predictions on the scene_0070 holdout views.

Writes outputs/world_holdout_zero_shot.json (Table 4, zero-shot TSHRNet/DHAN-SHR rows).
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

PRED = ROOT / "data" / "world_benchmark_v2" / "predictions"
GT = ROOT / "data" / "world_benchmark_v2" / "flat" / "test" / "gt"
VIEWS = (0, 4, 8)


def rgb(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]


def main() -> None:
    result = {}
    for method in ("tshrnet", "dhan_shr"):
        rows = []
        for view in VIEWS:
            name = f"scene_0070_v{view:03d}.png"
            pred, gt = rgb(PRED / method / name), rgb(GT / name)
            rows.append({"view": view, "psnr": psnr_torch(pred, gt), "ssim": ssim_torch(pred, gt)})
        result[method] = {"psnr": float(np.mean([r["psnr"] for r in rows])),
                          "ssim": float(np.mean([r["ssim"] for r in rows])), "per_view": rows}
        print(method, round(result[method]["psnr"], 2), round(result[method]["ssim"], 3))
    (ROOT / "outputs" / "world_holdout_zero_shot.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
