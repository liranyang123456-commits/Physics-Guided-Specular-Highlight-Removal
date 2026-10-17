"""Scores the matched fine-tuning baselines of Table 3 on their saved 8-bit predictions, like every other row of the table.

scripts/finetune_dhan_world_v2.py, scripts/finetune_tshrnet_world_v2.py, and scripts/train_hrnet_world_v2.py score the
floating-point outputs of their networks, whereas the other rows of Table 3 (scripts/_collect_baseline_metrics.py) are
scored on the saved predictions, which viz.save_image truncates to uint8.  This script re-scores the saved predictions
of the three models with the functions of _collect_baseline_metrics.py and rewrites the "test" and "hard_test" entries
of their metric files; the floating-point scores are kept under "float_outputs".

    python scripts/rescore_saved_predictions.py
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

BENCH = ROOT / "data" / "world_benchmark_v2"
MODELS = {"dhan_shr_world_v2_finetuned": "dhan_world_v2_finetuned", "tshrnet_world_v2_finetuned": "tshrnet_world_v2_finetuned",
          "hrnet_world_v2_trained": "hrnet_world_v2_trained"}


def rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def score(pred_dir: Path, names: list[str]) -> dict:
    p, s = [], []
    for n in names:
        x, y = rgb(pred_dir / n), rgb(BENCH / "flat" / "test" / "gt" / n)
        p.append(psnr_torch(x, y))
        s.append(ssim_torch(x, y))
    return {"n": len(p), "psnr": float(np.mean(p)), "ssim": float(np.mean(s))}


def main() -> None:
    names = sorted(p.name for p in (BENCH / "flat" / "test" / "input").glob("*.png"))
    hard = sorted(p.name for p in (BENCH / "flat_hard" / "test" / "input").glob("*.png"))
    for pred, out in MODELS.items():
        saved = {"test": score(BENCH / "predictions" / pred, names), "hard_test": score(BENCH / "predictions" / pred, hard)}
        for path in (BENCH / "predictions" / pred / "metrics.json", ROOT / "outputs" / out / "metrics.json"):
            if not path.exists():
                continue
            m = json.loads(path.read_text(encoding="utf-8"))
            if "float_outputs" not in m:
                m["float_outputs"] = {"test": m["test"], "hard_test": m["hard_test"]}
            m["test"], m["hard_test"] = saved["test"], saved["hard_test"]
            m["scoring"] = "saved 8-bit predictions (scripts/rescore_saved_predictions.py), as every other row of Table 3"
            path.write_text(json.dumps(m, indent=2), encoding="utf-8")
        f = m["float_outputs"]
        print(f"{pred}: saved {saved['test']['psnr']:.2f}/{saved['test']['ssim']:.4f}, hard {saved['hard_test']['psnr']:.2f}/"
              f"{saved['hard_test']['ssim']:.4f}  (float {f['test']['psnr']:.2f}, hard {f['hard_test']['psnr']:.2f})")


if __name__ == "__main__":
    main()
