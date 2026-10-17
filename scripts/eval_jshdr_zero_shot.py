"""Evaluate the released JSHDR checkpoint (CVPR 2021; zero-shot) with the metric functions of the other rows.

JSHDR is distributed as compiled CPython-3.6 modules plus a checkpoint (github.com/KosRud/SHIQ, network/); there is no
training code, so it can be evaluated but not fine-tuned.  It was run on the CPU of a Python 3.6 / PyTorch 1.5.1
environment in WSL (scripts/jshdr_batch.sh, scripts/export_jshdr_inputs.py) on exactly the inputs that the other
baselines see:
  * SHIQ: the 1,000 official test images at 200 x 200 (SHIQAdapter, split test_full; ground truth loaded by the adapter);
  * world_benchmark_v2: the 240 test frames (and the 121 hard frames among them) at 128 x 128.
usage: python scripts/eval_jshdr_zero_shot.py --preds <folder holding shiq/ and bench/ with D_<name>.png files>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvbrdf_shr.data.shiq_adapter import SHIQAdapter  # noqa: E402
from mvbrdf_shr.metrics import psnr_torch, ssim_torch  # noqa: E402

SHIQ_ROOT = ROOT / "data" / "raw" / "SHIQ_extracted"


def load(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def summarize(rows: list[tuple[float, float, float]]) -> dict:
    return {"n": len(rows), "psnr": float(np.mean([r[0] for r in rows])), "ssim": float(np.mean([r[1] for r in rows])),
            "input_psnr": float(np.mean([r[2] for r in rows]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "jshdr_zero_shot"))
    args = parser.parse_args()
    preds = Path(args.preds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- SHIQ
    dataset = SHIQAdapter(SHIQ_ROOT, 200, "test_full", augment=False)
    rows, split_rows = [], {"val": [], "test": []}
    missing = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        a = Path(sample.meta["path"])
        pred_path = preds / "shiq" / f"D_{a.name}"
        if not pred_path.exists():
            missing += 1
            continue
        pred = load(pred_path)
        assert pred.shape[-2:] == sample.image_clean.shape[-2:], (pred.shape, sample.image_clean.shape)
        row = (psnr_torch(pred, sample.image_clean[None]), ssim_torch(pred, sample.image_clean[None]),
               psnr_torch(sample.image[None], sample.image_clean[None]))
        rows.append(row)
        sid = SHIQAdapter._sample_id_from_path(a)
        split_rows["val" if sid in SHIQAdapter._VAL_IDS else "test"].append(row)
    shiq = {"all_1000": summarize(rows), "val_500": summarize(split_rows["val"]), "test_500": summarize(split_rows["test"]),
            "missing": missing}
    print("SHIQ", json.dumps(shiq))

    # ---------------------------------------------------------------- world_benchmark_v2
    bench = ROOT / "data" / "world_benchmark_v2"
    hard_names = {p.name for p in (bench / "flat_hard" / "test" / "input").glob("*.png")}
    rows_all, rows_hard = [], []
    pred_dir = bench / "predictions" / "jshdr_zero_shot"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((bench / "flat" / "test" / "input").glob("*.png")):
        pred_path = preds / "bench" / f"D_{path.name}"
        pred = load(pred_path)
        clean = load(bench / "flat" / "test" / "gt" / path.name)
        row = (psnr_torch(pred, clean), ssim_torch(pred, clean), psnr_torch(load(path), clean))
        rows_all.append(row)
        if path.name in hard_names:
            rows_hard.append(row)
        Image.open(pred_path).convert("RGB").save(pred_dir / path.name)
    benchmark = {"test": summarize(rows_all), "hard_test": summarize(rows_hard)}
    print("world_benchmark_v2", json.dumps(benchmark))

    result = {
        "method": "jshdr_zero_shot",
        "protocol": "released CVPR 2021 checkpoint (jshdr_d.pth, jshdr_r.pth), CPU, Python 3.6 / PyTorch 1.5.1 in WSL; "
                    "no fine-tuning (no training code in the release); same inputs, ground truth and metric functions as the other rows",
        "shiq": shiq,
        "world_benchmark_v2": benchmark,
        "prediction_dir": str(pred_dir),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", out / "metrics.json")


if __name__ == "__main__":
    main()
