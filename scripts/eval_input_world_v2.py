"""Raw-input reference of Table 3 and the split of every Table 3 row into the hard subset and the remaining frames.

PSNR / SSIM of the highlighted input against the diffuse ground truth on world_benchmark_v2 (240 test frames, of which the
121 hard frames form a subset), and the same metrics of the saved predictions of every Table 3 model on the 119 frames
outside the hard subset, with the number of frames on which a model is below the input.  Writes
data/world_benchmark_v2/predictions/input/metrics.json and outputs/v2/benchmark_by_subset.json.

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

BENCH = ROOT / "data" / "world_benchmark_v2"
MODELS = {"ours_ft": "mvbrdf_shr_world_v2_finetuned", "dhan_ft": "dhan_shr_world_v2_finetuned",
          "tshr_ft": "tshrnet_world_v2_finetuned", "ours_zs": "mvbrdf_shr_shiq_pretrained", "dhan_zs": "dhan_shr",
          "tshr_zs": "tshrnet"}


def rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def scores(pred_dir: Path, names: list[str]) -> dict[str, tuple[float, float]]:
    out = {}
    for n in names:
        x, y = rgb(pred_dir / n), rgb(BENCH / "flat" / "test" / "gt" / n)
        out[n] = (float(psnr_torch(x, y)), float(ssim_torch(x, y)))
    return out


def summary(s: dict, names: list[str]) -> dict:
    return {"n": len(names), "psnr": float(np.mean([s[n][0] for n in names])), "ssim": float(np.mean([s[n][1] for n in names]))}


def main() -> None:
    names = sorted(p.name for p in (BENCH / "flat" / "test" / "input").glob("*.png"))
    hard = sorted(p.name for p in (BENCH / "flat_hard" / "test" / "input").glob("*.png"))
    assert set(hard) <= set(names)
    rest = [n for n in names if n not in set(hard)]
    inp = scores(BENCH / "flat" / "test" / "input", names)
    ref = {"method": "input (no removal)", "test": summary(inp, names), "hard_test": summary(inp, hard), "non_hard": summary(inp, rest)}
    dest = BENCH / "predictions" / "input"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(ref, indent=2), encoding="utf-8")

    split = {"input": ref}
    for key, folder in MODELS.items():
        s = scores(BENCH / "predictions" / folder, names)
        split[key] = {"test": summary(s, names), "hard_test": summary(s, hard), "non_hard": summary(s, rest),
                      "non_hard_frames_below_input": int(sum(s[n][0] < inp[n][0] for n in rest)),
                      "hard_frames_below_input": int(sum(s[n][0] < inp[n][0] for n in hard))}
    (ROOT / "outputs" / "v2" / "benchmark_by_subset.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    for key, v in split.items():
        extra = "" if key == "input" else f"  below input: non-hard {v['non_hard_frames_below_input']}/{len(rest)}, hard {v['hard_frames_below_input']}/{len(hard)}"
        print(f"{key:8s} all {v['test']['psnr']:.2f}  hard {v['hard_test']['psnr']:.2f}  non-hard {v['non_hard']['psnr']:.2f}{extra}")


if __name__ == "__main__":
    main()
