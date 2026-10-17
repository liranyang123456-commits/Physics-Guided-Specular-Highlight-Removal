"""A combined check against failures that the training views do not show (offline, on outputs/v2/view_check.json).

Rule: keep the fused output if the field fits its training views at 40 dB or more AND the median absolute error of its full
render against the input images of the views being processed is at most theta; otherwise return the single-image prediction.
Both quantities use only images available at test time (training images; inputs of the processed views).  theta is fixed
a priori from fields known to be accurate: the largest median error over the ten benchmark scenes and the eight
material-controlled scenes, times a margin of 1.5.  Reports false accepts (fused output kept although it is below the
single-image prediction), false rejects, and the mean output PSNR over the 119 fields, against the training-view check alone,
the oracle, and a sweep of theta.  Writes outputs/v2/combined_check.json.

    python scripts/analyze_combined_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs" / "v2"
TAU = 40.0
MARGIN = 1.5


def outcome(rows, keep):
    out = np.array([r["fuse_xc0"] if k else r["single"] for r, k in zip(rows, keep)])
    worse = np.array([r["fuse_xc0"] < r["single"] for r in rows])
    keep = np.array(keep)
    return {"false_accepts": int((keep & worse).sum()), "false_rejects": int((~keep & ~worse).sum()),
            "mean_output_psnr": float(out.mean()), "max_accepted_loss": float(max([r["single"] - r["fuse_xc0"] for r, k in zip(rows, keep) if k and r["fuse_xc0"] < r["single"]] or [0.0]))}


def main() -> None:
    vc = json.loads((V2 / "view_check.json").read_text(encoding="utf-8"))
    items = list(vc.items()) if isinstance(vc, dict) else [(r["name"], r) for r in vc]
    rows = [dict(v, name=k) for k, v in items]
    accurate = [r for r in rows if r["group"] in ("ten", "material")]
    theta = MARGIN * max(r["holdout"]["med_err"] for r in accurate)
    train_ok = [r["train"]["full_psnr"] >= TAU for r in rows]
    combined = [t and r["holdout"]["med_err"] <= theta for t, r in zip(train_ok, rows)]
    res = {"n": len(rows), "n_fusion_worse": int(sum(r["fuse_xc0"] < r["single"] for r in rows)), "theta": theta,
           "accurate_fields": len(accurate), "accurate_med_err_max": max(r["holdout"]["med_err"] for r in accurate),
           "rules": {"always_fuse": outcome(rows, [True] * len(rows)), "training_view_check": outcome(rows, train_ok),
                     "combined_check": outcome(rows, combined),
                     "oracle": outcome(rows, [r["fuse_xc0"] >= r["single"] for r in rows])},
           "combined_rejects_accurate": int(sum(1 for r, c in zip(rows, combined) if r in accurate and not c))}
    res["sweep"] = {f"{t:.4f}": outcome(rows, [ok and r["holdout"]["med_err"] <= t for ok, r in zip(train_ok, rows)])
                    for t in (0.003, 0.004, 0.006, 0.008, 0.012, 0.02, 0.0337)}
    res["false_rejects_combined"] = [{"name": r["name"], "fuse_xc0": r["fuse_xc0"], "single": r["single"], "med_err": r["holdout"]["med_err"]}
                                     for r, c in zip(rows, combined) if not c and r["fuse_xc0"] >= r["single"]]
    (V2 / "combined_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "false_rejects_combined"}, indent=2))
    for r in res["false_rejects_combined"]:
        print("false reject:", r)


if __name__ == "__main__":
    main()
