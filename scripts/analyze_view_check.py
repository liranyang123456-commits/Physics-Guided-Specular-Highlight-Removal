"""Decision rules for accepting a fitted world field, scored on the fields of outputs/v2/view_check.json.

Rules (the output of an accepted field is the fusion, the output of a rejected one the single-image prediction):
  always   no check
  train    training-view fit check of the manuscript: accept iff the full-radiance PSNR on the training views is >= 40 dB
  view     held-out-view check: accept iff the median absolute error of the full-radiance render against the *input* image of
           the new view (object pixels) is <= theta; theta is calibrated on the fields of scene_0070 only
  both     accept iff both checks accept
  oracle   per field, the better of fusion and single-image (reference, not a rule)

The held-out-view statistic uses the input image of the new view, which is available at test time; the diffuse ground truth
enters only the scores below.

    python scripts/analyze_view_check.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAU_DB = 40.0
FUSE_KEY = "fuse_xc0"          # fusion output: the cross-fitted gate (seed 0) of Section 4.5; "fuse" is the gate of the original protocol


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability that a random positive has a higher score than a random negative (ties count half)."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = x.argsort().argsort().astype(float), y.argsort().argsort().astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def output(rule: str, row: dict, theta: float) -> float:
    train_ok = row["train"]["full_psnr"] >= TAU_DB
    view_ok = row["holdout"]["med_err"] <= theta
    accept = {"always": True, "train": train_ok, "view": view_ok, "both": train_ok and view_ok}.get(rule)
    if rule == "oracle":
        return max(row[FUSE_KEY], row["single"])
    return row[FUSE_KEY] if accept else row["single"]


def main() -> None:
    global FUSE_KEY
    if len(sys.argv) > 1 and sys.argv[1] == "orig":
        FUSE_KEY = "fuse"
    data = json.loads((ROOT / "outputs/v2/view_check.json").read_text(encoding="utf-8"))
    names = sorted(data)
    rows = [data[n] for n in names]
    med = np.array([r["holdout"]["med_err"] for r in rows])
    phys = np.array([r["phys"] for r in rows])
    fuse = np.array([r[FUSE_KEY] for r in rows])
    single = np.array([r["single"] for r in rows])
    train_fit = np.array([r["train"]["full_psnr"] for r in rows])
    calib = [i for i, n in enumerate(names) if "/s00" not in n and not n.startswith(("ten/", "material/"))]   # scene_0070 sensitivity fields
    calib += [i for i, n in enumerate(names) if n == "ten/scene_0070"]
    print(f"{len(rows)} fields; calibration fields (scene_0070): {len(calib)}")

    print("\nrank correlation with the diffuse PSNR of the physical render (all fields)")
    print(f"  held-out median error  : {spearman(-np.log(med), phys):+.3f}")
    print(f"  training-view fit PSNR : {spearman(train_fit, phys):+.3f}")

    better = fuse > single       # accepting the fusion pays off
    print(f"\nfusion better than single-image in {int(better.sum())} of {len(rows)} fields")
    print(f"  AUC of -log(held-out median error): {auc(-np.log(med), better):.3f};  of the training-view fit: {auc(train_fit, better):.3f}")

    # theta: largest threshold (on the grid) that maximizes the mean output on the calibration fields
    grid = np.unique(np.round(np.geomspace(5e-4, 5e-2, 200), 5))
    best_theta, best_score = grid[0], -1.0
    for theta in grid:
        score = np.mean([output("view", data[names[i]], theta) for i in calib])
        if score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and theta > best_theta):
            best_theta, best_score = theta, score
    print(f"\ncalibrated theta (scene_0070 fields only): {best_theta:.4f}")

    groups = {"all fields": list(range(len(rows))),
              "scene_0070 (calibration)": calib,
              "other scenes (not calibrated)": [i for i in range(len(rows)) if i not in calib],
              "fields with fusion < single": [i for i in range(len(rows)) if not better[i]]}
    result = {"theta": float(best_theta), "tau_db": TAU_DB, "n": len(rows), "groups": {}, "fusion_gate": FUSE_KEY,
              "spearman_med_phys": spearman(-np.log(med), phys), "spearman_trainfit_phys": spearman(train_fit, phys),
              "auc_med": auc(-np.log(med), better), "auc_fit": auc(train_fit, better), "n_fusion_worse": int((~better).sum())}
    for label, idx in groups.items():
        if not idx:
            continue
        line = {rule: float(np.mean([output(rule, data[names[i]], best_theta) for i in idx])) for rule in
                ("always", "train", "view", "both", "oracle")}
        fa = {rule: int(sum(1 for i in idx if (output(rule, data[names[i]], best_theta) == data[names[i]][FUSE_KEY]) and not better[i]))
              for rule in ("always", "train", "view", "both")}
        fr = {rule: int(sum(1 for i in idx if (output(rule, data[names[i]], best_theta) == data[names[i]]["single"]) and better[i]))
              for rule in ("train", "view", "both")}
        result["groups"][label] = {"n": len(idx), "mean_output_psnr": line, "false_accepts": fa, "false_rejects": fr}
        print(f"\n{label} (n={len(idx)}): mean output PSNR")
        print("  " + "  ".join(f"{k} {v:6.2f}" for k, v in line.items()))
        print("  false accepts (fusion used but worse than single): " + str(fa))
        print("  false rejects (single used but fusion better)    : " + str(fr))
    print("\nper-field table (sorted by held-out median error)")
    for i in np.argsort(med):
        r = rows[i]
        print(f"  {names[i]:50s} med {med[i]:.4f} trainfit {train_fit[i]:6.2f} | phys {phys[i]:6.2f} fuse {fuse[i]:6.2f} single {single[i]:6.2f}")
    (ROOT / "outputs/v2/view_check_analysis.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

