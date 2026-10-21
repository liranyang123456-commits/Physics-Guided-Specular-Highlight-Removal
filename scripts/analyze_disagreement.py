"""Does the disagreement between the two branches on a new view identify the fields whose fused output is below the
single-image prediction?  Scored on outputs/v2/disagreement.json (scripts/eval_disagreement.py), restricted to the 119
fields of the held-out-view study (outputs/v2/view_check.json) so that the numbers are comparable with Table 15.

Signals (no ground truth):
  d_hold   PSNR between the fused output and the single-image prediction on the held-out views
  d_train  the same on the training views (there the field reproduces its images, so d_train is close to the accuracy
           of the single-image branch in that scene)
  drop     d_train - d_hold
  dp_hold  PSNR between the physical diffuse render and the single-image prediction on the held-out views
Rules: the training-view fit check (accept iff fit >= 40 dB) alone, and combined with a threshold on each signal.
Reports the AUC of every signal for the label "fused output below single-image prediction", the values of the fields that
the fit check accepts although their fused output is below the single-image prediction, and, for the combined rules, the
false accepts and false rejects over a sweep of the threshold.  Writes outputs/v2/disagreement_analysis.json.

    python scripts/analyze_disagreement.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs" / "v2"
TAU = 40.0
GATE = "xc0"


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def outcome(fuse, single, keep, base=None):
    """Scores of a rule; ``base`` is the acceptance of the training-view check alone, so that ``false_rejects_among_accepted``
    counts only the correctly accepted fields that the additional threshold removes."""
    keep = np.asarray(keep, dtype=bool)
    worse = fuse < single
    out = np.where(keep, fuse, single)
    res = {"false_accepts": int((keep & worse).sum()), "false_rejects": int((~keep & ~worse).sum()),
           "mean_output_psnr": float(out.mean()),
           "max_accepted_loss": float(max([(s - f) for f, s, k in zip(fuse, single, keep) if k and f < s] or [0.0]))}
    if base is not None:
        res["false_rejects_among_accepted"] = int((np.asarray(base, dtype=bool) & ~keep & ~worse).sum())
    return res


def main() -> None:
    dis = json.loads((V2 / "disagreement.json").read_text(encoding="utf-8"))
    vc = json.loads((V2 / "view_check.json").read_text(encoding="utf-8"))
    names = sorted(n for n in vc if n in dis)
    missing = sorted(n for n in vc if n not in dis)
    assert not missing, f"fields of the study without disagreement record: {missing}"
    rows = [dis[n] for n in names]
    ref = [vc[n] for n in names]
    # ground-truth scores and the fit check come from view_check.json (the numbers of Table 15); the disagreement signals
    # from the re-evaluation.  Two checkpoints were re-run by the repeatability study after view_check.json was written,
    # which is why the recomputed scores deviate for them; the deviation is reported, the labels are unchanged.
    fuse = np.array([v[f"fuse_{GATE}"] for v in ref])
    single = np.array([v["single"] for v in ref])
    phys = np.array([v["phys"] for v in ref])
    dev = {n: max(abs(r[f"holdout_fuse_{GATE}_gt"] - v[f"fuse_{GATE}"]), abs(r["holdout_single_gt"] - v["single"]),
                  abs(r["holdout_phys_gt"] - v["phys"])) for n, r, v in zip(names, rows, ref)}
    dev = {n: float(x) for n, x in dev.items() if x > 0.05}
    fit = np.array([v["train"]["full_psnr"] for v in ref])
    d_hold = np.array([r[f"holdout_fuse_{GATE}_vs_single"] for r in rows])
    d_train = np.array([r[f"train_fuse_{GATE}_vs_single"] for r in rows])
    dp_hold = np.array([r["holdout_phys_vs_single"] for r in rows])
    drop = d_train - d_hold
    worse = fuse < single
    fit_ok = fit >= TAU
    acc = fit_ok & worse                      # false accepts of the fit check
    correct = np.array([n.startswith(("ten/", "material/")) for n in names])

    res = {"n": len(names), "n_fusion_worse": int(worse.sum()), "gate": GATE, "rerun_checkpoints_dev_db": dev,
           "fit_check": outcome(fuse, single, fit_ok)}
    # how much the single-image accuracy differs between the training and the held-out views of the same scene
    s_train = np.array([r["train_single_gt"] for r in rows])
    res["single_train_minus_holdout_db"] = {"accurate_fields_min": float((s_train - single)[correct].min()),
                                            "accurate_fields_max": float((s_train - single)[correct].max()),
                                            "all_min": float((s_train - single).min()), "all_max": float((s_train - single).max())}
    # AUCs for the label "fused output below single-image prediction"
    for label, idx in (("all", np.ones(len(names), bool)), ("fit_accepted", fit_ok)):
        res[f"auc_{label}"] = {"neg_d_hold": auc(-d_hold[idx], worse[idx]), "drop": auc(drop[idx], worse[idx]),
                               "neg_dp_hold": auc(-dp_hold[idx], worse[idx]), "neg_fit": auc(-fit[idx], worse[idx]),
                               "n": int(idx.sum()), "n_worse": int(worse[idx].sum())}
    # what the disagreement measures on the fields that the fit check accepts: the accuracy of the single-image branch
    sel = fit_ok
    res["d_hold_correlation_fit_accepted"] = {
        "pearson_with_single_gt": float(np.corrcoef(d_hold[sel], single[sel])[0, 1]),
        "pearson_with_fuse_gt": float(np.corrcoef(d_hold[sel], fuse[sel])[0, 1]),
        "pearson_with_fuse_minus_single": float(np.corrcoef(d_hold[sel], (fuse - single)[sel])[0, 1]),
        "mean_abs_d_hold_minus_single_gt": float(np.abs(d_hold[sel] - single[sel]).mean()),
        "n_correct_accepted_below_min_failing": int(((d_hold < d_hold[acc].min()) & fit_ok & ~worse).sum()),
        "n_correct_accepted": int((fit_ok & ~worse).sum())}
    res["false_accepts_of_fit_check"] = [
        {"name": n, "fit": float(fit[i]), "fuse": float(fuse[i]), "single": float(single[i]), "phys": float(phys[i]),
         "d_hold": float(d_hold[i]), "d_train": float(d_train[i]), "drop": float(drop[i]), "dp_hold": float(dp_hold[i])}
        for i, n in enumerate(names) if acc[i]]
    res["accurate_fields"] = {"d_hold_min": float(d_hold[correct].min()), "d_hold_max": float(d_hold[correct].max()),
                              "drop_min": float(drop[correct].min()), "drop_max": float(drop[correct].max()),
                              "dp_hold_min": float(dp_hold[correct].min()), "dp_hold_max": float(dp_hold[correct].max())}
    # combined rules: fit check AND signal threshold; sweep, and the cheapest threshold that removes every false accept
    sweeps = {}
    for key, sig, direction in (("d_hold", d_hold, "gt"), ("drop", drop, "lt"), ("dp_hold", dp_hold, "gt")):
        # thresholds at the observed values; a field is kept when its signal is strictly beyond the threshold, so the threshold
        # equal to the worst failing value is the cheapest one that removes every failing field
        grid = np.unique(sig)
        table = []
        for t in grid:
            keep = fit_ok & ((sig > t) if direction == "gt" else (sig < t))
            o = outcome(fuse, single, keep, base=fit_ok)
            table.append({"threshold": float(t), **o})
        # thresholds with no false accept: the one with the fewest false rejects
        clean = [e for e in table if e["false_accepts"] == 0]
        best_clean = min(clean, key=lambda e: (e["false_rejects"], -e["mean_output_psnr"])) if clean else None
        # thresholds with no false reject beyond those of the fit check alone: fewest false accepts
        base_fr = res["fit_check"]["false_rejects"]
        safe = [e for e in table if e["false_rejects"] <= base_fr]
        best_safe = min(safe, key=lambda e: (e["false_accepts"], -e["mean_output_psnr"])) if safe else None
        best_mean = max(table, key=lambda e: e["mean_output_psnr"])
        sweeps[key] = {"direction": direction, "no_false_accept": best_clean, "no_extra_false_reject": best_safe,
                       "best_mean_output": best_mean}
    res["combined_rules"] = sweeps
    res["oracle"] = outcome(fuse, single, fuse >= single)
    (V2 / "disagreement_analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    print(f"{len(names)} fields (checkpoints re-run after the study, recomputed score deviates: {dev}); "
          f"fused < single in {int(worse.sum())}; fit check: {res['fit_check']}")
    print(f"single-image accuracy, training minus held-out views (dB): accurate fields "
          f"{res['single_train_minus_holdout_db']['accurate_fields_min']:+.2f} .. {res['single_train_minus_holdout_db']['accurate_fields_max']:+.2f}")
    for label in ("all", "fit_accepted"):
        print(f"AUC ({label}, n={res[f'auc_{label}']['n']}, worse={res[f'auc_{label}']['n_worse']}): " +
              "  ".join(f"{k} {v:.3f}" for k, v in res[f"auc_{label}"].items() if k not in ("n", "n_worse")))
    print("accurate fields: d_hold %.2f..%.2f  drop %+.2f..%+.2f  dp_hold %.2f..%.2f" % (
        res["accurate_fields"]["d_hold_min"], res["accurate_fields"]["d_hold_max"], res["accurate_fields"]["drop_min"],
        res["accurate_fields"]["drop_max"], res["accurate_fields"]["dp_hold_min"], res["accurate_fields"]["dp_hold_max"]))
    print("d_hold on the fit-accepted fields:", res["d_hold_correlation_fit_accepted"])
    print("false accepts of the fit check:")
    for e in res["false_accepts_of_fit_check"]:
        print(f"  {e['name']:50s} fit {e['fit']:6.2f} fuse {e['fuse']:6.2f} single {e['single']:6.2f} phys {e['phys']:6.2f} | "
              f"d_hold {e['d_hold']:6.2f} d_train {e['d_train']:6.2f} drop {e['drop']:+6.2f} dp_hold {e['dp_hold']:6.2f}")
    for key, s in sweeps.items():
        print(f"combined fit AND {key} ({s['direction']}):")
        for k in ("no_false_accept", "no_extra_false_reject", "best_mean_output"):
            print(f"  {k:22s} {s[k]}")
    print("oracle:", res["oracle"])


if __name__ == "__main__":
    main()
