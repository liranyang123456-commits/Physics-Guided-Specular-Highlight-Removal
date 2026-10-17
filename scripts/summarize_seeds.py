"""Seeds 0-2 of every setting of the prior-sensitivity study (Table 10): physical and fused PSNR and the training-view fit.
Seeds 1 and 2 use other perturbation noise and view order (scripts/run_sensitivity_v2.py --seed k).
Writes outputs/v2/sensitivity_seeds.json.

    python scripts/summarize_seeds.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_train_fit import fit  # noqa: E402

AXES = ["geo_mild", "geo_severe", "albedo15", "albedo30", "light", "drop25", "drop50", "sparse6", "sparse3"]
FAILING = {"geo_mild", "geo_severe", "light", "drop25", "drop50"}


def main() -> None:
    device = torch.device("cuda")
    S = ROOT / "outputs" / "v2" / "sensitivity"
    fits0 = json.loads((ROOT / "outputs/v2/train_fit.json").read_text(encoding="utf-8"))
    per = {}
    for axis in AXES:
        rows = []
        for seed in range(3):
            d = json.loads((S / f"{axis}_seed{seed}.json").read_text(encoding="utf-8"))
            key = f"sens/{axis}_seed{seed}"
            f = fits0[key]["full_psnr"] if key in fits0 else fit(S / f"{axis}_seed{seed}.pt", device)["full_psnr"]
            rows.append({"seed": seed, "phys": d["phys"]["psnr"], "fuse": d["fuse"]["psnr"], "single": d["single"]["psnr"],
                         "gate_mean": d.get("gate_mean"), "train_fit": f})
        ph, fu = np.array([r["phys"] for r in rows]), np.array([r["fuse"] for r in rows])
        per[axis] = {"seeds": rows, "phys_min": float(ph.min()), "phys_max": float(ph.max()), "phys_mean": float(ph.mean()),
                     "fuse_min": float(fu.min()), "fuse_max": float(fu.max()), "failing": axis in FAILING}
        print(f"{axis:10s}: phys {', '.join(f'{x:.2f}' for x in ph)}  fuse {', '.join(f'{x:.2f}' for x in fu)}  fit "
              f"{', '.join(f'{r['train_fit']:.1f}' for r in rows)}", flush=True)
    fail = [r for a in AXES if a in FAILING for r in per[a]["seeds"]]
    ok = [r for a in AXES if a not in FAILING for r in per[a]["seeds"]]
    summary = {"failing_phys_min": min(r["phys"] for r in fail), "failing_phys_max": max(r["phys"] for r in fail),
               "failing_fuse_below_single_min": min(r["single"] - r["fuse"] for r in fail),
               "failing_fuse_below_single_max": max(r["single"] - r["fuse"] for r in fail),
               "failing_fit_max": max(r["train_fit"] for r in fail), "accurate_fit_min": min(r["train_fit"] for r in ok),
               "check_separates_all": max(r["train_fit"] for r in fail) < 40 <= min(r["train_fit"] for r in ok),
               "n_failing_runs": len(fail), "n_accurate_runs": len(ok)}
    (ROOT / "outputs/v2/sensitivity_seeds.json").write_text(json.dumps({"per_axis": per, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
