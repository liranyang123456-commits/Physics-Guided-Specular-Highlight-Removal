"""Summarize the drift-diagnosis runs of diag_drift.py into outputs/drift_diag/summary.json and print a table."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "drift_diag"

RUNS = [
    ("e0_baseline", "paper recipe (geometry+albedo exact, specular default; lr 2e-3, refiner)"),
    ("e4_norefiner", "... without refiner"),
    ("e8_noreg", "... without refiner, smoothness and texture regularizers"),
    ("e12_nospec", "... without refiner, without specular supervision"),
    ("e14_lowlr", "... without refiner, lr 2e-4"),
    ("e1_oracle_init", "complete generating state (oracle), lr 2e-3, refiner"),
    ("e5_oracle_norefiner", "oracle, without refiner"),
    ("e11_oracle_nospec", "oracle, without refiner, without specular supervision"),
    ("e13_oracle_lowlr", "oracle, without refiner, lr 2e-4"),
    ("e15_speconly", "paper init, without diffuse supervision (lr 2e-3)"),
    ("e16_speconly_hilr", "paper init, without diffuse supervision (lr 1e-2)"),
]


def main() -> None:
    rows = []
    for name, label in RUNS:
        f = OUT / f"{name}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        first, last = d["curve"][0], d["curve"][-1]
        row = {"run": name, "setting": label, "holdout_diffuse_psnr_start": first["holdout_diffuse_psnr"],
               "holdout_diffuse_psnr_end": last["holdout_diffuse_psnr"], "train_diffuse_psnr_end": last["train_diffuse_psnr"]}
        if "after" in d:
            for k in ("specular_weight_mae", "metalness_mae", "roughness_mae", "albedo_mae"):
                row[k + "_start"] = d["before"]["materials"][k]
                row[k + "_end"] = d["after"]["materials"][k]
            row["specular_l1_start"] = d["before"]["terms_train"]["specular_l1"]
            row["specular_l1_end"] = d["after"]["terms_train"]["specular_l1"]
        rows.append(row)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{'run':<22s} {'holdout start->end':>20s} {'spec.w MAE':>14s} {'metal. MAE':>14s} {'rough. MAE':>14s} {'spec L1':>16s}")
    for r in rows:
        mat = ""
        if "specular_weight_mae_start" in r:
            mat = (f"{r['specular_weight_mae_start']:.3f}->{r['specular_weight_mae_end']:.3f} "
                   f"{r['metalness_mae_start']:.3f}->{r['metalness_mae_end']:.3f} "
                   f"{r['roughness_mae_start']:.3f}->{r['roughness_mae_end']:.3f} "
                   f"{r['specular_l1_start']:.4f}->{r['specular_l1_end']:.4f}")
        print(f"{r['run']:<22s} {r['holdout_diffuse_psnr_start']:8.2f} -> {r['holdout_diffuse_psnr_end']:6.2f}   {mat}")


if __name__ == "__main__":
    main()
