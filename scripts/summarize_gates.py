"""Summarize the fusion-gate study (scene_0070): training repeats, cross-fitting, corruption augmentation.

Reads the metrics written by the gate trainings and by scripts/eval_sensitivity_gate.py and writes
outputs/v2/gate_seeds.json (read by the manuscript number scripts).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs" / "v2"


def hold(rel: str) -> dict:
    return json.loads((V2 / rel / "metrics.json").read_text(encoding="utf-8"))["holdout"]


def main() -> None:
    orig = hold("fusion_scene_0070")
    ins = [hold(f"fusion_seeds/seed{s}") for s in range(5)]
    xc = [hold("fusion_crossfit_clean_scene_0070")] + [hold(f"fusion_crossfit_clean_seeds/seed{s}") for s in range(1, 5)]
    gates = json.loads((V2 / "sensitivity" / "gates_all.json").read_text(encoding="utf-8"))["results"]
    fail = ["geo_mild_seed0", "geo_severe_seed0", "light_seed0", "drop25_seed0", "drop50_seed0"]
    clean = ["clean_seed0", "clean_seed1", "clean_seed2"]

    def over(names: list[str], states: list[str]) -> dict:
        vals = [gates[s][g]["psnr"] for s in states for g in names]
        return {"min": float(min(vals)), "max": float(max(vals)), "mean": float(np.mean(vals))}

    out = {
        "phys": orig["world_psnr"], "single": orig["single_psnr"], "orig": orig["hybrid_psnr"],
        "orig_ssim": orig["hybrid_ssim"],
        "insample": [h["hybrid_psnr"] for h in ins], "insample_ssim": [h["hybrid_ssim"] for h in ins],
        "crossfit": [h["hybrid_psnr"] for h in xc], "crossfit_ssim": [h["hybrid_ssim"] for h in xc],
        "crossfit_mix08": hold("fusion_crossfit_mix08_scene_0070")["hybrid_psnr"],
        "crossfit_robust": hold("fusion_crossfit_robust_scene_0070")["hybrid_psnr"],
        "augment_only_robust": hold("fusion_robust_scene_0070")["hybrid_psnr"],
        "insample_2000_steps": hold("fusion_cleanctl_scene_0070")["hybrid_psnr"],
        "corrupt_min": over(["orig", "ins0", "ins1", "ins2", "ins3", "ins4"], fail)["min"],
        "corrupt_max": over(["orig", "ins0", "ins1", "ins2", "ins3", "ins4"], fail)["max"],
        "corrupt": {
            "orig": over(["orig"], fail), "insample": over([f"ins{k}" for k in range(5)], fail),
            "crossfit": over([f"xc{k}" for k in range(5)], fail), "mix08": over(["xmix08"], fail),
            "robust": over(["xrobust"], fail), "analytic": over(["analytic"], fail),
        },
        "clean_field": {
            "orig": over(["orig"], clean), "insample": over([f"ins{k}" for k in range(5)], clean),
            "crossfit": over([f"xc{k}" for k in range(5)], clean), "mix08": over(["xmix08"], clean),
            "robust": over(["xrobust"], clean), "analytic": over(["analytic"], clean),
        },
    }
    (V2 / "gate_seeds.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    ins_a, xc_a = np.array(out["insample"]), np.array(out["crossfit"])
    print(f"physical {out['phys']:.2f} | single {out['single']:.2f} | original gate {out['orig']:.2f}")
    print(f"in-sample gates (5 seeds): {ins_a.round(2).tolist()}  mean {ins_a.mean():.2f} sd {ins_a.std(ddof=1):.2f}")
    print(f"cross-fitted gates (5 seeds): {xc_a.round(2).tolist()}  mean {xc_a.mean():.2f} sd {xc_a.std(ddof=1):.2f}")
    print("corruption-augmented cross-fitted: p_clean 0.8 ->", round(out["crossfit_mix08"], 2), "; 0.4 ->", round(out["crossfit_robust"], 2))
    for k, v in out["corrupt"].items():
        print(f"under corruption {k:9s}: {v['min']:.2f}-{v['max']:.2f} (mean {v['mean']:.2f})")


if __name__ == "__main__":
    main()
