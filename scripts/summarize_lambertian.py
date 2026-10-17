"""Lambertian field (no specular lobe; scripts/run_sensitivity_unfrozen.py --no-specular) against the BRDF field on the twenty
test scenes.  Writes outputs/v2/lambertian_baseline.json.

    python scripts/summarize_lambertian.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs" / "v2"


def main() -> None:
    unused = {r["scene"]: r for r in json.loads((V2 / "unused_scenes.json").read_text(encoding="utf-8"))["per_scene"]}
    per = {}
    for n in range(70, 90):
        name = f"scene_00{n}"
        stem = ("" if n == 70 else f"s00{n}_") + "none_none_seed0_lamb"
        d = json.loads((V2 / "sensitivity_unfrozen" / f"{stem}.json").read_text(encoding="utf-8"))
        if n < 80:
            brdf = json.loads((V2 / f"ablation_{name}" / "ablation_metrics.json").read_text(encoding="utf-8"))["variants"]["ours_phys"]
        else:
            brdf = unused[name]["ours_phys"]
        per[name] = {"lambertian": d["phys"]["psnr"], "lambertian_ssim": d["phys"]["ssim"], "brdf": brdf["psnr"], "brdf_ssim": brdf["ssim"],
                     "single": d["single"]["psnr"], "train_fit": d["fit_train"]["full_psnr"]}
    summary = {}
    for group, ns in (("ten", range(70, 80)), ("unused", range(80, 90)), ("all", range(70, 90))):
        rows = [per[f"scene_00{n}"] for n in ns]
        lam, brdf = np.array([r["lambertian"] for r in rows]), np.array([r["brdf"] for r in rows])
        summary[group] = {"lambertian": float(lam.mean()), "lambertian_min": float(lam.min()), "lambertian_max": float(lam.max()),
                          "brdf": float(brdf.mean()), "brdf_above": int((brdf > lam).sum()), "n": len(rows),
                          "single": float(np.mean([r["single"] for r in rows])),
                          "lambertian_above_single": int(sum(r["lambertian"] > r["single"] for r in rows)),
                          "lambertian_train_fit": float(np.mean([r["train_fit"] for r in rows]))}
    (V2 / "lambertian_baseline.json").write_text(json.dumps({"per_scene": per, "summary": summary}, indent=2), encoding="utf-8")
    for n in range(70, 90):
        r = per[f"scene_00{n}"]
        print(f"scene_00{n}: Lambertian {r['lambertian']:.2f}  BRDF {r['brdf']:.2f}  single {r['single']:.2f}  train fit {r['train_fit']:.2f}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
