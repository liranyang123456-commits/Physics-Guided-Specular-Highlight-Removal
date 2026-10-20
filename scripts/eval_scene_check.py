"""Effect of the training-view fit check on every evaluated condition (sensitivity states, ten scenes, material scenes, real MVS).

Reads the training-view fits (scripts/eval_train_fit.py -> outputs/v2/train_fit.json) and the fused results of the gates
(scripts/eval_sensitivity_gate.py -> sensitivity/gates_all.json; scripts/eval_gate_scenes.py -> gate_scenes_all.json), applies
the check with tau = 40 dB, and writes outputs/v2/scene_check.json.  tau is a round number inside the gap between the fits of
correct fields and of failing fields (printed below); no value in the gap changes any decision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs" / "v2"
TAU = 40.0


def main() -> None:
    fit = json.loads((V2 / "train_fit.json").read_text(encoding="utf-8"))
    sens = json.loads((V2 / "sensitivity" / "gates_all.json").read_text(encoding="utf-8"))["results"]
    scenes = json.loads((V2 / "gate_scenes_all.json").read_text(encoding="utf-8"))["results"]
    out = {"tau": TAU, "sensitivity": {}, "ten": {}, "material": {}, "mvs": {}}

    def entry(res: dict, f: float) -> dict:
        ok = f >= TAU
        xc = [res[f"xc{k}"]["psnr"] for k in range(5)]
        ins = [res[f"ins{k}"]["psnr"] for k in range(5)]
        single = res["single"]["psnr"]
        chk = lambda v: v if ok else single
        return {"fit": f, "accepted": bool(ok), "phys": res["phys"]["psnr"], "single": single,
                "orig": res["orig"]["psnr"], "orig_check": chk(res["orig"]["psnr"]),
                "insample_mean": float(np.mean(ins)), "insample_check_mean": float(np.mean([chk(v) for v in ins])),
                "crossfit_mean": float(np.mean(xc)), "crossfit_sd": float(np.std(xc, ddof=1)),
                "crossfit_check_mean": float(np.mean([chk(v) for v in xc])), "xc0": xc[0], "xc0_check": chk(xc[0])}

    for key, res in sens.items():
        out["sensitivity"][key] = entry(res, fit[f"sens/{key}"]["full_psnr"])
    for key, res in scenes.items():
        group, name = key.split("/")
        out[group][name] = entry(res, fit[key]["full_psnr"])
    for name, folder in (("dtu", "dtu_scan1_corrected_s3000"), ("blended", "blendedmvs_57f8d9bb_corrected_s3000")):
        f = json.loads((ROOT / "outputs" / folder / "world_metrics_train.json").read_text(encoding="utf-8"))["summary"]["full_psnr"]
        out["mvs"][name] = {"fit": f, "accepted": bool(f >= TAU)}
    fits = {**{f"sens/{k}": v["fit"] for k, v in out["sensitivity"].items()},
            **{f"ten/{k}": v["fit"] for k, v in out["ten"].items()}, **{f"material/{k}": v["fit"] for k, v in out["material"].items()}}
    accepted = [v for k, v in fits.items() if v >= TAU]
    rejected = [v for k, v in fits.items() if v < TAU]
    out["separation"] = {"min_accepted": float(min(accepted)), "max_rejected": float(max(rejected)),
                         "gap": float(min(accepted) - max(rejected)), "n_accepted": len(accepted), "n_rejected": len(rejected)}
    (V2 / "scene_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    sep = out["separation"]
    print(f"accepted {sep['n_accepted']} fields (fit >= {sep['min_accepted']:.2f} dB), rejected {sep['n_rejected']} (fit <= {sep['max_rejected']:.2f} dB); "
          f"gap {sep['gap']:.2f} dB; real MVS fits: " + ", ".join(f"{k} {v['fit']:.2f}" for k, v in out["mvs"].items()))
    for key, v in out["sensitivity"].items():
        print(f"  {key:18s} fit {v['fit']:6.2f} {'accept' if v['accepted'] else 'REJECT'}  single {v['single']:.2f}  xc0 {v['xc0']:.2f} -> {v['xc0_check']:.2f}")


if __name__ == "__main__":
    main()
