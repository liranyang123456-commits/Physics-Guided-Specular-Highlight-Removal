"""Summary of the estimation study (scripts/run_sensitivity_unfrozen.py) for the manuscript.

Writes outputs/v2/estimation_summary.json with
  scene_0070   the perturbations of Table 10 with the perturbed quantity taken as given (frozen) and estimated
  lighting_ten the lighting perturbation on the ten benchmark scenes: given (frozen), estimated, and the exact-prior controls,
               with the albedo-free initialization of the main results (lighting_ten_exact_albedo: exact albedo at the start)
  geometry     mild geometry noise on scenes 0071-0073: given and estimated, and the exact-geometry control
  held_out     fields whose training views are reproduced well but whose held-out views are not
  view_check   rank correlation of the held-out median residual with the accuracy of the physical render (scripts/analyze_view_check.py)

    python scripts/summarize_estimation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs" / "v2"
UNF = V2 / "sensitivity_unfrozen"
FROZEN = V2 / "sensitivity"


def load(folder: Path, stem: str) -> dict | None:
    p = folder / f"{stem}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    out = {"phys": d["phys"]["psnr"], "fuse": d["fuse"]["psnr"], "single": d["single"]["psnr"], "gate_mean": d["gate_mean"]}
    if "fit_train" in d:
        out["train_fit"] = d["fit_train"]["full_psnr"]
        out["holdout_med_err"] = d["fit_holdout"]["med_err"]
        out["light_angle_error_deg"] = d["light_angle_error_deg"]["unseen"]
        out["position_error"] = d["position_error_mean"]
    else:                                   # fields of scripts/run_sensitivity_v2.py: training-view fit from outputs/v2/train_fit.json
        fits = json.loads((V2 / "train_fit.json").read_text(encoding="utf-8"))
        key = f"sens/{stem}"
        if key in fits:
            out["train_fit"] = fits[key]["full_psnr"]
    return out


def scene70_rows() -> list[dict]:
    spec = [
        # label, perturbation, frozen source, estimated sources {variant: stem}
        ("lighting", "light (+25% intensity, direction noise)", (FROZEN, "light_seed0"),
         {"lighting": "light_lighting_seed0"}),
        ("positions", "positions (sigma 0.03)", (UNF, "geo_pos_none_seed0"),
         {"positions": "geo_pos_geometry_seed0_means", "all geometry": "geo_pos_geometry_seed0_all"}),
        ("normals", "normals (sigma 0.25)", (UNF, "geo_nrm_none_seed0"),
         {"normals": "geo_nrm_geometry_seed0_normal", "all geometry": "geo_nrm_geometry_seed0_all"}),
        ("geo_mild", "positions and normals (mild)", (FROZEN, "geo_mild_seed0"),
         {"positions+normals": "geo_mild_geometry_seed0_means_normal", "all geometry": "geo_mild_geometry_seed0"}),
        ("geo_severe", "positions and normals (severe)", (FROZEN, "geo_severe_seed0"),
         {"positions+normals": "geo_severe_geometry_seed0_means_normal", "all geometry": "geo_severe_geometry_seed0"}),
        ("drop25", "coverage (25% of the Gaussians removed)", (FROZEN, "drop25_seed0"),
         {"opacity": "drop25_geometry_seed0_opacity", "opacity+scale": "drop25_geometry_seed0_opacity_scale",
          "all geometry": "drop25_geometry_seed0"}),
        ("drop50", "coverage (50% removed)", (FROZEN, "drop50_seed0"),
         {"opacity": "drop50_geometry_seed0_opacity", "opacity+scale": "drop50_geometry_seed0_opacity_scale",
          "all geometry": "drop50_geometry_seed0"}),
        ("geolight", "mild geometry noise and lighting perturbation", (UNF, "geolight_none_seed0"),
         {"lighting": "geolight_lighting_seed0", "lighting+positions+normals": "geolight_both_seed0_means_normal",
          "lighting+all geometry": "geolight_both_seed0"}),
        ("exact", "none (exact priors; estimation unnecessary)", (FROZEN, "clean_seed0"),
         {"lighting": "clean_lighting_seed0", "positions+normals": "clean_geometry_seed0_means_normal",
          "all geometry": "clean_geometry_seed0"}),
    ]
    rows = []
    for key, text, (folder, stem), est in spec:
        row = {"key": key, "perturbation": text, "given": load(folder, stem),
               "estimated": {name: load(UNF, s) for name, s in est.items()}}
        rows.append(row)
    return rows


SEED_SPEC = {
    # key: (given folder, given stem, {variant of Table estimate: stem}); "{s}" is the seed
    "lighting": ((FROZEN, "light_seed{s}"), {"lighting": "light_lighting_seed{s}"}),
    "positions": ((UNF, "geo_pos_none_seed{s}"), {"positions": "geo_pos_geometry_seed{s}_means", "all geometry": "geo_pos_geometry_seed{s}_all"}),
    "normals": ((UNF, "geo_nrm_none_seed{s}"), {"normals": "geo_nrm_geometry_seed{s}_normal", "all geometry": "geo_nrm_geometry_seed{s}_all"}),
    "geo_mild": ((FROZEN, "geo_mild_seed{s}"), {"positions+normals": "geo_mild_geometry_seed{s}_means_normal",
                                               "all geometry": "geo_mild_geometry_seed{s}"}),
    "geo_severe": ((FROZEN, "geo_severe_seed{s}"), {"positions+normals": "geo_severe_geometry_seed{s}_means_normal",
                                                   "all geometry": "geo_severe_geometry_seed{s}"}),
    "drop25": ((FROZEN, "drop25_seed{s}"), {"opacity": "drop25_geometry_seed{s}_opacity", "all geometry": "drop25_geometry_seed{s}"}),
    "drop50": ((FROZEN, "drop50_seed{s}"), {"opacity": "drop50_geometry_seed{s}_opacity", "all geometry": "drop50_geometry_seed{s}"}),
    "geolight": ((UNF, "geolight_none_seed{s}"), {"lighting+positions+normals": "geolight_both_seed{s}_means_normal",
                                                 "lighting+all geometry": "geolight_both_seed{s}"}),
    "exact": ((FROZEN, "clean_seed{s}"), {"lighting": "clean_lighting_seed{s}", "positions+normals": "clean_geometry_seed{s}_means_normal",
                                          "all geometry": "clean_geometry_seed{s}"}),
}


def scene70_seeds(seeds=(0, 1, 2)) -> dict:
    """Seeds 0-2 of every entry of Table estimate (they differ in perturbation noise and view order), with the min-max range."""
    out = {}
    for key, ((folder, given), est) in SEED_SPEC.items():
        runs = {"given": [load(folder, given.format(s=s)) for s in seeds]}
        runs.update({name: [load(UNF, stem.format(s=s)) for s in seeds] for name, stem in est.items()})
        out[key] = {name: {"runs": r, "complete": all(x is not None for x in r),
                           **({"phys_min": min(x["phys"] for x in r), "phys_max": max(x["phys"] for x in r),
                               "fuse_min": min(x["fuse"] for x in r), "fuse_max": max(x["fuse"] for x in r)}
                              if all(x is not None for x in r) else {})}
                    for name, r in runs.items()}
    return out


def lighting_ten(suffix: str = "_af") -> dict:
    """suffix "_af": albedo initialized to 0.25 as in the main results (--init geometry); "": exact albedo at initialization."""
    per = []
    for n in range(70, 80):
        s = f"s00{n}_" if n != 70 else ""
        row = {"scene": n,
               "frozen": load(UNF, f"{s}light_none_seed0{suffix}"),
               "estimated": load(UNF, f"{s}light_lighting_seed0{suffix}"),
               "exact_estimated": load(UNF, f"{s}clean_lighting_seed0{suffix}")}
        # exact priors, frozen: the field of the ten-scene study
        ten = V2 / f"world_scene_00{n}" / "split_metrics.json"
        if ten.exists():
            row["exact_frozen_diffuse_train"] = None
        per.append(row)
    done = [r for r in per if r["frozen"] and r["estimated"] and r["exact_estimated"]]
    out = {"per_scene": per, "n_complete": len(done)}
    for key in ("frozen", "estimated", "exact_estimated"):
        for m in ("phys", "fuse", "single"):
            vals = [r[key][m] for r in done]
            if vals:
                out[f"{key}_{m}_mean"] = float(np.mean(vals))
                out[f"{key}_{m}_min"] = float(np.min(vals))
                out[f"{key}_{m}_max"] = float(np.max(vals))
    if done:
        out["estimated_light_angle_error_max"] = float(max(r["estimated"]["light_angle_error_deg"] for r in done))
        out["estimated_beats_single"] = int(sum(r["estimated"]["phys"] > r["estimated"]["single"] for r in done))
        out["estimated_fuse_beats_single"] = int(sum(r["estimated"]["fuse"] > r["estimated"]["single"] for r in done))
    return out


def geometry_scenes() -> dict:
    per, targeted = [], []
    for n in (71, 72, 73):
        s = f"s00{n}_"
        per.append({"scene": n, "frozen": load(UNF, f"{s}geo_mild_none_seed0"),
                    "estimated": load(UNF, f"{s}geo_mild_geometry_seed0"),
                    "exact_estimated": load(UNF, f"{s}clean_geometry_seed0")})
        targeted.append({"scene": n, "frozen": load(UNF, f"{s}geo_mild_none_seed0"),
                         "estimated": load(UNF, f"{s}geo_mild_geometry_seed0_means_normal"),
                         "exact_estimated": load(UNF, f"{s}clean_geometry_seed0_means_normal")})
    out = {"per_scene": per}
    if all(r["estimated"] and r["exact_estimated"] for r in targeted):
        out["targeted"] = targeted
    return out


def held_out() -> list[dict]:
    rows = []
    for label, folder, stem in (
            ("sparse3 (views 1, 5, 9)", FROZEN, "sparse3_seed0"), ("sparse6 (views 1, 3, 5, 7, 9, 11)", FROZEN, "sparse6_seed0"),
            ("one-sided, 6 views (1, 2, 3, 5, 6, 7)", UNF, "arc6_none_seed0"), ("one-sided, 3 views (1, 2, 3)", UNF, "arc3_none_seed0"),
            ("free geometry, mild noise", UNF, "geo_mild_geometry_seed0"),
            ("free positions and normals, mild noise", UNF, "geo_mild_geometry_seed0_means_normal"),
            ("free geometry, exact start", UNF, "clean_geometry_seed0")):
        r = load(folder, stem)
        if r:
            rows.append({"label": label, **r})
    return rows


def main() -> None:
    summary = {"scene_0070": scene70_rows(), "scene_0070_seeds": scene70_seeds(), "lighting_ten": lighting_ten("_af"),
               "lighting_ten_exact_albedo": lighting_ten(""), "geometry": geometry_scenes(), "held_out": held_out()}
    vc = V2 / "view_check_analysis.json"
    if vc.exists():
        summary["view_check"] = json.loads(vc.read_text(encoding="utf-8"))
    (V2 / "estimation_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    def fmt(r):
        return "   --   " if r is None else f"{r['phys']:5.2f}/{r['fuse']:5.2f}"

    print("scene_0070: perturbation | given (phys/fuse) | estimated variants")
    for row in summary["scene_0070"]:
        est = "  ".join(f"{k}: {fmt(v)}" for k, v in row["estimated"].items())
        print(f"  {row['key']:11s} {fmt(row['given'])} | {est}")
    print("\nscene_0070, physical branch over seeds 0-2 (min-max)")
    for key, entry in summary["scene_0070_seeds"].items():
        print(f"  {key:11s} " + "  ".join(f"{name}: " + (f"{v['phys_min']:5.2f}-{v['phys_max']:5.2f}" if v["complete"] else "incomplete")
                                          for name, v in entry.items()))
    li = summary["lighting_ten"]
    print(f"\nlighting, {li['n_complete']} complete scenes: frozen phys {li.get('frozen_phys_mean', float('nan')):.2f}, "
          f"estimated phys {li.get('estimated_phys_mean', float('nan')):.2f} (fuse {li.get('estimated_fuse_mean', float('nan')):.2f}, "
          f"single {li.get('estimated_single_mean', float('nan')):.2f}), exact+estimated {li.get('exact_estimated_phys_mean', float('nan')):.2f}")
    for r in li["per_scene"]:
        print(f"  scene_00{r['scene']}: frozen {fmt(r['frozen'])}  estimated {fmt(r['estimated'])}  exact+est {fmt(r['exact_estimated'])}")
    print("\ngeometry (mild noise), scenes 71-73")
    for r in summary["geometry"]["per_scene"]:
        print(f"  scene_00{r['scene']}: frozen {fmt(r['frozen'])}  estimated {fmt(r['estimated'])}  exact+free {fmt(r['exact_estimated'])}")
    print("\nheld-out-only failures: label | train fit | phys / fuse / single")
    for r in summary["held_out"]:
        print(f"  {r['label']:42s} {r.get('train_fit', float('nan')):6.2f} | {r['phys']:5.2f} / {r['fuse']:5.2f} / {r['single']:5.2f}")


if __name__ == "__main__":
    main()
