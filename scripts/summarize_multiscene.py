"""Summarize the 10-scene strict-holdout evaluation for the revision.

Checks that re-evaluation reproduces the archived per-view numbers, then
reports per-variant statistics, per-scene consistency, per-shape-family
means, and the confidence--error correlation.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import os

    os.chdir(ROOT)
    paths = sorted(glob.glob("outputs/world_ablation_scene_00*/ablation_metrics.json"))
    max_diff, compared = 0.0, 0
    for path in paths:
        backup = Path(path.replace("ablation_metrics.json", "ablation_metrics_v1_backup.json"))
        if not backup.exists():  # archived first evaluation; not part of the release
            continue
        new, old = json.load(open(path)), json.load(open(backup))
        compared += 1
        for name, variant in old["variants"].items():
            for a, b in zip(variant.get("per_view", []), new["variants"][name].get("per_view", [])):
                max_diff = max(max_diff, abs(a["psnr"] - b["psnr"]), 100 * abs(a["ssim"] - b["ssim"]))
    if compared:
        print(f"max |diff| vs archived per-view values (dB, SSIM x100): {max_diff:.4f}")

    agg = json.load(open("outputs/world_holdout_multiscene.json"))
    keys = ["input", "ours_phys", "ours_pro", "fixed_blend", "fusion_analytic",
            "fusion_no_correction", "fusion_full", "ours_refined",
            "dhan_ft", "dhan_fixed_blend", "dhan_analytic", "dhan_gate", "tshrnet_ft"]
    for key in keys:
        p, s = agg["variants"][key]["psnr"], agg["variants"][key]["ssim"]
        print(f"{key:21s} PSNR mean {p['mean']:.2f} med {p['median']:.2f} IQR [{p['iqr_lo']:.2f}, {p['iqr_hi']:.2f}]"
              f" | SSIM mean {s['mean']:.4f} med {s['median']:.4f}")

    scenes = [d["scene"][-4:] for d in agg["variants"]["fusion_full"]["per_scene"]]
    per_scene = {k: np.array([d["psnr"] for d in agg["variants"][k]["per_scene"]]) for k in keys}
    per_scene_ssim = {k: np.array([d["ssim"] for d in agg["variants"][k]["per_scene"]]) for k in keys}
    fuse = per_scene["fusion_full"]
    for key in ["ours_phys", "ours_pro", "fixed_blend", "fusion_analytic", "input"]:
        print(f"fusion_full > {key:16s}: PSNR {int((fuse > per_scene[key]).sum())}/10, "
              f"SSIM {int((per_scene_ssim['fusion_full'] > per_scene_ssim[key]).sum())}/10 scenes")
    print(f"phys > single: PSNR {int((per_scene['ours_phys'] > per_scene['ours_pro']).sum())}/10 scenes")
    for key in ["dhan_fixed_blend", "dhan_analytic", "dhan_gate", "fusion_full"]:
        print(f"{key:16s} > dhan_ft: PSNR {int((per_scene[key] > per_scene['dhan_ft']).sum())}/10, "
              f"SSIM {int((per_scene_ssim[key] > per_scene_ssim['dhan_ft']).sum())}/10 scenes")
    print("per-scene PSNR (dhan_ft / dhan_fixed / dhan_analytic / dhan_gate):")
    for i, scene in enumerate(scenes):
        print(f"  {scene} " + " / ".join(f"{per_scene[k][i]:.2f}" for k in
                                        ["dhan_ft", "dhan_fixed_blend", "dhan_analytic", "dhan_gate"]))

    manifest = json.load(open("data/world_benchmark_v2/manifest.json"))
    shapes = {f"{f['scene']:04d}": f["shape"] for f in manifest["frames"] if f["split"] == "test"}
    print("per-scene PSNR (phys / single / fixed / fuse) and shape:")
    for i, scene in enumerate(scenes):
        print(f"  {scene} {shapes[scene]:9s} {per_scene['ours_phys'][i]:.2f} / {per_scene['ours_pro'][i]:.2f} / "
              f"{per_scene['fixed_blend'][i]:.2f} / {fuse[i]:.2f}")
    families = {}
    for i, scene in enumerate(scenes):
        families.setdefault(shapes[scene], []).append(i)
    family_table = {}
    for family, idx in sorted(families.items()):
        family_table[family] = {k: float(per_scene[k][idx].mean()) for k in ["ours_phys", "ours_pro", "fusion_full"]}
        family_table[family]["n_scenes"] = len(idx)
        print(f"  family {family:9s} n={len(idx)}: phys {family_table[family]['ours_phys']:.2f} "
              f"single {family_table[family]['ours_pro']:.2f} fuse {family_table[family]['fusion_full']:.2f}")

    rows = []
    for path in paths:
        rows += json.load(open(path))["confidence_correlation"]
    r = np.array([x["pearson_e_vs_diffuse_error"] for x in rows])
    low = np.array([x["diffuse_l1_low_conf_half"] for x in rows])
    high = np.array([x["diffuse_l1_high_conf_half"] for x in rows])
    corr = {
        "n_views": len(rows),
        "pearson_mean": float(r.mean()),
        "pearson_median": float(np.median(r)),
        "pearson_min": float(r.min()),
        "positive_views": int((r > 0).sum()),
        "low_conf_l1_mean": float(low.mean()),
        "high_conf_l1_mean": float(high.mean()),
        "ratio_of_means": float(low.mean() / high.mean()),
        "low_gt_high_views": int((low > high).sum()),
        "per_view": rows,
    }
    print(f"correlation over {len(rows)} views: Pearson mean {r.mean():.3f}, median {np.median(r):.3f}, "
          f"min {r.min():.3f}, positive {int((r > 0).sum())}/{len(r)}; diffuse L1 low-conf {low.mean():.4f} vs "
          f"high-conf {high.mean():.4f} (x{low.mean() / high.mean():.2f}); low>high in {int((low > high).sum())}/{len(r)}")
    Path("outputs/world_confidence_correlation.json").write_text(json.dumps(corr, indent=2), encoding="utf-8")
    Path("outputs/world_multiscene_family.json").write_text(json.dumps(family_table, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
