"""Old-vs-new comparison of the physical-branch experiments (revision, step A).

v1 = recipe of the submitted manuscript (7x7 SSIM photometric term, scene lr 2e-3, 3000 steps)  -> outputs/
v2 = corrected recipe (RGB L1 photometric loss only, scene lr 0.03, 1500 steps)                 -> outputs/v2/

Nothing is re-used between the two: the v2 numbers come from freshly trained world fields, a freshly trained fusion gate,
and the same evaluation script (scripts/eval_world_ablation.py), the same strict holdout views {0, 4, 8}, and the same
metric functions as v1.  The script reads the JSON files of both runs and writes a side-by-side report to
outputs/v2/comparison_v1_v2.{md,json}.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from aggregate_world_holdout import collect  # noqa: E402

KEYS = ["input", "ours_phys", "ours_pro", "fixed_blend", "fusion_analytic", "fusion_no_correction", "fusion_full",
        "dhan_ft", "dhan_fixed_blend", "dhan_analytic", "dhan_gate", "tshrnet_ft"]
LABEL = {
    "input": "Input (no removal)", "ours_phys": "Ours-Phys", "ours_pro": "Ours single-image",
    "fixed_blend": "Fixed blend (Phys+single)", "fusion_analytic": "Fusion, analytic gate only",
    "fusion_no_correction": "Fusion, no correction", "fusion_full": "Ours-Fuse",
    "dhan_ft": "DHAN-SHR (FT)", "dhan_fixed_blend": "DHAN-SHR fixed blend",
    "dhan_analytic": "DHAN-SHR analytic gate", "dhan_gate": "DHAN-SHR + learned gate",
    "tshrnet_ft": "TSHRNet (FT)",
}
V1_GLOB = "outputs/world_ablation_scene_00[7][0-9]/ablation_metrics.json"
V2_GLOB = "outputs/v2/ablation_scene_00[7][0-9]/ablation_metrics.json"   # scenes 0070-0079; 0080-0089: eval_unused_scenes.py
lines: list[str] = []
report: dict = {}


def emit(text: str = "") -> None:
    lines.append(text)
    print(text, flush=True)


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def scene0070() -> None:
    old, new = load("outputs/world_ablation_scene_0070/ablation_metrics.json"), \
        load("outputs/v2/ablation_scene_0070/ablation_metrics.json")
    emit("## 1. scene_0070, strict holdout (3 views {0,4,8}), mean over views\n")
    emit("| variant | v1 PSNR | v1 SSIM | v2 PSNR | v2 SSIM | dPSNR |")
    emit("|---|---|---|---|---|---|")
    rows = {}
    for key in KEYS:
        a, b = old["variants"].get(key), new["variants"].get(key)
        if a is None or b is None:
            continue
        rows[key] = {"v1": [a["psnr"], a["ssim"]], "v2": [b["psnr"], b["ssim"]]}
        emit(f"| {LABEL[key]} | {a['psnr']:.2f} | {a['ssim']:.4f} | {b['psnr']:.2f} | {b['ssim']:.4f} | "
             f"{b['psnr'] - a['psnr']:+.2f} |")
    report["scene_0070"] = rows
    emit("\nPer view (PSNR), v1 -> v2:")
    for key in ["ours_phys", "fusion_full", "fixed_blend", "dhan_gate"]:
        pa = [v["psnr"] for v in old["variants"][key]["per_view"]]
        pb = [v["psnr"] for v in new["variants"][key]["per_view"]]
        emit(f"- {LABEL[key]}: " + ", ".join(f"v{old['variants'][key]['per_view'][i]['frame']}: "
                                              f"{pa[i]:.2f} -> {pb[i]:.2f}" for i in range(len(pa))))

    emit("\nRepeated optimisation of scene_0070 (identical recipe, different random view order):")
    runs = {"v1": [], "v2": []}
    runs["v1"] = [old["variants"]["ours_phys"]["psnr"]]
    for path in sorted(glob.glob(str(ROOT / "outputs/world_sens*/clean_seed*.json"))) + \
            sorted(glob.glob(str(ROOT / "outputs/sensitivity_reopt/clean_seed*.json"))):
        pass  # v1 repeated runs are reported in the sensitivity section (axis 'clean')
    v2_runs = [new["variants"]["ours_phys"]["psnr"]]
    v2_fuse = [new["variants"]["fusion_full"]["psnr"]]
    for k in (1, 2):
        path = f"outputs/v2/ablation_scene_0070_run{k}/ablation_metrics.json"
        if (ROOT / path).exists():
            data = load(path)
            v2_runs.append(data["variants"]["ours_phys"]["psnr"])
            v2_fuse.append(data["variants"]["fusion_full"]["psnr"])
    emit(f"- v2 physical PSNR over {len(v2_runs)} runs: " + ", ".join(f"{v:.2f}" for v in v2_runs)
         + f" (mean {np.mean(v2_runs):.2f}, sd {np.std(v2_runs, ddof=1) if len(v2_runs) > 1 else float('nan'):.2f})")
    emit(f"- v2 fused PSNR over {len(v2_fuse)} runs: " + ", ".join(f"{v:.2f}" for v in v2_fuse))
    report["scene_0070_repeats"] = {"v2_phys": v2_runs, "v2_fuse": v2_fuse}
    emit()


def ten_scene() -> tuple[dict, dict]:
    old, new = collect(str(ROOT / V1_GLOB)), collect(str(ROOT / V2_GLOB))
    emit(f"## 2. Ten scenes ({old['n_views_total']} v1 / {new['n_views_total']} v2 held-out views)\n")
    emit("| variant | v1 mean | v1 median | v1 IQR | v2 mean | v2 median | v2 IQR | v1 SSIM | v2 SSIM |")
    emit("|---|---|---|---|---|---|---|---|---|")
    table = {}
    for key in KEYS:
        if key not in old["variants"] or key not in new["variants"]:
            continue
        a, b = old["variants"][key], new["variants"][key]
        table[key] = {"v1": {"psnr_mean": a["psnr"]["mean"], "psnr_median": a["psnr"]["median"],
                             "ssim_mean": a["ssim"]["mean"]},
                      "v2": {"psnr_mean": b["psnr"]["mean"], "psnr_median": b["psnr"]["median"],
                             "ssim_mean": b["ssim"]["mean"]}}
        emit(f"| {LABEL[key]} | {a['psnr']['mean']:.2f} | {a['psnr']['median']:.2f} | "
             f"[{a['psnr']['iqr_lo']:.1f}, {a['psnr']['iqr_hi']:.1f}] | {b['psnr']['mean']:.2f} | "
             f"{b['psnr']['median']:.2f} | [{b['psnr']['iqr_lo']:.1f}, {b['psnr']['iqr_hi']:.1f}] | "
             f"{a['ssim']['mean']:.4f} | {b['ssim']['mean']:.4f} |")
    report["ten_scene"] = table
    emit()
    return old, new


def per_scene(old: dict, new: dict) -> None:
    scenes = [d["scene"][-4:] for d in new["variants"]["fusion_full"]["per_scene"]]
    manifest = json.loads((ROOT / "data/world_benchmark_v2/manifest.json").read_text(encoding="utf-8"))
    shapes = {f"{f['scene']:04d}": f["shape"] for f in manifest["frames"] if f["split"] == "test"}
    pick = ["ours_phys", "ours_pro", "fixed_blend", "fusion_full", "dhan_ft", "dhan_gate"]
    emit("## 3. Per-scene held-out PSNR (mean of 3 views), v1 -> v2\n")
    emit("| scene | shape | " + " | ".join(LABEL[k] for k in pick) + " |")
    emit("|---|---|" + "---|" * len(pick))
    arr = {"v1": {}, "v2": {}}
    for k in pick:
        arr["v1"][k] = np.array([d["psnr"] for d in old["variants"][k]["per_scene"]])
        arr["v2"][k] = np.array([d["psnr"] for d in new["variants"][k]["per_scene"]])
    for i, scene in enumerate(scenes):
        emit(f"| {scene} | {shapes.get(scene, '?')} | " + " | ".join(
            f"{arr['v1'][k][i]:.2f} -> {arr['v2'][k][i]:.2f}" for k in pick) + " |")
    emit("| mean | | " + " | ".join(f"{arr['v1'][k].mean():.2f} -> {arr['v2'][k].mean():.2f}" for k in pick) + " |")
    emit()

    emit("## 4. Counts over the ten scenes (PSNR / SSIM), v1 -> v2\n")
    ssim = {"v1": {}, "v2": {}}
    for ver, agg in (("v1", old), ("v2", new)):
        for k in KEYS:
            ssim[ver][k] = np.array([d["ssim"] for d in agg["variants"][k]["per_scene"]])
    psnr = {"v1": {k: np.array([d["psnr"] for d in old["variants"][k]["per_scene"]]) for k in KEYS},
            "v2": {k: np.array([d["psnr"] for d in new["variants"][k]["per_scene"]]) for k in KEYS}}
    pairs = [("fusion_full", "ours_pro"), ("fusion_full", "ours_phys"), ("ours_phys", "ours_pro"),
             ("fusion_full", "fixed_blend"), ("fusion_full", "fusion_analytic"), ("fusion_full", "dhan_ft"),
             ("dhan_gate", "dhan_ft"), ("dhan_gate", "dhan_fixed_blend"), ("dhan_gate", "dhan_analytic"),
             ("ours_phys", "dhan_ft"), ("ours_phys", "input")]
    counts = {}
    for a, b in pairs:
        row = {ver: (int((psnr[ver][a] > psnr[ver][b]).sum()), int((ssim[ver][a] > ssim[ver][b]).sum()))
               for ver in ("v1", "v2")}
        counts[f"{a}>{b}"] = row
        emit(f"- {LABEL[a]} > {LABEL[b]}: v1 {row['v1'][0]}/10 PSNR, {row['v1'][1]}/10 SSIM;  "
             f"v2 {row['v2'][0]}/10 PSNR, {row['v2'][1]}/10 SSIM")
    report["counts"] = counts
    emit()

    families: dict = {}
    for i, scene in enumerate(scenes):
        families.setdefault(shapes.get(scene, "?"), []).append(i)
    emit("## 5. Shape families (mean PSNR), v1 -> v2\n")
    emit("| family | n | Phys | single | Fuse |")
    emit("|---|---|---|---|---|")
    for family, idx in sorted(families.items()):
        emit(f"| {family} | {len(idx)} | " + " | ".join(
            f"{arr['v1'][k][idx].mean():.2f} -> {arr['v2'][k][idx].mean():.2f}"
            for k in ("ours_phys", "ours_pro", "fusion_full")) + " |")
    emit()


def correlation() -> None:
    def rows(glob_pattern: str) -> list[dict]:
        out: list[dict] = []
        for path in sorted(glob.glob(str(ROOT / glob_pattern))):
            out += json.loads(Path(path).read_text(encoding="utf-8"))["confidence_correlation"]
        return out

    emit("## 6. Confidence-error correlation (Pearson r between gate confidence and diffuse error, 30 views)\n")
    res = {}
    for ver, pattern in (("v1", V1_GLOB), ("v2", V2_GLOB)):
        data = rows(pattern)
        r = np.array([x["pearson_e_vs_diffuse_error"] for x in data])
        low = np.array([x["diffuse_l1_low_conf_half"] for x in data])
        high = np.array([x["diffuse_l1_high_conf_half"] for x in data])
        res[ver] = {"n": len(r), "mean": float(r.mean()), "median": float(np.median(r)), "min": float(r.min()),
                    "positive": int((r > 0).sum()), "low_l1": float(low.mean()), "high_l1": float(high.mean()),
                    "ratio": float(low.mean() / high.mean()), "low_gt_high": int((low > high).sum())}
        emit(f"- {ver}: n={len(r)}, Pearson mean {r.mean():.3f}, median {np.median(r):.3f}, min {r.min():.3f}, "
             f"positive {int((r > 0).sum())}/{len(r)}; diffuse L1 low-conf {low.mean():.4f} vs high-conf "
             f"{high.mean():.4f} (x{low.mean() / high.mean():.2f}); low>high in {int((low > high).sum())}/{len(r)}")
    report["confidence"] = res
    emit()


def material() -> None:
    old_path, new_path = "outputs/material_study/summary.json", "outputs/v2/material_study/summary.json"
    if not (ROOT / new_path).exists():
        emit("## 7. Material-controlled scenes: v2 summary not available yet\n")
        return
    old, new = load(old_path), load(new_path)
    emit("## 7. Material-controlled scenes (8 scenes), held-out PSNR (3 views), v1 -> v2\n")
    pick = ["ours_phys", "ours_pro", "fixed_blend", "fusion_full", "dhan_ft", "dhan_gate", "tshrnet_ft"]
    emit("| scene | input | " + " | ".join(LABEL[{"ours_phys": "ours_phys"}.get(k, k)] for k in pick) + " |")
    emit("|---|---|" + "---|" * len(pick))
    old_by = {s["name"]: s for s in old["scenes"]}
    for s in new["scenes"]:
        o = old_by[s["name"]]
        emit(f"| {s['name']} | {o['input_psnr_mean']:.2f} | " + " | ".join(
            f"{o[k + '_psnr']:.2f} -> {s[k + '_psnr']:.2f}" for k in pick) + " |")
    emit("\nFamily means (sphere + torus), v1 -> v2:")
    emit("| family | " + " | ".join(LABEL[k] for k in pick) + " |")
    emit("|---|" + "---|" * len(pick))
    for fam in new["families"]:
        emit(f"| {fam} | " + " | ".join(
            f"{old['families'][fam][k]:.2f} -> {new['families'][fam][k]:.2f}" for k in pick) + " |")
    emit(f"\nCounts v1: {old['counts']}")
    emit(f"Counts v2: {new['counts']}")
    report["material"] = {"v1_counts": old["counts"], "v2_counts": new["counts"],
                          "v1_families": old["families"], "v2_families": new["families"]}
    emit()


def sensitivity() -> None:
    old_dir, new_dir = ROOT / "outputs/sensitivity_reopt", ROOT / "outputs/v2/sensitivity"
    axes = ["clean", "geo_mild", "geo_severe", "albedo15", "albedo30", "light", "drop25", "drop50", "sparse6", "sparse3"]
    emit("## 8. Prior-sensitivity study (re-optimisation from perturbed priors), held-out PSNR, v1 -> v2\n")
    emit("| axis | seed | Phys | Fuse | single | gate mean |")
    emit("|---|---|---|---|---|---|")
    table = {}
    for axis in axes:
        for seed in (0, 1, 2):
            a, b = old_dir / f"{axis}_seed{seed}.json", new_dir / f"{axis}_seed{seed}.json"
            if not a.exists() and not b.exists():
                continue
            ja = json.loads(a.read_text(encoding="utf-8")) if a.exists() else None
            jb = json.loads(b.read_text(encoding="utf-8")) if b.exists() else None

            def cell(key: str) -> str:
                va = f"{ja[key]['psnr']:.2f}" if ja else "n/a"
                vb = f"{jb[key]['psnr']:.2f}" if jb else "pending"
                return f"{va} -> {vb}"

            emit(f"| {axis} | {seed} | {cell('phys')} | {cell('fuse')} | {cell('single')} | "
                 f"{(ja['gate_mean'] if ja else float('nan')):.2f} -> "
                 f"{(jb['gate_mean'] if jb else float('nan')):.2f} |")
            table[f"{axis}_seed{seed}"] = {"v1": ja, "v2": jb}
    report["sensitivity"] = {k: {ver: ({kk: v[kk]["psnr"] for kk in ("phys", "fuse", "single")} if v else None)
                                 for ver, v in d.items()} for k, d in table.items()}
    emit()


def ablation_rows() -> None:
    """Ablation rows that needed a dedicated training run under the corrected recipe (scene_0070)."""
    old = load("outputs/world_ablation_scene_0070/ablation_metrics.json")["variants"]
    new = load("outputs/v2/ablation_scene_0070/ablation_metrics.json")["variants"]
    strict = load("outputs/v2/ablation_strict_nossim_eval/ablation_metrics.json")["variants"]
    boot = load("outputs/v2/ablation_bootstrap_eval/ablation_metrics.json")["variants"]
    emit("## 9. Ablation rows of scene_0070 (3 held-out views), v1 -> corrected recipe\n")
    emit("| row | v1 PSNR / SSIM | corrected PSNR / SSIM | corrected run |")
    emit("|---|---|---|---|")
    rows = [
        ("Ours-Phys, strict protocol (oracle albedo init, paired D/S supervision, refiner on)",
         old["ours_phys"], strict["ours_phys"], "strict config with only photo_ssim=0"),
        ("World refiner (mask-confined), strict protocol", old["ours_refined"], strict["ours_refined"],
         "strict config with only photo_ssim=0"),
        ("Ours-Phys, albedo-free RGB-only (v2 main recipe)", old["inverse"], new["ours_phys"],
         "configs/world/v2 (v1 counterpart = 'Inverse (no albedo)' row)"),
        ("Inverse + single-image pseudo-diffuse bootstrap", old["inverse_pro"], boot["ours_phys"],
         "v2 recipe with pseudo-diffuse albedo init"),
        ("Ours-Fuse (full), v2 world field + v2 gate", old["fusion_full"], new["fusion_full"], "v2 gate"),
        ("Fusion with the strict-protocol physical branch (v2 gate)", old["fusion_full"], strict["fusion_full"],
         "strict config with only photo_ssim=0"),
    ]
    out = {}
    for label, a, b, note in rows:
        emit(f"| {label} | {a['psnr']:.2f} / {a['ssim']:.4f} | {b['psnr']:.2f} / {b['ssim']:.4f} | {note} |")
        out[label] = {"v1": [a["psnr"], a["ssim"]], "corrected": [b["psnr"], b["ssim"]]}
    report["ablation_rows"] = out
    emit("\nNote: the 'refiner' entry of the v2 bootstrap and v2 world runs is meaningless (those runs have no refiner).\n")


def exact_init_and_real_mvs() -> None:
    emit("## 10. Exact initialisation (0 optimisation steps) and real-MVS diagnostic\n")
    old_init = load("outputs/sensitivity_reopt/oracle_init.json")
    new_init = load("outputs/v2/sensitivity/oracle_init.json")
    emit(f"- exact initialisation, physical: {old_init['phys']['psnr']:.2f} dB (state is recipe-independent) -> "
         f"{new_init['phys']['psnr']:.2f} dB;  fused: {old_init['fuse']['psnr']:.2f} (v1 gate) -> "
         f"{new_init['fuse']['psnr']:.2f} dB (v2 gate)")
    emit("- optimisation from the exact state, held-out physical PSNR: v1 36.77 dB (drops by 17.1 dB), "
         "strict config with photo_ssim=0: 57.34 dB (gains 3.5 dB); v2 RGB-only from exact albedo: see 'clean' rows above")
    emit("\n| real MVS (full-radiance NVS PSNR / SSIM) | v1 train | v1 holdout | SSIM removed, other settings unchanged: train | holdout |")
    emit("|---|---|---|---|---|")
    mvs = {}
    for name, old_dir, new_dir in (("DTU scan1", "dtu_scan1", "dtu_scan1_nossim"),
                                   ("BlendedMVS 57f8d9bb", "blendedmvs_57f8d9bb", "blendedmvs_57f8d9bb_nossim")):
        cells = []
        for d in (old_dir, new_dir):
            for split in ("train", "holdout"):
                m = load(f"outputs/{d}/world_metrics_{split}.json")["summary"]
                cells.append(f"{m['full_psnr']:.2f} / {m['full_ssim']:.3f}")
                mvs[f"{d}_{split}"] = [m["full_psnr"], m["full_ssim"]]
        emit(f"| {name} | " + " | ".join(cells) + " |")
    report["exact_init"] = {"v1": [old_init["phys"]["psnr"], old_init["fuse"]["psnr"]],
                            "v2": [new_init["phys"]["psnr"], new_init["fuse"]["psnr"]]}
    report["real_mvs"] = mvs
    emit()


def main() -> None:
    import os

    os.chdir(ROOT)
    emit("# Old (v1: SSIM photometric term, lr 2e-3, 3000 steps) vs new (v2: RGB L1 only, lr 0.03, 1500 steps)\n")
    scene0070()
    old, new = ten_scene()
    per_scene(old, new)
    correlation()
    material()
    sensitivity()
    ablation_rows()
    exact_init_and_real_mvs()
    out = ROOT / "outputs" / "v2"
    (out / "comparison_v1_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "comparison_v1_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "world_holdout_multiscene.json").write_text(json.dumps(new, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
