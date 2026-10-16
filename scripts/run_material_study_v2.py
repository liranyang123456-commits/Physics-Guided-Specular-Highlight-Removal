"""Strict-holdout study on the material-controlled scenes, corrected recipe (v2).

Derived from run_material_study.py: the recipe is that of configs/world/v2 (oracle geometry and lighting, albedo and
specular BRDF unknown, RGB-only photometric loss without an SSIM term, learning rate 0.03, 1500 steps), and the gate is the
one trained on the v2 scene_0070 field.

Original description:

Every scene uses the scene_0070 strict configuration unchanged except for the
data path (3,000 steps, oracle geometry/albedo initialization, frozen geometry
and calibrated lighting, holdout views {0, 4, 8}), is optimized with gsplat,
and is evaluated with scripts/eval_world_ablation.py using the fusion gate
trained on scene_0070.

    python scripts/run_material_study.py --launch     # optimize all scenes in parallel
    python scripts/run_material_study.py --eval       # evaluate all variants
    python scripts/run_material_study.py --summary    # per-scene and per-family tables
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "world_benchmark_material"
OUT = ROOT / "outputs" / "v2" / "material_study"
VARIANTS = ["input", "ours_phys", "ours_pro", "fixed_blend", "fusion_full", "dhan_ft", "dhan_gate", "tshrnet_ft"]


def scenes() -> list[dict]:
    return json.loads((DATA / "manifest.json").read_text())["scenes"]


def train_one(name: str) -> None:
    from mvbrdf_shr.world.train import train

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_v2_configs import v2_config

    cfg = v2_config(70, 1500, 0.03, "outputs/v2")
    cfg["data_root"] = str((DATA / "scenes" / "test" / name).relative_to(ROOT)).replace("\\", "/")
    cfg["n_gaussians"] = int(len(np.load(ROOT / cfg["data_root"] / cfg["gaussian_prior"])["position"]))
    cfg["output_dir"] = str((OUT / name).relative_to(ROOT)).replace("\\", "/")
    cfg["raster_backend"] = "gsplat"
    os.chdir(ROOT)
    train(cfg)


def launch(workers: int = 4) -> None:
    import time

    queue = [scene["name"] for scene in scenes()]
    running = []
    while queue or running:
        while queue and len(running) < workers:
            name = queue.pop(0)
            (OUT / name).mkdir(parents=True, exist_ok=True)
            log = open(OUT / name / "train.log", "w", encoding="utf-8")
            running.append((name, subprocess.Popen([sys.executable, __file__, "--train", name], cwd=ROOT, stdout=log,
                                                   stderr=subprocess.STDOUT)))
        for name, proc in list(running):
            if proc.poll() is not None:
                running.remove((name, proc))
                print(name, "exit", proc.returncode, flush=True)
        time.sleep(3)


def evaluate() -> None:
    for scene in scenes():
        name = scene["name"]
        subprocess.run([sys.executable, "scripts/eval_world_ablation.py",
                        "--world-checkpoint", f"outputs/v2/material_study/{name}/last.pt",
                        "--fusion-checkpoint", "outputs/v2/fusion_scene_0070/fusion.pt",
                        "--output", f"outputs/v2/material_study/{name}",
                        "--raster-backend", "gsplat"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
                       env=dict(os.environ, PYTHONPATH=str(ROOT)))
        print("evaluated", name, flush=True)


def summary() -> None:
    rows = []
    for scene in scenes():
        data = json.loads((OUT / scene["name"] / "ablation_metrics.json").read_text())
        var = data["variants"]
        corr = data["confidence_correlation"]
        rows.append({**scene,
                     **{f"{k}_psnr": var[k]["psnr"] for k in VARIANTS if k in var},
                     **{f"{k}_ssim": var[k]["ssim"] for k in VARIANTS if k in var},
                     "pearson_mean": float(np.mean([c["pearson_e_vs_diffuse_error"] for c in corr])),
                     "low_high_ratio": float(np.mean([c["diffuse_l1_low_conf_half"] for c in corr])
                                             / np.mean([c["diffuse_l1_high_conf_half"] for c in corr]))})
    keys = [k for k in VARIANTS if f"{k}_psnr" in rows[0]]
    print("scene".ljust(28) + " ".join(k[:11].rjust(11) for k in keys) + "   r    ratio")
    for r in rows:
        print(r["name"].ljust(28) + " ".join(f"{r[k + '_psnr']:11.2f}" for k in keys)
              + f"  {r['pearson_mean']:.2f}  {r['low_high_ratio']:.2f}")
    families = {}
    for r in rows:
        families.setdefault(r["family"], []).append(r)
    family_table = {f: {k: float(np.mean([r[k + "_psnr"] for r in rs])) for k in keys} for f, rs in families.items()}
    print("\nfamily means (sphere + torus)")
    for f, t in family_table.items():
        print(f.ljust(28) + " ".join(f"{t[k]:11.2f}" for k in keys))
    counts = {
        "fuse_gt_single": sum(r["fusion_full_psnr"] > r["ours_pro_psnr"] for r in rows),
        "fuse_gt_phys": sum(r["fusion_full_psnr"] > r["ours_phys_psnr"] for r in rows),
        "phys_gt_single": sum(r["ours_phys_psnr"] > r["ours_pro_psnr"] for r in rows),
        "fuse_gt_fixed": sum(r["fusion_full_psnr"] > r["fixed_blend_psnr"] for r in rows),
        "dhan_gate_gt_dhan": sum(r["dhan_gate_psnr"] > r["dhan_ft_psnr"] for r in rows),
        "fuse_gt_dhan": sum(r["fusion_full_psnr"] > r["dhan_ft_psnr"] for r in rows),
        "n": len(rows),
    }
    print("\ncounts:", counts)
    (OUT / "summary.json").write_text(json.dumps({"scenes": rows, "families": family_table, "counts": counts},
                                                 indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.train:
        train_one(args.train)
    if args.launch:
        launch()
    if args.eval:
        evaluate()
    if args.summary:
        summary()


if __name__ == "__main__":
    main()
