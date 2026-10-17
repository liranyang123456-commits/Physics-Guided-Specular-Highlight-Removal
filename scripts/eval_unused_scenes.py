"""One-shot evaluation of the ten test scenes that no other experiment used (scene_0080 - scene_0089).

Every setting is frozen before this run: the recipe of configs/world/v2 (scripts/make_v2_configs.py --steps 1500 --scenes 80 ... 89),
the gate of the tables (outputs/v2/fusion_scene_0070), the seed-0 cross-fitted gate (outputs/v2/fusion_crossfit_clean_scene_0070),
and the threshold of the training-view fit check (40 dB).  Run after
    python scripts/run_v2_world.py --scenes 80 81 82 83 84 85 86 87 88 89
    python scripts/eval_world_ablation.py --world-checkpoint outputs/v2/world_scene_00NN/last.pt \
        --fusion-checkpoint outputs/v2/fusion_scene_0070/fusion.pt --output outputs/v2/ablation_scene_00NN --raster-backend gsplat
    (and the same with outputs/v2/fusion_crossfit_clean_scene_0070/fusion.pt into outputs/v2/ablation_xc0_scene_00NN)
Writes outputs/v2/unused_scenes.json.
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

SCENES = list(range(80, 90))
KEYS = ["input", "ours_phys", "ours_pro", "fixed_blend", "fusion_no_correction", "fusion_full", "dhan_ft", "dhan_gate", "tshrnet_ft"]
TAU = 40.0


def main() -> None:
    device = torch.device("cuda")
    manifest = json.loads((ROOT / "data/world_benchmark_v2/manifest.json").read_text(encoding="utf-8"))
    shapes = {f["scene"]: f["shape"] for f in manifest["frames"] if f["split"] == "test"}
    per = []
    for n in SCENES:
        ab = json.loads((ROOT / f"outputs/v2/ablation_scene_00{n}/ablation_metrics.json").read_text(encoding="utf-8"))["variants"]
        xc = json.loads((ROOT / f"outputs/v2/ablation_xc0_scene_00{n}/ablation_metrics.json").read_text(encoding="utf-8"))["variants"]
        f = fit(ROOT / f"outputs/v2/world_scene_00{n}/last.pt", device)
        row = {"scene": f"scene_00{n}", "shape": shapes[n], "train_fit": f["full_psnr"], "accepted": f["full_psnr"] >= TAU}
        row.update({k: {"psnr": ab[k]["psnr"], "ssim": ab[k]["ssim"]} for k in KEYS})
        row["fusion_xc0"] = {"psnr": xc["fusion_full"]["psnr"], "ssim": xc["fusion_full"]["ssim"]}
        per.append(row)
        print(row["scene"], row["shape"], {k: round(row[k]["psnr"], 2) for k in ("ours_phys", "fusion_full", "fusion_xc0", "ours_pro", "dhan_ft")},
              "fit", round(f["full_psnr"], 2), flush=True)
    col = lambda k: np.array([r[k]["psnr"] for r in per])
    summary = {k: float(col(k).mean()) for k in KEYS + ["fusion_xc0"]}
    summary.update({
        "phys_gt_single": int((col("ours_phys") > col("ours_pro")).sum()),
        "phys_gt_dhan": int((col("ours_phys") > col("dhan_ft")).sum()),
        "fuse_gt_single": int((col("fusion_full") > col("ours_pro")).sum()),
        "fuse_gt_phys": int((col("fusion_full") > col("ours_phys")).sum()),
        "xc0_gt_phys": int((col("fusion_xc0") > col("ours_phys")).sum()),
        "xc0_gt_single": int((col("fusion_xc0") > col("ours_pro")).sum()),
        "dhan_gate_gt_dhan": int((col("dhan_gate") > col("dhan_ft")).sum()),
        "phys_min": float(col("ours_phys").min()), "phys_max": float(col("ours_phys").max()),
        "fit_min": float(min(r["train_fit"] for r in per)), "fit_max": float(max(r["train_fit"] for r in per)),
        "accepted": int(sum(r["accepted"] for r in per)), "n": len(per),
    })
    out = {"frozen": {"recipe": "configs/world/v2 (1500 steps)", "gate": "outputs/v2/fusion_scene_0070/fusion.pt",
                      "crossfit_gate": "outputs/v2/fusion_crossfit_clean_scene_0070/fusion.pt", "tau_db": TAU},
           "per_scene": per, "summary": summary}
    (ROOT / "outputs/v2/unused_scenes.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
