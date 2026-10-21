"""Disagreement between the two branches on the same view, for every field of the held-out-view study.

Section 4.6 (``Failures that the training views do not show'') reports that the training-view fit check accepts fields that
fit their training views but fail on a new one.  The Future directions named a comparison of the two branches on the same
view, through their disagreement, as an untested candidate for a fall-back.  This script measures it.

For every field of outputs/v2/view_check.json (ten benchmark scenes, eight material scenes, the re-optimized sensitivity
fields with frozen and with free geometry/lighting) and for every training and held-out view it records
  * the disagreement PSNR between the fused output and the single-image prediction, PSNR(F, S), and between the physical
    diffuse render and the single-image prediction, PSNR(I_d, S) -- both computable without ground truth;
  * for reference only, the PSNR of F, I_d, and S against the diffuse ground truth.
The fused output F is the cross-fitted gate (seed 0) of Section 4.5 unless stated; the gate of the tables is recorded too.

Rationale.  On the training views the field reproduces its images, so F is close to the ground truth there and the
disagreement PSNR(F, S) is the accuracy of the single-image branch in that scene.  On a new view of a correct field the
disagreement stays at that level; if the fused output degrades to the accuracy of the single-image branch, the squared
difference doubles (independent errors) and the disagreement drops by 3 dB.  A drop larger than 3 dB therefore flags a
new view on which the fused output is probably below the single-image prediction.  scripts/analyze_disagreement.py scores
this rule, alone and combined with the training-view fit check.

Writes outputs/v2/disagreement.json.

    python scripts/eval_disagreement.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_view_check import collect  # noqa: E402
from eval_world_ablation import psnr  # noqa: E402
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro  # noqa: E402
from mvbrdf_shr.world.fusion import HybridMVBRDFSHR  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402

GATES = {"xc0": "outputs/v2/fusion_crossfit_clean_scene_0070/fusion.pt",   # cross-fitted gate (seed 0), Section 4.5
         "tab": "outputs/v2/fusion_scene_0070/fusion.pt"}                   # gate of the tables (original protocol)
SINGLE = "outputs/world_benchmark_single/best.pt"


@torch.no_grad()
def load_branches(device):
    pro = MVBRDFSHRPro(128, 48, 96).to(device)
    state = torch.load(ROOT / SINGLE, map_location=device, weights_only=False)
    pro.load_state_dict(state.get("model", state), strict=False)
    pro.eval()
    hybrids = {}
    for key, path in GATES.items():
        hybrid = HybridMVBRDFSHR(resolution=128).to(device)
        hybrid.load_single_checkpoint(ROOT / SINGLE)
        fusion = torch.load(ROOT / path, map_location=device, weights_only=False)
        hybrid.gate.load_state_dict(fusion["gate"])
        hybrid.correction.load_state_dict(fusion["correction"])
        hybrid.eval()
        hybrids[key] = hybrid
    return pro, hybrids


@torch.no_grad()
def per_view(pipeline, dataset, view_ids, device, pro, hybrids) -> list[dict]:
    rows = []
    for frame_id in view_ids:
        frame = dataset[frame_id]
        image = frame.image.to(device)
        target = frame.diffuse_gt.to(device)
        out = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)
        phys = out["render"].diffuse
        x = F.interpolate(image.unsqueeze(0), (128, 128), mode="bilinear", align_corners=False)
        single = pro(x, use_refine=True, detach_physics=True)["pred"]
        single = F.interpolate(single, size=image.shape[-2:], mode="bilinear", align_corners=False)[0]
        row = {"frame": int(frame_id),
               "phys_vs_single": psnr(phys, single), "phys_gt": psnr(phys, target), "single_gt": psnr(single, target)}
        for key, hybrid in hybrids.items():
            fused = hybrid(image.unsqueeze(0), out, detach_backbones=True)["pred"][0]
            row[f"fuse_{key}_vs_single"] = psnr(fused, single)
            row[f"fuse_{key}_gt"] = psnr(fused, target)
        rows.append(row)
    return rows


def main() -> None:
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = ROOT / "outputs/v2/disagreement.json"
    result = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    pro, hybrids = load_branches(device)
    for name, group, path in collect():
        if name in result:
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(state["config"])
        cfg["raster_backend"] = "gsplat"
        dataset = load_dataset(cfg)
        train_ids, holdout = list(state["train_ids"]), list(state["holdout_ids"])
        pipeline = build_pipeline(dataset, cfg, device, train_ids)
        pipeline.load_state_dict(state["model"], strict=False)
        pipeline.eval().requires_grad_(False)
        row = {"group": group,
               "train": per_view(pipeline, dataset, train_ids, device, pro, hybrids),
               "holdout": per_view(pipeline, dataset, holdout, device, pro, hybrids)}
        for split in ("train", "holdout"):
            for key in ("phys_vs_single", "fuse_xc0_vs_single", "fuse_tab_vs_single", "phys_gt", "single_gt", "fuse_xc0_gt", "fuse_tab_gt"):
                row[f"{split}_{key}"] = float(np.mean([v[key] for v in row[split]]))
        result[name] = row
        print(f"{name:52s} d(F,S) train {row['train_fuse_xc0_vs_single']:6.2f} holdout {row['holdout_fuse_xc0_vs_single']:6.2f} | "
              f"d(Id,S) train {row['train_phys_vs_single']:6.2f} holdout {row['holdout_phys_vs_single']:6.2f} | "
              f"gt: fuse {row['holdout_fuse_xc0_gt']:6.2f} single {row['holdout_single_gt']:6.2f}", flush=True)
        out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
