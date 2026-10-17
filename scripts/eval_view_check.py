"""Held-out-view check of a fitted world field, evaluated over all fields of the study.

The training-view fit check (mvbrdf_shr/world/fit_check.py) rejects fields that cannot reproduce their own training images.
It cannot see a field that fits its training views but fails on a new one (free geometry that drifts, too few views).  The
input image of the new view is available at test time, so the field can also be checked against it: the *median* absolute
error of the full-radiance render over the object pixels is robust to the specular misfit that dominates the mean
residual (the specular lobe is not identified from nine views), whereas it grows when the diffuse layer is wrong.

For every field (the ten benchmark scenes, the eight material scenes, the re-optimized sensitivity fields with frozen and with
free geometry/lighting) this script records, on the held-out views,
  * the diffuse PSNR of the physical render, of the single-image branch, and of the fusion (diffuse ground truth, reported only),
  * the no-ground-truth statistics: full-radiance PSNR, median and 90th-percentile error against the input, and the same
    statistics on the training views,
and writes outputs/v2/view_check.json.  scripts/analyze_view_check.py turns it into the decision rules and their scores.

    python scripts/eval_view_check.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_sensitivity_unfrozen import fit_statistics  # noqa: E402
from run_sensitivity_v2 import evaluate  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402


XC0_GATE = "outputs/v2/fusion_crossfit_clean_scene_0070/fusion.pt"   # cross-fitted gate (seed 0) of Section 4.5


def collect() -> list[tuple[str, str, Path]]:
    out = []
    for n in range(70, 80):
        out.append((f"ten/scene_00{n}", "ten", ROOT / f"outputs/v2/world_scene_00{n}/last.pt"))
    for p in sorted((ROOT / "outputs/v2/material_study").glob("*/last.pt")):
        out.append((f"material/{p.parent.name}", "material", p))
    for p in sorted((ROOT / "outputs/v2/sensitivity").glob("*_seed*.pt")):
        out.append((f"frozen/{p.stem}", "frozen", p))
    for p in sorted((ROOT / "outputs/v2/sensitivity_unfrozen").glob("*.pt")):
        if "_val" in p.stem:
            continue
        out.append((f"unfrozen/{p.stem}", "unfrozen", p))
    return out


def main() -> None:
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = ROOT / "outputs/v2/view_check.json"
    result = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
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
        summary = evaluate(pipeline, dataset, holdout, device)
        summary_xc = evaluate(pipeline, dataset, holdout, device, gate_path=XC0_GATE)
        row = {
            "group": group,
            "phys": summary["phys"]["psnr"], "fuse": summary["fuse"]["psnr"], "single": summary["single"]["psnr"],
            "fuse_xc0": summary_xc["fuse"]["psnr"], "gate_mean": summary["gate_mean"],
            "train": fit_statistics(pipeline, dataset, train_ids, device),
            "holdout": fit_statistics(pipeline, dataset, holdout, device),
        }
        result[name] = row
        print(f"{name:52s} phys {row['phys']:6.2f} fuse {row['fuse']:6.2f} xc0 {row['fuse_xc0']:6.2f} single {row['single']:6.2f} | "
              f"train fit {row['train']['full_psnr']:6.2f} med {row['train']['med_err']:.4f} | "
              f"holdout med {row['holdout']['med_err']:.4f} p90 {row['holdout']['p90_err']:.4f}", flush=True)
        out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
