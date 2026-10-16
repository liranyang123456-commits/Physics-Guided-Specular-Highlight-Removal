"""Evaluate several fusion gates on the ten benchmark scenes and the eight material-controlled scenes (strict holdout).

The gates are trained on scene_0070 only and transferred without retraining.  For every field the physical render, the
single-image branch, and each gate's fused output are scored on the three held-out views (mean PSNR / SSIM), with the
metric functions of eval_world_ablation.py.  Writes outputs/v2/gate_scenes_<tag>.json.

    python scripts/eval_gate_scenes.py --gate orig=... --gate xc0=... --tag all
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_sensitivity_gate import evaluate_field, load_hybrid  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--tag", default="all")
    args = parser.parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gates = dict(item.split("=", 1) for item in args.gate)
    hybrids = {name: load_hybrid(path, device) for name, path in gates.items()}
    fields = {f"ten/scene_00{n}": ROOT / f"outputs/v2/world_scene_00{n}/last.pt" for n in range(70, 80)}
    fields |= {f"material/{p.parent.name}": p for p in sorted((ROOT / "outputs/v2/material_study").glob("*/last.pt"))}
    result = {}
    for name, path in fields.items():
        st = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(st["config"])
        cfg["raster_backend"] = "gsplat"
        dataset = load_dataset(cfg)
        pipeline = build_pipeline(dataset, cfg, device, list(st["train_ids"]))
        pipeline.load_state_dict(st["model"], strict=False)
        result[name] = evaluate_field(pipeline, dataset, list(st["holdout_ids"]), device, hybrids)
        print(f"{name:34s} " + " ".join(f"{k}={result[name][k]['psnr']:.2f}" for k in result[name]), flush=True)
    out = ROOT / "outputs/v2" / f"gate_scenes_{args.tag}.json"
    out.write_text(json.dumps({"gates": gates, "results": result}, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
