"""Distance of the optimized per-Gaussian materials from the generating field (corrected recipe, ten benchmark scenes).

The diffuse albedo (c_i w_d,i) is what lobe selection needs; the individual budget weights, roughness and metalness are
reported to show which quantities the RGB-only optimization identifies.  Writes outputs/v2/material_error.json.
"""
from __future__ import annotations

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

from diag_drift import material_error  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402


def main() -> None:
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for n in range(70, 80):
        st = torch.load(f"outputs/v2/world_scene_00{n}/last.pt", map_location="cpu", weights_only=False)
        cfg = dict(st["config"])
        cfg["raster_backend"] = "gsplat"
        dataset = load_dataset(cfg)
        pipeline = build_pipeline(dataset, cfg, device, list(st["train_ids"]))
        pipeline.load_state_dict(st["model"], strict=False)
        prior = ROOT / cfg["data_root"] / cfg["gaussian_prior"]
        out[f"scene_00{n}"] = material_error(pipeline.scene, prior)
        print(f"scene_00{n}", {k: round(v, 3) for k, v in out[f"scene_00{n}"].items()}, flush=True)
    keys = list(next(iter(out.values())))
    out["mean"] = {k: float(np.mean([v[k] for v in out.values() if isinstance(v, dict) and k in v])) for k in keys}
    print("mean", {k: round(v, 3) for k, v in out["mean"].items()})
    Path("outputs/v2/material_error.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
