"""Gradient norms of the L1 and the weighted SSIM photometric terms at the initial state of the albedo-free scene_0070 field.

Uses configs/world/benchmark_scene_0070_inverse.yaml (the first-submission inverse configuration, SSIM weight 0.2).  For three
training views it back-propagates each term separately into the material parameters and reports the ratio of the norms.
Writes outputs/inverse_diag/gradient_ratio.json.
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
import yaml

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.losses import _ssim_loss  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset, split_frame_indices  # noqa: E402


def main() -> None:
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open("configs/world/benchmark_scene_0070_inverse.yaml", encoding="utf-8"))
    cfg["n_gaussians"] = int(len(np.load(Path(cfg["data_root"]) / cfg["gaussian_prior"])["position"]))
    dataset = load_dataset(cfg)
    train_ids, _ = split_frame_indices(len(dataset), cfg)
    pipe = build_pipeline(dataset, cfg, device, train_ids)
    params = pipe.scene.material_parameters()

    def gradnorm(fn) -> float:
        pipe.zero_grad(set_to_none=True)
        fn().backward()
        return float(np.sqrt(sum(float(p.grad.norm()) ** 2 for p in params if p.grad is not None)))

    rows = []
    for fid in train_ids[:3]:
        fr = dataset[fid]
        img = fr.image.to(device)

        def render():
            return pipe(fr.camera.to(device), fid, image=img, refine=False)["render"].full

        g_l1 = gradnorm(lambda: F.l1_loss(render(), img))
        g_ssim = gradnorm(lambda: 0.2 * _ssim_loss(render(), img))
        rows.append({"view": int(fid), "l1": g_l1, "ssim_0.2": g_ssim, "ratio": g_ssim / g_l1})
        print(rows[-1], flush=True)
    out = {"views": rows, "ratio_min": min(r["ratio"] for r in rows), "ratio_max": max(r["ratio"] for r in rows)}
    Path("outputs/inverse_diag/gradient_ratio.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
