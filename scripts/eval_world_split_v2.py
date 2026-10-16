"""Diffuse and full-radiance PSNR/SSIM of a world field on its training and held-out views (corrected recipe, scene_0070).

    python scripts/eval_world_split_v2.py --checkpoint outputs/v2/world_scene_0070/last.pt

Uses the metric functions of eval_world_ablation.py; writes split_metrics.json next to the checkpoint.
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

from eval_world_ablation import psnr, ssim  # noqa: E402
from mvbrdf_shr.world.train_fusion import load_world  # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/v2/world_scene_0070/last.pt")
    args = parser.parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world, dataset, state = load_world(args.checkpoint, device)
    result = {}
    for split in ("train", "holdout"):
        ids = list(state["train_ids"] if split == "train" else state["holdout_ids"])
        rows = []
        for frame_id in ids:
            frame = dataset[frame_id]
            image = frame.image.to(device)
            target = frame.diffuse_gt.to(device)
            render = world(frame.camera.to(device), frame_id, image=image, refine=False)["render"]
            full = (render.diffuse + render.specular).clamp(0, 1)
            rows.append({"frame": frame_id, "diffuse_psnr": psnr(render.diffuse, target), "diffuse_ssim": ssim(render.diffuse, target),
                         "full_psnr": psnr(full, image), "full_ssim": ssim(full, image)})
        result[split] = {"n": len(rows), "per_view": rows,
                         **{k: float(np.mean([r[k] for r in rows])) for k in ("diffuse_psnr", "diffuse_ssim", "full_psnr", "full_ssim")}}
        print(split, {k: round(v, 4) for k, v in result[split].items() if k not in ("n", "per_view")}, flush=True)
    out = Path(args.checkpoint).parent / "split_metrics.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
