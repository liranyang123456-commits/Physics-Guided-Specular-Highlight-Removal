"""Evaluate full reconstruction and highlight-free rendering across views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..metrics import psnr_torch, ssim_torch
from .pipeline import WorldGaussianBRDFPipeline
from .scene import GaussianBRDFField
from .train import load_dataset


@torch.no_grad()
def evaluate(checkpoint: str, split: str = "all") -> dict:
    if split not in {"all", "train", "holdout"}:
        raise ValueError("split must be all, train, or holdout")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = state["config"]
    dataset = load_dataset(cfg)
    colors = state.get("colors")
    scene = GaussianBRDFField.from_point_cloud(
        state["points"].to(device), colors.to(device) if colors is not None else None
    )
    pipeline = WorldGaussianBRDFPipeline(
        scene,
        len(dataset),
        lighting_mode=str(cfg.get("lighting_mode", "sh")),
        use_refiner=bool(cfg.get("use_refiner", True)),
        refiner_base=int(cfg.get("refiner_base", 96)),
        chunk_size=int(cfg.get("chunk_size", 32)),
        raster_backend=str(cfg.get("raster_backend", "auto")),
        shared_lighting=bool(cfg.get("shared_lighting", False)),
        background_color=cfg.get("background_color"),
    ).to(device)
    pipeline.load_state_dict(state["model"], strict=False)
    pipeline.eval()
    train_ids = state.get("train_ids", list(range(len(dataset))))
    holdout_ids = state.get("holdout_ids", [])
    selected = {
        "all": set(range(len(dataset))),
        "train": set(train_ids),
        "holdout": set(holdout_ids),
    }[split]
    if not selected:
        raise ValueError(f"checkpoint contains no {split} views")

    rows = []
    for index, frame in enumerate(dataset.frames):
        if index not in selected:
            continue
        image = frame.image.to(device)
        output = pipeline(frame.camera.to(device), index, image=image, refine=True)
        row = {
            "frame": index,
            "full_psnr": psnr_torch(output["render"].full, image),
            "full_ssim": ssim_torch(output["render"].full, image),
        }
        if frame.diffuse_gt is not None:
            target = frame.diffuse_gt.to(device)
            row.update(
                diffuse_psnr=psnr_torch(output["render"].diffuse, target),
                diffuse_ssim=ssim_torch(output["render"].diffuse, target),
                refined_psnr=psnr_torch(output["pred"], target),
                refined_ssim=ssim_torch(output["pred"], target),
            )
        rows.append(row)
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key != "frame"
    }
    result = {
        "split": split,
        "train_ids": train_ids,
        "holdout_ids": holdout_ids,
        "n": len(rows),
        "summary": summary,
        "frames": rows,
    }
    output_path = Path(checkpoint).parent / f"world_metrics_{split}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("all", "train", "holdout"), default="all")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.split)


if __name__ == "__main__":
    main()

