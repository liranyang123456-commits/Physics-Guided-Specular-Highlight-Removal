"""Why does the albedo-unknown run of scene_0070 stop at ~13 dB?  A diagnostic around benchmark_scene_0070_inverse.yaml.

Runs the world optimization from a YAML config with optional overrides and records, at intervals, the diffuse and
full-radiance PSNR of the training and held-out views (physical render, no refiner).  Results go to
outputs/inverse_diag/<name>.json.

  python scripts/diag_inverse.py --name p0 --config configs/world/benchmark_scene_0070_inverse.yaml
  python scripts/diag_inverse.py --name r1 --set scene_lr=0.03 --loss diffuse_gt=0 --loss specular_gt=0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.utils.cpp_extension
import yaml

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvbrdf_shr.metrics import psnr_torch  # noqa: E402
from mvbrdf_shr.world.losses import WorldLossWeights, world_total_loss  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset, split_frame_indices  # noqa: E402


@torch.no_grad()
def eval_views(pipeline, dataset, ids, device) -> dict:
    was = pipeline.training
    pipeline.eval()
    d, f = [], []
    for i in ids:
        frame = dataset[i]
        image = frame.image.to(device)
        render = pipeline(frame.camera.to(device), i, image=image, refine=False)["render"]
        d.append(float(psnr_torch(render.diffuse, frame.diffuse_gt.to(device))))
        f.append(float(psnr_torch(render.full, image)))
    pipeline.train(was)
    return {"diffuse_psnr": float(np.mean(d)), "full_psnr": float(np.mean(f))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", default="configs/world/benchmark_scene_0070_inverse.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config entry (YAML value)")
    parser.add_argument("--loss", action="append", default=[], metavar="KEY=VALUE", help="override a loss weight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perturb", default="", help="axis of run_sensitivity_reopt.perturb applied at initialization, e.g. albedo30")
    parser.add_argument("--eval-every", type=int, default=250)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.chdir(ROOT)
    device = torch.device("cuda")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for item in args.set:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    loss_cfg = dict(cfg.get("loss", {}))
    for item in args.loss:
        key, value = item.split("=")
        assert key in WorldLossWeights.__dataclass_fields__, key
        loss_cfg[key] = float(value)

    prior = Path(cfg["data_root"]) / str(cfg.get("gaussian_prior", "gaussian_brdf_gt.npz"))
    if prior.exists():
        cfg["n_gaussians"] = int(len(np.load(prior)["position"]))
    dataset = load_dataset(cfg)
    train_ids, holdout_ids = split_frame_indices(len(dataset), cfg)
    pipeline = build_pipeline(dataset, cfg, device, train_ids)
    if args.perturb:
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_sensitivity_reopt import perturb  # noqa: E402
        perturb(pipeline, args.perturb, torch.Generator().manual_seed(1000 + args.seed))
    steps = int(cfg["steps"])
    views_per_step = min(int(cfg.get("views_per_step", 1)), len(train_ids))
    optimizer = torch.optim.Adam([{"params": pipeline.scene.material_parameters(), "lr": float(cfg.get("scene_lr", 2e-3))}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    weights = WorldLossWeights(**loss_cfg)
    curve = []

    def record(step: int) -> None:
        row = {"step": step, "train": eval_views(pipeline, dataset, train_ids, device),
               "holdout": eval_views(pipeline, dataset, holdout_ids, device)}
        curve.append(row)
        print(f"[{args.name}] step {step:5d}  train diffuse {row['train']['diffuse_psnr']:6.2f} full {row['train']['full_psnr']:6.2f}"
              f"  |  holdout diffuse {row['holdout']['diffuse_psnr']:6.2f} full {row['holdout']['full_psnr']:6.2f}", flush=True)

    start = time.time()
    record(0)
    pipeline.train()
    for step in range(steps):
        frame_ids = train_ids if views_per_step == len(train_ids) else random.sample(train_ids, views_per_step)
        optimizer.zero_grad(set_to_none=True)
        for frame_id in frame_ids:
            frame = dataset[frame_id]
            image = frame.image.to(device)
            diffuse = frame.diffuse_gt.to(device)
            output = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)
            loss, _ = world_total_loss(output, image, pipeline.scene, diffuse_gt=diffuse, specular_gt=image - diffuse, weights=weights)
            (loss / views_per_step).backward()
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % args.eval_every == 0 and step + 1 != steps:
            record(step + 1)
    record(steps)
    out = ROOT / "outputs" / "inverse_diag"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps({"name": args.name, "args": vars(args), "config": cfg, "loss": loss_cfg,
                                                       "minutes": (time.time() - start) / 60, "curve": curve}, indent=1), encoding="utf-8")
    print(f"saved outputs/inverse_diag/{args.name}.json ({(time.time() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
