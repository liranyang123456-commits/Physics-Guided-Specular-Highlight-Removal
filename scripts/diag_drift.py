"""Diagnose why optimization from the exact scene_0070 initialization lowers diffuse accuracy.

Reuses the scene_0070 strict configuration and the training loop of
run_sensitivity_reopt.py, but (i) can start from the full oracle state
(``--init oracle``) or from the paper's setting (``--init geometry_albedo``),
(ii) can change the optimizer settings, and (iii) records the diffuse PSNR of the
training and held-out views along the way.  Results go to outputs/drift_diag/,
never to the folders used by the paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import random
import sys
import time

import numpy as np
import torch
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mvbrdf_shr.metrics import psnr_torch  # noqa: E402
from mvbrdf_shr.world.losses import WorldLossWeights, world_total_loss  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402


@torch.no_grad()
def diffuse_psnr(pipeline, dataset, ids, device) -> float:
    was_training = pipeline.training
    pipeline.eval()
    values = []
    for frame_id in ids:
        frame = dataset[frame_id]
        out = pipeline(frame.camera.to(device), frame_id, image=frame.image.to(device), refine=False)
        values.append(float(psnr_torch(out["render"].diffuse, frame.diffuse_gt.to(device))))
    pipeline.train(was_training)
    return float(np.mean(values))


@torch.no_grad()
def term_losses(pipeline, dataset, ids, device) -> dict:
    """Mean L1 of the data terms over the given views (no refiner)."""
    import torch.nn.functional as F

    pipeline.eval()
    rows = {"photo_l1": [], "diffuse_l1": [], "specular_l1": []}
    for frame_id in ids:
        frame = dataset[frame_id]
        image = frame.image.to(device)
        diffuse = frame.diffuse_gt.to(device)
        render = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)["render"]
        rows["photo_l1"].append(float(F.l1_loss(render.full, image)))
        rows["diffuse_l1"].append(float(F.l1_loss(render.diffuse, diffuse)))
        rows["specular_l1"].append(float(F.l1_loss(render.specular, image - diffuse)))
    return {k: float(np.mean(v)) for k, v in rows.items()}


@torch.no_grad()
def material_error(scene, prior_path) -> dict:
    """Distance of the current materials from the generating Gaussian field."""
    arrays = np.load(prior_path)
    m = scene.materials()
    gt = {k: torch.from_numpy(arrays[k]).to(m.albedo) for k in ("albedo", "specular", "absorption", "roughness", "metalness")}
    out = {
        "albedo_mae": float((m.albedo - gt["albedo"]).abs().mean()),
        "specular_weight_mae": float((m.specular - gt["specular"]).abs().mean()),
        "absorption_mae": float((m.absorption - gt["absorption"]).abs().mean()),
        "roughness_mae": float((m.roughness - gt["roughness"]).abs().mean()),
        "metalness_mae": float((m.metalness - gt["metalness"]).abs().mean()),
        "diffuse_weight_mean": float(m.diffuse_weight.mean()),
        "gt_diffuse_weight_mean": float((1 - gt["specular"] - gt["absorption"]).mean()),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--init", choices=("geometry_albedo", "oracle"), default="geometry_albedo")
    parser.add_argument("--scene-lr", type=float, default=2e-3)
    parser.add_argument("--views-per-step", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--no-refiner", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--loss", action="append", default=[], metavar="KEY=VALUE",
                        help="override one loss weight, e.g. --loss material_smooth=0 (repeatable)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.chdir(ROOT)
    device = torch.device("cuda")
    reference = torch.load("outputs/world_benchmark_v2_scene_0070_strict/last.pt", map_location="cpu", weights_only=False)
    cfg = dict(reference["config"])
    cfg.update(raster_backend="auto", gaussian_init=args.init, scene_lr=args.scene_lr,
               views_per_step=args.views_per_step, steps=args.steps)
    if args.no_refiner:
        cfg["use_refiner"] = False
    dataset = load_dataset(cfg)
    train_ids, holdout_ids = list(reference["train_ids"]), list(reference["holdout_ids"])
    pipeline = build_pipeline(dataset, cfg, device, train_ids)

    steps = args.steps
    refine_start = int(cfg.get("refine_start", steps // 2))
    optimizer = torch.optim.Adam([
        {"params": pipeline.scene.material_parameters(), "lr": args.scene_lr},
        *([{"params": pipeline.refiner.parameters(), "lr": float(cfg.get("refiner_lr", 1e-4))}] if pipeline.refiner is not None else []),
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    loss_cfg = dict(cfg.get("loss", {}))
    for item in args.loss:
        key, value = item.split("=")
        assert key in WorldLossWeights.__dataclass_fields__, key
        loss_cfg[key] = float(value)
    weights = WorldLossWeights(**loss_cfg)
    views_per_step = min(args.views_per_step, len(train_ids))
    curve = []

    def record(step: int, loss: float | None) -> None:
        row = {"step": step, "train_diffuse_psnr": diffuse_psnr(pipeline, dataset, train_ids, device),
               "holdout_diffuse_psnr": diffuse_psnr(pipeline, dataset, holdout_ids, device)}
        if loss is not None:
            row["loss"] = loss
        curve.append(row)
        print(f"[{args.name}] step {step:5d}  train {row['train_diffuse_psnr']:6.2f}  holdout {row['holdout_diffuse_psnr']:6.2f}"
              + (f"  loss {loss:.4f}" if loss is not None else ""), flush=True)

    prior_path = Path(cfg["data_root"]) / str(cfg.get("gaussian_prior", "gaussian_brdf_gt.npz"))
    start = time.time()
    record(0, None)
    before = {"terms_train": term_losses(pipeline, dataset, train_ids, device), "materials": material_error(pipeline.scene, prior_path)}
    pipeline.train()
    pipeline.train()
    last_loss = None
    for step in range(steps):
        frame_ids = train_ids if views_per_step == len(train_ids) else random.sample(train_ids, views_per_step)
        optimizer.zero_grad(set_to_none=True)
        for frame_id in frame_ids:
            frame = dataset[frame_id]
            image = frame.image.to(device)
            diffuse = frame.diffuse_gt.to(device)
            output = pipeline(frame.camera.to(device), frame_id, image=image,
                              refine=(step >= refine_start and pipeline.refiner is not None))
            loss, _ = world_total_loss(output, image, pipeline.scene, diffuse_gt=diffuse,
                                       specular_gt=image - diffuse, weights=weights)
            (loss / views_per_step).backward()
            last_loss = float(loss.detach())
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % args.eval_every == 0 and step + 1 != steps:
            record(step + 1, last_loss)
    record(steps, last_loss)
    after = {"terms_train": term_losses(pipeline, dataset, train_ids, device), "materials": material_error(pipeline.scene, prior_path)}
    print("terms before:", json.dumps(before["terms_train"]), flush=True)
    print("terms after :", json.dumps(after["terms_train"]), flush=True)
    print("materials before:", json.dumps({k: round(v, 4) for k, v in before["materials"].items()}), flush=True)
    print("materials after :", json.dumps({k: round(v, 4) for k, v in after["materials"].items()}), flush=True)

    out_dir = ROOT / "outputs" / "drift_diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"name": args.name, "args": vars(args), "minutes": (time.time() - start) / 60, "curve": curve,
              "before": before, "after": after}
    (out_dir / f"{args.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {out_dir / (args.name + '.json')}  ({result['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
