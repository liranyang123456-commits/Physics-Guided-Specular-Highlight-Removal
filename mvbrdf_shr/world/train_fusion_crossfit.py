"""Train the confidence gate on cross-fitted, optionally corrupted physical renders (robust-gate extension).

Two changes with respect to ``train_fusion.py``; everything else (frozen single-image backbone, gate and bounded correction,
loss, optimizer, 1,000 steps by default) is unchanged:

1. *Cross-fitting.*  The physical buffers are rendered from the fold field that did **not** see the view
   (``scripts/train_crossfit_fields.py``: three fields, each optimized on six of the nine training views, rendering one of its
   three left-out views), so that the gate meets the residual statistics of a held-out view instead of those of an
   in-sample view.
2. *Corruption augmentation* (``--p-clean < 1``).  With probability ``1 - p_clean`` the field is corrupted at render time
   (geometry or lighting; see ``train_fusion_robust.corrupted``) while the target stays the true diffuse image.  Coverage
   corruption is never used in training and serves as an unseen type.

The final gate is evaluated, as in ``train_fusion.py``, on the strict holdout views with the field optimized on all nine
training views.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from ..models.pipeline_pro import MVBRDFSHRPro
from .fusion import HybridMVBRDFSHR
from .train import build_pipeline, load_dataset
from .train_fusion import evaluate, load_world
from .train_fusion_robust import corrupted


def load_fold(path: str, device: torch.device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(state["config"])
    cfg["raster_backend"] = "gsplat"
    dataset = load_dataset(cfg)
    pipeline = build_pipeline(dataset, cfg, device, list(state["train_ids"]))
    pipeline.load_state_dict(state["model"], strict=False)
    pipeline.eval().requires_grad_(False)
    return pipeline, dataset, list(state["left_out_ids"])


def train(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world, dataset, world_state = load_world(args.world_checkpoint, device)  # nine-view field, used for evaluation only
    folds = [load_fold(path, device) for path in args.fold_checkpoint]
    single = MVBRDFSHRPro(128, 48, 96).to(device)
    single_state = torch.load(args.single_checkpoint, map_location=device, weights_only=False)
    single.load_state_dict(single_state["model"], strict=True)
    single.eval().requires_grad_(False)
    hybrid = HybridMVBRDFSHR(single=single, gate_hidden=args.gate_hidden).to(device)
    holdout_ids = list(world_state.get("holdout_ids", []))
    train_ids = list(world_state.get("train_ids", []))
    optimizer = torch.optim.AdamW(hybrid.fusion_parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    kinds = ["clean", "geo", "light"]
    probs = [args.p_clean, (1 - args.p_clean) / 2, (1 - args.p_clean) / 2]
    history = []
    for step in range(args.steps):
        pipeline, fold_dataset, left_out = random.choice(folds)
        frame_id = random.choice(left_out)
        frame = fold_dataset[frame_id]
        image = frame.image.to(device).unsqueeze(0)
        target = frame.diffuse_gt.to(device).unsqueeze(0)
        kind = random.choices(kinds, weights=probs)[0]
        with torch.no_grad(), corrupted(pipeline, kind):
            world_output = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)
        hybrid.gate.train()
        hybrid.correction.train()
        result = hybrid(image, world_output, detach_backbones=True)
        world_error = (result["world_diffuse"] - target).abs().mean(1, keepdim=True)
        single_error = (result["single_pred"] - target).abs().mean(1, keepdim=True)
        gate_target = (world_error < single_error).to(result["gate"])
        reconstruction = F.l1_loss(result["pred"], target)
        mse = F.mse_loss(result["pred"], target)
        gate_loss = F.binary_cross_entropy(result["gate"], gate_target)
        correction_anchor = F.l1_loss(result["pred"], result["fused"].detach())
        loss = reconstruction + 4 * mse + 0.01 * gate_loss + 0.05 * correction_anchor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(hybrid.fusion_parameters()), 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            row = {"step": step, "kind": kind, "loss": float(loss.detach()),
                   "reconstruction": float(reconstruction.detach()), "gate_mean": float(result["gate"].detach().mean())}
            history.append(row)
            print(json.dumps(row), flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"gate": hybrid.gate.state_dict(), "correction": hybrid.correction.state_dict(),
                "world_checkpoint": args.world_checkpoint, "single_checkpoint": args.single_checkpoint,
                "fold_checkpoints": list(args.fold_checkpoint), "train_ids": train_ids, "holdout_ids": holdout_ids,
                "recipe": {"p_clean": args.p_clean, "steps": args.steps, "lr": args.lr, "seed": args.seed,
                           "cross_fitted": True, "families": ["geo", "light"] if args.p_clean < 1 else []}},
               output / "fusion.pt")
    metrics = evaluate(hybrid, world, dataset, holdout_ids, device, output / "holdout_predictions")
    result = {"holdout": metrics, "history": history}
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--single-checkpoint", required=True)
    parser.add_argument("--fold-checkpoint", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gate-hidden", type=int, default=48)
    parser.add_argument("--p-clean", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
