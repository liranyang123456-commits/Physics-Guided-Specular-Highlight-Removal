"""Train the confidence gate with corrupted physical priors (robust-gate extension).

``train_fusion.py`` trains the gate on renders of a correctly optimised world field.  Such a gate has never seen a failing
physical branch and does not fall back to the single-image branch when geometry, lighting, or coverage are wrong
(see ``scripts/compare_v2.py``, sensitivity section).  This script keeps everything of ``train_fusion.py`` -- frozen world
field, frozen single-image backbone, gate and bounded correction, loss, optimizer -- and changes one thing: at every step,
with probability ``1 - p_clean`` the frozen field is corrupted *at render time* (no re-optimization) before the physical
buffers are rendered, then restored.  The training target stays the true diffuse image, so the gate is rewarded for trusting
the physical branch where it is right and for falling back to the single-image branch where it is wrong.

Corruption families and strengths are fixed a priori (they bracket, but are not tuned to, the evaluation settings of the
sensitivity study, which re-optimizes the perturbed field):

* ``geo``   : Gaussian noise on the Gaussian centres (sigma U(0.01, 0.08)) and on the raw normals (sigma U(0.10, 0.60));
* ``light`` : global log-intensity shift U(-0.4, 0.4) and Gaussian noise on the raw light directions (sigma U(0.10, 0.50)).

Coverage corruption (random removal of Gaussians) is *not* used in training and serves as an unseen corruption type.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from ..models.pipeline_pro import MVBRDFSHRPro
from .fusion import HybridMVBRDFSHR
from .train_fusion import evaluate, load_world


@contextmanager
def corrupted(pipeline, kind: str):
    """Corrupt the frozen world field in place for the duration of the ``with`` block."""
    scene, lighting = pipeline.scene, pipeline.frame_lighting
    tensors = {
        "means": scene.means,
        "normal_raw": scene.normal_raw,
        "log_intensity": lighting.log_intensity,
        "direction_raw": lighting.direction_raw,
    }
    saved = {name: tensor.detach().clone() for name, tensor in tensors.items()}
    with torch.no_grad():
        if kind == "geo":
            sigma_mu, sigma_n = random.uniform(0.01, 0.08), random.uniform(0.10, 0.60)
            scene.means.add_(torch.randn_like(scene.means) * sigma_mu)
            scene.normal_raw.add_(torch.randn_like(scene.normal_raw) * sigma_n)
        elif kind == "light":
            shift, sigma_d = random.uniform(-0.4, 0.4), random.uniform(0.10, 0.50)
            lighting.log_intensity.add_(shift)
            lighting.direction_raw.add_(torch.randn_like(lighting.direction_raw) * sigma_d)
        elif kind != "clean":
            raise ValueError(kind)
    try:
        yield
    finally:
        with torch.no_grad():
            for name, tensor in tensors.items():
                tensor.copy_(saved[name])


def train(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world, dataset, world_state = load_world(args.world_checkpoint, device)
    single = MVBRDFSHRPro(128, 48, 96).to(device)
    single_state = torch.load(args.single_checkpoint, map_location=device, weights_only=False)
    single.load_state_dict(single_state["model"], strict=True)
    single.eval().requires_grad_(False)
    hybrid = HybridMVBRDFSHR(single=single, gate_hidden=args.gate_hidden).to(device)
    train_ids = list(world_state.get("train_ids", range(len(dataset))))
    holdout_ids = list(world_state.get("holdout_ids", []))
    if not holdout_ids:
        raise ValueError("world checkpoint has no strict holdout split")
    optimizer = torch.optim.AdamW(hybrid.fusion_parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    kinds = ["clean", "geo", "light"]
    probs = [args.p_clean, (1 - args.p_clean) / 2, (1 - args.p_clean) / 2]
    history = []
    for step in range(args.steps):
        frame_id = random.choice(train_ids)
        frame = dataset[frame_id]
        image = frame.image.to(device).unsqueeze(0)
        target = frame.diffuse_gt.to(device).unsqueeze(0)
        kind = random.choices(kinds, weights=probs)[0]
        with torch.no_grad(), corrupted(world, kind):
            world_output = world(frame.camera.to(device), frame_id, image=image, refine=False)
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
                "train_ids": train_ids, "holdout_ids": holdout_ids,
                "recipe": {"p_clean": args.p_clean, "steps": args.steps, "lr": args.lr, "seed": args.seed,
                           "families": ["geo", "light"]}}, output / "fusion.pt")
    metrics = evaluate(hybrid, world, dataset, holdout_ids, device, output / "holdout_predictions")
    result = {"holdout": metrics, "history": history}
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--single-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gate-hidden", type=int, default=48)
    parser.add_argument("--p-clean", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
