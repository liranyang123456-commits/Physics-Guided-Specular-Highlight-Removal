"""Prior-sensitivity study with re-optimization (scene_0070 strict holdout).

Each run perturbs one prior at initialization and then re-optimizes the world
field with the unchanged scene_0070 strict configuration (3000 Adam steps,
refiner stage from step 1500, frozen geometry and calibrated lighting) on the
training views only.  Held-out views {0, 4, 8} are evaluated with the metric
functions of eval_world_ablation.py and the frozen scene_0070 fusion gate.

Axes: clean (re-run), geo_mild, geo_severe, albedo15, albedo30, light,
drop25, drop50, sparse6, sparse3.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_world_ablation import psnr, ssim  # noqa: E402
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro  # noqa: E402
from mvbrdf_shr.world.fusion import HybridMVBRDFSHR  # noqa: E402
from mvbrdf_shr.world.losses import WorldLossWeights, world_total_loss  # noqa: E402
from mvbrdf_shr.world.scene import inverse_sigmoid  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402

AXES = (
    "clean", "geo_mild", "geo_severe", "albedo15", "albedo30",
    "light", "drop25", "drop50", "sparse6", "sparse3",
)
SPARSE_IDS = {"sparse6": [1, 3, 5, 7, 9, 11], "sparse3": [1, 5, 9]}


def perturb(pipeline, axis: str, generator: torch.Generator) -> None:
    scene = pipeline.scene
    device = scene.means.device

    def noise(t: torch.Tensor, sigma: float) -> torch.Tensor:
        return torch.randn(t.shape, generator=generator, device="cpu").to(device) * sigma

    with torch.no_grad():
        if axis in ("geo_mild", "geo_severe"):
            s_mu, s_n = (0.03, 0.25) if axis == "geo_mild" else (0.06, 0.50)
            scene.means.add_(noise(scene.means, s_mu))
            scene.normal_raw.add_(noise(scene.normal_raw, s_n))
        elif axis in ("albedo15", "albedo30"):
            rel = 0.15 if axis == "albedo15" else 0.30
            base = torch.sigmoid(scene.base_color_logits)
            noisy = (base * (1 + noise(base, rel))).clamp(1e-4, 1 - 1e-4)
            scene.base_color_logits.copy_(inverse_sigmoid(noisy))
        elif axis == "light":
            lighting = pipeline.frame_lighting
            lighting.log_intensity.add_(math.log(1.25))
            lighting.direction_raw.add_(noise(lighting.direction_raw, 0.35))
        elif axis in ("drop25", "drop50"):
            rate = 0.25 if axis == "drop25" else 0.50
            drop = torch.rand(scene.n_gaussians, generator=generator).to(device) < rate
            scene.opacity_logits[drop] = -10.0


def train_world(pipeline, dataset, cfg, train_ids, device) -> None:
    steps = int(cfg["steps"])
    refine_start = int(cfg.get("refine_start", steps // 2))
    params = (
        pipeline.scene.material_parameters()
        if bool(cfg.get("freeze_geometry", False))
        else list(pipeline.scene.parameters())
    )
    groups = [{"params": params, "lr": float(cfg.get("scene_lr", 2e-3))}]
    if not bool(cfg.get("freeze_lighting", False)):
        groups.append({"params": pipeline.frame_lighting.parameters(), "lr": float(cfg.get("light_lr", 1e-3))})
    if pipeline.refiner is not None:
        groups.append({"params": pipeline.refiner.parameters(), "lr": float(cfg.get("refiner_lr", 1e-4))})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    weights = WorldLossWeights(**cfg.get("loss", {}))
    views_per_step = min(int(cfg.get("views_per_step", 1)), len(train_ids))
    pipeline.train()
    for step in range(steps):
        frame_ids = train_ids if views_per_step == len(train_ids) else random.sample(train_ids, views_per_step)
        optimizer.zero_grad(set_to_none=True)
        for frame_id in frame_ids:
            frame = dataset[frame_id]
            image = frame.image.to(device)
            diffuse = frame.diffuse_gt.to(device)
            output = pipeline(frame.camera.to(device), frame_id, image=image, refine=step >= refine_start)
            loss, _ = world_total_loss(
                output, image, pipeline.scene,
                diffuse_gt=diffuse, specular_gt=image - diffuse, weights=weights,
            )
            (loss / views_per_step).backward()
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"[step {step}/{steps}] loss={float(loss):.4f}", flush=True)


@torch.no_grad()
def evaluate(pipeline, dataset, holdout, device) -> dict:
    pipeline.eval()
    pro = MVBRDFSHRPro(128, 48, 96).to(device)
    state = torch.load(ROOT / "outputs/world_benchmark_single/best.pt", map_location=device, weights_only=False)
    pro.load_state_dict(state.get("model", state), strict=False)
    pro.eval()
    hybrid = HybridMVBRDFSHR(resolution=128).to(device)
    hybrid.load_single_checkpoint(ROOT / "outputs/world_benchmark_single/best.pt")
    fusion = torch.load(ROOT / "outputs/world_benchmark_v2_scene_0070_hybrid/fusion.pt", map_location=device, weights_only=False)
    hybrid.gate.load_state_dict(fusion["gate"])
    hybrid.correction.load_state_dict(fusion["correction"])
    hybrid.eval()
    rows = {"phys": [], "fuse": [], "single": [], "gate": []}
    for frame_id in holdout:
        frame = dataset[frame_id]
        image = frame.image.to(device)
        target = frame.diffuse_gt.to(device)
        out = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)
        phys = out["render"].diffuse
        fused = hybrid(image.unsqueeze(0), out, detach_backbones=True)
        x = F.interpolate(image.unsqueeze(0), (128, 128), mode="bilinear", align_corners=False)
        single = pro(x, use_refine=True, detach_physics=True)["pred"]
        single = F.interpolate(single, size=image.shape[-2:], mode="bilinear", align_corners=False)[0]
        rows["phys"].append((psnr(phys, target), ssim(phys, target)))
        rows["fuse"].append((psnr(fused["pred"][0], target), ssim(fused["pred"][0], target)))
        rows["single"].append((psnr(single, target), ssim(single, target)))
        rows["gate"].append(float(fused["gate"].mean()))
    summary = {}
    for key in ("phys", "fuse", "single"):
        summary[key] = {
            "psnr": float(np.mean([v[0] for v in rows[key]])),
            "ssim": float(np.mean([v[1] for v in rows[key]])),
            "per_view": [{"frame": holdout[i], "psnr": rows[key][i][0], "ssim": rows[key][i][1]} for i in range(len(holdout))],
        }
    summary["gate_mean"] = float(np.mean(rows["gate"]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", choices=AXES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import os

    os.chdir(ROOT)
    reference = torch.load("outputs/world_benchmark_v2_scene_0070_strict/last.pt", map_location="cpu", weights_only=False)
    cfg = dict(reference["config"])
    cfg["raster_backend"] = "auto"
    dataset = load_dataset(cfg)
    holdout = list(reference["holdout_ids"])
    train_ids = SPARSE_IDS.get(args.axis, list(reference["train_ids"]))

    pipeline = build_pipeline(dataset, cfg, device, train_ids)
    generator = torch.Generator().manual_seed(1000 + args.seed)
    perturb(pipeline, args.axis, generator)

    start = time.time()
    train_world(pipeline, dataset, cfg, train_ids, device)
    summary = evaluate(pipeline, dataset, holdout, device)
    summary.update({
        "axis": args.axis,
        "seed": args.seed,
        "train_ids": train_ids,
        "holdout_ids": holdout,
        "steps": int(cfg["steps"]),
        "minutes": (time.time() - start) / 60,
    })
    out_dir = ROOT / "outputs" / "sensitivity_reopt"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.axis}_seed{args.seed}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("axis", "gate_mean", "minutes")}), flush=True)
    for key in ("phys", "fuse", "single"):
        print(f"{key}: {summary[key]['psnr']:.2f} / {summary[key]['ssim']:.4f}", flush=True)


if __name__ == "__main__":
    main()
