"""Systematic world/fusion ablations on scene_0070 strict holdout.

Uses existing checkpoints where possible; reports physical / refined / fusion
variants without fabricating numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension

# MSVC prints a localized (non-UTF-8) banner that breaks the gsplat JIT check.
torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro
from mvbrdf_shr.world.fusion import HybridMVBRDFSHR
from mvbrdf_shr.world.pipeline import WorldGaussianBRDFPipeline
from mvbrdf_shr.world.scene import GaussianBRDFField
from mvbrdf_shr.world.train import load_dataset


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = (pred.clamp(0, 1) - target.clamp(0, 1)).square().mean().clamp(min=1e-12)
    return float((-10 * torch.log10(mse)).item())


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.float().clamp(0, 1).unsqueeze(0)
    target = target.float().clamp(0, 1).unsqueeze(0)
    pred = F.pad(pred, (5, 5, 5, 5), mode="reflect")
    target = F.pad(target, (5, 5, 5, 5), mode="reflect")
    mp = F.avg_pool2d(pred, 11, 1)
    mt = F.avg_pool2d(target, 11, 1)
    vp = F.avg_pool2d(pred.square(), 11, 1) - mp.square()
    vt = F.avg_pool2d(target.square(), 11, 1) - mt.square()
    cov = F.avg_pool2d(pred * target, 11, 1) - mp * mt
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mp * mt + c1) * (2 * cov + c2)) / (
        (mp.square() + mt.square() + c1) * (vp + vt + c2) + 1e-12
    )
    return float(score.mean().item())


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world-checkpoint",
        default="outputs/world_benchmark_v2_scene_0070_strict/last.pt",
    )
    parser.add_argument(
        "--single-checkpoint",
        default="outputs/world_benchmark_single/best.pt",
    )
    parser.add_argument(
        "--fusion-checkpoint",
        default="outputs/world_benchmark_v2_scene_0070_hybrid/fusion.pt",
    )
    parser.add_argument("--output", default="outputs/world_ablation_scene_0070")
    parser.add_argument(
        "--raster-backend",
        default="auto",
        help="Must match the rasterizer used to optimize the scene "
        "(see scripts/check_raster_backend.py).",
    )
    parser.add_argument(
        "--dhan-checkpoint",
        default="outputs/dhan_world_v2_finetuned/best.pt",
        help="DHAN-SHR fine-tuned with the matched protocol "
        "(scripts/finetune_dhan_world_v2.py); empty string disables it.",
    )
    parser.add_argument(
        "--tshrnet-checkpoint",
        default="outputs/tshrnet_world_v2_finetuned/best.pt",
        help="TSHRNet fine-tuned with the matched protocol "
        "(scripts/finetune_tshrnet_world_v2.py); empty string disables it.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.world_checkpoint, map_location=device, weights_only=False)
    cfg = state["config"]
    dataset = load_dataset(cfg)
    holdout = state.get("holdout_ids", [0, 4, 8])
    scene = GaussianBRDFField.from_point_cloud(
        state["points"].to(device),
        state["colors"].to(device) if state.get("colors") is not None else None,
    )
    pipe = WorldGaussianBRDFPipeline(
        scene,
        len(dataset),
        lighting_mode=str(cfg.get("lighting_mode", "sh")),
        use_refiner=True,
        refiner_base=int(cfg.get("refiner_base", 96)),
        chunk_size=int(cfg.get("chunk_size", 32)),
        raster_backend=args.raster_backend,
        shared_lighting=bool(cfg.get("shared_lighting", False)),
        background_color=cfg.get("background_color"),
    ).to(device)
    pipe.load_state_dict(state["model"], strict=False)
    pipe.eval()

    pro = MVBRDFSHRPro(128, 48, 96).to(device)
    pstate = torch.load(args.single_checkpoint, map_location=device, weights_only=False)
    pro.load_state_dict(pstate.get("model", pstate), strict=False)
    pro.eval()

    # Hybrid owns its own Pro backbone; must load the same single checkpoint
    # used during fusion training (not a randomly initialized single).
    hybrid = HybridMVBRDFSHR(resolution=128).to(device)
    hybrid.load_single_checkpoint(args.single_checkpoint)
    if Path(args.fusion_checkpoint).exists():
        fstate = torch.load(args.fusion_checkpoint, map_location=device, weights_only=False)
        # fusion.pt stores gate/correction modules, not a full model state_dict.
        if "gate" in fstate and "correction" in fstate:
            hybrid.gate.load_state_dict(fstate["gate"])
            hybrid.correction.load_state_dict(fstate["correction"])
        else:
            hybrid.load_state_dict(fstate.get("model", fstate), strict=False)
    hybrid.eval()

    dhan = None
    if args.dhan_checkpoint and Path(args.dhan_checkpoint).exists():
        sys.path.insert(0, str(Path("baselines_ext/DHAN-SHR").resolve()))
        from models import Model  # DHAN-SHR repository

        dhan = Model()
        dhan.load_state_dict(torch.load(args.dhan_checkpoint, map_location="cpu", weights_only=False)["model"])
        dhan = dhan.to(device).eval()

    tshrnet = None
    if args.tshrnet_checkpoint and Path(args.tshrnet_checkpoint).exists():
        import importlib.util

        # Loaded by path: DHAN-SHR and TSHRNet both ship a top-level "models" package.
        spec = importlib.util.spec_from_file_location("tshrnet_unet", "baselines_ext/TSHRNet/models/UNet.py")
        unet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(unet)
        weights = torch.load(args.tshrnet_checkpoint, map_location="cpu", weights_only=False)["networks"]
        tshrnet = []
        for channels, state_dict in zip((3, 3, 6, 9), weights):
            net = unet.UNet(input_channels=channels, output_channels=3)
            net.load_state_dict(state_dict)
            tshrnet.append(net.to(device).eval())

    rows = {
        "ours_phys": [],
        "ours_refined": [],
        "ours_pro": [],
        "fusion_full": [],
        "fusion_analytic": [],
        "fusion_no_correction": [],
        "fixed_blend": [],
        "input": [],
    }
    confidence_rows = []

    for fid in holdout:
        frame = dataset[fid]
        image = frame.image.to(device)
        gt = frame.diffuse_gt.to(device)
        out = pipe(frame.camera.to(device), fid, image=image, refine=True)
        phys = out["render"].diffuse
        refined = out["pred"]
        x = F.interpolate(image.unsqueeze(0), (128, 128), mode="bilinear", align_corners=False)
        pro_pred = pro(x, use_refine=True, detach_physics=True)["pred"]
        pro_pred = F.interpolate(pro_pred, size=image.shape[-2:], mode="bilinear", align_corners=False)[0]

        fout = hybrid(image.unsqueeze(0), out, detach_backbones=True)
        full = fout["pred"][0]
        # Analytic-only gate: zero the learned residual by using sigmoid(analytic_logit)
        world_d = out["render"].diffuse.unsqueeze(0)
        world_s = out["render"].specular.unsqueeze(0)
        alpha = out["render"].alpha.unsqueeze(0)
        err = (image.unsqueeze(0) - (world_d + world_s).clamp(0, 1)).abs().mean(1, keepdim=True)
        c = (alpha * torch.exp(-err / hybrid.reconstruction_temperature)).clamp(1e-4, 1 - 1e-4)
        g_an = c  # already in (0,1)
        analytic = (g_an * world_d + (1 - g_an) * pro_pred.unsqueeze(0))[0]
        # Learned gate, no correction
        no_corr = fout["fused"][0]
        fixed = (0.5 * phys + 0.5 * pro_pred).clamp(0, 1)

        # Does the full-radiance residual localize the actual diffuse error?
        covered = (alpha[0, 0] > 0.1).flatten()
        e_full = err[0, 0].flatten()[covered]
        e_diff = (phys - gt).abs().mean(0).flatten()[covered]
        conf = c[0, 0].flatten()[covered]
        if e_full.numel() > 10 and float(e_full.std()) > 1e-8 and float(e_diff.std()) > 1e-8:
            pearson = float(torch.corrcoef(torch.stack([e_full, e_diff]))[0, 1])
            low = conf < conf.median()
            confidence_rows.append({
                "frame": fid,
                "covered_pixels": int(covered.sum()),
                "pearson_e_vs_diffuse_error": pearson,
                "diffuse_l1_low_conf_half": float(e_diff[low].mean()),
                "diffuse_l1_high_conf_half": float(e_diff[~low].mean()),
            })

        rows["ours_phys"].append((psnr(phys, gt), ssim(phys, gt)))
        rows["ours_refined"].append((psnr(refined, gt), ssim(refined, gt)))
        rows["ours_pro"].append((psnr(pro_pred, gt), ssim(pro_pred, gt)))
        rows["fusion_full"].append((psnr(full, gt), ssim(full, gt)))
        rows["fusion_analytic"].append((psnr(analytic, gt), ssim(analytic, gt)))
        rows["fusion_no_correction"].append((psnr(no_corr, gt), ssim(no_corr, gt)))
        rows["fixed_blend"].append((psnr(fixed, gt), ssim(fixed, gt)))
        rows["input"].append((psnr(image, gt), ssim(image, gt)))

        # The same world branch combined with a stronger single-image restorer:
        # the analytic gate depends only on the world residual, and the learned
        # gate is reused without retraining.
        if dhan is not None:
            restored = dhan(image.unsqueeze(0)).clamp(0, 1)[0]
            gate = fout["gate"][0]
            for key, pred in (
                ("dhan_ft", restored),
                ("dhan_fixed_blend", (0.5 * phys + 0.5 * restored).clamp(0, 1)),
                ("dhan_analytic", c[0] * phys + (1 - c[0]) * restored),
                ("dhan_gate", gate * phys + (1 - gate) * restored),
            ):
                rows.setdefault(key, []).append((psnr(pred, gt), ssim(pred, gt)))
        if tshrnet is not None:
            # Same 256x256 working resolution as the zero-shot and fine-tuning protocols.
            x = F.interpolate(image.unsqueeze(0), size=(256, 256), mode="bicubic", align_corners=False) * 2 - 1
            d1, s1 = tshrnet[0](x), tshrnet[1](x)
            d2 = tshrnet[2](torch.cat([d1, x], dim=1))
            d4 = tshrnet[3](torch.cat([d2, s1, x], dim=1))
            pred = ((F.interpolate(d4, size=image.shape[-2:], mode="bicubic", align_corners=False) + 1) / 2)
            pred = pred.clamp(0, 1)[0]
            rows.setdefault("tshrnet_ft", []).append((psnr(pred, gt), ssim(pred, gt)))

    # Pull inverse numbers from existing JSON if present.
    inv = Path("outputs/world_benchmark_v2_scene_0070_inverse/world_metrics_holdout.json")
    invp = Path("outputs/world_benchmark_v2_scene_0070_inverse_pro/world_metrics_holdout.json")
    summary = {}
    for key, vals in rows.items():
        summary[key] = {
            "psnr": float(np.mean([v[0] for v in vals])),
            "ssim": float(np.mean([v[1] for v in vals])),
            "per_view": [
                {"frame": holdout[i], "psnr": vals[i][0], "ssim": vals[i][1]}
                for i in range(len(vals))
            ],
        }
    if inv.exists():
        d = json.loads(inv.read_text())
        summary["inverse"] = {
            "psnr": d["summary"]["diffuse_psnr"],
            "ssim": d["summary"]["diffuse_ssim"],
            "source": str(inv),
        }
    if invp.exists():
        d = json.loads(invp.read_text())
        summary["inverse_pro"] = {
            "psnr": d["summary"]["diffuse_psnr"],
            "ssim": d["summary"]["diffuse_ssim"],
            "source": str(invp),
        }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_metrics.json").write_text(
        json.dumps(
            {
                "protocol": "strict holdout; local checkpoints only; fusion gate from --fusion-checkpoint",
                "world_checkpoint": args.world_checkpoint,
                "fusion_checkpoint": args.fusion_checkpoint,
                "raster_backend": args.raster_backend,
                "dhan_checkpoint": args.dhan_checkpoint if dhan is not None else None,
                "tshrnet_checkpoint": args.tshrnet_checkpoint if tshrnet is not None else None,
                "holdout_ids": holdout,
                "variants": summary,
                "confidence_correlation": confidence_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
