"""Rasterizer consistency check for the DTU / BlendedMVS diagnostic runs.

Reports, per backend, the material/geometry gradient norm of the photometric
training loss (small for the backend used during optimization) and the
held-out full-radiance PSNR, which is compared with world_metrics_holdout.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.losses import WorldLossWeights, world_total_loss  # noqa: E402
from mvbrdf_shr.world.pipeline import WorldGaussianBRDFPipeline  # noqa: E402
from mvbrdf_shr.world.scene import GaussianBRDFField  # noqa: E402
from mvbrdf_shr.world.train import load_dataset  # noqa: E402


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = (pred.clamp(0, 1) - target.clamp(0, 1)).square().mean().clamp(min=1e-12)
    return float(-10 * torch.log10(mse))


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["dtu_scan1", "blendedmvs_57f8d9bb"])
    parser.add_argument("--backends", nargs="+", default=["gsplat", "torch"])
    args = parser.parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda")
    report_path = Path("outputs/mvs_raster_backend_check.json")
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for run in args.runs:
        state = torch.load(f"outputs/{run}/last.pt", map_location=device, weights_only=False)
        cfg = state["config"]
        dataset = load_dataset(cfg)
        archived = json.loads(Path(f"outputs/{run}/world_metrics_holdout.json").read_text())["summary"]
        result = report.get(run, {})
        result["archived_holdout_full_psnr"] = archived["full_psnr"]
        for backend in args.backends:
            scene = GaussianBRDFField.from_point_cloud(state["points"].to(device), state["colors"].to(device))
            pipe = WorldGaussianBRDFPipeline(
                scene, len(dataset), lighting_mode=str(cfg.get("lighting_mode", "sh")), use_refiner=False,
                chunk_size=64, raster_backend=backend, shared_lighting=bool(cfg.get("shared_lighting", False)),
                background_color=cfg.get("background_color"),
            ).to(device)
            pipe.load_state_dict(state["model"], strict=False)
            pipe.eval()
            weights = WorldLossWeights(**cfg.get("loss", {}))
            ids = state["train_ids"][:: max(1, len(state["train_ids"]) // 8)][:8]
            for p in pipe.parameters():
                p.grad = None
            for frame_id in ids:
                frame = dataset[frame_id]
                image = frame.image.to(device)
                out = pipe(frame.camera.to(device), frame_id, image=image, refine=False)
                loss, _ = world_total_loss(out, image, pipe.scene, weights=weights)
                (loss / len(ids)).backward()
            grad = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in pipe.scene.parameters() if p.grad is not None)))
            values = []
            with torch.no_grad():
                for frame_id in state["holdout_ids"]:
                    frame = dataset[frame_id]
                    out = pipe(frame.camera.to(device), frame_id, image=frame.image.to(device), refine=False)
                    values.append(psnr(out["render"].full, frame.image.to(device)))
            result[backend] = {"grad_norm": grad, "holdout_full_psnr": sum(values) / len(values)}
            print(f"{run} {backend}: grad {grad:.3e} holdout full PSNR {result[backend]['holdout_full_psnr']:.2f} "
                  f"(archived {archived['full_psnr']:.2f})", flush=True)
        report[run] = result
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
