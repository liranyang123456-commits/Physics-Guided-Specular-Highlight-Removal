"""Identify which rasterizer each strict-holdout scene was optimized with.

Training ends with a cosine learning rate that decays to zero, so the final
material parameters sit near a stationary point of the training objective
rendered with the backend that was used during optimization.  The full-batch
gradient norm of the training loss is therefore much smaller for that backend.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.losses import WorldLossWeights, world_total_loss  # noqa: E402
from mvbrdf_shr.world.pipeline import WorldGaussianBRDFPipeline  # noqa: E402
from mvbrdf_shr.world.scene import GaussianBRDFField  # noqa: E402
from mvbrdf_shr.world.train import load_dataset  # noqa: E402


def main() -> None:
    import os

    os.chdir(ROOT)
    device = torch.device("cuda")
    report = {}
    for scene_id in range(70, 80):
        run = Path(f"outputs/world_benchmark_v2_scene_0{scene_id:03d}_strict")
        state = torch.load(run / "last.pt", map_location=device, weights_only=False)
        cfg = state["config"]
        dataset = load_dataset(cfg)
        history = json.loads((run / "history.json").read_text())
        tail = history[-5:]
        logged = {
            "photo": sum(h["photo"] for h in tail) / len(tail),
            "diffuse": sum(h.get("diffuse", 0.0) for h in tail) / len(tail),
        }
        measured = {}
        for backend in ("gsplat", "torch"):
            scene = GaussianBRDFField.from_point_cloud(state["points"].to(device), state["colors"].to(device))
            pipe = WorldGaussianBRDFPipeline(
                scene, len(dataset), lighting_mode="sh", use_refiner=True, refiner_base=96,
                chunk_size=32, raster_backend=backend, background_color=cfg.get("background_color"),
            ).to(device).eval()
            pipe.load_state_dict(state["model"], strict=False)
            weights = WorldLossWeights(**cfg.get("loss", {}))
            params = pipe.scene.material_parameters()
            for p in pipe.parameters():
                p.grad = None
            photo, diffuse = [], []
            for frame_id in state["train_ids"]:
                frame = dataset[frame_id]
                image = frame.image.to(device)
                target = frame.diffuse_gt.to(device)
                out = pipe(frame.camera.to(device), frame_id, image=image, refine=False)
                out["pred"] = out["render"].diffuse
                loss, _ = world_total_loss(
                    out, image, pipe.scene, diffuse_gt=target,
                    specular_gt=image - target, weights=weights,
                )
                (loss / len(state["train_ids"])).backward()
                photo.append(float(F.l1_loss(out["render"].full, image)))
                diffuse.append(float(F.l1_loss(out["render"].diffuse, target)))
            grad_norm = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))
            measured[backend] = {
                "photo": sum(photo) / len(photo),
                "diffuse": sum(diffuse) / len(diffuse),
                "material_grad_norm": grad_norm,
            }
        backend_used = min(measured, key=lambda b: measured[b]["material_grad_norm"])
        report[scene_id] = {"logged": logged, "measured": measured, "training_backend": backend_used}
        print(
            f"scene {scene_id}: |grad| gsplat {measured['gsplat']['material_grad_norm']:.2e} "
            f"torch {measured['torch']['material_grad_norm']:.2e} | train diffuse L1 "
            f"gsplat {measured['gsplat']['diffuse']:.4f} torch {measured['torch']['diffuse']:.4f} -> {backend_used}",
            flush=True,
        )
    Path("outputs/world_raster_backend_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
