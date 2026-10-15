"""Train a persistent world-space 3D Gaussian BRDF scene from many views."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from ..models.pipeline_pro import MVBRDFSHRPro
from .data import MultiViewScene
from .losses import WorldLossWeights, world_total_loss
from .pipeline import WorldGaussianBRDFPipeline
from .scene import GaussianBRDFField


def initialize_from_first_view(
    dataset: MultiViewScene,
    n_gaussians: int,
    depth_near: float = 1.0,
    depth_far: float = 3.0,
    frame_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback initialization when COLMAP has no sparse points."""
    frame = dataset[frame_id]
    origins, directions = frame.camera.pixel_rays()
    flat_origins = origins.reshape(-1, 3)
    flat_directions = directions.reshape(-1, 3)
    image = frame.image.permute(1, 2, 0).reshape(-1, 3)
    count = min(n_gaussians, flat_directions.shape[0])
    ids = torch.linspace(0, flat_directions.shape[0] - 1, count).long()
    depths = torch.linspace(depth_near, depth_far, count)
    points = flat_origins[ids] + flat_directions[ids] * depths[:, None]
    return points, image[ids]


@torch.inference_mode()
def predict_pseudo_diffuse(
    dataset: MultiViewScene,
    checkpoint: str | Path,
    frame_indices: list[int],
    device: torch.device,
    confidence_floor: float = 0.2,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Predict train-view diffuse priors without touching holdout images."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = state.get("model", state)
    # Auto-detect restorer kind: DHAN checkpoints contain 'restorer.proc.',
    # UNet checkpoints contain 'restorer.stem.'.
    restorer_kind = "dhan" if any("restorer.proc." in k for k in sd) else "unet"
    model = MVBRDFSHRPro(128, 48, 96, restorer_kind=restorer_kind).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    images, confidences = {}, {}
    for frame_id in frame_indices:
        image = dataset[frame_id].image.to(device).unsqueeze(0)
        _, _, h, w = image.shape
        # DHAN + geometry/material U-Nets require square (or evenly-divisible)
        # inputs for their skip-connections. Pad to square, predict, crop back.
        side = max(h, w)
        pad_h, pad_w = side - h, side - w
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), value=1.0)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(image, use_refine=True, detach_physics=True)
        pred = output["pred"][0, :, :h, :w].float().cpu()
        mask_pred = output["mask_pred"][0, :, :h, :w]
        images[frame_id] = pred
        # Prefer views where the single-image branch reports little ambiguity,
        # but retain a non-zero contribution in repaired highlight regions.
        confidence = confidence_floor + (1 - confidence_floor) * (1 - mask_pred.float())
        confidences[frame_id] = confidence.clamp(0, 1).cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return images, confidences


def build_pipeline(
    dataset: MultiViewScene,
    cfg: dict,
    device: torch.device,
    train_ids: list[int] | None = None,
):
    n_gaussians = int(cfg.get("n_gaussians", 512))
    pseudo_checkpoint = cfg.get("pseudo_diffuse_checkpoint")
    if dataset.points is not None:
        count = min(n_gaussians, len(dataset.points))
        ids = torch.linspace(0, len(dataset.points) - 1, count).long()
        points = dataset.points[ids]
        projected = dataset.fuse_projected_texture(points, train_ids)
        point_color_weight = float(cfg.get("point_color_weight", 0.5))
        colors = (
            point_color_weight * dataset.point_colors[ids]
            + (1 - point_color_weight) * projected
            if dataset.point_colors is not None
            else projected
        )
    else:
        first_train_id = train_ids[0] if train_ids else 0
        points, colors = initialize_from_first_view(
            dataset, n_gaussians, frame_id=first_train_id
        )
        colors = dataset.fuse_projected_texture(points, train_ids)
    if pseudo_checkpoint:
        selected_ids = train_ids or list(range(len(dataset)))
        pseudo_images, pseudo_confidences = predict_pseudo_diffuse(
            dataset,
            pseudo_checkpoint,
            selected_ids,
            device,
            float(cfg.get("pseudo_diffuse_confidence_floor", 0.2)),
        )
        pseudo_colors = dataset.fuse_projected_texture(
            points,
            selected_ids,
            source_images=pseudo_images,
            confidence_maps=pseudo_confidences,
        )
        blend = float(cfg.get("pseudo_diffuse_blend", 1.0))
        colors = colors.lerp(pseudo_colors, blend)
    elif cfg.get("initial_albedo") is not None:
        colors = torch.full_like(colors, float(cfg["initial_albedo"]))
    scene = GaussianBRDFField.from_point_cloud(
        points.to(device), colors.to(device) if colors is not None else None
    )
    gaussian_init = str(cfg.get("gaussian_init", "point_cloud"))
    if gaussian_init in {"geometry", "geometry_albedo", "oracle"}:
        prior_path = Path(cfg["data_root"]) / str(
            cfg.get("gaussian_prior", "gaussian_brdf_gt.npz")
        )
        if not prior_path.exists():
            raise FileNotFoundError(f"Gaussian prior not found: {prior_path}")
        scene.initialize_from_npz(str(prior_path), mode=gaussian_init)
    pipeline = WorldGaussianBRDFPipeline(
        scene=scene,
        n_frames=len(dataset),
        lighting_mode=str(cfg.get("lighting_mode", "sh")),
        use_refiner=bool(cfg.get("use_refiner", True)),
        refiner_base=int(cfg.get("refiner_base", 96)),
        chunk_size=int(cfg.get("chunk_size", 32)),
        raster_backend=str(cfg.get("raster_backend", "auto")),
        shared_lighting=bool(cfg.get("shared_lighting", False)),
        background_color=cfg.get("background_color"),
    ).to(device)

    shiq_checkpoint = cfg.get("shiq_refiner_checkpoint")
    if shiq_checkpoint and pipeline.refiner is not None and Path(shiq_checkpoint).exists():
        copied = pipeline.load_shiq_refiner(shiq_checkpoint)
        print(f"transferred {copied} SHIQ Restorer tensors")

    # Initialize known per-frame capture lighting when metadata is available.
    with torch.no_grad():
        for index, frame in enumerate(dataset.frames):
            if frame.light_position is not None:
                pipeline.frame_lighting.position[index].copy_(
                    frame.light_position.to(device)
                )
            if frame.light_direction is not None:
                pipeline.frame_lighting.direction_raw[index].copy_(
                    frame.light_direction.to(device)
                )
            if frame.light_color is not None:
                color = frame.light_color.to(device).clamp(1e-4, 1 - 1e-4)
                pipeline.frame_lighting.color_logits[index].copy_(
                    torch.logit(color)
                )
            if frame.point_light_weight is not None:
                weight = min(max(frame.point_light_weight, 1e-4), 1.0 - 1e-4)
                pipeline.frame_lighting.point_light_logits[index].fill_(
                    torch.logit(torch.tensor(weight)).item()
                )
            elif frame.light_position is not None and frame.light_direction is None:
                pipeline.frame_lighting.point_light_logits[index].fill_(8.0)
            if frame.light_intensity is not None:
                pipeline.frame_lighting.log_intensity[index].fill_(
                    max(frame.light_intensity, 1e-4)
                )
                pipeline.frame_lighting.log_intensity[index].log_()
            if frame.env_sh is not None:
                pipeline.frame_lighting.env_sh[index].copy_(frame.env_sh.to(device))
    return pipeline


def split_frame_indices(n_frames: int, cfg: dict) -> tuple[list[int], list[int]]:
    """Return disjoint train/holdout frame indices from the scene config."""
    stride = int(cfg.get("holdout_stride", 0))
    offset = int(cfg.get("holdout_offset", 0))
    if stride <= 1:
        return list(range(n_frames)), []
    holdout = [index for index in range(n_frames) if index % stride == offset % stride]
    train = [index for index in range(n_frames) if index not in set(holdout)]
    if not train or not holdout:
        raise ValueError("holdout split must contain both train and holdout frames")
    return train, holdout


def load_dataset(cfg: dict) -> MultiViewScene:
    root = cfg["data_root"]
    kind = str(cfg.get("data_type", "blender")).lower()
    scale = float(cfg.get("image_scale", 1.0))
    if kind == "blender":
        return MultiViewScene.from_blender(
            root, str(cfg.get("transforms", "transforms.json")), scale
        )
    if kind == "colmap":
        return MultiViewScene.from_colmap(
            root,
            str(cfg.get("sparse_dir", "sparse/0")),
            str(cfg.get("images_dir", "images")),
            scale,
        )
    if kind == "mvsnet":
        return MultiViewScene.from_mvsnet(
            root,
            str(cfg.get("images_dir", "blended_images")),
            str(cfg.get("cams_dir", "cams")),
            str(cfg.get("depth_dir", "rendered_depth_maps")),
            scale,
            int(cfg.get("max_init_points", 50000)),
            int(cfg.get("max_point_views", 16)),
        )
    if kind == "dtu":
        return MultiViewScene.from_dtu(
            root,
            int(cfg.get("scan", 1)),
            int(cfg.get("light", 3)),
            scale,
            int(cfg.get("max_init_points", 50000)),
        )
    raise ValueError(f"unknown data_type {kind}")


def train(cfg: dict) -> Path:
    requested_device = cfg.get("device")
    device = torch.device(
        requested_device
        if requested_device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset = load_dataset(cfg)
    train_ids, holdout_ids = split_frame_indices(len(dataset), cfg)
    pipeline = build_pipeline(dataset, cfg, device, train_ids)
    steps = int(cfg.get("steps", 3000))
    refine_start = int(cfg.get("refine_start", steps // 2))

    scene_parameters = (
        pipeline.scene.material_parameters()
        if bool(cfg.get("freeze_geometry", False))
        else list(pipeline.scene.parameters())
    )
    optimize_lighting = not bool(cfg.get("freeze_lighting", False))
    optimizer = torch.optim.Adam(
        [
            {"params": scene_parameters, "lr": float(cfg.get("scene_lr", 2e-3))},
            *(
                [
                    {
                        "params": pipeline.frame_lighting.parameters(),
                        "lr": float(cfg.get("light_lr", 1e-3)),
                    }
                ]
                if optimize_lighting
                else []
            ),
            *(
                [
                    {
                        "params": pipeline.neural_light.parameters(),
                        "lr": float(cfg.get("neilf_lr", 1e-4)),
                    }
                ]
                if pipeline.neural_light is not None
                else []
            ),
            *(
                [
                    {
                        "params": pipeline.refiner.parameters(),
                        "lr": float(cfg.get("refiner_lr", 1e-4)),
                    }
                ]
                if pipeline.refiner is not None
                else []
            ),
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    weights = WorldLossWeights(**cfg.get("loss", {}))
    output_dir = Path(cfg.get("output_dir", "outputs/world_scene"))
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    print(f"train views={train_ids} holdout views={holdout_ids}")
    views_per_step = min(int(cfg.get("views_per_step", 1)), len(train_ids))

    pipeline.train()
    for step in range(steps):
        frame_ids = (
            train_ids
            if views_per_step == len(train_ids)
            else random.sample(train_ids, views_per_step)
        )
        optimizer.zero_grad(set_to_none=True)
        logs: dict[str, float] = {}
        for frame_id in frame_ids:
            frame = dataset[frame_id]
            camera = frame.camera.to(device)
            image = frame.image.to(device)
            has_diffuse = frame.diffuse_gt is not None
            use_refiner = has_diffuse and step >= refine_start
            output = pipeline(camera, frame_id, image=image, refine=use_refiner)
            diffuse_gt = frame.diffuse_gt.to(device) if has_diffuse else None
            specular_gt = image - diffuse_gt if has_diffuse else None
            loss, frame_logs = world_total_loss(
                output,
                image,
                pipeline.scene,
                diffuse_gt=diffuse_gt,
                specular_gt=specular_gt,
                weights=weights,
            )
            (loss / views_per_step).backward()
            for key, value in frame_logs.items():
                logs[key] = logs.get(key, 0.0) + value / views_per_step
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if step % int(cfg.get("log_every", 25)) == 0 or step == steps - 1:
            print(
                f"[world {step}/{steps}] "
                + " ".join(f"{key}={value:.4f}" for key, value in logs.items())
            )
            history.append({"step": step, **logs})

    checkpoint = output_dir / "last.pt"
    torch.save(
        {
            "model": pipeline.state_dict(),
            "config": cfg,
            "points": pipeline.scene.means.detach().cpu(),
            "colors": pipeline.scene.texture.detach().cpu(),
            "train_ids": train_ids,
            "holdout_ids": holdout_ids,
        },
        checkpoint,
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"saved {checkpoint}")
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if args.steps is not None:
        cfg["steps"] = args.steps
    train(cfg)


if __name__ == "__main__":
    main()

