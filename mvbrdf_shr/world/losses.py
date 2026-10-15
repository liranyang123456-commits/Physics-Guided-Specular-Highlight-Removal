"""Losses for joint world-space geometry, material, light, and refinement."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .scene import GaussianBRDFField


@dataclass(slots=True)
class WorldLossWeights:
    photo_l1: float = 1.0
    photo_ssim: float = 0.2
    diffuse_gt: float = 1.0
    refine_gt: float = 1.0
    specular_gt: float = 0.5
    energy: float = 0.1
    material_smooth: float = 0.02
    normal_axis: float = 0.02
    scale: float = 0.005
    opacity_sparse: float = 0.001
    texture_consistency: float = 0.02


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 3:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    c1, c2 = 0.01**2, 0.03**2
    mean_p = F.avg_pool2d(pred, 7, 1, 3)
    mean_t = F.avg_pool2d(target, 7, 1, 3)
    var_p = F.avg_pool2d(pred.square(), 7, 1, 3) - mean_p.square()
    var_t = F.avg_pool2d(target.square(), 7, 1, 3) - mean_t.square()
    covariance = F.avg_pool2d(pred * target, 7, 1, 3) - mean_p * mean_t
    ssim = ((2 * mean_p * mean_t + c1) * (2 * covariance + c2)) / (
        (mean_p.square() + mean_t.square() + c1) * (var_p + var_t + c2) + 1e-8
    )
    return 1 - ssim.mean()


def _material_smoothness(scene: GaussianBRDFField, max_points: int = 512) -> torch.Tensor:
    count = min(scene.n_gaussians, max_points)
    ids = torch.randperm(scene.n_gaussians, device=scene.means.device)[:count]
    points = scene.means[ids]
    distance = torch.cdist(points.detach(), points.detach())
    distance.fill_diagonal_(float("inf"))
    neighbor = distance.argmin(dim=1)
    material = scene.materials()
    features = torch.cat(
        [
            material.albedo,
            material.specular,
            material.roughness,
            material.metalness,
            material.absorption,
        ],
        dim=-1,
    )[ids]
    spatial_weight = torch.exp(-distance[torch.arange(count), neighbor].detach() * 20)
    return (
        (features - features[neighbor]).abs().mean(-1) * spatial_weight
    ).mean()


def world_total_loss(
    output: dict,
    target: torch.Tensor,
    scene: GaussianBRDFField,
    diffuse_gt: torch.Tensor | None = None,
    specular_gt: torch.Tensor | None = None,
    weights: WorldLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or WorldLossWeights()
    render = output["render"]
    target = target.to(render.full)
    loss = target.new_zeros(())
    logs: dict[str, float] = {}

    photo = F.l1_loss(render.full, target)
    photo_ssim = _ssim_loss(render.full, target)
    loss = loss + weights.photo_l1 * photo + weights.photo_ssim * photo_ssim
    logs["photo"] = float(photo.detach())
    logs["photo_ssim"] = float(photo_ssim.detach())

    if diffuse_gt is not None:
        diffuse_gt = diffuse_gt.to(render.diffuse)
        diffuse_loss = F.l1_loss(render.diffuse, diffuse_gt)
        refine_loss = F.l1_loss(output["pred"], diffuse_gt)
        loss = (
            loss
            + weights.diffuse_gt * diffuse_loss
            + weights.refine_gt * refine_loss
        )
        logs["diffuse"] = float(diffuse_loss.detach())
        logs["refine"] = float(refine_loss.detach())

    if specular_gt is not None:
        spec_loss = F.l1_loss(render.specular, specular_gt.to(render.specular))
        loss = loss + weights.specular_gt * spec_loss
        logs["specular"] = float(spec_loss.detach())

    energy = (scene.energy_sum() - 1).abs().mean()
    smooth = _material_smoothness(scene)
    axis = scene.rotations[:, :, 2]
    normal_axis = (1 - (axis * scene.normals).sum(-1).abs()).mean()
    scale_reg = scene.scales.mean()
    opacity_sparse = scene.opacity.mean()
    texture_consistency = F.l1_loss(
        scene.texture, scene.materials().albedo.detach()
    )
    loss = (
        loss
        + weights.energy * energy
        + weights.material_smooth * smooth
        + weights.normal_axis * normal_axis
        + weights.scale * scale_reg
        + weights.opacity_sparse * opacity_sparse
        + weights.texture_consistency * texture_consistency
    )
    logs.update(
        energy=float(energy.detach()),
        material_smooth=float(smooth.detach()),
        normal_axis=float(normal_axis.detach()),
        total=float(loss.detach()),
    )
    return loss, logs

