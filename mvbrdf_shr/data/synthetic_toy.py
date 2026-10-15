"""Procedural multi-view / multi-light toy dataset for Phase A."""
from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset

from ..types import Batch


def _make_sphere_buffers(H: int, W: int, light_dir: torch.Tensor, roughness: float):
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    rr = xx ** 2 + yy ** 2
    mask = (rr <= 1.0).float()
    zz = torch.sqrt((1.0 - rr).clamp(min=0.0))
    normal = torch.stack([xx, yy, zz], dim=0)  # (3,H,W)
    normal = normal / (normal.norm(dim=0, keepdim=True) + 1e-6)
    albedo = torch.tensor([0.7, 0.35, 0.25]).view(3, 1, 1).expand(3, H, W).clone()
    albedo = albedo * mask.unsqueeze(0)

    L = light_dir / (light_dir.norm() + 1e-6)
    V = torch.tensor([0.0, 0.0, 1.0])
    n_dot_l = (normal * L.view(3, 1, 1)).sum(0).clamp(min=0.0)
    diffuse = albedo * (0.15 + 0.85 * n_dot_l.unsqueeze(0))

    Hvec = L + V
    Hvec = Hvec / (Hvec.norm() + 1e-6)
    n_dot_h = (normal * Hvec.view(3, 1, 1)).sum(0).clamp(min=0.0)
    # Blinn-Phong style highlight for toy GT
    spec = (n_dot_h ** (1.0 / max(roughness, 0.05) * 8.0)).unsqueeze(0)
    specular = spec.expand(3, H, W) * mask.unsqueeze(0)
    full = (diffuse + 0.6 * specular).clamp(0, 1) * mask.unsqueeze(0)
    clean = diffuse.clamp(0, 1) * mask.unsqueeze(0)
    spec_mask = (specular.mean(0, keepdim=True) > 0.15).float() * mask.unsqueeze(0)
    return full, clean, spec_mask


class SyntheticMultiViewToy(Dataset):
    """Synthetic glossy sphere under rotating lights / slight view jitter."""

    def __init__(
        self,
        size: int = 64,
        resolution: int = 128,
        n_views: int = 4,
        seed: int = 0,
    ):
        self.size = size
        self.resolution = resolution
        self.n_views = n_views
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Batch:
        H = W = self.resolution
        images = []
        cleans = []
        masks = []
        light_ints = []
        base_angle = (idx / max(self.size, 1)) * 2 * math.pi
        for v in range(self.n_views):
            ang = base_angle + v * (2 * math.pi / self.n_views)
            elev = 0.4 + 0.2 * math.sin(ang)
            light_dir = torch.tensor(
                [math.cos(ang) * 0.7, elev, 0.6 + 0.2 * math.cos(ang * 0.5)]
            )
            intensity = 0.8 + 0.4 * ((v % 3) / 2.0)
            full, clean, mask = _make_sphere_buffers(H, W, light_dir, roughness=0.2)
            full = (full * intensity).clamp(0, 1)
            images.append(full)
            cleans.append(clean)
            masks.append(mask)
            light_ints.append(intensity)
        image = torch.stack(images, dim=0)          # (V,3,H,W)
        image_clean = cleans[0]                     # reference clean (view 0 shading)
        mask_gt = masks[0]
        return Batch(
            image=image,
            image_clean=image_clean,
            mask_gt=mask_gt,
            light_ints=torch.tensor(light_ints, dtype=torch.float32),
            meta={"idx": idx},
        )


def collate_batch(samples: list[Batch]) -> Batch:
    """Stack samples; if multi-view (V,3,H,W) mixed with single (3,H,W), take view-0."""
    images = []
    for s in samples:
        img = s.image
        if img.dim() == 4:  # (V,3,H,W)
            images.append(img)
        elif img.dim() == 3:
            images.append(img)
        else:
            raise ValueError(f"bad image dim {img.shape}")
    # Unify: if any multi-view, keep 5D; if shapes differ, collapse to view-0
    if any(im.dim() == 4 for im in images):
        if not all(im.dim() == 4 and im.shape[0] == images[0].shape[0] for im in images):
            images = [im[0] if im.dim() == 4 else im for im in images]
            image = torch.stack(images, dim=0)
        else:
            image = torch.stack(images, dim=0)  # (B,V,3,H,W)
    else:
        image = torch.stack(images, dim=0)

    def stack_opt(attr: str):
        vals = [getattr(s, attr) for s in samples]
        if vals[0] is None:
            return None
        return torch.stack(vals, dim=0)

    return Batch(
        image=image,
        image_clean=stack_opt("image_clean"),
        specular_gt=stack_opt("specular_gt"),
        mask_gt=stack_opt("mask_gt"),
        poses=stack_opt("poses"),
        light_ints=stack_opt("light_ints"),
    )
