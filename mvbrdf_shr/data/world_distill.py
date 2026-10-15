"""World-GT distillation dataset for the single-image physics heads (path E).

The world_benchmark scenes carry oracle normal / albedo GT per view (rendered
from the known Gaussian-BRDF scene parameters). This dataset exposes
(image, normal_gt, albedo_gt) tuples so the single-image ScreenGeometryField
and MaterialField can be pre-trained on physics supervision — improving the
quality of the decomposition prior that feeds the restorer.

IMPORTANT provenance: world GT comes from the method's own analytic renderer
(see generate_world_benchmark.py gt_provenance). Distillation here only
improves the physics prior; the final SHR number is still measured on SHIQ
independently. Do not cite world-distill metrics as SOTA evidence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..types import Batch


def _load_unit(path: Path, size: int | None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class WorldDistillDataset(Dataset):
    """Yield (image, normal_gt, albedo_gt) over all views of all world scenes.

    normal_gt is decoded from [0,1] to signed [-1,1] and L2-normalized per pixel.
    albedo_gt stays in [0,1]. Both are resized to ``resolution``.
    """

    def __init__(self, root: str | Path, resolution: int = 128, splits=("train", "val", "test")):
        self.root = Path(root)
        self.resolution = resolution
        self.samples: list[tuple[Path, Path, Path]] = []  # (image, normal, albedo)
        scenes_root = self.root / "scenes"
        if not scenes_root.is_dir():
            return
        for split in splits:
            split_dir = scenes_root / split
            if not split_dir.is_dir():
                continue
            for scene_dir in sorted(split_dir.iterdir()):
                img_dir = scene_dir / "images"
                nrm_dir = scene_dir / "normal"
                alb_dir = scene_dir / "albedo"
                if not (img_dir.is_dir() and nrm_dir.is_dir() and alb_dir.is_dir()):
                    continue
                for img in sorted(img_dir.glob("*.png")):
                    stem = img.stem
                    nrm = nrm_dir / f"{stem}.png"
                    alb = alb_dir / f"{stem}.png"
                    if nrm.exists() and alb.exists():
                        self.samples.append((img, nrm, alb))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_p, nrm_p, alb_p = self.samples[idx]
        image = _load_unit(img_p, self.resolution)
        normal_unit = _load_unit(nrm_p, self.resolution)
        # Decode signed normal: stored as (n+1)/2 in [0,1] -> n in [-1,1].
        normal = normal_unit * 2.0 - 1.0
        normal = torch.nn.functional.normalize(normal, dim=0, eps=1e-6)
        albedo = _load_unit(alb_p, self.resolution)
        return {
            "image": image,
            "normal_gt": normal,
            "albedo_gt": albedo,
            "meta": {"path": str(img_p)},
        }


def collate_distill(batch: list[dict]) -> dict:
    """Stack a distill batch into batched tensors."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "normal_gt": torch.stack([b["normal_gt"] for b in batch]),
        "albedo_gt": torch.stack([b["albedo_gt"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }
