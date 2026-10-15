"""Demo multi-view dataset written by scripts/generate_demo_data.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..types import Batch

_PKG_ROOT = Path(__file__).resolve().parents[2]  # mvbrdf_shr/
_DEFAULT = _PKG_ROOT / "data" / "demo_mv"


def _load_rgb(path: Path, size: int | None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_gray(path: Path, size: int | None) -> torch.Tensor:
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class DemoMultiViewDataset(Dataset):
    """Load demo scenes. If group_views=True, returns (V,3,H,W) per scene."""

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        resolution: int = 128,
        group_views: bool = True,
    ):
        self.root = Path(root) if root else _DEFAULT
        self.split_dir = self.root / split
        self.resolution = resolution
        self.group_views = group_views
        self.samples: list[list[Path]] = []
        if self.split_dir.exists():
            self._scan()

    def _scan(self) -> None:
        by_scene: dict[str, list[Path]] = {}
        for a in sorted(self.split_dir.glob("*_A.png")):
            # scene_0000_v0_A.png
            name = a.name
            scene = name.rsplit("_v", 1)[0]
            by_scene.setdefault(scene, []).append(a)
        for scene, paths in sorted(by_scene.items()):
            paths = sorted(paths)
            if self.group_views:
                self.samples.append(paths)
            else:
                for p in paths:
                    self.samples.append([p])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Batch:
        paths = self.samples[idx]
        images, cleans, masks, ints = [], [], [], []
        for i, a in enumerate(paths):
            d = a.with_name(a.name.replace("_A.png", "_D.png"))
            s = a.with_name(a.name.replace("_A.png", "_S.png"))
            images.append(_load_rgb(a, self.resolution))
            cleans.append(_load_rgb(d, self.resolution))
            masks.append(_load_gray(s, self.resolution) if s.exists() else torch.zeros(1, self.resolution, self.resolution))
            ints.append(0.8 + 0.4 * ((i % 3) / 2.0))
        if len(images) == 1:
            image = images[0]
            clean = cleans[0]
            mask = masks[0]
            light = torch.tensor(ints[0], dtype=torch.float32)
        else:
            image = torch.stack(images, dim=0)
            clean = cleans[0]
            mask = masks[0]
            light = torch.tensor(ints, dtype=torch.float32)
        return Batch(
            image=image,
            image_clean=clean,
            mask_gt=mask,
            light_ints=light if light.ndim > 0 else light,
            meta={"paths": [str(p) for p in paths]},
        )
