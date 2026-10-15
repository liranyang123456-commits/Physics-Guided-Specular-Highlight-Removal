"""Hard-synth paired dataset (SHIQ-like protocol)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..types import Batch

_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "hard_synth"


def _rgb(path: Path, size: int | None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)


def _gray(path: Path, size: int | None) -> torch.Tensor:
    img = Image.open(path).convert("L")
    if size:
        img = img.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).unsqueeze(0)


class HardSynthDataset(Dataset):
    def __init__(self, root=None, split="train", resolution=200, group_lights=True):
        self.root = Path(root) if root else _DEFAULT
        self.dir = self.root / split
        self.resolution = resolution
        self.group_lights = group_lights
        self.samples: list[list[Path]] = []
        if self.dir.exists():
            by = {}
            for a in sorted(self.dir.glob("*_A.png")):
                key = a.name.rsplit("_v", 1)[0]
                by.setdefault(key, []).append(a)
            for _, paths in sorted(by.items()):
                paths = sorted(paths)
                if group_lights:
                    self.samples.append(paths)
                else:
                    for p in paths:
                        self.samples.append([p])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]
        imgs, cleans, masks, ints = [], [], [], []
        for i, a in enumerate(paths):
            imgs.append(_rgb(a, self.resolution))
            cleans.append(_rgb(a.with_name(a.name.replace("_A.png", "_D.png")), self.resolution))
            s = a.with_name(a.name.replace("_A.png", "_S.png"))
            masks.append(_gray(s, self.resolution))
            ints.append(0.7 + 0.5 * ((i + 1) / max(len(paths), 1)))
        if len(imgs) == 1:
            return Batch(
                image=imgs[0],
                image_clean=cleans[0],
                mask_gt=masks[0],
                light_ints=torch.tensor(ints[0]),
            )
        return Batch(
            image=torch.stack(imgs, 0),
            image_clean=cleans[0],
            mask_gt=masks[0],
            light_ints=torch.tensor(ints, dtype=torch.float32),
        )
