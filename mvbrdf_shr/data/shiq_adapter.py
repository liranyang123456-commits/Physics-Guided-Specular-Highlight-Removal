"""Adapters for existing SHIQ / SSHR layouts under repo `data/raw/` (read-only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..types import Batch

# mvbrdf_shr/mvbrdf_shr/data/ → repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SHIQ = _REPO_ROOT / "data" / "raw" / "SHIQ_extracted"
_DEFAULT_SSHR = _REPO_ROOT / "data" / "raw" / "SSHR_extracted"
_DEFAULT_NSH = _REPO_ROOT / "data" / "raw" / "NSH_extracted"
_DEFAULT_PSD = _REPO_ROOT / "data" / "raw" / "PSD_extracted"


def _load_rgb(path: Path, size: int | None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _augment_pair(tensors: dict, seed_state: tuple) -> dict:
    """Apply synchronized geometric + photometric augmentation to paired tensors.

    All images in ``tensors`` (image / clean / specular / mask) undergo the SAME
    geometric transform so pixel correspondence is preserved; only the input
    ``image`` gets brightness jitter (the clean diffuse GT must not be recolored).
    Used only on the train split — val/test/test_full pass through unchanged.

    Transforms: horizontal flip (p=0.5), 90/180/270 deg rotation (p=0.5 pick
    one), brightness jitter on image only (p=0.3). Rotations are 90-degree
    multiples (torch.rot90) so there is zero interpolation loss.
    """
    import random as _random

    rng = _random.Random(seed_state)
    out = {k: v.clone() for k, v in tensors.items() if v is not None}
    # Horizontal flip.
    if rng.random() < 0.5:
        for k, v in out.items():
            out[k] = torch.flip(v, dims=[-1])
    # 90-degree rotation (0/90/180/270).
    rot = rng.choice([0, 1, 2, 3])
    if rot:
        for k, v in out.items():
            out[k] = torch.rot90(v, k=rot, dims=[-2, -1])
    # Brightness jitter on the input image only (preserve GT color).
    if "image" in out and rng.random() < 0.3:
        scale = rng.uniform(0.85, 1.15)
        out["image"] = (out["image"] * scale).clamp(0.0, 1.0)
    return out


class SHIQAdapter(Dataset):
    """Load SHIQ-style pairs if present; never modifies files under data/.

    Split semantics (read-only data, no files are moved):

    - ``train``  : original ``SHIQ_data_10825/train/`` (9825 samples), unchanged.
    - ``val``    : first half of the original ``test/`` directory (sample ids
      14001-14500, 500 samples). Used ONLY for model selection / early stopping.
    - ``test``   : second half of the original ``test/`` directory (sample ids
      14501-15000, 500 samples). Held out from training and model selection; the
      ONLY split on which final numbers are reported.
    - ``test_full`` : the entire original ``test/`` directory (1000 samples),
      kept so numbers stay comparable to literature (Neural DRM, DHAN-SHR, ...)
      which evaluate on the official 1000-sample set. Use ``test_full`` for the
      literature-comparison table and ``test`` for the strict protocol table.

    The val/test 500/500 split is deterministic (by integer sample id) so it is
    fully reproducible without copying files. This fixes a test-set leakage in
    which the test split was previously used for both model selection and final
    reporting.
    """

    # Original SHIQ test ids span 14001..15000 (1000 samples). We split them
    # deterministically: lower half -> val (model selection), upper half -> test
    # (final report). These constants make the split reproducible and explicit.
    _TEST_ID_LO = 14001
    _TEST_ID_HI = 15000
    _VAL_IDS = frozenset(range(_TEST_ID_LO, _TEST_ID_LO + 500))  # 14001..14500
    _TEST_IDS = frozenset(range(_TEST_ID_LO + 500, _TEST_ID_HI + 1))  # 14501..15000

    def __init__(
        self,
        root: str | Path | None = None,
        resolution: int = 200,
        split: str = "train",
        augment: bool | None = None,
    ):
        if split not in {"train", "val", "test", "test_full"}:
            raise ValueError(
                "SHIQ split must be train|val|test|test_full "
                "(val/test are a deterministic 500/500 split of the official test set)"
            )
        self.root = Path(root) if root else _DEFAULT_SHIQ
        self.resolution = resolution
        self.split = split
        # Augment only the train split by default (val/test/test_full must be
        # deterministic for reproducible evaluation).
        self.augment = (split == "train") if augment is None else bool(augment)
        self.pairs: list[tuple[Path, Path, Path | None, Path | None]] = []
        if self.root.exists():
            self._scan(split)

    @staticmethod
    def _sample_id_from_path(path: Path) -> int | None:
        """Extract the integer SHIQ sample id (e.g. '14001' from '14001_A.png')."""
        stem = path.stem  # e.g. '14001_A'
        digits = stem.split("_", 1)[0]
        try:
            return int(digits)
        except ValueError:
            return None

    def _scan(self, split: str) -> None:
        # For val/test we always load from the official 'test' directory, then
        # filter by deterministic sample-id ranges. 'train' uses 'train/'.
        if split in {"val", "test", "test_full"}:
            scan_split = "test"
            id_filter: frozenset[int] | None = {
                "val": self._VAL_IDS,
                "test": self._TEST_IDS,
                "test_full": None,
            }[split]
        else:
            scan_split = "train"
            id_filter = None

        # Prefer explicit split directory: .../train or .../test
        split_dirs = [p for p in self.root.rglob(scan_split) if p.is_dir()]
        search_roots = split_dirs if split_dirs else [self.root]

        for lst in self.root.rglob(f"*{scan_split}*.lst"):
            for line in lst.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    a = self.root / parts[0]
                    d = self.root / parts[1]
                    m = self.root / parts[2] if len(parts) > 2 else None
                    if a.exists() and d.exists():
                        self.pairs.append((a, d, m if m and m.exists() else None, None))
            if self.pairs:
                self._apply_id_filter(id_filter)
                return

        for base in search_roots:
            for a in sorted(base.glob("*_A.png")):
                d = a.with_name(a.name.replace("_A.png", "_D.png"))
                s = a.with_name(a.name.replace("_A.png", "_S.png"))
                t = a.with_name(a.name.replace("_A.png", "_T.png"))
                if d.exists():
                    self.pairs.append(
                        (a, d, s if s.exists() else None, t if t.exists() else None)
                    )
            if self.pairs:
                self._apply_id_filter(id_filter)
                return
        # Fallback: recursive but filter by scan_split name in path
        for a in sorted(self.root.rglob("*_A.png")):
            if scan_split not in a.parts:
                continue
            d = a.with_name(a.name.replace("_A.png", "_D.png"))
            s = a.with_name(a.name.replace("_A.png", "_S.png"))
            t = a.with_name(a.name.replace("_A.png", "_T.png"))
            if d.exists():
                self.pairs.append(
                    (a, d, s if s.exists() else None, t if t.exists() else None)
                )
        self._apply_id_filter(id_filter)

    def _apply_id_filter(self, id_filter: frozenset[int] | None) -> None:
        """Keep only pairs whose sample id is in id_filter (None keeps all)."""
        if id_filter is None:
            return
        kept: list[tuple[Path, Path, Path | None, Path | None]] = []
        for pair in self.pairs:
            sid = self._sample_id_from_path(pair[0])
            if sid is not None and sid in id_filter:
                kept.append(pair)
        self.pairs = kept

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Batch:
        a, d, s, t = self.pairs[idx]
        image = _load_rgb(a, self.resolution)
        clean = _load_rgb(d, self.resolution)
        specular = _load_rgb(s, self.resolution) if s is not None else image - clean
        # SHIQ semantics: S is RGB highlight intensity; T is the binary mask.
        mask = _load_rgb(t, self.resolution)[:1] if t is not None else None
        if self.augment:
            aug = _augment_pair(
                {"image": image, "clean": clean, "specular": specular},
                seed_state=(idx, id(self)),
            )
            image, clean, specular = aug["image"], aug["clean"], aug["specular"]
            if mask is not None:
                mask = _augment_pair({"mask": mask}, seed_state=(idx, id(self)))["mask"]
        return Batch(
            image=image,
            image_clean=clean,
            specular_gt=specular,
            mask_gt=mask,
            meta={"path": str(a)},
        )


class SSHRAdapter(Dataset):
    """SSHR synthetic adapter — A/S/D style tuples."""

    def __init__(
        self,
        root: str | Path | None = None,
        resolution: int = 256,
        split: str = "train",
        tone_corrected: bool = True,
        trust_list: bool = True,
    ):
        self.root = Path(root) if root else _DEFAULT_SSHR
        self.resolution = resolution
        self.split = split
        self.pairs: list[tuple[Path, Path, Path | None]] = []
        if self.root.exists():
            base = self.root / "SSHR" if (self.root / "SSHR").is_dir() else self.root
            # SSHR ships both 4-tuple and 7-tuple list files; prefer whichever
            # exists for the requested split (avoids a slow rglob over 117k files
            # on mounted Windows drives).
            list_path = None
            for cand in (f"{split}_7_tuples.lst", f"{split}_4_tuples.lst"):
                if (base / cand).exists():
                    list_path = base / cand
                    break
            if list_path is not None:
                for line in list_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines():
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    prefix = base.parent if parts[0].startswith("SSHR/") else base
                    # Column order differs between the list files (4-tuple:
                    # i r d d_tc; 7-tuple: i a s r d d_tc m), so select by suffix.
                    by_suffix = {}
                    for part in parts:
                        stem = part.rsplit(".", 1)[0]
                        for suffix in ("_d_tc", "_i", "_a", "_s", "_r", "_d", "_m"):
                            if stem.endswith(suffix):
                                by_suffix.setdefault(suffix, part)
                                break
                    target = by_suffix.get("_d_tc" if tone_corrected else "_d") or by_suffix.get("_d")
                    if "_i" not in by_suffix or "_r" not in by_suffix or target is None:
                        continue
                    image = prefix / by_suffix["_i"]
                    specular = prefix / by_suffix["_r"]
                    diffuse = prefix / target
                    # trust_list skips per-file exists() checks (117k x 2 stat
                    # calls on mounted Windows drives take ~4 min). The official
                    # list files reference only valid files, so this is safe and
                    # makes joint-training startup practical. Missing files are
                    # caught lazily at __getitem__ time.
                    if trust_list:
                        self.pairs.append((image, diffuse, specular))
                    elif image.exists() and diffuse.exists():
                        self.pairs.append(
                            (image, diffuse, specular if specular.exists() else None)
                        )
            else:
                for image in sorted(base.rglob("*_i.jpg")):
                    diffuse = image.with_name(
                        image.name.replace(
                            "_i.jpg", "_d_tc.jpg" if tone_corrected else "_d.jpg"
                        )
                    )
                    specular = image.with_name(image.name.replace("_i.jpg", "_r.jpg"))
                    if diffuse.exists():
                        self.pairs.append(
                            (image, diffuse, specular if specular.exists() else None)
                        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Batch:
        image_path, diffuse_path, specular_path = self.pairs[idx]
        image = _load_rgb(image_path, self.resolution)
        clean = _load_rgb(diffuse_path, self.resolution)
        specular = (
            _load_rgb(specular_path, self.resolution)
            if specular_path is not None
            else (image - clean).clamp(min=0)
        )
        return Batch(
            image=image,
            image_clean=clean,
            specular_gt=specular,
            mask_gt=(specular.mean(dim=0, keepdim=True) > 0.02).float(),
            meta={"path": str(image_path)},
        )


class NSHAdapter(Dataset):
    """Paired NSH adapter using the official Specular/Diffuse test folders."""

    def __init__(
        self,
        root: str | Path | None = None,
        resolution: int = 256,
        split: str = "test",
    ):
        self.root = Path(root) if root else _DEFAULT_NSH
        self.resolution = resolution
        self.pairs: list[tuple[Path, Path]] = []
        base = self.root / "NSH" if (self.root / "NSH").is_dir() else self.root
        if split == "test":
            specular_dir = base / "Test_DataSet" / "Specular"
            diffuse_dir = base / "Test_DataSet" / "Diffuse"
        elif split in ("val", "eval"):
            specular_dir = base / "eval" / "specular"
            diffuse_dir = base / "eval" / "diffuse"
        else:
            specular_dir = base / "Train" / "Train_Specular"
            diffuse_dir = base / "Train" / "Train_Diffuse"
        if specular_dir.is_dir() and diffuse_dir.is_dir():
            for image_path in sorted(specular_dir.iterdir()):
                if not image_path.is_file():
                    continue
                diffuse_path = diffuse_dir / image_path.name
                if diffuse_path.exists():
                    self.pairs.append((image_path, diffuse_path))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Batch:
        image_path, diffuse_path = self.pairs[idx]
        image = _load_rgb(image_path, self.resolution)
        clean = _load_rgb(diffuse_path, self.resolution)
        specular = (image - clean).clamp(min=0)
        return Batch(
            image=image,
            image_clean=clean,
            specular_gt=specular,
            mask_gt=(specular.mean(dim=0, keepdim=True) > 0.02).float(),
            meta={"path": str(image_path)},
        )


class PSDAdapter(Dataset):
    """Fixed-polarization PSD group protocol.

    The SpecularityNet ``GroupDataset`` defines ``idx-01`` as diffuse target,
    excludes the invalid ``idx-07``, and evaluates all other available views.

    The public archive only ships ``PSD_Test``. For domain fine-tuning we expose
    a deterministic group-level split of that archive:

    - ``test``: all readable pairs (official zero-shot protocol)
    - ``train`` / ``val`` / ``ft_test``: 70% / 10% / 20% of *groups*
      (not pairs), so the same object group never leaks across FT splits.
      Report FT numbers only on ``ft_test`` and label the protocol clearly.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        resolution: int = 256,
        split: str = "test",
    ):
        if split not in {"test", "train", "val", "ft_test"}:
            raise ValueError("PSD split must be test|train|val|ft_test")
        self.root = Path(root) if root else _DEFAULT_PSD
        self.resolution = resolution
        self.split = split
        self.pairs: list[tuple[Path, Path]] = []
        base = self.root / "PSD_Dataset" if (self.root / "PSD_Dataset").is_dir() else self.root
        group_dir = base / "PSD_Test" / "PSD_Test_group"
        # Collect by group prefix so FT splits do not leak views of the same object.
        groups: dict[str, list[tuple[Path, Path]]] = {}
        for target in sorted(group_dir.glob("test-*-idx-01.png")):
            prefix = target.name.removesuffix("idx-01.png")
            group_pairs: list[tuple[Path, Path]] = []
            for image_path in sorted(group_dir.glob(f"{prefix}idx-*.png")):
                index = int(image_path.stem.rsplit("-", 1)[-1])
                if index not in (1, 7):
                    try:
                        with Image.open(image_path) as image:
                            image.verify()
                    except (OSError, ValueError):
                        continue
                    group_pairs.append((image_path, target))
            if group_pairs:
                groups[prefix] = group_pairs

        prefixes = sorted(groups)
        if split == "test":
            selected = prefixes
        else:
            # Deterministic 70/10/20 group split.
            n = len(prefixes)
            n_train = int(n * 0.7)
            n_val = int(n * 0.1)
            if split == "train":
                selected = prefixes[:n_train]
            elif split == "val":
                selected = prefixes[n_train : n_train + n_val]
            else:  # ft_test
                selected = prefixes[n_train + n_val :]
        for prefix in selected:
            self.pairs.extend(groups[prefix])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Batch:
        image_path, diffuse_path = self.pairs[idx]
        image = _load_rgb(image_path, self.resolution)
        clean = _load_rgb(diffuse_path, self.resolution)
        specular = (image - clean).clamp(min=0)
        return Batch(
            image=image,
            image_clean=clean,
            specular_gt=specular,
            mask_gt=(specular.mean(dim=0, keepdim=True) > 0.02).float(),
            meta={"path": str(image_path)},
        )
