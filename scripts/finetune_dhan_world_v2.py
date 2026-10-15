"""Matched in-domain fine-tuning of DHAN-SHR on world_benchmark_v2.

The optimization protocol mirrors scripts/train_flat_benchmark.py, which
produced our in-domain single-image checkpoint: same train/val/test splits,
3000 steps, batch size 8, AdamW (lr 1e-4, weight decay 1e-4), cosine schedule,
horizontal/vertical flip augmentation, validation every 300 steps, and the
checkpoint with the best validation PSNR is evaluated once on the 240 test and
121 hard frames.  DHAN-SHR keeps its native objective (MSE + 0.4 * (1 - SSIM)).
Metrics use the same psnr_torch/ssim_torch functions as the other rows of the
world_benchmark_v2 table.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.image import structural_similarity_index_measure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvbrdf_shr.metrics import psnr_torch, ssim_torch  # noqa: E402
from mvbrdf_shr.viz import save_image  # noqa: E402

STEPS = 3000
BATCH = 8
EVAL_EVERY = 300
SEED = 0


def rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class FlatPairs(Dataset):
    def __init__(self, root: Path, split: str, flat: str = "flat", augment: bool = False):
        self.root = root / flat / split
        self.paths = sorted((self.root / "input").glob("*.png"))
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = rgb(path)
        clean = rgb(self.root / "gt" / path.name)
        if self.augment:
            if random.random() < 0.5:
                image, clean = image.flip(-1), clean.flip(-1)
            if random.random() < 0.5:
                image, clean = image.flip(-2), clean.flip(-2)
        return image, clean, path.name


def load_dhan(device: torch.device) -> torch.nn.Module:
    dhan_root = ROOT / "baselines_ext" / "DHAN-SHR"
    sys.path.insert(0, str(dhan_root))
    from models import Model  # type: ignore

    model = Model()
    state = torch.load(dhan_root / "weights" / "model.pth", map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = OrderedDict((k.removeprefix("module."), v) for k, v in state.items())
    model.load_state_dict(state, strict=False)
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device, save_dir: Path | None = None) -> dict:
    model.eval()
    psnr, ssim = [], []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for image, clean, names in loader:
        image, clean = image.to(device), clean.to(device)
        prediction = model(image).clamp(0, 1)
        for i in range(len(prediction)):
            psnr.append(psnr_torch(prediction[i : i + 1], clean[i : i + 1]))
            ssim.append(ssim_torch(prediction[i : i + 1], clean[i : i + 1]))
            if save_dir is not None:
                save_image(prediction[i], save_dir / names[i])
    return {"n": len(psnr), "psnr": float(np.mean(psnr)), "ssim": float(np.mean(ssim))}


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench = ROOT / "data" / "world_benchmark_v2"
    train_loader = DataLoader(FlatPairs(bench, "train", augment=True), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(FlatPairs(bench, "val"), batch_size=BATCH)
    test_loader = DataLoader(FlatPairs(bench, "test"), batch_size=BATCH)
    hard_loader = DataLoader(FlatPairs(bench, "test", flat="flat_hard"), batch_size=BATCH)

    model = load_dhan(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
    out_dir = ROOT / "outputs" / "dhan_world_v2_finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)

    best = {"psnr": -1.0}
    history = []
    iterator = iter(train_loader)
    for step in range(STEPS):
        try:
            image, clean, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            image, clean, _ = next(iterator)
        image, clean = image.to(device), clean.to(device)
        model.train()
        prediction = model(image)
        loss = F.mse_loss(prediction, clean) + 0.4 * (
            1 - structural_similarity_index_measure(prediction, clean, data_range=1.0)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            metrics = evaluate(model, val_loader, device)
            history.append({"step": step, **metrics})
            print(f"[dhan-ft {step}/{STEPS}] loss={float(loss):.4f} val={metrics['psnr']:.3f}/{metrics['ssim']:.4f}", flush=True)
            if metrics["psnr"] > best["psnr"]:
                best = {**metrics, "step": step}
                torch.save({"model": model.state_dict(), "metrics": best}, out_dir / "best.pt")

    state = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    pred_dir = bench / "predictions" / "dhan_shr_world_v2_finetuned"
    test_metrics = evaluate(model, test_loader, device, pred_dir)
    hard_metrics = evaluate(model, hard_loader, device)
    result = {
        "method": "dhan_shr_world_v2_finetuned",
        "protocol": "same splits/steps/batch/optimizer/schedule/augmentation/selection as train_flat_benchmark.py; native DHAN loss",
        "steps": STEPS,
        "batch_size": BATCH,
        "validation_best": best,
        "test": test_metrics,
        "hard_test": hard_metrics,
        "history": history,
        "prediction_dir": str(pred_dir),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (pred_dir / "metrics.json").write_text(
        json.dumps({k: result[k] for k in ("method", "test", "hard_test", "prediction_dir")}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: result[k] for k in ("validation_best", "test", "hard_test")}, indent=2))


if __name__ == "__main__":
    main()
