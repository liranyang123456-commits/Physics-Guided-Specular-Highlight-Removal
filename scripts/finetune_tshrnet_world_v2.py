"""Matched in-domain fine-tuning of TSHRNet on world_benchmark_v2.

The optimization protocol is the one used for our single-image branch and for
DHAN-SHR (scripts/finetune_dhan_world_v2.py): same train/val/test splits, 3000
steps, batch size 8, AdamW (lr 1e-4, weight decay 1e-4), cosine schedule,
horizontal/vertical flips, gradient clipping at 1.0, validation every 300
steps, and the checkpoint with the best validation PSNR is evaluated once on
the 240 test and 121 hard frames.

TSHRNet keeps its native objective (train_4_networks_mix.py): the sum of four
MSE terms, in [-1, 1], on the stage-1 diffuse, the stage-1 specular residue,
the stage-2 refined diffuse, and the stage-3 tone-corrected diffuse.  The
benchmark has no separate tone-corrected target, so the stage-3 target is the
diffuse ground truth, and the specular-residue target is the exported specular
buffer.  As in the zero-shot protocol (scripts/eval_tshrnet_world.py), images
are processed at 256x256 and predictions are resized back to 128x128.
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

ROOT = Path(__file__).resolve().parents[1]
TSHR_ROOT = ROOT / "baselines_ext" / "TSHRNet"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TSHR_ROOT))

from models.UNet import UNet  # noqa: E402  (TSHRNet repository)
from mvbrdf_shr.metrics import psnr_torch, ssim_torch  # noqa: E402
from mvbrdf_shr.viz import save_image  # noqa: E402

STEPS = 3000
BATCH = 8
EVAL_EVERY = 300
SEED = 0
WORK_SIZE = 256


def rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class FlatTriples(Dataset):
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
        specular = rgb(self.root / "specular" / path.name)
        if self.augment:
            if random.random() < 0.5:
                image, clean, specular = image.flip(-1), clean.flip(-1), specular.flip(-1)
            if random.random() < 0.5:
                image, clean, specular = image.flip(-2), clean.flip(-2), specular.flip(-2)
        return image, clean, specular, path.name


def load_tshrnet(device: torch.device) -> list[torch.nn.Module]:
    checkpoints = TSHR_ROOT / "checkpoints_mix_SSHR_SHIQ_PSD"
    networks = []
    for index, channels in enumerate((3, 3, 6, 9), start=1):
        model = UNet(input_channels=channels, output_channels=3)
        state = torch.load(checkpoints / f"UNet{index}_60.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(OrderedDict((k.removeprefix("module."), v) for k, v in state.items()))
        networks.append(model.to(device))
    return networks


def to_work(x01: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x01, size=(WORK_SIZE, WORK_SIZE), mode="bicubic", align_corners=False) * 2 - 1


def forward(networks, image01: torch.Tensor):
    x = to_work(image01)
    diffuse = networks[0](x)
    specular = networks[1](x)
    refined = networks[2](torch.cat([diffuse, x], dim=1))
    corrected = networks[3](torch.cat([refined, specular, x], dim=1))
    return diffuse, specular, refined, corrected


def to_output(corrected: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return ((F.interpolate(corrected, size=size, mode="bicubic", align_corners=False) + 1) / 2).clamp(0, 1)


@torch.no_grad()
def evaluate(networks, loader, device, save_dir: Path | None = None) -> dict:
    for net in networks:
        net.eval()
    psnr, ssim = [], []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for image, clean, _, names in loader:
        image, clean = image.to(device), clean.to(device)
        prediction = to_output(forward(networks, image)[3], tuple(clean.shape[-2:]))
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
    train_loader = DataLoader(FlatTriples(bench, "train", augment=True), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(FlatTriples(bench, "val"), batch_size=BATCH)
    test_loader = DataLoader(FlatTriples(bench, "test"), batch_size=BATCH)
    hard_loader = DataLoader(FlatTriples(bench, "test", flat="flat_hard"), batch_size=BATCH)

    networks = load_tshrnet(device)
    zero_shot = {"test": evaluate(networks, test_loader, device), "hard_test": evaluate(networks, hard_loader, device)}
    print(f"[tshrnet zero-shot, in-script] test {zero_shot['test']['psnr']:.3f} hard {zero_shot['hard_test']['psnr']:.3f}",
          flush=True)
    parameters = [p for net in networks for p in net.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
    out_dir = ROOT / "outputs" / "tshrnet_world_v2_finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)

    best = {"psnr": -1.0}
    history = []
    iterator = iter(train_loader)
    for step in range(STEPS):
        try:
            image, clean, specular, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            image, clean, specular, _ = next(iterator)
        image, clean, specular = image.to(device), clean.to(device), specular.to(device)
        for net in networks:
            net.train()
        diffuse, residue, refined, corrected = forward(networks, image)
        target, target_specular = to_work(clean), to_work(specular)
        loss = (F.mse_loss(diffuse, target) + F.mse_loss(residue, target_specular)
                + F.mse_loss(refined, target) + F.mse_loss(corrected, target))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            metrics = evaluate(networks, val_loader, device)
            history.append({"step": step, **metrics})
            print(f"[tshrnet-ft {step}/{STEPS}] loss={float(loss):.4f} val={metrics['psnr']:.3f}/{metrics['ssim']:.4f}",
                  flush=True)
            if metrics["psnr"] > best["psnr"]:
                best = {**metrics, "step": step}
                torch.save({"networks": [net.state_dict() for net in networks], "metrics": best}, out_dir / "best.pt")

    state = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    for net, weights in zip(networks, state["networks"]):
        net.load_state_dict(weights)
    pred_dir = bench / "predictions" / "tshrnet_world_v2_finetuned"
    test_metrics = evaluate(networks, test_loader, device, pred_dir)
    hard_metrics = evaluate(networks, hard_loader, device)
    result = {
        "method": "tshrnet_world_v2_finetuned",
        "protocol": "same splits/steps/batch/optimizer/schedule/augmentation/selection as train_flat_benchmark.py; "
                    "native TSHRNet loss (four MSE terms); 256x256 working resolution as in the zero-shot protocol",
        "steps": STEPS,
        "batch_size": BATCH,
        "zero_shot_in_script": zero_shot,
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
    print(json.dumps({k: result[k] for k in ("zero_shot_in_script", "validation_best", "test", "hard_test")}, indent=2))


if __name__ == "__main__":
    main()
