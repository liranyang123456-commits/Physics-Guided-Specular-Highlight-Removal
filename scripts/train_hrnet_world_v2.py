"""Matched-budget training of the HighlightRNet architecture on world_benchmark_v2 (no public weights).

The public repository of HighlightRNet (HighlightRemover, ACM MM 2024; github.com/zz0223/HRNet) contains the network
(models/arch/unet.py, RefinedNet) and its training objective (models/specularitynet_model.py, backward_G), but no
checkpoint and no ``options`` package, its scripts point to the authors' directories, and the NSH data are not
released.  A fine-tuning from official weights is therefore impossible.  This script trains the released architecture
from its default (random) initialization with the optimization protocol of scripts/finetune_dhan_world_v2.py: the same
train/val/test splits, batch size 8, AdamW (lr 1e-4, weight decay 1e-4), cosine schedule, horizontal/vertical flips,
gradient clipping at 1.0, validation every EVAL_EVERY steps, and the checkpoint with the best validation PSNR is
evaluated once on the 240 test and 121 hard frames.  The pyramid pooling of the network (average pooling with kernels up
to 32 on features at 1/8 of the input size) needs inputs of at least 256 x 256 (the original loader uses 512 x 512), so,
as for TSHRNet, images are processed at a 256 x 256 working resolution (bilinear up, area down) and the predictions are
resized back to 128 x 128 before the metrics.

The objective is the native one of the repository with the options of its train-refined.sh and the defaults of the
SpecularityNet-PSD code that it extends: refined and coarse pixel loss 0.2 MSE + 0.4 gradient L1, VGG19 perceptual loss
(weight 0.1) on the refined and the coarse output, coarse weight 0.6, and a focal detection loss (weight 1.0) on the
highlight mask that data/spec.py derives from the input-target difference (0.3 R + 0.59 G + 0.11 B > 0.707 of its
maximum); the input noise of the original data loader (sigma 2/255) is kept.  The adversarial term (weight 0.01, enabled
only after epoch 20 of 100 in the original recipe) is not used.

Mixed precision: at 256 x 256 and batch size 8 the network needs about 24 GB and 7.5 s per step in float32 on an RTX 5090
(15 GB and 0.6 s with bf16 autocast and channels-last tensors), so training and evaluation run under bf16 autocast;
``--no-amp`` restores float32.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import types

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
HR_ROOT = ROOT / "baselines_ext" / "HRNet"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HR_ROOT))

for _name in ("tensorboardX", "yaml", "skimage"):  # imported by util/*.py of the repository, not needed for training
    try:
        __import__(_name)
    except ImportError:
        _module = types.ModuleType(_name)
        if _name == "tensorboardX":
            _module.SummaryWriter = object
        if _name == "skimage":
            _metrics = types.ModuleType("skimage.metrics")

            def _unavailable(*args, **kwargs):
                raise RuntimeError("scikit-image is not installed (its metrics are not used by this script)")

            _metrics.structural_similarity = _metrics.peak_signal_noise_ratio = _unavailable
            _module.metrics = _metrics
            sys.modules["skimage.metrics"] = _metrics
        sys.modules[_name] = _module

from models import arch  # noqa: E402  (HighlightRNet repository)
import models.losses as hr_losses  # noqa: E402
from mvbrdf_shr.metrics import psnr_torch, ssim_torch  # noqa: E402
from mvbrdf_shr.viz import save_image  # noqa: E402

BATCH = 8
WORK = 256


def rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class HRPairs(Dataset):
    """input / gt pairs of the flat benchmark; the highlight mask is defined as in data/spec.py of the repository."""

    def __init__(self, root: Path, split: str, flat: str = "flat", augment: bool = False, noise: bool = False):
        self.root = root / flat / split
        self.paths = sorted((self.root / "input").glob("*.png"))
        self.augment = augment
        self.noise = noise

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
        native = clean
        image = F.interpolate(image[None], size=(WORK, WORK), mode="bilinear", align_corners=False)[0]
        clean = F.interpolate(clean[None], size=(WORK, WORK), mode="bilinear", align_corners=False)[0]
        delta = image - clean
        luma = 0.3 * delta[0] + 0.59 * delta[1] + 0.11 * delta[2]
        mask = (luma > 0.707 * luma.max()).float()
        if self.noise:
            image = image + torch.randn_like(image) * (2 / 255.0)
        return image, clean, mask, native, path.name


class Vgg19Features(nn.Module):
    """models/vgg.py:Vgg19 of the repository (torchvision weights); the removed ``pretrained=`` argument is replaced."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        self.vgg_pretrained_features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, indices=None):
        indices = indices or [2, 7, 12, 21, 30]
        out = []
        for i in range(indices[-1]):
            x = self.vgg_pretrained_features[i](x)
            if (i + 1) in indices:
                out.append(x)
        return out


class NativeLoss:
    """backward_G of models/specularitynet_model.py for aligned data, without the adversarial term."""

    def __init__(self, device, lambda_vgg=0.1, lambda_coarse=0.6, lambda_detect=1.0):
        self.pixel = hr_losses.MultipleLoss([nn.MSELoss(), hr_losses.GradientLoss()], [0.2, 0.4]).to(device)
        self.vgg = hr_losses.VGGLoss(Vgg19Features().to(device).eval()).to(device)
        self.focal = hr_losses.BinaryFocalLoss()
        self.lambda_vgg, self.lambda_coarse, self.lambda_detect = lambda_vgg, lambda_coarse, lambda_detect

    def __call__(self, out, target, mask):
        refined = out["refined"]
        loss = self.pixel(refined, target)
        coarse_loss = 0.0
        for coarse in out["coarse"]:
            coarse_loss = coarse_loss + self.pixel(coarse, target) + self.vgg(coarse, target) * self.lambda_vgg
            coarse_loss = coarse_loss / len(out["coarse"])  # inside the loop, as in the repository
        loss = loss + coarse_loss * self.lambda_coarse
        loss = loss + self.vgg(refined, target) * self.lambda_vgg
        detect_loss = 0.0
        for detect in out["detect"]:
            detect_loss = detect_loss + self.focal(detect, mask)
        detect_loss = detect_loss / len(out["detect"])
        return loss + detect_loss * self.lambda_detect


def forward(net, image: torch.Tensor, amp: bool) -> dict:
    """The network in bf16 autocast (channels-last); outputs are returned in float32."""
    if not amp:
        return net(image)
    with torch.autocast(device_type=image.device.type, dtype=torch.bfloat16):
        out = net(image.contiguous(memory_format=torch.channels_last))
    return {k: ([o.float() for o in v] if isinstance(v, list) else v.float()) for k, v in out.items()}


@torch.no_grad()
def evaluate(net, loader, device, amp: bool, save_dir: Path | None = None) -> dict:
    net.eval()
    psnr, ssim = [], []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for image, _, _, clean, names in loader:
        image, clean = image.to(device), clean.to(device)
        prediction = forward(net, image, amp)["refined"].clamp(0, 1)
        prediction = F.interpolate(prediction, size=tuple(clean.shape[-2:]), mode="area")  # back to the benchmark size
        for i in range(len(prediction)):
            psnr.append(psnr_torch(prediction[i : i + 1], clean[i : i + 1]))
            ssim.append(ssim_torch(prediction[i : i + 1], clean[i : i + 1]))
            if save_dir is not None:
                save_image(prediction[i], save_dir / names[i])
    return {"n": len(psnr), "psnr": float(np.mean(psnr)), "ssim": float(np.mean(ssim))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="hrnet_world_v2_trained")
    parser.add_argument("--bench", default=str(ROOT / "data" / "world_benchmark_v2"))
    parser.add_argument("--out-root", default=str(ROOT / "outputs"))
    parser.add_argument("--no-amp", action="store_true", help="float32 instead of bf16 autocast (about 12x slower on RTX 5090)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    bench = Path(args.bench)
    train_loader = DataLoader(HRPairs(bench, "train", augment=True, noise=True), batch_size=BATCH, shuffle=True, drop_last=False)
    val_loader = DataLoader(HRPairs(bench, "val"), batch_size=BATCH)
    test_loader = DataLoader(HRPairs(bench, "test"), batch_size=BATCH)
    hard_loader = DataLoader(HRPairs(bench, "test", flat="flat_hard"), batch_size=BATCH)

    net = arch.refined(3, 3, enhance="de", ppm=True, bilinear=True).to(device)  # default initialization ('edsr' = PyTorch default)
    if amp:
        net = net.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[hrnet] RefinedNet parameters: {n_params / 1e6:.2f} M", flush=True)
    criterion = NativeLoss(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    out_dir = Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    best = {"psnr": -1.0}
    history = []
    iterator = iter(train_loader)
    for step in range(args.steps):
        try:
            image, clean, mask, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            image, clean, mask, _, _ = next(iterator)
        image, clean, mask = image.to(device), clean.to(device), mask.to(device)
        net.train()
        loss = criterion(forward(net, image, amp), clean, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            metrics = evaluate(net, val_loader, device, amp)
            history.append({"step": step, "loss": float(loss), **metrics})
            print(f"[hrnet {step}/{args.steps}] loss={float(loss):.4f} val={metrics['psnr']:.3f}/{metrics['ssim']:.4f}", flush=True)
            if metrics["psnr"] > best["psnr"]:
                best = {**metrics, "step": step}
                torch.save({"model": net.state_dict(), "metrics": best}, out_dir / "best.pt")

    state = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    net.load_state_dict(state["model"])
    pred_dir = bench / "predictions" / args.tag
    test_metrics = evaluate(net, test_loader, device, amp, pred_dir)
    hard_metrics = evaluate(net, hard_loader, device, amp)
    result = {
        "method": args.tag,
        "protocol": "released HighlightRNet architecture trained from its default initialization (no public weights); same splits, "
                    "batch size, optimizer, schedule, augmentation, gradient clipping and validation-based selection as "
                    "finetune_dhan_world_v2.py; native objective of the repository without the adversarial term; "
                    "256x256 working resolution; bf16 autocast",
        "steps": args.steps,
        "batch_size": BATCH,
        "parameters": n_params,
        "bf16_autocast": amp,
        "work_size": WORK,
        "validation_best": best,
        "test": test_metrics,
        "hard_test": hard_metrics,
        "history": history,
        "prediction_dir": str(pred_dir),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (pred_dir / "metrics.json").write_text(
        json.dumps({k: result[k] for k in ("method", "test", "hard_test", "prediction_dir")}, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("validation_best", "test", "hard_test")}, indent=2))


if __name__ == "__main__":
    main()
