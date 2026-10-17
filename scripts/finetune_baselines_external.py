"""Matched domain fine-tuning of DHAN-SHR and TSHRNet on SSHR, NSH, and PSD.

Our single-image branch is adapted to SSHR/NSH/PSD by scripts/train_paired_pro.py: it starts from the SHIQ checkpoint, uses
the dataset adapters of mvbrdf_shr.data at 256x256 (the validation images are the last 5% of the SSHR training list, the
``eval`` folder of NSH, and the ``val`` groups of PSD), runs 1,000 steps with batch size 4 at a learning rate of 5e-5
(AdamW, weight decay 1e-4, cosine schedule, gradient clipping at 1.0, no augmentation), validates every 200 steps, and the
checkpoint with the best validation PSNR is evaluated once on the test split.

This script applies the same protocol to the baselines, starting from their official checkpoints (DHAN-SHR; the TSHRNet
checkpoint trained on the mix of SSHR, SHIQ, and PSD) and keeping each method's native objective:

  * DHAN-SHR : MSE + 0.4 * (1 - SSIM) on the output
  * TSHRNet  : the sum of four MSE terms, in [-1, 1], on the stage-1 diffuse, the stage-1 specular residue, the stage-2
               refined diffuse, and the stage-3 tone-corrected diffuse (train_4_networks_mix.py)

The zero-shot score of the starting checkpoint on the same test split is recorded for reference.  The metric functions are
those of scripts/train_paired_pro.py.

    python scripts/finetune_baselines_external.py --method dhan_shr --dataset sshr
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional.image import structural_similarity_index_measure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mvbrdf_shr.data import NSHAdapter, PSDAdapter, SSHRAdapter, collate_batch  # noqa: E402
from train_paired_pro import batch_metrics  # noqa: E402

RAW = ROOT.parent / "data" / "raw"
ROOTS = {"sshr": RAW / "SSHR_extracted", "nsh": RAW / "NSH_extracted", "psd": RAW / "PSD_extracted"}
SIZE = 256


def build_sets(dataset: str):
    """Train / validation / test sets exactly as in scripts/train_paired_pro.py (test: ft_test for PSD)."""
    root = ROOTS[dataset]
    if dataset == "sshr":
        train_full = SSHRAdapter(root, SIZE, "train", tone_corrected=True)
        n_val = min(512, max(1, len(train_full) // 20))
        val = Subset(train_full, range(len(train_full) - n_val, len(train_full)))
        train = Subset(train_full, range(len(train_full) - n_val))
        tests = {"test": SSHRAdapter(root, SIZE, "test", tone_corrected=True)}
        tests["test_first2000"] = Subset(tests["test"], range(2000))
    elif dataset == "psd":
        train, val = PSDAdapter(root, SIZE, "train"), PSDAdapter(root, SIZE, "val")
        tests = {"ft_test": PSDAdapter(root, SIZE, "ft_test")}
    else:
        train, val = NSHAdapter(root, SIZE, "train"), NSHAdapter(root, SIZE, "val")
        tests = {"test": NSHAdapter(root, SIZE, "test")}
    return train, val, tests


def load_dhan(device):
    dhan_root = ROOT / "baselines_ext" / "DHAN-SHR"
    sys.path.insert(0, str(dhan_root))
    from models import Model  # type: ignore

    model = Model()
    state = torch.load(dhan_root / "weights" / "model.pth", map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(OrderedDict((k.removeprefix("module."), v) for k, v in state.items()), strict=False)
    return model.to(device)


def load_tshrnet(device):
    tshr_root = ROOT / "baselines_ext" / "TSHRNet"
    sys.path.insert(0, str(tshr_root))
    from models.UNet import UNet  # type: ignore

    ckpt = tshr_root / "checkpoints_mix_SSHR_SHIQ_PSD"
    nets = []
    for index, channels in enumerate((3, 3, 6, 9), start=1):
        net = UNet(input_channels=channels, output_channels=3)
        state = torch.load(ckpt / f"UNet{index}_60.pth", map_location="cpu", weights_only=True)
        net.load_state_dict(OrderedDict((k.removeprefix("module."), v) for k, v in state.items()))
        nets.append(net.to(device))
    return nets


class Method:
    """Uniform wrapper: modules, forward to a [0, 1] prediction, native loss."""

    def __init__(self, name: str, device):
        self.name = name
        if name == "dhan_shr":
            self.modules_ = [load_dhan(device)]
        else:
            self.modules_ = load_tshrnet(device)

    def parameters(self):
        return [p for m in self.modules_ for p in m.parameters()]

    def train(self):
        for m in self.modules_:
            m.train()

    def eval(self):
        for m in self.modules_:
            m.eval()

    def state(self):
        return [m.state_dict() for m in self.modules_]

    def load_state(self, states):
        for m, s in zip(self.modules_, states):
            m.load_state_dict(s)

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        if self.name == "dhan_shr":
            return self.modules_[0](image).clamp(0, 1)
        x = image * 2 - 1
        d, s, r, c = self._tshr(x)
        return ((c + 1) / 2).clamp(0, 1)

    def _tshr(self, x):
        n1, n2, n3, n4 = self.modules_
        diffuse = n1(x)
        specular = n2(x)
        refined = n3(torch.cat([diffuse, x], dim=1))
        corrected = n4(torch.cat([refined, specular, x], dim=1))
        return diffuse, specular, refined, corrected

    def loss(self, image, clean, specular) -> torch.Tensor:
        if self.name == "dhan_shr":
            pred = self.modules_[0](image)
            return F.mse_loss(pred, clean) + 0.4 * (1 - structural_similarity_index_measure(pred, clean, data_range=1.0))
        x = image * 2 - 1
        d, s, r, c = self._tshr(x)
        t, ts = clean * 2 - 1, specular * 2 - 1
        return F.mse_loss(d, t) + F.mse_loss(s, ts) + F.mse_loss(r, t) + F.mse_loss(c, t)


@torch.inference_mode()
def evaluate(method: Method, loader, device) -> dict:
    method.eval()
    psnr, ssim = [], []
    for batch in loader:
        image = batch.image.to(device, non_blocking=True)
        target = batch.image_clean.to(device, non_blocking=True)
        p, s = batch_metrics(method.predict(image), target)
        psnr.extend(p.cpu().tolist())
        ssim.extend(s.cpu().tolist())
    return {"n": len(psnr), "psnr": float(np.mean(psnr)), "ssim": float(np.mean(ssim))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dhan_shr", "tshrnet"), required=True)
    parser.add_argument("--dataset", choices=("sshr", "nsh", "psd"), required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop", type=int, default=0,
                        help="train on random crops of this size (DHAN-SHR: one training step at 256x256 needs 28.6 GB); "
                             "validation and test stay at 256x256")
    parser.add_argument("--skip-full-sshr", action="store_true", help="evaluate SSHR on its first 2000 test images only")
    args = parser.parse_args()
    os.chdir(ROOT)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, val_set, test_sets = build_sets(args.dataset)
    if args.skip_full_sshr and args.dataset == "sshr":
        test_sets.pop("test")
    kwargs = dict(num_workers=4, pin_memory=True, collate_fn=collate_batch)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, **kwargs)
    test_loaders = {k: DataLoader(v, batch_size=8, shuffle=False, **kwargs) for k, v in test_sets.items()}
    print(f"{args.method}/{args.dataset}: train {len(train_set)} val {len(val_set)} " +
          " ".join(f"{k} {len(v)}" for k, v in test_sets.items()), flush=True)

    method = Method(args.method, device)
    zero_shot = {k: evaluate(method, v, device) for k, v in test_loaders.items()}
    zero_shot["val"] = evaluate(method, val_loader, device)
    print("zero-shot", json.dumps(zero_shot), flush=True)

    optimizer = torch.optim.AdamW(method.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    out_dir = ROOT / "outputs" / "baselines_external_ft" / f"{args.method}_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best, history = {"psnr": -1.0}, []
    iterator = iter(train_loader)
    start = time.time()
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        image = batch.image.to(device, non_blocking=True)
        clean = batch.image_clean.to(device, non_blocking=True)
        specular = batch.specular_gt.to(device, non_blocking=True)
        if args.crop:
            y0 = random.randint(0, image.shape[-2] - args.crop)
            x0 = random.randint(0, image.shape[-1] - args.crop)
            crop = lambda t: t[..., y0:y0 + args.crop, x0:x0 + args.crop]
            image, clean, specular = crop(image), crop(clean), crop(specular)
        method.train()
        loss = method.loss(image, clean, specular)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(method.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            metrics = evaluate(method, val_loader, device)
            history.append({"step": step, **metrics})
            print(f"[{args.method}-ft {args.dataset} {step}/{args.steps}] loss={float(loss):.4f} "
                  f"val={metrics['psnr']:.3f}/{metrics['ssim']:.4f}", flush=True)
            if metrics["psnr"] > best["psnr"]:
                best = {**metrics, "step": step}
                torch.save({"states": method.state(), "metrics": best}, out_dir / "best.pt")

    state = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    method.load_state(state["states"])
    tests = {k: evaluate(method, v, device) for k, v in test_loaders.items()}
    result = {
        "method": args.method, "dataset": args.dataset,
        "protocol": "scripts/train_paired_pro.py: SHIQ-trained official checkpoint start, same adapters/splits, 1000 steps, batch 4, "
                    "AdamW lr 5e-5 wd 1e-4, cosine, clip 1.0, no augmentation, best validation PSNR; native loss",
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr, "train_crop": args.crop or None,
        "n_train": len(train_set), "n_val": len(val_set),
        "zero_shot": zero_shot, "validation_best": best, "test": tests, "history": history,
        "minutes": (time.time() - start) / 60,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("zero_shot", "validation_best", "test")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
