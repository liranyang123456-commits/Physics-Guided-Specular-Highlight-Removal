"""K-fold cross-fitted world fields for calibrating the fusion gate on views the field has not seen.

The fusion gate of the original protocol is trained on renders of a field that was optimized on the very views it is
trained on; there the physical branch is almost exact, which says little about how it behaves on a held-out view.  Here the
nine training views of scene_0070 are split into three folds of three views; fold ``k`` trains a world field with the
corrected recipe (configs/world/v2) on the other six views and keeps its three left-out views for gate calibration.  The
held-out evaluation views {0, 4, 8} are used by none of the folds.

    python scripts/train_crossfit_fields.py --fold 0   # likewise 1 and 2
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_sensitivity_v2 import train_world  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402

FOLDS = [[1, 5, 9], [2, 6, 10], [3, 7, 11]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    random.seed(args.fold)
    np.random.seed(args.fold)
    torch.manual_seed(args.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = torch.load("outputs/v2/world_scene_0070/last.pt", map_location="cpu", weights_only=False)
    cfg = dict(reference["config"])
    cfg["raster_backend"] = "gsplat"
    dataset = load_dataset(cfg)
    left_out = FOLDS[args.fold]
    train_ids = [v for v in reference["train_ids"] if v not in left_out]
    pipeline = build_pipeline(dataset, cfg, device, train_ids)
    start = time.time()
    train_world(pipeline, dataset, cfg, train_ids, device)
    out = ROOT / "outputs" / "v2" / "crossfit"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": pipeline.state_dict(), "config": cfg, "train_ids": train_ids, "left_out_ids": left_out,
                "holdout_ids": list(reference["holdout_ids"]), "fold": args.fold}, out / f"fold{args.fold}.pt")
    print(f"fold {args.fold}: trained on {train_ids}, left out {left_out}, {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
