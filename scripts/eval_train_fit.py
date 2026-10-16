"""Training-view fit of every world field: full-radiance and diffuse PSNR on the views the field was optimized on.

The full-radiance fit on the training views needs no ground truth beyond the training images themselves, so it is available
at test time; a field that cannot reproduce its own training images (frozen but wrong geometry or lighting) fails this check
by a wide margin.  The output feeds the scene-level validity check of the fusion (scripts/eval_scene_check.py).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_world_ablation import psnr  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402
from mvbrdf_shr.world.train_fusion_robust import corrupted  # noqa: E402


def fit(path: Path, device, corrupt: str = "clean", views: str = "train_ids") -> dict:
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(st["config"])
    cfg["raster_backend"] = "gsplat"
    dataset = load_dataset(cfg)
    pipeline = build_pipeline(dataset, cfg, device, list(st["train_ids"]))
    pipeline.load_state_dict(st["model"] if "model" in st else st, strict=False)
    pipeline.eval().requires_grad_(False)
    ids = list(st[views])
    rows = []
    with torch.no_grad():
        for frame_id in ids:
            frame = dataset[frame_id]
            image = frame.image.to(device)
            with corrupted(pipeline, corrupt):
                render = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)["render"]
            full = (render.diffuse + render.specular).clamp(0, 1)
            rows.append((psnr(full, image), psnr(render.diffuse, frame.diffuse_gt.to(device))))
    return {"full_psnr": float(np.mean([r[0] for r in rows])), "diffuse_psnr": float(np.mean([r[1] for r in rows])), "n_views": len(rows)}


def main() -> None:
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {}
    # re-optimized sensitivity fields (training views of the field)
    for p in sorted((ROOT / "outputs/v2/sensitivity").glob("*_seed*.pt")):
        result[f"sens/{p.stem}"] = fit(p, device)
        print(p.stem, {k: round(v, 2) for k, v in result[f"sens/{p.stem}"].items()}, flush=True)
    # ten benchmark scenes and material-controlled scenes
    for n in range(70, 80):
        p = ROOT / f"outputs/v2/world_scene_00{n}/last.pt"
        result[f"ten/scene_00{n}"] = fit(p, device)
        print(f"scene_00{n}", {k: round(v, 2) for k, v in result[f"ten/scene_00{n}"].items()}, flush=True)
    for p in sorted((ROOT / "outputs/v2/material_study").glob("*/last.pt")):
        result[f"material/{p.parent.name}"] = fit(p, device)
        print(p.parent.name, {k: round(v, 2) for k, v in result[f"material/{p.parent.name}"].items()}, flush=True)
    # cross-fitted fold fields: fit on their own training views, clean and render-time corrupted
    for fold in range(3):
        for kind in ("clean", "geo", "light"):
            torch.manual_seed(fold)
            result[f"fold{fold}/{kind}"] = fit(ROOT / f"outputs/v2/crossfit/fold{fold}.pt", device, kind)
            print(f"fold{fold}/{kind}", {k: round(v, 2) for k, v in result[f"fold{fold}/{kind}"].items()}, flush=True)
    out = ROOT / "outputs/v2/train_fit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
