"""Evaluation of the renderer-mismatch runs (scripts/make_mismatch_scenes.py, run_sensitivity_unfrozen.py --data-root ...).

For every trained field: held-out diffuse PSNR over the whole image (as everywhere in the paper) and over the object interior,
the pixels at least 3 px inside both the rendered silhouette and the silhouette of the target; the latter excludes the ring
where splatted discs and ray-traced triangles cover different pixels, so it measures the appearance and not the silhouette of
the given geometry.  The single-image branch is scored on the same pixels.  Training-view fit likewise.
Writes outputs/v2/mismatch_summary.json.

    python scripts/eval_mismatch.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import binary_erosion
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402

UNF = ROOT / "outputs" / "v2" / "sensitivity_unfrozen"
VARIANTS = {"ours": "none_none_seed0_mm_ours", "mitsuba": "none_none_seed0_mm_mitsuba", "mitsuba_light": "none_lighting_seed0_mm_mitsuba",
            "mitsuba_broad": "none_none_seed0_mm_broad", "smooth_ours": "none_none_seed0_mms_ours", "smooth_mitsuba": "none_none_seed0_mms_mitsuba",
            "smooth_mitsuba_masked": "none_none_seed0_mmsk_mitsuba"}


def masked_psnr(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor | None = None) -> float:
    d = (a.clamp(0, 1) - b.clamp(0, 1)).square().mean(0)
    mse = d.mean() if m is None else d[m].mean()
    return float(-10 * torch.log10(mse.clamp(min=1e-12)))


@torch.no_grad()
def main() -> None:
    os.chdir(ROOT)
    device = torch.device("cuda")
    pro = MVBRDFSHRPro(128, 48, 96).to(device)
    pro.load_state_dict(torch.load(ROOT / "outputs/world_benchmark_single/best.pt", map_location=device, weights_only=False)["model"])
    pro.eval()
    per = {}
    for n in range(70, 80):
        for var, stem in VARIANTS.items():
            name = (f"s00{n}_" if n != 70 else "") + stem
            p = UNF / f"{name}.pt"
            if not p.exists():
                continue
            st = torch.load(p, map_location="cpu", weights_only=False)
            cfg = dict(st["config"])
            cfg["raster_backend"] = "gsplat"
            dataset = load_dataset(cfg)
            pipe = build_pipeline(dataset, cfg, device, list(st["train_ids"]))
            pipe.load_state_dict(st["model"], strict=False)
            pipe.eval()
            rows = {"full": [], "interior": [], "single_full": [], "single_interior": [], "fit_full": [], "fit_interior": []}
            for fid in list(st["holdout_ids"]) + list(st["train_ids"]):
                frame = dataset[fid]
                image, gt = frame.image.to(device), frame.diffuse_gt.to(device)
                out = pipe(frame.camera.to(device), fid, image=image, refine=False)["render"]
                alpha = out.alpha.reshape(gt.shape[-2:]).cpu().numpy() > 0.5
                target = gt.amax(0).cpu().numpy() > 0.02
                inner = torch.from_numpy(binary_erosion(alpha & target, iterations=3)).to(device)
                if fid in st["holdout_ids"]:
                    rows["full"].append(masked_psnr(out.diffuse, gt))
                    rows["interior"].append(masked_psnr(out.diffuse, gt, inner))
                    single = pro(image.unsqueeze(0), use_refine=True, detach_physics=True)["pred"][0]
                    rows["single_full"].append(masked_psnr(single, gt))
                    rows["single_interior"].append(masked_psnr(single, gt, inner))
                else:
                    full = (out.diffuse + out.specular).clamp(0, 1)
                    rows["fit_full"].append(masked_psnr(full, image))
                    rows["fit_interior"].append(masked_psnr(full, image, inner))
            per.setdefault(f"scene_00{n}", {})[var] = {k: float(np.mean(v)) for k, v in rows.items()}
            r = per[f"scene_00{n}"][var]
            print(f"scene_00{n} {var:14s}: held-out diffuse {r['full']:.2f} (interior {r['interior']:.2f}); single {r['single_full']:.2f} "
                  f"({r['single_interior']:.2f}); training fit {r['fit_full']:.2f} ({r['fit_interior']:.2f})", flush=True)
            del pipe
            torch.cuda.empty_cache()
    summary = {}
    for var in VARIANTS:
        rows = [per[s][var] for s in per if var in per[s]]
        if rows:
            summary[var] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | {"n": len(rows),
                            "interior_min": float(min(r["interior"] for r in rows)), "interior_max": float(max(r["interior"] for r in rows))}
    (ROOT / "outputs/v2/mismatch_summary.json").write_text(json.dumps({"per_scene": per, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
