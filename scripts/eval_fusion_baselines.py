"""Fusion baselines on the held-out views of the twenty test scenes (scene_0070 - scene_0089).

For each held-out view: the physical render W, the single-image prediction S, the per-pixel weight g of the gate of the
tables and of the seed-0 cross-fitted gate, and the diffuse ground truth G.
  * constant blends  w W + (1 - w) S  for w = 0, 0.05, ..., 1; reported for the best w over the ten first scenes (tuned on the
    test scenes themselves, an upper bound for any constant blend) and for the best w on scene_0070 applied to the others;
  * scene-level switch: the physical render when the field passes the training-view fit check (40 dB), else S;
  * per-pixel AUROC of each gate: does g rank object pixels by whether W is closer to G than S is?
  * per-pixel oracle selection between W and S (an upper bound for any gate that only selects).
Writes outputs/v2/fusion_baselines.json.

    python scripts/eval_fusion_baselines.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_world_ablation import psnr  # noqa: E402
from mvbrdf_shr.world.fusion import HybridMVBRDFSHR  # noqa: E402
from mvbrdf_shr.world.pipeline import WorldGaussianBRDFPipeline  # noqa: E402
from mvbrdf_shr.world.scene import GaussianBRDFField  # noqa: E402
from mvbrdf_shr.world.train import load_dataset  # noqa: E402

SINGLE = "outputs/world_benchmark_single/best.pt"
GATES = {"orig": "outputs/v2/fusion_scene_0070/fusion.pt", "xc0": "outputs/v2/fusion_crossfit_clean_scene_0070/fusion.pt"}
WS = [round(0.05 * k, 2) for k in range(21)]


def auroc(score: np.ndarray, label: np.ndarray) -> float | None:
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def hybrid_for(path: str, device) -> HybridMVBRDFSHR:
    h = HybridMVBRDFSHR(resolution=128).to(device)
    h.load_single_checkpoint(SINGLE)
    st = torch.load(path, map_location=device, weights_only=False)
    h.gate.load_state_dict(st["gate"])
    h.correction.load_state_dict(st["correction"])
    return h.eval()


@torch.no_grad()
def main() -> None:
    os.chdir(ROOT)
    device = torch.device("cuda")
    hybrids = {k: hybrid_for(p, device) for k, p in GATES.items()}
    fits = json.loads((ROOT / "outputs/v2/train_fit.json").read_text(encoding="utf-8"))
    unused_fit = {r["scene"]: r["train_fit"] for r in json.loads((ROOT / "outputs/v2/unused_scenes.json").read_text(encoding="utf-8"))["per_scene"]}
    per = {}
    for n in range(70, 90):
        name = f"scene_00{n}"
        state = torch.load(ROOT / f"outputs/v2/world_{name}/last.pt", map_location=device, weights_only=False)
        cfg = state["config"]
        dataset = load_dataset(cfg)
        scene = GaussianBRDFField.from_point_cloud(state["points"].to(device),
                                                   state["colors"].to(device) if state.get("colors") is not None else None)
        pipe = WorldGaussianBRDFPipeline(scene, len(dataset), lighting_mode=str(cfg.get("lighting_mode", "sh")), use_refiner=True,
                                         refiner_base=int(cfg.get("refiner_base", 96)), chunk_size=int(cfg.get("chunk_size", 32)),
                                         raster_backend="gsplat", shared_lighting=bool(cfg.get("shared_lighting", False)),
                                         background_color=cfg.get("background_color")).to(device)
        pipe.load_state_dict(state["model"], strict=False)
        pipe.eval()
        fit = fits.get(f"ten/{name}", {}).get("full_psnr", unused_fit.get(name))
        rows = {"blend": {w: [] for w in WS}, "oracle": [], "auroc_orig": [], "auroc_xc0": [], "phys": [], "single": []}
        for fid in state.get("holdout_ids", [0, 4, 8]):
            frame = dataset[fid]
            image = frame.image.to(device)
            gt = frame.diffuse_gt.to(device)
            out = pipe(frame.camera.to(device), fid, image=image, refine=True)
            W = out["render"].diffuse
            res = {k: h(image.unsqueeze(0), out, detach_backbones=True) for k, h in hybrids.items()}
            S = res["orig"]["single_pred"][0]
            rows["phys"].append(psnr(W, gt))
            rows["single"].append(psnr(S, gt))
            for w in WS:
                rows["blend"][w].append(psnr(w * W + (1 - w) * S, gt))
            ew, es = (W - gt).abs().mean(0), (S - gt).abs().mean(0)
            rows["oracle"].append(psnr(torch.where((ew <= es).unsqueeze(0), W, S), gt))
            obj = (out["render"].alpha.reshape(ew.shape) > 0.5).cpu().numpy()
            label = (ew < es).cpu().numpy()[obj]
            for k in GATES:
                g = res[k]["gate"][0, 0].cpu().numpy()[obj]
                a = auroc(g, label)
                if a is not None:
                    rows[f"auroc_{k}"].append(a)
        per[name] = {"fit": fit, "phys": float(np.mean(rows["phys"])), "single": float(np.mean(rows["single"])),
                     "blend": {str(w): float(np.mean(v)) for w, v in rows["blend"].items()}, "oracle_select": float(np.mean(rows["oracle"])),
                     "auroc_orig": float(np.mean(rows["auroc_orig"])) if rows["auroc_orig"] else None,
                     "auroc_xc0": float(np.mean(rows["auroc_xc0"])) if rows["auroc_xc0"] else None}
        r = per[name]
        best = max(r["blend"], key=lambda w: r["blend"][w])
        print(f"{name}: phys {r['phys']:.2f} single {r['single']:.2f} best blend w={best} {r['blend'][best]:.2f} oracle {r['oracle_select']:.2f} "
              f"AUROC orig {r['auroc_orig']:.3f} xc0 {r['auroc_xc0']:.3f} fit {r['fit']:.2f}", flush=True)
        del pipe, scene
        torch.cuda.empty_cache()
    ten, unused = [f"scene_00{n}" for n in range(70, 80)], [f"scene_00{n}" for n in range(80, 90)]
    mean = lambda names, f: float(np.mean([f(per[s]) for s in names]))
    w_ten = max(WS, key=lambda w: mean(ten, lambda r: r["blend"][str(w)]))
    w_70 = max(WS, key=lambda w: per["scene_0070"]["blend"][str(w)])
    others = [s for s in ten + unused if s != "scene_0070"]
    summary = {
        "w_tuned_on_ten": w_ten, "blend_tuned_ten_on_ten": mean(ten, lambda r: r["blend"][str(w_ten)]),
        "blend_tuned_ten_on_unused": mean(unused, lambda r: r["blend"][str(w_ten)]),
        "w_tuned_on_0070": w_70, "blend_0070_on_others": mean(others, lambda r: r["blend"][str(w_70)]),
        "phys_ten": mean(ten, lambda r: r["phys"]), "phys_unused": mean(unused, lambda r: r["phys"]),
        "switch_ten": mean(ten, lambda r: r["phys"] if r["fit"] >= 40 else r["single"]),
        "switch_unused": mean(unused, lambda r: r["phys"] if r["fit"] >= 40 else r["single"]),
        "oracle_ten": mean(ten, lambda r: r["oracle_select"]), "oracle_unused": mean(unused, lambda r: r["oracle_select"]),
        "auroc_orig_ten": mean(ten, lambda r: r["auroc_orig"]), "auroc_xc0_ten": mean(ten, lambda r: r["auroc_xc0"]),
        "auroc_orig_unused": mean(unused, lambda r: r["auroc_orig"]), "auroc_xc0_unused": mean(unused, lambda r: r["auroc_xc0"]),
        "blend_curve_ten": {str(w): mean(ten, lambda r, w=w: r["blend"][str(w)]) for w in WS},
    }
    (ROOT / "outputs/v2/fusion_baselines.json").write_text(json.dumps({"per_scene": per, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "blend_curve_ten"}, indent=2))


if __name__ == "__main__":
    main()
