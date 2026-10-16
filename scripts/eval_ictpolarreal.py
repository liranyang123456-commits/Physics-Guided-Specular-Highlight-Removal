"""Zero-shot single-image SHR test on real light-stage captures with a polarization-measured diffuse reference.

Data: the public sample of ICTPolarReal (Yang et al., CVPR 2026): 8 objects, 8 views, cross- and parallel-polarized
one-light-at-a-time (OLAT) images and all-lights-on ("static") images.  Following the dataset paper, the diffuse
reflection is 2 x cross and the full reflection is 2 x parallel, so (parallel -> cross) is a (highlight image ->
diffuse reference) pair on the same scale.  Only the views and frames that were downloaded are used (the script
adapts to whatever is present); frames 0, 1, 347, 348, 349 (dark frames and polarization-free reference frames)
are never used.

Conditions
  olat       : for every view, 6 random groups of 8 OLAT frames; the sum of the parallel frames is the input and the
               sum of the cross frames the reference (light transport is linear).  A group is accepted only if the
               99th percentile of the input luminance inside the object mask is >= 0.05, so that the exposure
               normalisation below does not amplify sensor noise.
  olat_hard  : per view, the accepted group (out of 40) with the largest specular share of the input energy,
               which selects strong highlights using the reference only, not the methods.
  static     : all lights on (static_parallel -> static_cross), where available.

Pre-processing (identical for the input and the reference): linear radiance x s, clipped to [0, 1], gamma 1/2.2,
where s puts the 99th percentile of the input luminance inside the mask at 1; square crop around the mask, resized
to 256 x 256.  Metrics: PSNR/SSIM over object pixels (the background is nearly black in both images) and over the
whole crop (as elsewhere in the paper).  Confidence intervals are bootstrap intervals over views.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_baseline_paired import load_dhan, load_tshrnet, predict_dhan, predict_tshrnet  # noqa: E402
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro  # noqa: E402

DATA = ROOT / "data" / "raw" / "ICTPolarReal_sample"
EXCLUDED = {0, 1, 347, 348, 349}
SIZE = 256
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def read_exr(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32)


def read_mask(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED) > 127


def square_box(mask: np.ndarray, margin: float = 1.1):
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * margin) + 2
    return int(round(cy - side / 2)), int(round(cx - side / 2)), side


def crop_resize(x: np.ndarray, box, interp) -> np.ndarray:
    y0, x0, side = box
    padded = cv2.copyMakeBorder(x, side, side, side, side, cv2.BORDER_CONSTANT, value=0)
    crop = padded[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side]
    return cv2.resize(crop, (SIZE, SIZE), interpolation=interp)


def p99_luma(img: np.ndarray, mask: np.ndarray) -> float:
    return float(np.percentile((img @ LUMA)[mask], 99))


def prepare(inp_lin: np.ndarray, gt_lin: np.ndarray, mask: np.ndarray):
    inp_lin, gt_lin = np.clip(inp_lin, 0, None), np.clip(gt_lin, 0, None)
    scale = 1.0 / max(p99_luma(inp_lin, mask), 1e-6)
    inp = np.clip(inp_lin * scale, 0, 1) ** (1 / 2.2)
    gt = np.clip(gt_lin * scale, 0, 1) ** (1 / 2.2)
    box = square_box(mask)
    inp_t = torch.from_numpy(crop_resize(inp, box, cv2.INTER_AREA)).permute(2, 0, 1)
    gt_t = torch.from_numpy(crop_resize(gt, box, cv2.INTER_AREA)).permute(2, 0, 1)
    m = crop_resize(mask.astype(np.float32), box, cv2.INTER_AREA) > 0.5
    return inp_t, gt_t, torch.from_numpy(m)


def ssim_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, target = pred.clamp(0, 1), target.clamp(0, 1)
    pred = F.pad(pred, (5, 5, 5, 5), mode="reflect")
    target = F.pad(target, (5, 5, 5, 5), mode="reflect")
    mp, mt = F.avg_pool2d(pred, 11, 1), F.avg_pool2d(target, 11, 1)
    vp = F.avg_pool2d(pred.square(), 11, 1) - mp.square()
    vt = F.avg_pool2d(target.square(), 11, 1) - mt.square()
    cov = F.avg_pool2d(pred * target, 11, 1) - mp * mt
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mp * mt + c1) * (2 * cov + c2)) / ((mp.square() + mt.square() + c1) * (vp + vt + c2) + 1e-12)


def metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict:
    """pred, gt: [1,3,H,W]; mask: [1,H,W] bool."""
    pred, gt = pred.float().clamp(0, 1), gt.float().clamp(0, 1)
    se = (pred - gt).square()
    m = mask.float()
    msum = m.flatten(1).sum(1).clamp(min=1)
    smap = ssim_map(pred, gt).mean(1)
    return {
        "psnr": float(-10 * torch.log10(se.flatten(1).mean(1).clamp(min=1e-12))),
        "ssim": float(smap.flatten(1).mean(1)),
        "psnr_obj": float(-10 * torch.log10(((se.mean(1) * m).flatten(1).sum(1) / msum).clamp(min=1e-12))),
        "ssim_obj": float((smap * m).flatten(1).sum(1) / msum),
    }


def load_ours(device, checkpoint: str):
    state = torch.load(ROOT / checkpoint, map_location=device, weights_only=False)
    weights = state.get("model", state)
    for kind in ("unet", "dhan"):
        model = MVBRDFSHRPro(SIZE, 48, 96, restorer_kind=kind).to(device)
        result = model.load_state_dict(weights, strict=False)
        bad = [k for k in result.missing_keys + result.unexpected_keys if not k.endswith("num_batches_tracked")]
        if not bad:
            return model.eval()
    raise RuntimeError("checkpoint does not match a known restorer backbone")


@torch.inference_mode()
def predict_ours(model, image: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type="cuda", enabled=image.is_cuda):
        return model(image, use_refine=True, detach_physics=True)["pred"].float()


def bootstrap(per_view: dict[str, np.ndarray], rng: np.random.Generator, n: int = 10000):
    """per_view[name]: array [views]; returns mean and 95% interval over views, and of the difference to 'input'."""
    views = len(next(iter(per_view.values())))
    idx = rng.integers(0, views, size=(n, views))
    out = {}
    for name, v in per_view.items():
        means = v[idx].mean(1)
        out[name] = {"mean": float(v.mean()), "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}
        if name != "input":
            d = (v - per_view["input"])[idx].mean(1)
            out[name]["delta_vs_input"] = float((v - per_view["input"]).mean())
            out[name]["delta_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/shiq_mask/best.pt")
    parser.add_argument("--output", default="outputs/ictpolarreal_sample")
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--hard-pool", type=int, default=40)
    parser.add_argument("--min-p99", type=float, default=0.05)
    args = parser.parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ours, dhan = load_ours(device, args.checkpoint), load_dhan(device)
    # DHAN-SHR and TSHRNet both ship a top-level package called `models`; unload the first before importing the second.
    for name in [m for m in sys.modules if m == "models" or m.startswith("models.")]:
        del sys.modules[name]
    sys.path[:] = [q for q in sys.path if "DHAN-SHR" not in q]
    tshr = load_tshrnet(device)
    methods = {"input": lambda x: x, "ours": lambda x: predict_ours(ours, x), "dhan": lambda x: predict_dhan(dhan, x),
               "tshrnet": lambda x: predict_tshrnet(tshr, x)}

    objects = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    records, sheet = [], []

    def evaluate(inp_lin, gt_lin, mask, meta):
        inp, gt, m = prepare(inp_lin, gt_lin, mask)
        x = inp.unsqueeze(0).to(device)
        rec = dict(meta)
        preds = {}
        for name, fn in methods.items():
            preds[name] = fn(x)
            for k, v in metrics(preds[name].cpu(), gt.unsqueeze(0), m.unsqueeze(0)).items():
                rec[f"{name}_{k}"] = v
        records.append(rec)
        return inp, gt, preds

    for oi, obj in enumerate(objects):
        for ci, cam in enumerate(sorted((DATA / obj).glob("cam0?"))):
            if not (cam / "mask.png").exists():
                continue
            mask = read_mask(cam / "mask.png")
            usable = sorted(int(f.stem) for f in (cam / "cross").glob("*.exr")
                            if (cam / "parallel" / f.name).exists() and int(f.stem) not in EXCLUDED) \
                if (cam / "cross").exists() and (cam / "parallel").exists() else []
            if len(usable) >= max(2 * args.group_size, 24):
                cross = {k: read_exr(cam / "cross" / f"{k:06d}.exr") for k in usable}
                parallel = {k: read_exr(cam / "parallel" / f"{k:06d}.exr") for k in usable}
                rng = np.random.default_rng([2026, oi, ci])
                accepted = []
                attempts = 0
                while len(accepted) < max(args.groups, args.hard_pool) and attempts < 600:
                    attempts += 1
                    g = sorted(rng.choice(usable, size=args.group_size, replace=False).tolist())
                    par = sum(parallel[k] for k in g)
                    if p99_luma(np.clip(par, 0, None), mask) < args.min_p99:
                        continue
                    cro = sum(cross[k] for k in g)
                    share = float((np.clip(par - cro, 0, None) @ LUMA)[mask].sum() / max((np.clip(par, 0, None) @ LUMA)[mask].sum(), 1e-9))
                    accepted.append((g, par, cro, share))
                if len(accepted) >= args.groups:
                    for gi, (g, par, cro, share) in enumerate(accepted[:args.groups]):
                        evaluate(par, cro, mask, {"object": obj, "cam": cam.name, "condition": "olat", "frames": g, "spec_share": share})
                    pool = accepted[:args.hard_pool]
                    g, par, cro, share = max(pool, key=lambda t: t[3])
                    inp, gt, preds = evaluate(par, cro, mask, {"object": obj, "cam": cam.name, "condition": "olat_hard", "frames": g, "spec_share": share})
                    if cam.name == "cam03" or (obj, cam.name) in (("candle", "cam00"), ("coachbag", "cam01")):
                        sheet.append(torch.cat([inp, preds["ours"][0].cpu(), preds["dhan"][0].cpu(), preds["tshrnet"][0].cpu(), gt], dim=2))
                del cross, parallel, accepted
            if (cam / "static_parallel.exr").exists() and (cam / "static_cross.exr").exists():
                sp, sc = read_exr(cam / "static_parallel.exr"), read_exr(cam / "static_cross.exr")
                share = float((np.clip(sp - sc, 0, None) @ LUMA)[mask].sum() / max((np.clip(sp, 0, None) @ LUMA)[mask].sum(), 1e-9))
                evaluate(sp, sc, mask, {"object": obj, "cam": cam.name, "condition": "static", "frames": [], "spec_share": share})
        print(f"{obj}: {sum(r['object'] == obj for r in records)} images", flush=True)

    summary = {}
    rng = np.random.default_rng(7)
    for cond in ("olat", "olat_hard", "static"):
        sub = [r for r in records if r["condition"] == cond]
        summary[cond] = {"n": len(sub), "views": len({(r["object"], r["cam"]) for r in sub}),
                         "objects": len({r["object"] for r in sub})}
        if not sub:
            continue
        summary[cond]["mean_spec_share"] = float(np.mean([r["spec_share"] for r in sub]))
        for k in ("psnr", "ssim", "psnr_obj", "ssim_obj"):
            summary[cond][k] = {name: float(np.mean([r[f"{name}_{k}"] for r in sub])) for name in methods}
        view_keys = sorted({(r["object"], r["cam"]) for r in sub})
        for k in ("psnr_obj", "psnr", "ssim_obj"):
            per_view = {name: np.array([np.mean([r[f"{name}_{k}"] for r in sub if (r["object"], r["cam"]) == vk]) for vk in view_keys])
                        for name in methods}
            summary[cond][f"bootstrap_{k}"] = bootstrap(per_view, rng)
        per_view_obj = {name: np.array([np.mean([r[f"{name}_psnr_obj"] for r in sub if (r["object"], r["cam"]) == vk]) for vk in view_keys])
                        for name in methods}
        idx = rng.integers(0, len(view_keys), size=(10000, len(view_keys)))
        summary[cond]["paired_vs_ours_psnr_obj"] = {}
        for name in ("dhan", "tshrnet"):
            diff = per_view_obj[name] - per_view_obj["ours"]
            ci = np.percentile(diff[idx].mean(1), [2.5, 97.5])
            summary[cond]["paired_vs_ours_psnr_obj"][name] = {"mean": float(diff.mean()), "ci": [float(ci[0]), float(ci[1])]}
        if cond == "olat":
            order = np.argsort([r["spec_share"] for r in sub])
            thirds = np.array_split(order, 3)
            strata = []
            for t in thirds:
                rows_t = [sub[i] for i in t]
                strata.append({"n": len(rows_t),
                               "spec_share_range": [float(min(r["spec_share"] for r in rows_t)), float(max(r["spec_share"] for r in rows_t))],
                               "psnr_obj": {name: float(np.mean([r[f"{name}_psnr_obj"] for r in rows_t])) for name in methods}})
            summary[cond]["by_spec_share_tercile"] = strata
        summary[cond]["win_rate_vs_input_psnr_obj"] = {
            name: float(np.mean([r[f"{name}_psnr_obj"] > r["input_psnr_obj"] for r in sub])) for name in methods if name != "input"}
        summary[cond]["per_object_psnr_obj"] = {
            obj: {name: float(np.mean([r[f"{name}_psnr_obj"] for r in sub if r["object"] == obj])) for name in methods}
            for obj in objects if any(r["object"] == obj for r in sub)}
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"checkpoint": args.checkpoint, "settings": vars(args), "excluded_frames": sorted(EXCLUDED),
                                                  "summary": summary, "records": records}, indent=1), encoding="utf-8")
    if sheet:
        grid = torch.cat(sheet, dim=1).permute(1, 2, 0).numpy()
        cv2.imwrite(str(out / "examples.png"), (np.clip(grid, 0, 1)[..., ::-1] * 255).astype(np.uint8))
    for cond in ("olat", "olat_hard", "static"):
        s = summary[cond]
        if s["n"] == 0:
            continue
        print(f"\n[{cond}] n={s['n']} images, {s['views']} views, {s['objects']} objects; mean specular share of the input {100 * s['mean_spec_share']:.1f}%")
        for name, v in s["paired_vs_ours_psnr_obj"].items():
            print(f"  {name} minus ours: {v['mean']:+.2f} dB [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]")
        for t in s.get("by_spec_share_tercile", []):
            lo, hi = t["spec_share_range"]
            print(f"  specular share {100 * lo:4.1f}-{100 * hi:4.1f}% (n={t['n']}): " + "  ".join(f"{k} {v:.2f}" for k, v in t["psnr_obj"].items()))
        print("  method    object-pixel PSNR [95% CI over views]   delta vs input [95% CI]        SSIM(obj)   whole-crop PSNR   beats input")
        b = s["bootstrap_psnr_obj"]
        for name in methods:
            d = "" if name == "input" else f"{b[name]['delta_vs_input']:+6.2f} [{b[name]['delta_ci'][0]:+.2f}, {b[name]['delta_ci'][1]:+.2f}]"
            wr = "" if name == "input" else f"{100 * s['win_rate_vs_input_psnr_obj'][name]:.0f}%"
            print(f"  {name:<8s} {b[name]['mean']:6.2f} [{b[name]['ci'][0]:.2f}, {b[name]['ci'][1]:.2f}]     {d:<26s} {s['ssim_obj'][name]:.4f}      {s['psnr'][name]:6.2f}            {wr}")


if __name__ == "__main__":
    main()
