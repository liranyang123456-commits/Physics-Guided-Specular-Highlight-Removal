"""Information-matched multi-view baseline: the nine training views, the given geometry, and the given lighting, no optimization.

Every Gaussian of the given geometry is projected into the training views and kept in those where it faces the camera and is
not hidden by another surface.  Its observed colour is divided by the diffuse shading of a uniform material under the given
lighting, rendered from the given geometry with the renderer of the benchmark; specular reflection only adds energy, so the
observation with the smallest ratio estimates its diffuse albedo.  The held-out view is then rendered from the given geometry
and lighting with these albedos and no specular lobe.  This uses the information of the physical branch (training views,
geometry, lighting) and its renderer, but no BRDF field and no optimization.

Under the benchmark lighting many observations saturate at 1, which bounds the ratio only from below; per channel, the rule
takes the smallest unsaturated ratio, or the largest lower bound when every observation is saturated, and a Gaussian that no
training view sees takes the albedo of the nearest seen Gaussian.

Variants: smallest ratio (main), median ratio, smallest ratio including the held-out input itself, and, as the ceiling of the
rule, the smallest ratio taken from the diffuse ground-truth images of the training views instead of the inputs.
Writes outputs/v2/multiview_baseline.json.

    python scripts/eval_multiview_baseline.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_world_ablation import psnr, ssim  # noqa: E402
from mvbrdf_shr.world.scene import inverse_sigmoid  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402

GRAY = 0.1             # uniform albedo of the shading render (keeps the clamped diffuse output below 1)
OCCLUSION = 0.1        # a Gaussian is hidden if it lies this far behind the visible surface (scene units)
SCENES = list(range(70, 90))


def sample(img: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear sample of a (C, h, w) image at continuous pixel coordinates uv (N, 2), pixel centres at +0.5."""
    _, h, w = img.shape
    grid = torch.stack([uv[:, 0] / w * 2 - 1, uv[:, 1] / h * 2 - 1], dim=-1).view(1, 1, -1, 2)
    return F.grid_sample(img.unsqueeze(0), grid, mode="bilinear", align_corners=False)[0, :, 0, :]   # (C, N)


def set_material(sc, albedo: torch.Tensor) -> None:
    """Purely diffuse material with the given albedo per Gaussian (diffuse weight 1, specular and absorption 0)."""
    sc.base_color_logits.copy_(inverse_sigmoid(albedo.clamp(1e-4, 1 - 1e-4)))
    sc.material_budget_logits[:, 0] = 20.0
    sc.material_budget_logits[:, 1:] = -20.0


@torch.no_grad()
def run_scene(n: int, device) -> dict:
    reference = torch.load(ROOT / f"outputs/v2/world_scene_00{n}/last.pt", map_location="cpu", weights_only=False)
    cfg = dict(reference["config"])
    cfg["raster_backend"] = "gsplat"
    cfg["gaussian_init"] = "geometry_albedo"                          # the given geometry; the albedo is overwritten
    dataset = load_dataset(cfg)
    train_ids, holdout = list(reference["train_ids"]), list(reference["holdout_ids"])
    pipeline = build_pipeline(dataset, cfg, device, train_ids)
    pipeline.eval()
    sc = pipeline.scene
    render = lambda fid: pipeline(dataset[fid].camera.to(device), fid, image=dataset[fid].image.to(device), refine=False)["render"]

    set_material(sc, torch.full_like(sc.means, GRAY))
    shading = {t: render(t).diffuse / GRAY for t in train_ids + holdout}
    opacity = sc.opacity_logits.clone()
    sc.opacity_logits.fill_(12.0)                                   # opaque: depth of the visible surface
    depth = {t: render(t).depth.reshape(shading[t].shape[-2:]) for t in train_ids + holdout}
    sc.opacity_logits.copy_(opacity)

    means, normals = sc.means.detach(), F.normalize(sc.normal_raw.detach(), dim=-1)

    def ratios(ids, source):
        """Ratio of observation to shading per view, Gaussian, and channel; visibility; and saturation of the observation."""
        R, M, C = [], [], []
        for t in ids:
            cam = dataset[t].camera.to(device)
            uv, z, ok = cam.project(means)
            front = ((cam.center - means) * normals).sum(-1) / (cam.center - means).norm(dim=-1) > 0.05
            zt = sample(depth[t].unsqueeze(0), uv)[0]
            st = sample(shading[t], uv)
            img = dataset[t].image if source == "input" else dataset[t].diffuse_gt
            it = sample(img.to(device), uv)
            vis = ok & front & (z - zt < OCCLUSION) & (st.mean(0) > 0.02)
            R.append((it / st.clamp(min=1e-4)).T)
            M.append(vis)
            C.append((it > 0.995).T)                                 # saturated: the ratio is only a lower bound
        return torch.stack(R), torch.stack(M), torch.stack(C)       # (V, G, 3), (V, G), (V, G, 3)

    def fill_unseen(est, seen):
        """Gaussians that no view sees take the albedo of the nearest seen Gaussian."""
        if seen.all() or not seen.any():
            return est
        d = torch.cdist(means[~seen], means[seen])
        est = est.clone()
        est[~seen] = est[seen][d.argmin(1)]
        return est

    def smallest(R, M, C):
        """Per channel: smallest unsaturated ratio (specular light only adds); if every observation is saturated, the
        largest of the lower bounds."""
        vis = M.unsqueeze(-1)
        free = vis & ~C
        lo = torch.where(free, R, torch.full_like(R, float("inf"))).min(0).values
        bound = torch.where(vis & C, R, torch.full_like(R, -float("inf"))).max(0).values
        est = torch.where(free.any(0), lo, bound)
        return fill_unseen(torch.where(M.any(0).unsqueeze(-1), est, torch.zeros_like(est)), M.any(0))

    def median(R, M, C):
        vis = M.unsqueeze(-1)
        free = vis & ~C
        rr = torch.where(free, R, torch.full_like(R, float("nan")))
        med = torch.nanmedian(rr, dim=0).values
        bound = torch.where(vis & C, R, torch.full_like(R, -float("inf"))).max(0).values
        est = torch.where(free.any(0), med, bound)
        return fill_unseen(torch.where(M.any(0).unsqueeze(-1), est, torch.zeros_like(est)), M.any(0))

    Rin, Min, Cin = ratios(train_ids, "input")
    Rgt, Mgt, Cgt = ratios(train_ids, "diffuse")
    est = {"min_ratio": smallest(Rin, Min, Cin), "median": median(Rin, Min, Cin),
           "ceiling_from_diffuse_gt": smallest(Rgt, Mgt, Cgt)}
    out = {"share_gaussians_seen": float(Min.any(0).float().mean()), "mean_views_per_gaussian": float(Min.sum(0).float().mean()),
           "share_saturated_input_obs": float((Cin.any(-1) & Min).sum() / Min.sum().clamp(min=1)),
           "share_saturated_diffuse_obs": float((Cgt.any(-1) & Mgt).sum() / Mgt.sum().clamp(min=1))}
    scores = {k: [] for k in list(est) + ["min_ratio_with_heldout"]}
    for h in holdout:
        Rh, Mh, Ch = ratios([h], "input")
        est_h = smallest(torch.cat([Rin, Rh]), torch.cat([Min, Mh]), torch.cat([Cin, Ch]))
        gt = dataset[h].diffuse_gt.to(device)
        for name, alb in list(est.items()) + [("min_ratio_with_heldout", est_h)]:
            set_material(sc, alb)
            pred = render(h).diffuse
            scores[name].append((psnr(pred, gt), ssim(pred, gt)))
    for k, v in scores.items():
        out[k] = {"psnr": float(np.mean([a for a, _ in v])), "ssim": float(np.mean([b for _, b in v]))}
    return out


def main() -> None:
    os.chdir(ROOT)
    device = torch.device("cuda")
    results = {}
    for n in SCENES:
        r = run_scene(n, device)
        results[f"scene_00{n}"] = r
        print(f"scene_00{n}: min {r['min_ratio']['psnr']:.2f}  median {r['median']['psnr']:.2f}  +held-out {r['min_ratio_with_heldout']['psnr']:.2f}"
              f"  | ceiling {r['ceiling_from_diffuse_gt']['psnr']:.2f}  seen {r['share_gaussians_seen']:.3f}  views {r['mean_views_per_gaussian']:.1f}  saturated in/diffuse {r['share_saturated_input_obs']:.2f}/{r['share_saturated_diffuse_obs']:.2f}",
              flush=True)
        torch.cuda.empty_cache()
    summary = {}
    for group, ns in (("ten", range(70, 80)), ("unused", range(80, 90))):
        for k in ("min_ratio", "median", "min_ratio_with_heldout", "ceiling_from_diffuse_gt"):
            summary[f"{group}_{k}"] = float(np.mean([results[f"scene_00{n}"][k]["psnr"] for n in ns]))
            summary[f"{group}_{k}_ssim"] = float(np.mean([results[f"scene_00{n}"][k]["ssim"] for n in ns]))
            summary[f"{group}_{k}_min"] = float(min(results[f"scene_00{n}"][k]["psnr"] for n in ns))
            summary[f"{group}_{k}_max"] = float(max(results[f"scene_00{n}"][k]["psnr"] for n in ns))
    out = {"method": "per-Gaussian smallest ratio of observation to shading over the training views; given geometry and lighting",
           "per_scene": results, "summary": summary}
    (ROOT / "outputs/v2/multiview_baseline.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

