"""Prior-sensitivity study with the geometry and/or the lighting *unfrozen* (scene_0070 strict holdout).

The sensitivity study of the manuscript (scripts/run_sensitivity_v2.py) perturbs one prior at initialization and then
re-optimizes only the materials, with the geometry and the per-frame lighting frozen, so the perturbed prior is never
corrected.  This script repeats the study with the corrected recipe (L1, no SSIM term, lr 0.03, 1500 steps) and lets the
optimizer also estimate the perturbed quantity:

  --unfreeze geometry : positions, scales, rotations, normals, and opacities of the Gaussians are optimized
  --unfreeze lighting : direction and intensity of the dominant light of every *training* frame are optimized; for
                        every held-out frame they are then estimated from the held-out *input* image alone (the field
                        and the materials stay fixed), because the lighting of a new view is not part of the field
  --unfreeze both     : both of the above

The perturbation, the perturbation noise (seed 1000 + seed), the exact-albedo initialization, the evaluation, and the gate
(trained on the clean field) are those of run_sensitivity_v2.py, so ``--unfreeze none`` reproduces its numbers.

Besides the held-out diffuse PSNR of the physical branch and of the fusion, every run records quantities that need no
diffuse ground truth: the full-radiance fit on the training views, the full-radiance fit on the held-out input images,
and robust residual statistics.  ``--val-views N`` trains on the training views minus N of them and reports the same fit
quantities on those N unseen training views (the diffuse ground truth of the benchmark is used only to report the
accuracy next to them).

    python scripts/run_sensitivity_unfrozen.py --axis geo_severe --unfreeze geometry --seed 0
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
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

from eval_world_ablation import psnr  # noqa: E402
from run_sensitivity_v2 import AXES, SPARSE_IDS, evaluate, perturb  # noqa: E402
from mvbrdf_shr.world.losses import WorldLossWeights, world_total_loss  # noqa: E402
from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402

EXTRA_AXES = ("geolight", "geo_pos", "geo_nrm", "arc6", "arc3")   # mild geometry noise + lighting; positions only; normals only; one-sided views
ARC_IDS = {"arc6": [1, 2, 3, 5, 6, 7], "arc3": [1, 2, 3]}          # training views that cover one side of the object only (held-out views {0, 4, 8})
UNFREEZE = ("none", "geometry", "lighting", "both")
GEOMETRY_LR = {"means": 2e-3, "log_scales": 5e-3, "quaternions": 1e-3, "normal_raw": 1e-2, "opacity_logits": 3e-2}
LIGHT_LR = 1e-2


def apply_perturbation(pipeline, axis: str, generator: torch.Generator) -> None:
    if axis == "geolight":
        perturb(pipeline, "geo_mild", generator)
        perturb(pipeline, "light", generator)
    elif axis in ("geo_pos", "geo_nrm"):
        scene = pipeline.scene
        target, sigma = (scene.means, 0.03) if axis == "geo_pos" else (scene.normal_raw, 0.25)
        with torch.no_grad():
            target.add_(torch.randn(target.shape, generator=generator, device="cpu").to(target.device) * sigma)
    elif axis not in ("none",) + tuple(ARC_IDS):
        perturb(pipeline, axis, generator)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (3, H, W) image; used to widen the basin of convergence of the geometry early on."""
    radius = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    y = x.unsqueeze(0)
    c = y.shape[1]
    y = torch.nn.functional.pad(y, (radius, radius, radius, radius), mode="replicate")
    y = torch.nn.functional.conv2d(y, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    y = torch.nn.functional.conv2d(y, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return y[0]


def object_mask(image: torch.Tensor, erode: int) -> torch.Tensor:
    """Object pixels of an input image (any channel above 0.01), shrunk by ``erode`` pixels; shape (1, H, W)."""
    m = (image.amax(0, keepdim=True) > 0.01).float()
    if erode > 0:
        m = 1 - torch.nn.functional.max_pool2d(1 - m.unsqueeze(0), 2 * erode + 1, stride=1, padding=erode)[0]
    return m


def surface_regularizer(scene, k: int, spacing: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Local-plane prior of a surface: each normal agrees with the plane fitted to the k nearest centres (sign-free), and each
    centre lies near that plane.  The plane is fitted without the Gaussian itself and is not differentiated."""
    mu = scene.means
    with torch.no_grad():
        idx = torch.cdist(mu, mu).topk(k + 1, largest=False).indices[:, 1:]
        nb = mu[idx]
        c = nb.mean(dim=1)
        centred = nb - c[:, None]
        plane_normal = torch.linalg.eigh(centred.transpose(1, 2) @ centred / k).eigenvectors[:, :, 0]
    l_normal = (1 - (scene.normals * plane_normal).sum(-1).abs()).mean()
    l_plane = ((mu - c) * plane_normal).sum(-1).square().mean() / spacing ** 2
    return l_normal, l_plane


def install_brdf_extension(pipeline, kind: str) -> list[torch.nn.Parameter]:
    """Extra reflectance terms, learned per Gaussian and initialized at the model of the method.

    ``diffuse``  wrapped lighting max(0, (n.l + w) / (1 + w)) and an Oren-Nayar roughness sigma, both starting at 0 (Lambert)
    ``specular`` an anisotropic base lobe (aspect exp(+-a) and tangent angle, starting isotropic; the tangent reference is the
                 projected z axis), a second GGX lobe (F0 0.04, own weight and roughness), and a Charlie sheen lobe, the last two
                 starting at weight 0
    ``both``     both of the above
    The diffuse render of the field is the generalized diffuse lobe."""
    import types

    import torch.nn.functional as Fn
    from mvbrdf_shr.models.envlight import sh_to_irradiance
    from mvbrdf_shr.world import renderer as wr

    scene = pipeline.scene
    n, dev = scene.n_gaussians, scene.means.device
    init = {}
    if kind in ("diffuse", "both"):
        init.update(ext_wrap=-6.0, ext_oren=-6.0)
    if kind in ("specular", "both"):
        init.update(ext_aniso=0.0, ext_tangent=0.0, ext_lobe2_weight=-6.0, ext_lobe2_rough=-1.4, ext_sheen=-6.0)
    for name, value in init.items():
        scene.register_parameter(name, torch.nn.Parameter(torch.full((n, 1), value, device=dev)))

    def shade(self, scene, materials, camera, light):
        nrm = scene.normals
        view = Fn.normalize(camera.center.view(1, 3) - scene.means, dim=-1)
        directional = Fn.normalize(light.direction, dim=-1).view(1, 3).expand_as(nrm)
        point_vector = light.position.view(1, 3) - scene.means
        point_distance = point_vector.norm(dim=-1, keepdim=True).clamp(min=0.1)
        pw = light.point_weight.view(1, 1)
        direction = Fn.normalize(directional * (1 - pw) + (point_vector / point_distance) * pw, dim=-1)
        cos_l = (nrm * direction).sum(-1, keepdim=True)
        n_dot_l = cos_l.clamp(min=0)
        n_dot_v = (nrm * view).sum(-1, keepdim=True).clamp(min=1e-4)
        irradiance = sh_to_irradiance(nrm.T.unsqueeze(0).unsqueeze(-1), light.env_sh.unsqueeze(0))[0, :, :, 0].T
        radiance = light.color.view(1, 3) * light.intensity.view(1, 1) * ((1 - pw) + pw / point_distance.square())
        direct = radiance * n_dot_l
        if hasattr(scene, "ext_wrap"):
            w = torch.sigmoid(scene.ext_wrap)
            s2 = torch.sigmoid(scene.ext_oren).square()
            a, b = 1 - 0.5 * s2 / (s2 + 0.33), 0.45 * s2 / (s2 + 0.09)
            theta_i, theta_o = torch.acos(n_dot_l.clamp(0, 1 - 1e-6)), torch.acos(n_dot_v.clamp(0, 1 - 1e-6))
            lt, vt = direction - cos_l * nrm, view - (nrm * view).sum(-1, keepdim=True) * nrm
            cos_phi = ((lt * vt).sum(-1, keepdim=True) / (lt.norm(dim=-1, keepdim=True) * vt.norm(dim=-1, keepdim=True) + 1e-6)).clamp(min=0)
            oren = a + b * cos_phi * torch.sin(torch.maximum(theta_i, theta_o)) * torch.tan(torch.minimum(theta_i, theta_o)).clamp(max=10.0)
            diffuse = materials.albedo / wr._PI * (irradiance + radiance * oren * ((cos_l + w) / (1 + w)).clamp(min=0))
        else:
            diffuse = materials.albedo / wr._PI * (irradiance + direct)
        half = Fn.normalize(direction + view, dim=-1)
        n_dot_h = (nrm * half).sum(-1, keepdim=True).clamp(min=0)
        h_dot_v = (half * view).sum(-1, keepdim=True).clamp(min=0)
        alpha = materials.roughness.square()
        g_term = wr._smith_g1(n_dot_l, alpha) * wr._smith_g1(n_dot_v, alpha)
        f0 = 0.04 * (1 - materials.metalness) + materials.albedo * materials.metalness
        if hasattr(scene, "ext_aniso"):
            ref = torch.zeros_like(nrm)
            ref[:, 2] = 1
            alt = torch.zeros_like(nrm)
            alt[:, 0] = 1
            ref = torch.where((nrm * ref).sum(-1, keepdim=True).abs() > 0.95, alt, ref)
            t0 = Fn.normalize(ref - (nrm * ref).sum(-1, keepdim=True) * nrm, dim=-1)
            b0 = torch.cross(nrm, t0, dim=-1)
            t = torch.cos(scene.ext_tangent) * t0 + torch.sin(scene.ext_tangent) * b0
            bt = torch.cross(nrm, t, dim=-1)
            ax, ay = alpha * torch.exp(scene.ext_aniso), alpha * torch.exp(-scene.ext_aniso)
            d_term = 1.0 / (wr._PI * ax * ay * (((half * t).sum(-1, keepdim=True) / ax).square()
                                                  + ((half * bt).sum(-1, keepdim=True) / ay).square() + n_dot_h.square()).square() + wr._EPS)
        else:
            d_term = wr._ggx_d(n_dot_h, alpha)
        specular = (materials.specular * d_term * g_term * wr._schlick_fresnel(h_dot_v, f0)
                    / (4 * n_dot_l * n_dot_v + wr._EPS) * direct)
        if hasattr(scene, "ext_lobe2_weight"):
            a2 = (0.04 + 0.96 * torch.sigmoid(scene.ext_lobe2_rough)).square()
            g2 = wr._smith_g1(n_dot_l, a2) * wr._smith_g1(n_dot_v, a2)
            f2 = wr._schlick_fresnel(h_dot_v, torch.full_like(a2, 0.04))
            specular = specular + torch.sigmoid(scene.ext_lobe2_weight) * wr._ggx_d(n_dot_h, a2) * g2 * f2 / (4 * n_dot_l * n_dot_v + wr._EPS) * direct
            sin2 = (1 - n_dot_h.square()).clamp(min=0)
            d_sheen = (2 + 1 / 0.5) * sin2.pow(1.0) / (2 * wr._PI)
            v_sheen = 1.0 / (4 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v) + wr._EPS)
            specular = specular + 0.2 * torch.sigmoid(scene.ext_sheen) * d_sheen * v_sheen * direct
        return diffuse, specular

    pipeline.renderer._shade_sh = types.MethodType(shade, pipeline.renderer)
    return [getattr(scene, name) for name in init]


def train_world(pipeline, dataset, cfg, train_ids, device, unfreeze: str, geo_scale: float = 1.0,
                light_lr: float = LIGHT_LR, geo_start: int = 0, blur0: float = 0.0,
                geo_params: tuple[str, ...] = tuple(GEOMETRY_LR), mask_erode: int = -1,
                surf: dict | None = None, geo_eps: float = 1e-8, extra_material: list | None = None,
                mat_start: int = 0) -> list[float]:
    steps = int(cfg["steps"])
    scene = pipeline.scene
    groups = [{"params": scene.material_parameters() + list(extra_material or []), "lr": float(cfg.get("scene_lr", 2e-3))}]
    material_group = 0                   # the materials stay frozen for ``mat_start`` initial steps
    delayed = [False]                    # whether the group is a geometry group that starts after ``geo_start`` steps
    if unfreeze in ("geometry", "both"):
        for name, lr in GEOMETRY_LR.items():
            if name in geo_params:
                groups.append({"params": [getattr(scene, name)], "lr": lr * geo_scale, "eps": geo_eps})
                delayed.append(True)
    if unfreeze in ("lighting", "both"):
        fl = pipeline.frame_lighting
        groups.append({"params": [fl.direction_raw, fl.log_intensity], "lr": light_lr})
        delayed.append(False)
    optimizer = torch.optim.Adam(groups)

    def cosine(step: int) -> float:      # the cosine schedule of the recipe (CosineAnnealingLR, eta_min = 0)
        return 0.5 * (1 + math.cos(math.pi * step / steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [(lambda step, d=d, i=i: cosine(step) * (0.0 if (d and step < geo_start) or (i == material_group and step < mat_start) else 1.0))
                    for i, d in enumerate(delayed)])
    weights = WorldLossWeights(**cfg.get("loss", {}))
    views_per_step = min(int(cfg.get("views_per_step", 1)), len(train_ids))
    pipeline.train()
    history = []
    for step in range(steps):
        frame_ids = train_ids if views_per_step == len(train_ids) else random.sample(train_ids, views_per_step)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        sigma = blur0 * max(0.0, 1.0 - step / (0.7 * steps)) if blur0 > 0 else 0.0   # blur annealed to zero at 70% of the steps
        for frame_id in frame_ids:
            frame = dataset[frame_id]
            image = frame.image.to(device)
            output = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)
            if mask_erode >= 0:
                m = object_mask(image, mask_erode)
                output = dict(output)
                output["render"] = dataclasses.replace(output["render"], full=output["render"].full * m)
                image = image * m
            if sigma > 0.3:
                output = dict(output)
                output["render"] = dataclasses.replace(output["render"], full=gaussian_blur(output["render"].full, sigma))
                loss, _ = world_total_loss(output, gaussian_blur(image, sigma), scene, weights=weights)
            else:
                loss, _ = world_total_loss(output, image, scene, weights=weights)
            (loss / views_per_step).backward()
            total += float(loss) / views_per_step
        decay = max(0.0, 1.0 - step / (surf["decay"] * steps)) if surf and surf.get("decay") else 1.0
        if surf and unfreeze in ("geometry", "both") and decay > 0:
            l_normal, l_plane = surface_regularizer(scene, surf["k"], surf["spacing"])
            (decay * (surf["normal"] * l_normal + surf["plane"] * l_plane)).backward()
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        history.append(total)
        if step % 300 == 0 or step == steps - 1:
            print(f"[step {step}/{steps}] loss={total:.4f}", flush=True)
    return history


def fit_lighting_of_views(pipeline, dataset, view_ids, device, steps: int = 150, lr: float = 2e-2) -> None:
    """Estimate direction and intensity of the dominant light of the given views from their input images only."""
    pipeline.eval()
    saved = {id(p): p.requires_grad for p in pipeline.parameters()}
    for p in pipeline.parameters():
        p.requires_grad_(False)
    fl = pipeline.frame_lighting
    for p in (fl.direction_raw, fl.log_intensity):
        p.requires_grad_(True)
    optimizer = torch.optim.Adam([fl.direction_raw, fl.log_intensity], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for frame_id in view_ids:
            frame = dataset[frame_id]
            image = frame.image.to(device)
            render = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)["render"]
            full = render.diffuse + render.specular
            ((full - image).abs().mean() / len(view_ids)).backward()
        optimizer.step()
    for p in pipeline.parameters():
        p.requires_grad_(saved[id(p)])


@torch.no_grad()
def fit_statistics(pipeline, dataset, view_ids, device) -> dict:
    """Full-radiance fit of the field on the given views, measured against the input images (no diffuse ground truth).

    ``full_psnr``  mean PSNR over the whole image
    ``med_err``    median absolute error over the pixels brighter than 0.02 in the input (object pixels)
    ``p90_err``    90th percentile of the same errors
    ``diffuse_psnr`` PSNR of the diffuse render against the diffuse ground truth (reported, never used by a check)
    """
    pipeline.eval()
    full_psnr, med, p90, diff = [], [], [], []
    for frame_id in view_ids:
        frame = dataset[frame_id]
        image = frame.image.to(device)
        render = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)["render"]
        full = (render.diffuse + render.specular).clamp(0, 1)
        full_psnr.append(psnr(full, image))
        err = (full - image).abs().mean(dim=0)
        sel = image.max(dim=0).values > 0.02
        e = err[sel] if bool(sel.any()) else err.flatten()
        med.append(float(e.median()))
        p90.append(float(torch.quantile(e.float(), 0.9)))
        diff.append(psnr(render.diffuse, frame.diffuse_gt.to(device)))
    return {"full_psnr": float(np.mean(full_psnr)), "med_err": float(np.mean(med)), "p90_err": float(np.mean(p90)),
            "diffuse_psnr": float(np.mean(diff)), "n_views": len(view_ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", choices=AXES + EXTRA_AXES + ("none",), required=True)
    parser.add_argument("--unfreeze", choices=UNFREEZE, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-views", type=int, default=0, help="train on the training views minus this many (unseen-view check)")
    parser.add_argument("--geo-scale", type=float, default=1.0, help="multiplier of the geometry learning rates")
    parser.add_argument("--light-lr", type=float, default=LIGHT_LR)
    parser.add_argument("--geo-start", type=int, default=0, help="geometry stays frozen for this many initial steps")
    parser.add_argument("--mat-start", type=int, default=0, help="materials stay frozen for this many initial steps (geometry first)")
    parser.add_argument("--geo-params", default=",".join(GEOMETRY_LR), help="comma list of the geometry parameters that are optimized")
    parser.add_argument("--blur", type=float, default=0.0, help="initial sigma (px) of a blur on render and target, annealed to 0 at 70%% of the steps")
    parser.add_argument("--init", choices=("geometry_albedo", "geometry", "point_cloud"), default="geometry_albedo",
                        help="geometry_albedo: exact geometry and albedo (sensitivity study); geometry: exact geometry, albedo 0.25 "
                             "(the albedo-free recipe of the main results); point_cloud: positions from the point cloud of the scene")
    parser.add_argument("--steps", type=int, default=0, help="override the number of steps (0 = recipe)")
    parser.add_argument("--scene", default="0070", help="benchmark scene (reference field, data, and exact lighting)")
    parser.add_argument("--no-specular", action="store_true",
                        help="Lambertian field: every primitive is purely diffuse (energy budget fixed to w_d = 1), all else unchanged")
    parser.add_argument("--data-root", default="", help="another rendering of the scene (same cameras and lights), e.g. the mismatch scenes")
    parser.add_argument("--mask-erode", type=int, default=-1,
                        help="photometric loss only on the object pixels of each input image (value above 0.01), shrunk by this many "
                             "pixels; -1 (default) uses the whole image")
    parser.add_argument("--surf-normal", type=float, default=0.0,
                        help="weight of the local-plane normal term (normals agree with the plane of the k nearest centres)")
    parser.add_argument("--surf-plane", type=float, default=0.0, help="weight of the local-plane position term (centres near that plane)")
    parser.add_argument("--surf-k", type=int, default=8, help="neighbours of the local-plane prior")
    parser.add_argument("--surf-decay", type=float, default=0.0,
                        help="if > 0, the weight of the local-plane prior falls linearly to 0 at this fraction of the steps")
    parser.add_argument("--views-per-step", type=int, default=0, help="training views per step (0 = recipe, 3)")
    parser.add_argument("--brdf-ext", choices=("none", "diffuse", "specular", "both"), default="none",
                        help="learned extra reflectance terms that start at the model of the method (see install_brdf_extension)")
    parser.add_argument("--geo-eps", type=float, default=1e-8,
                        help="Adam epsilon of the geometry parameters; a larger value makes the step proportional to small gradients")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", default="outputs/v2/sensitivity_unfrozen")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(ROOT)
    reference_path = f"outputs/v2/world_scene_{args.scene}/last.pt"
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    cfg = dict(reference["config"])
    cfg["raster_backend"] = "gsplat"
    cfg["gaussian_init"] = args.init   # geometry_albedo: exact albedo at initialization, as in the sensitivity study
    if args.init == "point_cloud":
        cfg.pop("gaussian_prior", None)
    if args.steps:
        cfg["steps"] = args.steps
    if args.views_per_step:
        cfg["views_per_step"] = args.views_per_step
    if args.data_root:
        cfg["data_root"] = args.data_root
        cfg["n_gaussians"] = int(len(np.load(ROOT / args.data_root / cfg.get("gaussian_prior", "gaussian_brdf_gt.npz"))["position"]))
    dataset = load_dataset(cfg)
    holdout = list(reference["holdout_ids"])
    all_train = ARC_IDS.get(args.axis) or SPARSE_IDS.get(args.axis, list(reference["train_ids"]))
    if args.val_views:
        val_ids = all_train[len(all_train) - args.val_views:]      # unseen training views, never optimized on
        train_ids = all_train[: len(all_train) - args.val_views]
    else:
        val_ids, train_ids = [], list(all_train)

    pipeline = build_pipeline(dataset, cfg, device, train_ids)
    extra_material = install_brdf_extension(pipeline, args.brdf_ext) if args.brdf_ext != "none" else []
    if args.no_specular:
        with torch.no_grad():
            pipeline.scene.material_budget_logits[:, 0] = 12.0
            pipeline.scene.material_budget_logits[:, 1:] = -30.0
        pipeline.scene.material_budget_logits.requires_grad_(False)
    generator = torch.Generator().manual_seed(1000 + args.seed)
    apply_perturbation(pipeline, args.axis, generator)
    perturbed_light = (pipeline.frame_lighting.direction_raw.detach().clone(), pipeline.frame_lighting.log_intensity.detach().clone())

    surf = None
    if args.surf_normal > 0 or args.surf_plane > 0:
        with torch.no_grad():
            mu = pipeline.scene.means
            spacing = float(torch.cdist(mu, mu).topk(2, largest=False).values[:, 1].median())
        surf = {"normal": args.surf_normal, "plane": args.surf_plane, "k": args.surf_k, "spacing": spacing, "decay": args.surf_decay}
    start = time.time()
    history = train_world(pipeline, dataset, cfg, train_ids, device, args.unfreeze, args.geo_scale, args.light_lr,
                          args.geo_start, args.blur, tuple(args.geo_params.split(",")), args.mask_erode, surf, args.geo_eps,
                          extra_material, args.mat_start)
    unseen = val_ids + holdout           # views whose lighting is not part of the optimization
    if args.unfreeze in ("lighting", "both"):
        fit_lighting_of_views(pipeline, dataset, unseen, device)
    summary = evaluate(pipeline, dataset, holdout, device)
    summary["fit_train"] = fit_statistics(pipeline, dataset, train_ids, device)
    summary["fit_holdout"] = fit_statistics(pipeline, dataset, holdout, device)
    if val_ids:
        summary["fit_val"] = fit_statistics(pipeline, dataset, val_ids, device)
    with torch.no_grad():
        fl = pipeline.frame_lighting
        d0, i0 = perturbed_light
        clean = torch.load(reference_path, map_location="cpu", weights_only=False)["model"]
        true_dir = torch.nn.functional.normalize(clean["frame_lighting.direction_raw"].to(device), dim=-1)
        est_dir = torch.nn.functional.normalize(fl.direction_raw, dim=-1)
        angle = torch.rad2deg(torch.acos((true_dir * est_dir).sum(-1).clamp(-1, 1)))
        summary["light_angle_error_deg"] = {"train": float(angle[train_ids].mean()), "unseen": float(angle[unseen].mean())}
        summary["light_log_intensity_error"] = {
            "train": float((fl.log_intensity[train_ids] - clean["frame_lighting.log_intensity"].to(device)[train_ids]).abs().mean()),
            "unseen": float((fl.log_intensity[unseen] - clean["frame_lighting.log_intensity"].to(device)[unseen]).abs().mean()),
        }
        ref = pipeline.scene
        clean_means = clean["scene.means"].to(device)
        summary["position_error_mean"] = (float((ref.means - clean_means).norm(dim=-1).mean())
                                          if clean_means.shape == ref.means.shape else None)
        clean_normals = torch.nn.functional.normalize(clean["scene.normal_raw"].to(device), dim=-1)
        summary["normal_error_deg"] = (float(torch.rad2deg(torch.acos((ref.normals * clean_normals).sum(-1).clamp(-1, 1))).mean())
                                       if clean_normals.shape == ref.normals.shape else None)
    summary.update({
        "axis": args.axis, "unfreeze": args.unfreeze, "seed": args.seed, "train_ids": train_ids, "val_ids": val_ids,
        "holdout_ids": holdout, "steps": int(cfg["steps"]), "geo_scale": args.geo_scale, "light_lr": args.light_lr,
        "geo_start": args.geo_start, "blur": args.blur, "geo_params": args.geo_params, "surf": surf,
        "views_per_step": int(cfg.get("views_per_step", 1)), "geo_eps": args.geo_eps, "brdf_ext": args.brdf_ext,
        "mat_start": args.mat_start,
        "final_loss": history[-1], "minutes": (time.time() - start) / 60,
    })
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    name = (f"s{args.scene}_" if args.scene != "0070" else "") + f"{args.axis}_{args.unfreeze}_seed{args.seed}" + (f"_val{args.val_views}" if args.val_views else "") + (f"_{args.tag}" if args.tag else "")
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save({"model": pipeline.state_dict(), "config": cfg, "train_ids": train_ids, "val_ids": val_ids, "holdout_ids": holdout,
                "axis": args.axis, "unfreeze": args.unfreeze, "seed": args.seed}, out_dir / f"{name}.pt")
    print(json.dumps({"run": name, "gate_mean": summary["gate_mean"], "minutes": round(summary["minutes"], 1)}), flush=True)
    for key in ("phys", "fuse", "single"):
        print(f"{key}: {summary[key]['psnr']:.2f} / {summary[key]['ssim']:.4f}", flush=True)
    print("fit_train", {k: round(v, 3) for k, v in summary["fit_train"].items()}, flush=True)
    print("fit_holdout", {k: round(v, 3) for k, v in summary["fit_holdout"].items()}, flush=True)
    print("light err", summary["light_angle_error_deg"], "pos err", summary["position_error_mean"] and round(summary["position_error_mean"], 4), flush=True)


if __name__ == "__main__":
    main()
