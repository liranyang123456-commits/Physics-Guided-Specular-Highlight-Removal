"""High-readability physical-rendering figure from scene_0070.

Renders the calibrated GT Gaussian BRDF at 4x resolution so panels stay sharp
in a journal figure* layout. Adds explicit arrows and per-panel captions.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.camera import PerspectiveCamera
from mvbrdf_shr.world.lighting import WorldLight
from mvbrdf_shr.world.pipeline import WorldGaussianBRDFPipeline
from mvbrdf_shr.world.renderer import GaussianBRDFRenderer
from mvbrdf_shr.world.scene import GaussianBRDFField
from mvbrdf_shr.world.train import load_dataset

SCENE = ROOT / "data" / "world_benchmark_v2" / "scenes" / "test" / "scene_0070"
CKPT = ROOT / "outputs" / "world_benchmark_v2_scene_0070_strict" / "last.pt"
GT_NPZ = SCENE / "gaussian_brdf_gt.npz"
OUT = Path(__file__).resolve().parent / "figures" / "fig_physical_render.png"

MV_IDS = [0, 1, 2]
FOCUS = 0
SCALE = 4  # 128 -> 512
PANEL = 480
BG = np.array([250, 251, 252], dtype=np.float32)
SPEC_BG = np.array([12, 16, 22], dtype=np.float32)


def _scale_camera(cam: PerspectiveCamera, scale: int) -> PerspectiveCamera:
    k = cam.K.clone()
    k[0, 0] *= scale
    k[1, 1] *= scale
    k[0, 2] *= scale
    k[1, 2] *= scale
    return PerspectiveCamera(
        K=k,
        c2w=cam.c2w.clone(),
        width=int(cam.width * scale),
        height=int(cam.height * scale),
        frame_id=cam.frame_id,
    )


def _light_from_frame(frame, device) -> WorldLight:
    env = frame.env_sh
    if env is None:
        env = torch.zeros(9, 3, device=device)
        env[0] = 1.0
    else:
        env = env.to(device)
    direction = frame.light_direction
    if direction is None:
        direction = torch.tensor([0.3, -0.4, 0.85], device=device)
    else:
        direction = F.normalize(direction.to(device), dim=-1)
    intensity = frame.light_intensity
    intensity = (
        torch.tensor([float(intensity)], device=device)
        if intensity is not None
        else torch.ones(1, device=device)
    )
    color = (
        frame.light_color.to(device)
        if frame.light_color is not None
        else torch.ones(3, device=device)
    )
    position = (
        frame.light_position.to(device)
        if frame.light_position is not None
        else torch.tensor([0.0, -2.0, 2.0], device=device)
    )
    pw = (
        torch.tensor([float(frame.point_light_weight)], device=device)
        if frame.point_light_weight is not None
        else torch.zeros(1, device=device)
    )
    return WorldLight(
        env_sh=env,
        direction=direction,
        intensity=intensity,
        color=color,
        position=position,
        point_weight=pw,
    )


def _oracle_scene(device) -> GaussianBRDFField:
    arrays = np.load(GT_NPZ)
    means = torch.from_numpy(arrays["position"]).to(device)
    albedo = torch.from_numpy(arrays["albedo"]).to(device).clamp(1e-4, 1 - 1e-4)
    scene = GaussianBRDFField.from_point_cloud(means, albedo)
    scene.initialize_from_npz(str(GT_NPZ), mode="oracle")
    return scene


def _tensor_to_panel(
    t: torch.Tensor,
    alpha: torch.Tensor | None = None,
    bg: np.ndarray = BG,
    gain: float = 1.0,
    contrast: float = 1.12,
) -> np.ndarray:
    x = t.detach().float().cpu().clamp(0, 1)
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0).numpy()
    else:
        x = x.numpy()
    if x.ndim == 2:
        x = np.stack([x, x, x], -1)
    x = np.clip(x * gain, 0, 1)
    if alpha is not None:
        a = alpha.detach().float().cpu()
        if a.ndim == 3:
            a = a[0] if a.shape[0] == 1 else a.mean(0)
        a = a.numpy()[..., None]
        rgb = x * a + (bg / 255.0) * (1.0 - a)
    else:
        a = None
        rgb = x
    im = Image.fromarray((rgb * 255).round().clip(0, 255).astype(np.uint8))
    arr = np.asarray(im).astype(np.float32)
    if a is not None:
        m = (a[..., 0] > 0.05).astype(np.uint8)
    else:
        diff = np.abs(arr - bg[None, None, :]).sum(-1)
        m = (diff > 12).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if ys.size > 0:
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        pad = 8
        y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
        y1 = min(arr.shape[0], y1 + pad)
        x1 = min(arr.shape[1], x1 + pad)
        side = max(y1 - y0, x1 - x0)
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        y0 = int(np.clip(cy - side / 2, 0, arr.shape[0] - side))
        x0 = int(np.clip(cx - side / 2, 0, arr.shape[1] - side))
        im = Image.fromarray(arr[y0 : y0 + side, x0 : x0 + side].astype(np.uint8))
    im = im.resize((PANEL, PANEL), Image.Resampling.LANCZOS)
    im = ImageEnhance.Contrast(im).enhance(contrast)
    im = ImageEnhance.Sharpness(im).enhance(1.25)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=2))
    return np.asarray(im)


def _spec_zoom(full: np.ndarray, spec_t: torch.Tensor, alpha: torch.Tensor):
    s = spec_t.detach().float().cpu()
    if s.ndim == 3 and s.shape[0] in (1, 3):
        s = s.mean(0)
    a = alpha.detach().float().cpu()
    if a.ndim == 3:
        a = a[0] if a.shape[0] == 1 else a.mean(0)
    score = s.numpy() * (a.numpy() > 0.05)
    cy, cx = np.unravel_index(int(np.argmax(score)), score.shape)
    ys, xs = np.where(a.numpy() > 0.05)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad = 8
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1 = min(a.shape[0], y1 + pad)
    x1 = min(a.shape[1], x1 + pad)
    side = max(y1 - y0, x1 - x0)
    cy0, cx0 = (y0 + y1) / 2, (x0 + x1) / 2
    y0 = int(np.clip(cy0 - side / 2, 0, a.shape[0] - side))
    x0 = int(np.clip(cx0 - side / 2, 0, a.shape[1] - side))
    py = int((cy - y0) / side * PANEL)
    px = int((cx - x0) / side * PANEL)
    half = PANEL // 5
    yy = int(np.clip(py - half, 0, PANEL - 2 * half))
    xx = int(np.clip(px - half, 0, PANEL - 2 * half))
    crop = np.array(full[yy : yy + 2 * half, xx : xx + 2 * half], copy=True)
    zoom = np.array(
        Image.fromarray(crop).resize((PANEL, PANEL), Image.Resampling.NEAREST),
        copy=True,
    )
    zoom[:8, :] = (245, 158, 11)
    zoom[-8:, :] = (245, 158, 11)
    zoom[:, :8] = (245, 158, 11)
    zoom[:, -8:] = (245, 158, 11)
    return zoom, (xx, yy, 2 * half, 2 * half)


@torch.inference_mode()
def _render_hires_bundle():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {"data_root": str(SCENE), "dataset": "blender", "image_size": [128, 128]}
    if CKPT.exists():
        state = torch.load(CKPT, map_location="cpu", weights_only=False)
        cfg = state["config"]

    dataset = load_dataset(cfg)
    oracle = _oracle_scene(device)
    renderer = GaussianBRDFRenderer(chunk_size=64, backend="auto").to(device)

    views = {}
    for fid in MV_IDS:
        frame = dataset[fid]
        cam = _scale_camera(frame.camera.to(device), SCALE)
        light = _light_from_frame(frame, device)
        views[fid] = renderer(oracle, cam, light=light)

    pred = None
    if CKPT.exists():
        state = torch.load(CKPT, map_location=device, weights_only=False)
        scene = GaussianBRDFField.from_point_cloud(
            state["points"].to(device),
            state["colors"].to(device) if state.get("colors") is not None else None,
        )
        pipeline = WorldGaussianBRDFPipeline(
            scene,
            len(dataset),
            lighting_mode=str(cfg.get("lighting_mode", "sh")),
            use_refiner=False,
            chunk_size=int(cfg.get("chunk_size", 32)),
            raster_backend=str(cfg.get("raster_backend", "auto")),
            shared_lighting=bool(cfg.get("shared_lighting", False)),
            background_color=cfg.get("background_color"),
        ).to(device)
        pipeline.load_state_dict(state["model"], strict=False)
        pipeline.eval()
        frame = dataset[FOCUS]
        cam = _scale_camera(frame.camera.to(device), SCALE)
        light = pipeline.frame_lighting(FOCUS)
        pred = pipeline.renderer(pipeline.scene, cam, light=light)
    return views, pred


def _panel(ax, img, title, edge="#334155", lw=1.8):
    ax.imshow(img, interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(edge)
        s.set_linewidth(lw)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=3, color="#0f172a")


def _caption(fig, cx, y, lines, color="#475569", fs=8.4):
    if isinstance(lines, str):
        lines = [lines]
    fig.text(
        cx, y, "\n".join(lines),
        ha="center", va="top", fontsize=fs, color=color, linespacing=1.30,
    )


def _harrow(fig, x0, x1, y, color="#64748b", lw=1.55):
    """Strictly horizontal arrow in a gutter."""
    fig.add_artist(
        FancyArrowPatch(
            (x0, y), (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=11,
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=20,
        )
    )


def _varrow(fig, x, y0, y1, color="#64748b", lw=1.55):
    """Strictly vertical arrow."""
    fig.add_artist(
        FancyArrowPatch(
            (x, y0), (x, y1),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=11,
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=20,
        )
    )


def _glabel(fig, x, y, text, color="#64748b", fs=7.8):
    """Small label sitting in a gutter (above an arrow)."""
    fig.text(x, y, text, ha="center", va="bottom", fontsize=fs, color=color, fontweight="bold")


def _row_xs(n: int, left: float, right: float, width: float) -> list[float]:
    """Evenly place n panels of given width between left and right margins."""
    if n == 1:
        return [(left + right - width) / 2]
    span = right - left - width
    step = span / (n - 1)
    return [left + i * step for i in range(n)]


def _packed_xs(left: float, right: float, axes_width: float, visible_widths: list[float], gap: float) -> list[float]:
    """Center a row with a fixed gap between the *visible* panel frames."""
    total = sum(visible_widths) + gap * (len(visible_widths) - 1)
    cursor = (left + right - total) / 2
    xs = []
    for visible_width in visible_widths:
        xs.append(cursor - 0.5 * (axes_width - visible_width))
        cursor += visible_width + gap
    return xs


def _chain_arrows(fig, xs, w, y, labels, colors=None, lw=1.55, visual_widths=None):
    """Join the visible right edge of each panel to the next visible left edge."""
    colors = colors or ["#64748b"] * (len(xs) - 1)
    visual_widths = visual_widths or [w] * len(xs)
    for i in range(len(xs) - 1):
        # imshow keeps images square inside their wider Axes boxes.  Therefore
        # we use their visible square borders rather than the Axes borders;
        # otherwise arrows appear detached despite matching the Axes geometry.
        wi, wj = visual_widths[i], visual_widths[i + 1]
        x0 = xs[i] + 0.5 * (w - wi) + wi - 0.002
        x1 = xs[i + 1] + 0.5 * (w - wj) + 0.002
        if x1 <= x0:
            continue
        _harrow(fig, x0, x1, y, color=colors[i], lw=lw)
        if labels and labels[i]:
            _glabel(fig, (x0 + x1) / 2, y + 0.006, labels[i], color=colors[i])


def build():
    views, pred = _render_hires_bundle()
    focus = views[FOCUS]

    mv_imgs = [_tensor_to_panel(views[i].full, views[i].alpha, contrast=1.15) for i in MV_IDS]
    obs = _tensor_to_panel(focus.full, focus.alpha, contrast=1.18)
    diff = _tensor_to_panel(focus.diffuse, focus.alpha, contrast=1.16)
    spec = _tensor_to_panel(focus.specular, focus.alpha, bg=SPEC_BG, gain=2.8, contrast=1.05)
    albedo = _tensor_to_panel(focus.albedo, focus.alpha, contrast=1.2)
    normal = _tensor_to_panel((focus.normal * 0.5 + 0.5).clamp(0, 1), focus.alpha, contrast=1.1)
    zoom, box = _spec_zoom(obs, focus.specular, focus.alpha)
    pred_full = _tensor_to_panel(pred.full, pred.alpha) if pred is not None else obs
    pred_diff = _tensor_to_panel(pred.diffuse, pred.alpha) if pred is not None else diff

    # Dense page grid: larger panels, short captions, minimal vertical gaps
    L, R = 0.02, 0.98
    fig = plt.figure(figsize=(14.2, 9.6), dpi=200, facecolor="white")

    # ===================== (a) 4 equal columns =====================
    fig.text(
        L, 0.975,
        r"(a)  Multi-view observations — one shared world-space BRDF field $\mathcal{G}$",
        fontsize=12.5, fontweight="bold", color="#0f172a",
    )
    a_w, a_h, a_y = 0.205, 0.195, 0.745
    square_a = a_h * 9.6 / 14.2
    a_xs = _packed_xs(L, R, a_w, [square_a, square_a, square_a, a_w], gap=0.045)
    a_items = [
        (mv_imgs[0], rf"View $I_{{{MV_IDS[0]}}}$", "#2563eb", 2.6, [
            r"$\pi_0$ (analyzed): RGB + glare",
            r"Input to (b)",
        ]),
        (mv_imgs[1], rf"View $I_{{{MV_IDS[1]}}}$", "#94a3b8", 1.4, [
            r"$\pi_1$: same object, new pose",
            r"Specular moves; albedo fixed",
        ]),
        (mv_imgs[2], rf"View $I_{{{MV_IDS[2]}}}$", "#94a3b8", 1.4, [
            r"$\pi_2$: third viewpoint",
            r"Still queries the same $\mathcal{G}$",
        ]),
        (None, "Shared 3D state", "#3b82f6", 1.6, [
            r"$\theta_i=\{\rho_d,\rho_s,r,m\}$ on $g_i$",
            r"One BRDF $\rightarrow$ all views",
        ]),
    ]
    for k, (img, title, edge, lw, caps) in enumerate(a_items):
        ax = fig.add_axes([a_xs[k], a_y, a_w, a_h])
        if img is None:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#eff6ff")
            for s in ax.spines.values():
                s.set_color(edge); s.set_linewidth(lw)
            ax.text(
                0.5, 0.52,
                "Shared 3D state\n"
                r"$\mathcal{G}=\{g_i\}$" + "\n"
                r"material $\theta_i$ in world space",
                ha="center", va="center", fontsize=9.5, color="#1e3a5f",
                linespacing=1.35, transform=ax.transAxes, fontweight="bold",
            )
            ax.set_title(title, fontsize=11, fontweight="bold", pad=2, color="#0f172a")
        else:
            _panel(ax, img, title, edge=edge, lw=lw)
        _caption(fig, a_xs[k] + a_w / 2, a_y - 0.006, caps, fs=8.0)

    _chain_arrows(
        fig, a_xs, a_w, a_y + a_h * 0.50,
        labels=[r"same $\mathcal{G}$", r"same $\mathcal{G}$", r"couple"],
        colors=["#64748b", "#64748b", "#3b82f6"],
        visual_widths=[square_a, square_a, square_a, a_w],
    )

    # Vertical link (a)→(b)
    link_x = a_xs[0] + a_w / 2
    _varrow(fig, link_x, a_y - 0.040, 0.690, color="#2563eb", lw=1.6)
    fig.text(
        link_x + 0.026, 0.704, r"factorize $\pi_0$",
        ha="left", va="center", fontsize=7.4, color="#2563eb",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"),
    )

    # ===================== (b) =====================
    fig.text(
        L, 0.675,
        r"(b)  Physical factorization:  $I = I_{\mathrm{d}} + I_{\mathrm{s}}$"
        r"  $\Rightarrow$  SHR keeps $I_{\mathrm{d}}$",
        fontsize=12.5, fontweight="bold", color="#0f172a",
    )
    b_w, b_h, b_y = 0.205, 0.195, 0.445
    square_b = b_h * 9.6 / 14.2
    b_xs = _packed_xs(L, R, b_w, [square_b] * 4, gap=0.045)
    b_items = [
        (obs, r"Observed $I$", "#334155", 1.6, [
            r"$L_o=L_{\mathrm{d}}+L_{\mathrm{s}}$",
            r"Orange box = specular hotspot",
        ]),
        (diff, r"Diffuse $I_{\mathrm{d}}$ (SHR)", "#059669", 2.5, [
            r"$I_{\mathrm{d}}=\mathcal{S}(L_{\mathrm{d}})$; keep $f_{\mathrm{d}}$",
            r"Highlight-free Track-A output",
        ]),
        (spec, r"Specular $I_{\mathrm{s}}$", "#db2777", 2.5, [
            r"$I_{\mathrm{s}}=\mathcal{S}(L_{\mathrm{s}})$; drop $f_{\mathrm{s}}$",
            r"Disabled for physical SHR",
        ]),
        (zoom, r"Highlight zoom", "#d97706", 2.0, [
            r"Close-up of specular peak",
            r"Geometry unchanged after SHR",
        ]),
    ]
    for k, (img, title, edge, lw, caps) in enumerate(b_items):
        ax = fig.add_axes([b_xs[k], b_y, b_w, b_h])
        _panel(ax, img, title, edge=edge, lw=lw)
        if k == 0:
            x, y, w, h = box
            ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec="#f59e0b", lw=2.0))
        _caption(fig, b_xs[k] + b_w / 2, b_y - 0.006, caps, fs=8.0)

    _chain_arrows(
        fig, b_xs, b_w, b_y + b_h * 0.50,
        labels=[r"keep $f_d$", r"also $I_s$", r"inspect"],
        colors=["#059669", "#db2777", "#d97706"],
        lw=1.7,
        visual_widths=[square_b] * 4,
    )
    fig.text(
        b_xs[1] + b_w / 2, b_y + b_h + 0.006,
        r"$\hat{I}_A := I_{\mathrm{d}}$",
        ha="center", fontsize=9.5, color="#059669", fontweight="bold",
    )

    axf = fig.add_axes([L, 0.365, R - L, 0.028])
    axf.axis("off")
    axf.add_patch(FancyBboxPatch(
        (0, 0.08), 1, 0.84,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor="#ecfdf5", edgecolor="#34d399", linewidth=1.1,
        transform=axf.transAxes,
    ))
    axf.text(
        0.5, 0.5,
        r"$I_{\mathrm{full}}=\mathcal{S}(L_{\mathrm{d}}+L_{\mathrm{s}})\ \ |\ \ $"
        r"$I_{\mathrm{d}}=\mathcal{S}(L_{\mathrm{d}})\ \ |\ \ $"
        r"$I_{\mathrm{s}}=\mathcal{S}(L_{\mathrm{s}})\ \ |\ \ $"
        r"physical SHR: $\hat{I}_A := I_{\mathrm{d}}$",
        ha="center", va="center", fontsize=10.5, color="#065f46", transform=axf.transAxes,
    )

    # ===================== (c) =====================
    fig.text(
        L, 0.340,
        r"(c)  Material buffers and learned renderer vs.\ oracle diffuse",
        fontsize=12.5, fontweight="bold", color="#0f172a",
    )
    c_w, c_h, c_y = 0.165, 0.175, 0.125
    square_c = c_h * 9.6 / 14.2
    c_xs = _packed_xs(L, R, c_w, [square_c] * 5, gap=0.040)
    c_items = [
        (albedo, r"Albedo $\rho_d$", "#64748b", 1.4, [
            r"View-independent color",
            r"Enters $f_{\mathrm{d}}=\rho_d/\pi$",
        ]),
        (normal, r"Normal $\mathbf{n}$", "#64748b", 1.4, [
            r"Shading geometry",
            r"$c_\ell=\mathbf{n}\!\cdot\!\omega_\ell$",
        ]),
        (pred_full, r"Pred.\ $I_{\mathrm{full}}$", "#64748b", 1.4, [
            r"Our full shaded render",
            r"Learned $\mathcal{G}$ + light",
        ]),
        (pred_diff, r"Pred.\ $I_{\mathrm{d}}$", "#059669", 2.3, [
            r"Physical SHR (lobe off)",
            r"Same $\alpha$ as full render",
        ]),
        (diff, r"GT $I_{\mathrm{d}}$", "#059669", 2.3, [
            r"Oracle diffuse reference",
            r"Track-A supervision target",
        ]),
    ]
    for k, (img, title, edge, lw, caps) in enumerate(c_items):
        ax = fig.add_axes([c_xs[k], c_y, c_w, c_h])
        _panel(ax, img, title, edge=edge, lw=lw)
        _caption(fig, c_xs[k] + c_w / 2, c_y - 0.006, caps, fs=7.8)

    _chain_arrows(
        fig, c_xs, c_w, c_y + c_h * 0.50,
        labels=[r"$\rho_d$", r"shade", r"disable $f_s$", "compare"],
        colors=["#64748b", "#64748b", "#059669", "#059669"],
        visual_widths=[square_c] * 5,
    )

    # ===================== (d) =====================
    fig.text(L, 0.072, r"(d)  End-to-end physical path", fontsize=12.5, fontweight="bold")
    d_y, d_h = 0.015, 0.040
    d_w = 0.140
    d_xs = _row_xs(6, L, R, d_w)
    d_stages = [
        ("#dbeafe", r"Light $L_i$"),
        ("#ede9fe", r"Material $\theta_i$"),
        ("#d1fae5", r"BRDF $f_d{+}f_s$"),
        ("#ffedd5", r"Splat $\mathcal{S}$"),
        ("#fce7f3", r"Separate $I_d,I_s$"),
        ("#bbf7d0", r"SHR $\hat{I}_A{=}I_d$"),
    ]
    for k, (color, text) in enumerate(d_stages):
        ax = fig.add_axes([d_xs[k], d_y, d_w, d_h])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(color)
        for s in ax.spines.values():
            s.set_color("#64748b"); s.set_linewidth(1.0)
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9.5,
                fontweight="bold", transform=ax.transAxes, color="#0f172a")
    _chain_arrows(fig, d_xs, d_w, d_y + d_h * 0.50, labels=[""] * 5, colors=["#475569"] * 5, lw=1.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=280, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build()
