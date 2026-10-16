"""Generate all qualitative / comparison figures for the Neurocomputing manuscript."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
sys.path.insert(0, str(ROOT))
# Checkpoint configs store paths relative to the package root.
import os

os.chdir(ROOT)

from mvbrdf_shr.data.shiq_adapter import (
    NSHAdapter,
    PSDAdapter,
    SHIQAdapter,
    SSHRAdapter,
)
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro
from mvbrdf_shr.world.pipeline import WorldGaussianBRDFPipeline
from mvbrdf_shr.world.scene import GaussianBRDFField
from mvbrdf_shr.world.train import load_dataset

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

WB_PRED = ROOT / "data" / "world_benchmark_v2" / "predictions"
WB_FLAT = ROOT / "data" / "world_benchmark_v2" / "flat"


def _font(size: int = 16):
    for name in ("arial.ttf", "Arial.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().cpu()
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    else:
        x = x.clamp(0, 1).numpy()
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    return (x * 255).round().astype(np.uint8)


def _load_rgb_path(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize(size, Image.BILINEAR)
    return np.asarray(im)


PANEL = 168
GAP = 2
HEADER_H = 28
ROW_LABEL_W = 56
MARGIN = 16
BG = (255, 255, 255)
HEADER_BG = (248, 250, 252)
GRID = (226, 232, 240)
INK = (15, 23, 42)
MUTED = (71, 85, 105)


def _resize_square(img: np.ndarray, size: int = PANEL) -> Image.Image:
    im = Image.fromarray(img)
    return im.resize((size, size), Image.Resampling.BICUBIC)


def _error_heatmap(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-pixel L1 error mapped to a clean blue→yellow→red colormap."""
    err = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean(axis=-1)
    err = err / max(err.max(), 1e-6)
    # piecewise: blue (low) -> cyan -> yellow -> red (high)
    r = np.clip(1.5 * err - 0.2, 0, 1)
    g = np.clip(1.2 - np.abs(err - 0.55) * 2.2, 0, 1)
    b = np.clip(1.1 - err * 1.4, 0, 1)
    heat = np.stack([r, g, b], axis=-1)
    return (heat * 255).round().astype(np.uint8)


def _add_zoom(
    tile: Image.Image,
    src: np.ndarray,
    box: tuple[float, float, float, float] | None = None,
    zoom: int = 2,
) -> Image.Image:
    """Draw a red ROI and paste a magnified inset at bottom-right."""
    h, w = src.shape[:2]
    if box is None:
        # default: brightest 22% window (likely highlight)
        gray = src.mean(axis=-1) if src.ndim == 3 else src
        yy, xx = np.unravel_index(int(np.argmax(gray)), gray.shape)
        bw, bh = max(12, w // 5), max(12, h // 5)
        x0 = int(np.clip(xx - bw // 2, 0, w - bw))
        y0 = int(np.clip(yy - bh // 2, 0, h - bh))
        x1, y1 = x0 + bw, y0 + bh
    else:
        x0, y0, x1, y1 = [int(v) for v in box]
    crop = Image.fromarray(src[y0:y1, x0:x1]).resize(
        ((x1 - x0) * zoom, (y1 - y0) * zoom), Image.Resampling.NEAREST
    )
    out = tile.copy()
    draw = ImageDraw.Draw(out)
    # map ROI to panel coords
    sx, sy = out.width / w, out.height / h
    draw.rectangle(
        [x0 * sx, y0 * sy, x1 * sx - 1, y1 * sy - 1],
        outline=(220, 38, 38),
        width=2,
    )
    ix = out.width - crop.width - 4
    iy = out.height - crop.height - 4
    draw.rectangle(
        [ix - 1, iy - 1, ix + crop.width, iy + crop.height],
        outline=(220, 38, 38),
        width=2,
    )
    out.paste(crop, (ix, iy))
    return out


def _make_grid(
    rows: list[list[np.ndarray]],
    col_labels: list[str],
    row_labels: list[str] | None = None,
    panel: int = PANEL,
    zoom_cols: set[int] | None = None,
) -> Image.Image:
    """Publication-style comparison grid with shared column headers."""
    n_rows = len(rows)
    n_cols = len(col_labels)
    zoom_cols = zoom_cols or set()
    row_w = ROW_LABEL_W if row_labels else 0
    width = MARGIN * 2 + row_w + n_cols * panel + (n_cols - 1) * GAP
    height = MARGIN * 2 + HEADER_H + n_rows * panel + (n_rows - 1) * GAP
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, width, MARGIN + HEADER_H], fill=HEADER_BG)
    for j, lab in enumerate(col_labels):
        x = MARGIN + row_w + j * (panel + GAP)
        tw = draw.textlength(lab, font=_font(13))
        draw.text(
            (x + (panel - tw) / 2, MARGIN + 6),
            lab,
            fill=INK,
            font=_font(13),
        )

    for i, row in enumerate(rows):
        y = MARGIN + HEADER_H + i * (panel + GAP)
        if row_labels:
            lab = row_labels[i]
            tw = draw.textlength(lab, font=_font(12))
            draw.text(
                (MARGIN + (row_w - tw) / 2, y + panel / 2 - 8),
                lab,
                fill=MUTED,
                font=_font(12),
            )
        # share zoom ROI from first column (input)
        zoom_box = None
        if zoom_cols and row:
            gray = row[0].mean(axis=-1) if row[0].ndim == 3 else row[0]
            hh, ww = gray.shape
            yy, xx = np.unravel_index(int(np.argmax(gray)), gray.shape)
            bw, bh = max(12, ww // 5), max(12, hh // 5)
            x0 = int(np.clip(xx - bw // 2, 0, ww - bw))
            y0 = int(np.clip(yy - bh // 2, 0, hh - bh))
            zoom_box = (x0, y0, x0 + bw, y0 + bh)
        for j, arr in enumerate(row):
            x = MARGIN + row_w + j * (panel + GAP)
            tile = _resize_square(arr, panel)
            if j in zoom_cols:
                tile = _add_zoom(tile, arr, box=zoom_box, zoom=2)
            canvas.paste(tile, (x, y))
            draw.rectangle([x, y, x + panel - 1, y + panel - 1], outline=GRID, width=1)
    return canvas


def _label_panel(img: np.ndarray, text: str, bar_h: int = 26, width: int | None = None) -> Image.Image:
    """Legacy single-panel helper (kept for MVS / holdout composites)."""
    im = Image.fromarray(img)
    if width is not None and im.width != width:
        im = im.resize((width, int(im.height * width / im.width)), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (im.width, im.height + bar_h), BG)
    canvas.paste(im, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width, bar_h], fill=HEADER_BG)
    tw = draw.textlength(text, font=_font(13))
    draw.text(((canvas.width - tw) / 2, 5), text, fill=INK, font=_font(13))
    return canvas


def _hstack(panels: list[Image.Image], pad: int = GAP) -> Image.Image:
    h = max(p.height for p in panels)
    w = sum(p.width for p in panels) + pad * (len(panels) - 1)
    out = Image.new("RGB", (w, h), BG)
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + pad
    return out


def _vstack(panels: list[Image.Image], pad: int = 8) -> Image.Image:
    w = max(p.width for p in panels)
    h = sum(p.height for p in panels) + pad * (len(panels) - 1)
    out = Image.new("RGB", (w, h), BG)
    y = 0
    for p in panels:
        out.paste(p, (0, y))
        y += p.height + pad
    return out


def _framed_row(panels: list[Image.Image], title: str | None = None) -> Image.Image:
    row = _hstack(panels, pad=GAP)
    top = HEADER_H if title else 0
    canvas = Image.new("RGB", (row.width + 2 * MARGIN, row.height + top + 2 * MARGIN), BG)
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.rectangle([0, 0, canvas.width, MARGIN + HEADER_H], fill=HEADER_BG)
        draw.text((MARGIN, MARGIN + 5), title, fill=INK, font=_font(14))
    canvas.paste(row, (MARGIN, MARGIN + top))
    return canvas


def _save(fig: Image.Image, name: str):
    path = FIG_DIR / name
    fig.save(path, optimize=True)
    print(f"wrote {path}")


def make_bar_chart(
    title: str,
    labels: list[str],
    series: dict[str, list[float]],
    outfile: str,
    ymax: float | None = None,
    width: int = 980,
    height: int = 420,
):
    colors = {
        "Ours": (37, 99, 235),
        "Input": (148, 163, 184),
        "DHAN-SHR": (234, 88, 12),
        "TSHRNet": (22, 163, 74),
        "Physical": (124, 58, 237),
        "Fusion": (219, 39, 119),
        "Single": (8, 145, 178),
        "Zero-shot": (100, 116, 139),
        "Fine-tuned": (37, 99, 235),
        "Train": (59, 130, 246),
        "Holdout": (239, 68, 68),
        "PSNR": (37, 99, 235),
        "All": (37, 99, 235),
        "Hard": (239, 68, 68),
        "SOTA/Base": (148, 163, 184),
    }
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), title, fill=(15, 23, 42), font=_font(18))
    left, top, right, bottom = 160, 55, width - 30, height - 40
    draw.rectangle([left, top, right, bottom], outline=(100, 116, 139), width=1)
    keys = list(series.keys())
    n = len(labels)
    all_vals = [v for vals in series.values() for v in vals]
    ymax = ymax or max(all_vals) * 1.15
    group_w = (right - left) / n
    bar_w = group_w / (len(keys) + 1.2)
    for i, lab in enumerate(labels):
        gx = left + i * group_w
        for j, key in enumerate(keys):
            val = series[key][i]
            bh = (val / ymax) * (bottom - top - 8)
            x0 = gx + 12 + j * bar_w
            y0 = bottom - bh
            draw.rectangle(
                [x0, y0, x0 + bar_w * 0.85, bottom],
                fill=colors.get(key, (51, 65, 85)),
            )
            draw.text((x0, y0 - 16), f"{val:.2f}", fill=(30, 41, 59), font=_font(11))
        draw.text((gx + 8, bottom + 6), lab[:18], fill=(15, 23, 42), font=_font(12))
    lx = 180
    for key in keys:
        draw.rectangle([lx, 28, lx + 14, 42], fill=colors.get(key, (51, 65, 85)))
        draw.text((lx + 18, 26), key, fill=(15, 23, 42), font=_font(13))
        lx += 110
    _save(canvas, outfile)


@torch.inference_mode()
def make_shiq_figure(n_rows: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = ROOT / "outputs" / "shiq_mask" / "best.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = MVBRDFSHRPro(200, 48, 96).to(device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    dataset = SHIQAdapter(REPO / "data" / "raw" / "SHIQ_extracted", 200, "test")
    scores = []
    for i in range(min(150, len(dataset))):
        s = dataset[i]
        score = s.specular_gt.abs().mean().item() if s.specular_gt is not None else 0
        scores.append((score, i))
    scores.sort(reverse=True)
    grid_rows = []
    row_labels = []
    for k, idx in enumerate([i for _, i in scores[:n_rows]], start=1):
        s = dataset[idx]
        image = s.image.unsqueeze(0).to(device)
        out = model(image, use_refine=True, detach_physics=True)
        pred = _to_uint8(out["pred"][0])
        gt = _to_uint8(s.image_clean)
        grid_rows.append(
            [
                _to_uint8(image[0]),
                pred,
                gt,
                _error_heatmap(pred, gt),
            ]
        )
        row_labels.append(f"#{k}")
    _save(
        _make_grid(
            grid_rows,
            ["Input", "Ours", "Diffuse GT", "Error"],
            row_labels,
            panel=176,
            zoom_cols={0, 1, 2},
        ),
        "fig_shiq_qualitative.png",
    )


@torch.inference_mode()
def _paired_qualitative(
    name: str,
    adapter,
    ckpt: Path,
    outfile: str,
    n_rows: int = 3,
    size: int = 256,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = MVBRDFSHRPro(size, 48, 96).to(device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    if len(adapter) == 0:
        print(f"[warn] empty adapter for {name}")
        return
    scores = []
    for i in range(min(80, len(adapter))):
        s = adapter[i]
        score = (s.image - s.image_clean).abs().mean().item()
        scores.append((score, i))
    scores.sort(reverse=True)
    grid_rows = []
    row_labels = []
    for k, idx in enumerate([i for _, i in scores[:n_rows]], start=1):
        s = adapter[idx]
        image = s.image.unsqueeze(0).to(device)
        if image.shape[-1] != size:
            image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)
        out = model(image, use_refine=True, detach_physics=True)
        gt = s.image_clean
        if gt.shape[-1] != size:
            gt = F.interpolate(gt.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]
        pred = _to_uint8(out["pred"][0])
        gt_u = _to_uint8(gt)
        grid_rows.append([_to_uint8(image[0]), pred, gt_u, _error_heatmap(pred, gt_u)])
        row_labels.append(f"#{k}")
    _save(
        _make_grid(
            grid_rows,
            ["Input", "Ours", "Diffuse GT", "Error"],
            row_labels,
            panel=160,
            zoom_cols={0, 1, 2},
        ),
        outfile,
    )


def make_external_qualitative():
    shiq_ckpt = ROOT / "outputs" / "shiq_mask" / "best.pt"
    sshr_ckpt = ROOT / "outputs" / "sshr_finetuned_strict" / "best.pt"
    nsh_ckpt = ROOT / "outputs" / "nsh_finetuned_strict" / "best.pt"
    if not sshr_ckpt.exists():
        sshr_ckpt = shiq_ckpt
    if not nsh_ckpt.exists():
        nsh_ckpt = shiq_ckpt
    _paired_qualitative(
        "SSHR",
        SSHRAdapter(REPO / "data" / "raw" / "SSHR_extracted", 256, "test", tone_corrected=True),
        sshr_ckpt,
        "fig_sshr_qualitative.png",
    )
    _paired_qualitative(
        "NSH",
        NSHAdapter(REPO / "data" / "raw" / "NSH_extracted", 256, "test"),
        nsh_ckpt,
        "fig_nsh_qualitative.png",
    )
    _paired_qualitative(
        "PSD",
        PSDAdapter(REPO / "data" / "raw" / "PSD_extracted", 256, "test"),
        shiq_ckpt,
        "fig_psd_qualitative.png",
    )


def make_world_v2_comparison(n_rows: int = 3):
    methods = [
        ("Input", None),
        ("TSHRNet", WB_PRED / "tshrnet"),
        ("DHAN-SHR", WB_PRED / "dhan_shr"),
        ("Ours", WB_PRED / "mvbrdf_shr_world_v2_finetuned"),
        ("GT", None),
    ]
    ours_dir = WB_PRED / "mvbrdf_shr_world_v2_finetuned"
    if not ours_dir.exists():
        print("[warn] world v2 predictions missing")
        return
    names = sorted([p.name for p in ours_dir.glob("*.png")])[:80]
    hard = [n for n in names if "hard" in n.lower() or "_v0" in n]
    picks = (hard or names)[:n_rows]
    def _resolve_scene_pair(stem: str) -> tuple[Path | None, Path | None]:
        # stem like scene_0070_v000
        parts = stem.split("_")
        if len(parts) < 3:
            return None, None
        scene = "_".join(parts[:2])
        view = parts[2].lstrip("v")  # 000
        scene_dir = ROOT / "data" / "world_benchmark_v2" / "scenes" / "test" / scene
        if not scene_dir.is_dir():
            # try any split
            cands = list((ROOT / "data" / "world_benchmark_v2" / "scenes").rglob(scene))
            scene_dir = next((p for p in cands if p.is_dir()), None)
        if scene_dir is None:
            return None, None
        inp = scene_dir / "images" / f"{view}.png"
        gt = scene_dir / "diffuse" / f"{view}.png"
        if not inp.exists():
            inp = next(scene_dir.glob(f"images/*{view}*"), None)
        if not gt.exists():
            gt = next(scene_dir.glob(f"diffuse/*{view}*"), None)
        return inp if inp and Path(inp).exists() else None, gt if gt and Path(gt).exists() else None

    # Prefer diverse hard-ish views across scenes.
    picks = []
    for name in names:
        if name.endswith("_v000.png") or name.endswith("_v003.png"):
            picks.append(name)
        if len(picks) >= n_rows:
            break
    picks = picks or names[:n_rows]

    grid_rows = []
    row_labels = []
    for k, name in enumerate(picks, start=1):
        stem = Path(name).stem
        inp_path, gt_path = _resolve_scene_pair(stem)
        cells = []
        ok = True
        for label, folder in methods:
            if label == "Input":
                if inp_path is None:
                    ok = False
                    break
                img = _load_rgb_path(inp_path, (PANEL, PANEL))
            elif label == "GT":
                if gt_path is None:
                    ok = False
                    break
                img = _load_rgb_path(gt_path, (PANEL, PANEL))
            else:
                assert folder is not None
                f = folder / name
                if not f.exists():
                    ok = False
                    break
                img = _load_rgb_path(f, (PANEL, PANEL))
            cells.append(img)
        if ok and len(cells) == len(methods):
            grid_rows.append(cells)
            row_labels.append(f"#{k}")
    if grid_rows:
        _save(
            _make_grid(
                grid_rows,
                [m[0] for m in methods],
                row_labels,
                panel=148,
                zoom_cols={0, 1, 2, 3, 4},
            ),
            "fig_world_v2_compare.png",
        )
    else:
        print("[warn] could not assemble world v2 comparison rows")


@torch.inference_mode()
def make_world_holdout_figure():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = ROOT / "outputs" / "world_benchmark_v2_scene_0070_strict" / "last.pt"
    if not ckpt.exists():
        ckpt = ROOT / "outputs" / "world_benchmark_v2_scene_0070" / "last.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = state["config"]
    dataset = load_dataset(cfg)
    holdout = state.get("holdout_ids", [0, 4, 8])
    scene = GaussianBRDFField.from_point_cloud(
        state["points"].to(device),
        state["colors"].to(device) if state.get("colors") is not None else None,
    )
    pipeline = WorldGaussianBRDFPipeline(
        scene,
        len(dataset),
        lighting_mode=str(cfg.get("lighting_mode", "sh")),
        use_refiner=bool(cfg.get("use_refiner", True)),
        refiner_base=int(cfg.get("refiner_base", 96)),
        chunk_size=int(cfg.get("chunk_size", 32)),
        raster_backend=str(cfg.get("raster_backend", "auto")),
        shared_lighting=bool(cfg.get("shared_lighting", False)),
        background_color=cfg.get("background_color"),
    ).to(device)
    pipeline.load_state_dict(state["model"], strict=False)
    pipeline.eval()

    pro = None
    pro_ckpt = ROOT / "outputs" / "world_benchmark_single" / "best.pt"
    if pro_ckpt.exists():
        pstate = torch.load(pro_ckpt, map_location=device, weights_only=False)
        pro = MVBRDFSHRPro(128, 48, 96).to(device)
        pro.load_state_dict(pstate.get("model", pstate), strict=False)
        pro.eval()

    grid_rows = []
    row_labels = []
    col_labels = ["Input", "Full", "Physical", "Single", "Specular", "GT"]
    for k, fid in enumerate(holdout[:3], start=1):
        frame = dataset[fid]
        image = frame.image.to(device)
        out = pipeline(frame.camera.to(device), fid, image=image, refine=False)
        render = out["render"]
        if pro is not None:
            x = F.interpolate(image.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False)
            p = pro(x, use_refine=True, detach_physics=True)["pred"]
            p = F.interpolate(p, size=image.shape[-2:], mode="bilinear", align_corners=False)[0]
            single = _to_uint8(p)
        else:
            single = np.zeros_like(_to_uint8(image))
        gt = _to_uint8(frame.diffuse_gt) if frame.diffuse_gt is not None else np.zeros_like(_to_uint8(image))
        grid_rows.append(
            [
                _to_uint8(image),
                _to_uint8(render.full),
                _to_uint8(render.diffuse),
                single,
                _to_uint8(render.specular),
                gt,
            ]
        )
        row_labels.append(f"H{k}")
    _save(_make_grid(grid_rows, col_labels, row_labels, panel=140), "fig_world_holdout.png")


@torch.inference_mode()
def make_mvs_figure():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [
        (
            ROOT / "outputs" / "dtu_scan1" / "last.pt",
            "fig_dtu_nvs.png",
            "DTU scan1",
        ),
        (
            ROOT / "outputs" / "blendedmvs_57f8d9bb" / "last.pt",
            "fig_blendedmvs_nvs.png",
            "BlendedMVS",
        ),
    ]
    for ckpt, outfile, title in configs:
        if not ckpt.exists():
            continue
        state = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = state["config"]
        dataset = load_dataset(cfg)
        holdout = state.get("holdout_ids", [0])
        train_ids = state.get("train_ids", [1])
        scene = GaussianBRDFField.from_point_cloud(
            state["points"].to(device),
            state["colors"].to(device) if state.get("colors") is not None else None,
        )
        pipeline = WorldGaussianBRDFPipeline(
            scene,
            len(dataset),
            lighting_mode=str(cfg.get("lighting_mode", "sh")),
            use_refiner=False,
            chunk_size=64,
            # Table 11 was evaluated with the rasterizer the scenes were
            # optimized with (gsplat; scripts/check_mvs_backend.py).
            raster_backend=str(cfg.get("raster_backend", "auto")),
            shared_lighting=bool(cfg.get("shared_lighting", False)),
            background_color=cfg.get("background_color"),
        ).to(device)
        pipeline.load_state_dict(state["model"], strict=False)
        pipeline.eval()
        grid_rows = []
        row_labels = []
        for tag, ids in (("T", train_ids[:2]), ("H", holdout[:2])):
            for j, fid in enumerate(ids, start=1):
                frame = dataset[fid]
                image = frame.image.to(device)
                out = pipeline(frame.camera.to(device), fid, image=image, refine=False)
                render = out["render"]
                alpha = render.alpha
                if alpha.shape[0] == 1:
                    alpha = alpha.expand(3, -1, -1)
                grid_rows.append(
                    [
                        _to_uint8(image),
                        _to_uint8(render.full),
                        _to_uint8(render.diffuse),
                        _to_uint8(alpha),
                    ]
                )
                row_labels.append(f"{tag}{j}")
        if grid_rows:
            _save(
                _make_grid(
                    grid_rows,
                    ["Input", "Full NVS", "Diffuse", "Alpha"],
                    row_labels,
                    panel=220,
                ),
                outfile,
            )


def make_graphical_abstract():
    """Elsevier graphical abstract — left-right dual-branch merge schematic.

    Matches the redesigned Fig. 1 (pipeline): shared input fans out to a
    world-space physics branch (top) and a single-image branch (bottom), which
    merge at a confidence-gated fusion module. Bottom band shows the four
    headline numbers in an advantage-led framing.
    """
    w, h = 1400, 700
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Palette (matches LaTeX TikZ: chipA blue, chipB teal, chipC purple, goal green)
    BLUE = (37, 99, 235)
    BLUE_FILL = (239, 246, 255)
    TEAL = (13, 148, 136)
    TEAL_FILL = (240, 253, 250)
    PURPLE = (124, 58, 237)
    PURPLE_FILL = (245, 243, 255)
    GREEN = (22, 163, 74)
    GREEN_FILL = (240, 253, 244)
    GREY_FILL = (248, 250, 252)

    # Header
    draw.rectangle([0, 0, w, 64], fill=GREY_FILL)
    draw.text((28, 12), "Physics-Guided Specular Highlight Removal", fill=INK, font=_font(22))
    draw.text(
        (28, 40),
        "World-space 3D Gaussian BRDF field  ·  single-image restorer  ·  confidence-gated fusion",
        fill=MUTED,
        font=_font(13),
    )

    def box(xy, fill, outline, label, sub=None, lw=2):
        draw.rounded_rectangle(xy, radius=10, fill=fill, outline=outline, width=lw)
        x0, y0, x1, y1 = xy
        draw.text((x0 + 12, y0 + 10), label, fill=INK, font=_font(14))
        if sub:
            draw.multiline_text((x0 + 12, y0 + 32), sub, fill=MUTED, font=_font(11), spacing=2)

    def arrow(x0, y0, x1, y1, color, width=2):
        draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
        # arrowhead
        import math
        ang = math.atan2(y1 - y0, x1 - x0)
        L = 9
        for da in (0.45, -0.45):
            ax = x1 - L * math.cos(ang + da)
            ay = y1 - L * math.sin(ang + da)
            draw.polygon([(x1, y1), (ax, ay), (x1, y1)], fill=color)
            draw.line([(x1, y1), (ax, ay)], fill=color, width=width)

    # ===== Shared input (left) =====
    box((28, 300, 200, 410), (255, 255, 255), (100, 116, 139), "Input",
        "multi-view {I_v, π_v}\nor single image I")

    # ===== World branch (top row, blue) =====
    draw.text((250, 100), "World-space BRDF branch  (physics)", fill=BLUE, font=_font(13))
    box((250, 125, 430, 240), BLUE_FILL, BLUE, "3D Gaussian field",
        "Gaussians {g_i}\ngeometry + material\nw_d + ρ_s + a = 1")
    box((470, 125, 650, 240), BLUE_FILL, BLUE, "Cook–Torrance",
        "differentiable renderer\nSH / NeILF light")
    box((690, 125, 870, 240), BLUE_FILL, BLUE, "Disable specular",
        "lobe f_s → 0\n→ I_d (diffuse)")
    # forward arrows
    arrow(430, 182, 470, 182, BLUE)
    arrow(650, 182, 690, 182, BLUE)
    # input → world
    arrow(200, 340, 250, 200, BLUE)

    # ===== Image branch (bottom row, teal) =====
    draw.text((250, 470), "Single-image branch  (learning)", fill=TEAL, font=_font(13))
    box((250, 495, 430, 610), TEAL_FILL, TEAL, "Screen-space fields",
        "geometry + material\nphysical prior")
    box((470, 495, 650, 610), TEAL_FILL, TEAL, "Decompose",
        "(Ŝ, M̂)\nmask-aware")
    box((690, 495, 870, 610), TEAL_FILL, TEAL, "Restore",
        "mask-aware R\n→ I_pro")
    arrow(430, 552, 470, 552, TEAL)
    arrow(650, 552, 690, 552, TEAL)
    arrow(200, 375, 250, 552, TEAL)

    # ===== Fusion (right, purple) =====
    box((920, 320, 1150, 420), PURPLE_FILL, PURPLE, "Confidence gate",
        "G = σ(logit c + Δ)\nÎ = G·I_d + (1−G)·I_pro + δ", lw=3)
    # merge arrows
    arrow(870, 200, 920, 340, BLUE)
    arrow(870, 552, 920, 400, TEAL)

    # ===== Output (far right, green) =====
    box((1190, 330, 1372, 410), GREEN_FILL, GREEN, "Highlight-free",
        "diffuse output Î", lw=3)
    arrow(1150, 370, 1190, 370, PURPLE, width=3)

    # ===== Bottom headline band =====
    draw.rounded_rectangle([28, 640, 1372, 690], radius=8, fill=GREY_FILL, outline=(203, 213, 225), width=1)
    metrics = [
        ("Physical branch (oracle priors)", "36.8–39.4 dB (4 runs); fails if perturbed"),
        ("Fusion, 18 held-out scenes", "> single-image in 18/18; 38.27 dB (10 bench.)"),
        ("Same gate + DHAN-SHR (FT)", "37.45 → 38.59 dB"),
        ("Cross-dataset FT", "+1.26 / +0.49 / +3.33 dB"),
    ]
    for i, (k, v) in enumerate(metrics):
        x = 48 + i * 333
        draw.text((x, 648), k, fill=BLUE, font=_font(12))
        draw.text((x, 668), v, fill=INK, font=_font(11))

    _save(canvas, "graphical_abstract.png")


def make_all_charts():
    make_bar_chart(
        "SHIQ official test (PSNR)",
        ["TSHRNet", "HRNet", "Ours", "DHAN", "JSHDR", "NeuralDRM"],
        {"PSNR": [25.58, 30.23, 32.61, 33.81, 34.13, 34.50]},
        "fig_shiq_psnr_bars.png",
        ymax=36,
    )
    # rename key for legend
    # Fix colors by using known keys - redo with single series named consistently
    make_bar_chart(
        "world_benchmark_v2 test / hard subset (PSNR)",
        ["TSHRNet", "DHAN-SHR", "Ours v2 FT"],
        {
            "All": [19.01, 25.17, 36.41],
            "Hard": [18.38, 22.66, 30.46],
        },
        "fig_world_v2_psnr_bars.png",
        ymax=40,
    )
    make_bar_chart(
        "scene_0070 strict holdout (PSNR)",
        ["TSHRNet", "DHAN", "Input", "Single", "Fusion", "Physical"],
        {"PSNR": [20.08, 23.50, 30.98, 35.53, 36.28, 36.82]},
        "fig_holdout_psnr_bars.png",
        ymax=40,
    )
    make_bar_chart(
        "External datasets: Ours vs Input (PSNR)",
        ["SSHR-ZS", "SSHR-FT", "NSH-ZS", "NSH-FT", "PSD-ZS"],
        {
            "Ours": [26.37, 31.37, 31.09, 33.71, 26.18],
            "Input": [30.11, 30.11, 33.22, 33.22, 27.53],
        },
        "fig_external_psnr.png",
        ymax=36,
    )
    make_bar_chart(
        "Inverse rendering ablation on scene_0070 holdout (PSNR)",
        ["No albedo GT", "+Pro init", "Oracle physical"],
        {"PSNR": [12.89, 13.59, 36.82]},
        "fig_inverse_ablation.png",
        ymax=40,
    )
    make_bar_chart(
        "DTU / BlendedMVS diagnostic NVS (PSNR)",
        ["DTU train", "DTU holdout", "BMVS train", "BMVS holdout"],
        {"PSNR": [7.04, 7.43, 9.62, 9.48]},
        "fig_mvs_nvs_bars.png",
        ymax=12,
    )
    make_bar_chart(
        "Summary comparison across protocols (PSNR)",
        ["SHIQ", "WB-v2", "Holdout", "SSHR-FT", "NSH-FT"],
        {
            "Ours": [32.61, 36.41, 36.82, 31.37, 33.71],
            "SOTA/Base": [34.50, 25.17, 35.53, 30.11, 33.22],
        },
        "fig_summary_psnr.png",
        ymax=40,
    )


def main():
    if os.environ.get("GEN_MVS_ONLY"):
        try:
            make_mvs_figure()
            print("MVS figures OK")
        except Exception as exc:
            import traceback; traceback.print_exc()
        return
    make_graphical_abstract()
    make_all_charts()
    try:
        make_shiq_figure()
    except Exception as exc:
        print(f"[warn] SHIQ figure failed: {exc}")
    try:
        make_external_qualitative()
    except Exception as exc:
        print(f"[warn] external qualitative failed: {exc}")
    try:
        make_world_v2_comparison()
    except Exception as exc:
        print(f"[warn] world v2 compare failed: {exc}")
    try:
        make_mvs_figure()
    except Exception as exc:
        print(f"[warn] MVS figure failed: {exc}")


if __name__ == "__main__":
    main()
