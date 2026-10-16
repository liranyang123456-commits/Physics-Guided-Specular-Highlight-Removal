"""Build experiment figures/tables from verified local metrics and predictions only."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mvbrdf_shr.data.shiq_adapter import (  # noqa: E402
    NSHAdapter,
    PSDAdapter,
    SHIQAdapter,
    SSHRAdapter,
)
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro  # noqa: E402
from mvbrdf_shr.world.fusion import HybridMVBRDFSHR  # noqa: E402
from mvbrdf_shr.world.pipeline import WorldGaussianBRDFPipeline  # noqa: E402
from mvbrdf_shr.world.scene import GaussianBRDFField  # noqa: E402
from mvbrdf_shr.world.train import load_dataset  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
PRED = ROOT / "data" / "world_benchmark_v2" / "predictions"
OUT = ROOT / "outputs"

PANEL = 220
GAP = 3
HEADER_H = 34
ROW_W = 70
MARGIN = 14
BG = (255, 255, 255)
HDR = (248, 250, 252)
GRID = (226, 232, 240)
INK = (15, 23, 42)
MUTED = (71, 85, 105)
OURS = (91, 163, 168)  # soft teal, matches bar charts


def font(size=13):
    for n in ("C:/Windows/Fonts/arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


def to_u8(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().cpu().clamp(0, 1)
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0).numpy()
    else:
        x = x.numpy()
    if x.ndim == 2:
        x = np.stack([x, x, x], -1)
    return (x * 255).round().astype(np.uint8)


def load_rgb(path: Path, size: int | None = None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(im)


def load_rgb_tensor(t: torch.Tensor, size: int = PANEL) -> np.ndarray:
    return np.asarray(Image.fromarray(to_u8(t)).resize((size, size), Image.Resampling.BICUBIC))


def err_heat(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    e = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean(-1)
    e = e / max(float(e.max()), 1e-6)
    r = np.clip(1.5 * e - 0.2, 0, 1)
    g = np.clip(1.2 - np.abs(e - 0.55) * 2.2, 0, 1)
    b = np.clip(1.1 - e * 1.4, 0, 1)
    return (np.stack([r, g, b], -1) * 255).round().astype(np.uint8)


def make_grid(
    rows: list[list[np.ndarray]],
    cols: list[str],
    row_labels: list[str],
    panel: int = PANEL,
    ours_cols: set[int] | None = None,
) -> Image.Image:
    ours_cols = ours_cols or set()
    n_r, n_c = len(rows), len(cols)
    w = MARGIN * 2 + ROW_W + n_c * panel + (n_c - 1) * GAP
    h = MARGIN * 2 + HEADER_H + n_r * panel + (n_r - 1) * GAP
    canvas = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w, MARGIN + HEADER_H], fill=HDR)
    for j, lab in enumerate(cols):
        x = MARGIN + ROW_W + j * (panel + GAP)
        color = OURS if j in ours_cols else INK
        tw = draw.textlength(lab, font=font(12))
        draw.text((x + (panel - tw) / 2, MARGIN + 5), lab, fill=color, font=font(12))
        if j in ours_cols:
            draw.rectangle([x, MARGIN + HEADER_H - 3, x + panel, MARGIN + HEADER_H - 1], fill=OURS)
    for i, row in enumerate(rows):
        y = MARGIN + HEADER_H + i * (panel + GAP)
        tw = draw.textlength(row_labels[i], font=font(11))
        draw.text((MARGIN + (ROW_W - tw) / 2, y + panel / 2 - 7), row_labels[i], fill=MUTED, font=font(11))
        for j, arr in enumerate(row):
            x = MARGIN + ROW_W + j * (panel + GAP)
            tile = Image.fromarray(arr).resize((panel, panel), Image.Resampling.BICUBIC)
            canvas.paste(tile, (x, y))
            outline = OURS if j in ours_cols else GRID
            draw.rectangle([x, y, x + panel - 1, y + panel - 1], outline=outline, width=2 if j in ours_cols else 1)
    return canvas


def save(im: Image.Image, name: str):
    path = FIG / name
    im.save(path, optimize=True)
    print("wrote", path)


def collect_verified_metrics() -> dict:
    """Only numbers backed by local JSON / curated comparison report."""
    shiq = json.loads((OUT / "shiq_mse" / "shiq_results.json").read_text())
    wb = {}
    for name in [
        "tshrnet",
        "dhan_shr",
        "mvbrdf_shr_shiq_pretrained",
        "mvbrdf_shr_world_v2_finetuned",
    ]:
        wb[name] = json.loads((PRED / name / "metrics.json").read_text())
    hybrid = json.loads((OUT / "world_benchmark_v2_scene_0070_hybrid" / "metrics.json").read_text())
    strict = json.loads(
        (OUT / "world_benchmark_v2_scene_0070_strict" / "world_metrics_holdout.json").read_text()
    )
    strict_tr = json.loads(
        (OUT / "world_benchmark_v2_scene_0070_strict" / "world_metrics_train.json").read_text()
    )
    inv = json.loads(
        (OUT / "world_benchmark_v2_scene_0070_inverse" / "world_metrics_holdout.json").read_text()
    )
    invp = json.loads(
        (OUT / "world_benchmark_v2_scene_0070_inverse_pro" / "world_metrics_holdout.json").read_text()
    )
    dtu_h = json.loads((OUT / "dtu_scan1" / "world_metrics_holdout.json").read_text())
    dtu_t = json.loads((OUT / "dtu_scan1" / "world_metrics_train.json").read_text())
    bm_h = json.loads((OUT / "blendedmvs_57f8d9bb" / "world_metrics_holdout.json").read_text())
    bm_t = json.loads((OUT / "blendedmvs_57f8d9bb" / "world_metrics_train.json").read_text())

    def paired(path):
        d = json.loads(path.read_text())
        return {
            "ours_psnr": round(d["ours_psnr"], 2),
            "ours_ssim": round(d["ours_ssim"], 3),
            "input_psnr": round(d["input_psnr"], 2),
            "input_ssim": round(d["input_ssim"], 3),
            "n": d.get("n"),
        }

    def baseline(path):
        d = json.loads(path.read_text())
        return {
            "psnr": round(d["psnr"], 2),
            "ssim": round(d["ssim"], 3),
            "n": d["n"],
            "size": d.get("size"),
            "protocol": d.get("protocol", "local"),
        }

    abl = json.loads((OUT / "world_ablation_scene_0070" / "ablation_metrics.json").read_text())
    shiq_dhan = baseline(OUT / "baselines" / "dhan_shr_shiq" / "metrics.json")
    shiq_tshr = baseline(OUT / "baselines" / "tshrnet_shiq" / "metrics.json")

    metrics = {
        "source_note": "All values loaded from local JSON under mvbrdf_shr/outputs and predictions/*/metrics.json",
        "shiq": {
            "ours_psnr": round(shiq["mvbrdf_pro"]["psnr"], 2),
            "ours_ssim": round(shiq["mvbrdf_pro"]["ssim"], 3),
            "n": shiq["mvbrdf_pro"]["n"],
            "checkpoint": "outputs/shiq_mse (official test)",
            "literature": shiq["literature_sota"],
            "local_baselines": {
                "dhan_shr": shiq_dhan,
                "tshrnet": shiq_tshr,
            },
        },
        "world_v2": {
            "tshrnet": wb["tshrnet"],
            "dhan_shr": wb["dhan_shr"],
            "ours_zero_shot": wb["mvbrdf_shr_shiq_pretrained"],
            "ours_finetuned": wb["mvbrdf_shr_world_v2_finetuned"],
        },
        "holdout_0070": {
            "physical": {
                "psnr": round(strict["summary"]["diffuse_psnr"], 2),
                "ssim": round(strict["summary"]["diffuse_ssim"], 3),
            },
            "refined": {
                "psnr": round(strict["summary"]["refined_psnr"], 2),
                "ssim": round(strict["summary"]["refined_ssim"], 3),
            },
            "full": {
                "psnr": round(strict["summary"]["full_psnr"], 2),
                "ssim": round(strict["summary"]["full_ssim"], 3),
            },
            "hybrid": {
                "psnr": round(hybrid["holdout"]["hybrid_psnr"], 2),
                "ssim": round(hybrid["holdout"]["hybrid_ssim"], 4),
                "single_psnr": round(hybrid["holdout"]["single_psnr"], 2),
                "world_psnr": round(hybrid["holdout"]["world_psnr"], 2),
            },
            "inverse": {
                "psnr": round(inv["summary"]["diffuse_psnr"], 2),
                "ssim": round(inv["summary"]["diffuse_ssim"], 3),
            },
            "inverse_pro": {
                "psnr": round(invp["summary"]["diffuse_psnr"], 2),
                "ssim": round(invp["summary"]["diffuse_ssim"], 3),
            },
            "train_physical": {
                "psnr": round(strict_tr["summary"]["diffuse_psnr"], 2),
                "ssim": round(strict_tr["summary"]["diffuse_ssim"], 3),
            },
            "per_view_physical": [
                {
                    "frame": f["frame"],
                    "psnr": round(f["diffuse_psnr"], 2),
                    "ssim": round(f["diffuse_ssim"], 3),
                }
                for f in strict["frames"]
            ],
            # From curated report world_benchmark_v2_comparison.md (same 3 holdout views)
            "report_baselines": {
                "input": {"psnr": 30.98, "ssim": 0.974},
                "tshrnet": {"psnr": 20.08, "ssim": 0.832},
                "dhan_shr": {"psnr": 23.50, "ssim": 0.937},
                "single": {"psnr": 35.53, "ssim": 0.985},
            },
        },
        "external": {
            "sshr_zs": paired(OUT / "sshr_shiq_zero_shot" / "metrics.json"),
            "sshr_ft": paired(OUT / "sshr_finetuned_strict" / "test" / "metrics.json"),
            "sshr_head_ft": paired(OUT / "sshr_head_finetuned" / "test" / "metrics.json"),
            "nsh_zs": paired(OUT / "nsh_shiq_zero_shot" / "metrics.json"),
            "nsh_ft": paired(OUT / "nsh_finetuned_strict" / "test" / "metrics.json"),
            "nsh_head_ft": paired(OUT / "nsh_head_finetuned" / "test" / "metrics.json"),
            "psd_zs": paired(OUT / "psd_shiq_zero_shot" / "metrics.json"),
            "psd_ft": paired(OUT / "psd_finetuned_strict" / "ft_test" / "metrics.json"),
            "psd_zs_ft_test": paired(OUT / "psd_finetuned_strict" / "zs_ft_test" / "metrics.json"),
            "psd_ft_protocol": (
                "Public PSD archive has PSD_Test only; FT uses group-level "
                "70/10/20 split of groups; report FT on ft_test only."
            ),
            "local_baselines": {
                "sshr_n2000": {
                    "note": "First 2000 SSHR test images (same subset for all methods)",
                    "dhan_shr": baseline(OUT / "baselines" / "dhan_shr_sshr" / "metrics.json"),
                    "tshrnet": baseline(OUT / "baselines" / "tshrnet_sshr" / "metrics.json"),
                    "ours_ft": paired(OUT / "sshr_finetuned_strict" / "test_n2000" / "metrics.json"),
                },
                "psd_full_test": {
                    "note": "Official PSD_Test pairs (zero-shot protocol)",
                    "dhan_shr": baseline(OUT / "baselines" / "dhan_shr_psd" / "metrics.json"),
                    "tshrnet": baseline(OUT / "baselines" / "tshrnet_psd" / "metrics.json"),
                },
            },
        },
        "ablation_0070": {
            "protocol": abl["protocol"],
            "holdout_ids": abl["holdout_ids"],
            "variants": {
                k: {
                    "psnr": round(v["psnr"], 2),
                    "ssim": round(v["ssim"], 4 if v["psnr"] > 30 else 3),
                }
                for k, v in abl["variants"].items()
            },
        },
        "mvs": {
            "dtu_train": dtu_t["summary"],
            "dtu_holdout": dtu_h["summary"],
            "bmvs_train": bm_t["summary"],
            "bmvs_holdout": bm_h["summary"],
        },
    }
    out = Path(__file__).resolve().parent / "verified_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("wrote", out)
    return metrics


def scene_pair(stem: str):
    parts = stem.split("_")
    scene = "_".join(parts[:2])
    view = parts[2].lstrip("v")
    d = ROOT / "data" / "world_benchmark_v2" / "scenes" / "test" / scene
    return d / "images" / f"{view}.png", d / "diffuse" / f"{view}.png"


def fig_world_v2(n_rows: int = 6):
    methods = [
        ("Input", None, False),
        ("TSHRNet", PRED / "tshrnet", False),
        ("DHAN-SHR", PRED / "dhan_shr", False),
        ("DHAN-SHR-FT", PRED / "dhan_shr_world_v2_finetuned", False),
        ("Ours-ZS", PRED / "mvbrdf_shr_shiq_pretrained", True),
        ("Ours-FT", PRED / "mvbrdf_shr_world_v2_finetuned", True),
        ("GT", None, False),
        ("Err(Ours-FT)", None, True),
    ]
    names = sorted(p.name for p in (PRED / "mvbrdf_shr_world_v2_finetuned").glob("scene_*_v000.png"))
    # diversify scenes
    picks = names[:: max(1, len(names) // n_rows)][:n_rows]
    rows, labs = [], []
    ours_cols = {j for j, m in enumerate(methods) if m[2]}
    for k, name in enumerate(picks, 1):
        stem = Path(name).stem
        inp, gt = scene_pair(stem)
        if not inp.exists() or not gt.exists():
            continue
        inp_a, gt_a = load_rgb(inp, PANEL), load_rgb(gt, PANEL)
        cells = []
        ours_ft = None
        for lab, folder, _ in methods:
            if lab == "Input":
                cells.append(inp_a)
            elif lab == "GT":
                cells.append(gt_a)
            elif lab.startswith("Err"):
                cells.append(err_heat(ours_ft if ours_ft is not None else inp_a, gt_a))
            else:
                assert folder is not None
                arr = load_rgb(folder / name, PANEL)
                cells.append(arr)
                if lab == "Ours-FT":
                    ours_ft = arr
        rows.append(cells)
        labs.append(f"#{k}")
    save(make_grid(rows, [m[0] for m in methods], labs, ours_cols=ours_cols), "fig_world_v2_compare.png")


@torch.inference_mode()
def fig_holdout_compare():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = OUT / "v2" / "world_scene_0070" / "last.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = state["config"]
    dataset = load_dataset(cfg)
    holdout = state.get("holdout_ids", [0, 4, 8])
    scene = GaussianBRDFField.from_point_cloud(
        state["points"].to(device),
        state["colors"].to(device) if state.get("colors") is not None else None,
    )
    pipe = WorldGaussianBRDFPipeline(
        scene,
        len(dataset),
        lighting_mode=str(cfg.get("lighting_mode", "sh")),
        use_refiner=bool(cfg.get("use_refiner", True)),
        refiner_base=int(cfg.get("refiner_base", 96)),
        chunk_size=int(cfg.get("chunk_size", 32)),
        # Must be the rasterizer the scene was optimized with
        # (scripts/check_raster_backend.py); scene_0070 used gsplat.
        raster_backend=str(cfg.get("raster_backend", "auto")),
        shared_lighting=bool(cfg.get("shared_lighting", False)),
        background_color=cfg.get("background_color"),
    ).to(device)
    pipe.load_state_dict(state["model"], strict=False)
    pipe.eval()

    single_ckpt = OUT / "world_benchmark_single" / "best.pt"
    pro = MVBRDFSHRPro(128, 48, 96).to(device)
    pstate = torch.load(single_ckpt, map_location=device, weights_only=False)
    pro.load_state_dict(pstate.get("model", pstate), strict=False)
    pro.eval()

    # fusion.pt stores only the gate/correction modules; the single-image
    # backbone inside the hybrid must be loaded separately.
    fusion = HybridMVBRDFSHR(resolution=128).to(device)
    fusion.load_single_checkpoint(single_ckpt)
    fstate = torch.load(OUT / "v2" / "fusion_scene_0070" / "fusion.pt", map_location=device, weights_only=False)
    fusion.gate.load_state_dict(fstate["gate"])
    fusion.correction.load_state_dict(fstate["correction"])
    fusion.eval()

    def psnr_db(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(-10 * torch.log10((a.clamp(0, 1) - b.clamp(0, 1)).square().mean().clamp(min=1e-12)))

    cols = ["Input", "TSHRNet (ZS)", "DHAN-SHR (ZS)", "DHAN-SHR (FT)", "Ours (single)", "Ours-Phys",
            "Fixed blend", "Ours-Fuse", "GT"]
    ours_cols = {4, 5, 7}
    rows, labs = [], []
    for k, fid in enumerate(holdout[:3], 1):
        frame = dataset[fid]
        image = frame.image.to(device)
        out = pipe(frame.camera.to(device), fid, image=image, refine=False)
        phys_t = out["render"].diffuse
        x = F.interpolate(image.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False)
        p = pro(x, use_refine=True, detach_physics=True)["pred"]
        p = F.interpolate(p, size=image.shape[-2:], mode="bilinear", align_corners=False)[0]
        fuse_t = fusion(image.unsqueeze(0), out, detach_backbones=True)["pred"][0]
        fixed_t = (0.5 * phys_t + 0.5 * p).clamp(0, 1)
        gt_t = frame.diffuse_gt.to(device)
        print(
            f"holdout v{fid}: single {psnr_db(p, gt_t):.2f}  phys {psnr_db(phys_t, gt_t):.2f}  "
            f"fixed {psnr_db(fixed_t, gt_t):.2f}  fuse {psnr_db(fuse_t, gt_t):.2f}"
        )
        name = f"scene_0070_v{fid:03d}.png"
        tshr = load_rgb(PRED / "tshrnet" / name, PANEL)
        dhan = load_rgb(PRED / "dhan_shr" / name, PANEL)
        dhan_ft = load_rgb(PRED / "dhan_shr_world_v2_finetuned" / name, PANEL)
        rows.append(
            [load_rgb_tensor(t) for t in (image,)]
            + [tshr, dhan, dhan_ft]
            + [load_rgb_tensor(t) for t in (p, phys_t, fixed_t, fuse_t, gt_t)]
        )
        labs.append(f"H{fid}")
    save(make_grid(rows, cols, labs, panel=PANEL, ours_cols=ours_cols), "fig_holdout_compare.png")


def fig_ours_advantage(metrics: dict):
    """Cross-scene synthetic comparison; holdout comparison is shown separately."""
    world = Image.open(FIG / "fig_world_v2_compare.png").convert("RGB")

    def cell(image: Image.Image, col: int, row: int, panel: int = PANEL) -> Image.Image:
        x = MARGIN + ROW_W + col * (panel + GAP)
        y = MARGIN + HEADER_H + row * (panel + GAP)
        return image.crop((x, y, x + panel, y + panel))

    W, H = 1600, 1040
    pad, gap, tile = 40, 14, 226
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    f_sub, f_small = font(17), font(15)
    teal, ink, muted, grid = (91, 163, 168), (25, 35, 48), (84, 97, 112), (218, 226, 234)

    # Distinct blue, cyan, orange, and green scenes from the cross-scene benchmark.
    cols = [
        ("Input", 0), ("DHAN-SHR (ZS)", 2), ("DHAN-SHR (FT)", 3), ("Ours-ZS", 4), ("Ours-FT", 5),
        ("Diffuse GT", 6),
    ]
    grid_x, grid_y = (W - (64 + len(cols) * tile + (len(cols) - 1) * gap)) // 2, 48
    for j, (label, _) in enumerate(cols):
        x = grid_x + 64 + j * (tile + gap)
        draw.text((x, grid_y - 28), label, fill=teal if label == "Ours-FT" else ink, font=f_small)
    source_rows = (0, 1, 3, 5)
    for r, source_row in enumerate(source_rows):
        y = grid_y + r * (tile + gap)
        draw.text((grid_x, y + tile // 2 - 8), f"scene {source_row + 1}", fill=muted, font=f_small)
        for j, (_, source_col) in enumerate(cols):
            x = grid_x + 64 + j * (tile + gap)
            im = cell(world, source_col, source_row).resize((tile, tile), Image.Resampling.LANCZOS)
            im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=135, threshold=3))
            canvas.paste(im, (x, y))
            highlight = source_col in {4, 5}
            draw.rectangle((x, y, x + tile, y + tile), outline=teal if highlight else grid, width=3 if highlight else 1)
    caption = "Cross-scene synthetic benchmark: world_benchmark_v2"
    tw = draw.textlength(caption, font=f_sub)
    draw.text(((W - tw) / 2, grid_y + 4 * tile + 3 * gap + 18), caption, fill=ink, font=f_sub)

    save(canvas, "fig_ours_simulation_advantage.png")


@torch.inference_mode()
def fig_shiq(n_rows: int = 6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prefer shiq_mse checkpoint matching verified 32.61 number; fallback shiq_mask
    ckpt = OUT / "shiq_mse" / "best.pt"
    if not ckpt.exists():
        ckpt = OUT / "shiq_mask" / "best.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = MVBRDFSHRPro(200, 48, 96).to(device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    ds = SHIQAdapter(REPO / "data" / "raw" / "SHIQ_extracted", 200, "test")
    dhan_dir = OUT / "baselines" / "dhan_shr_shiq" / "predictions"
    tshr_dir = OUT / "baselines" / "tshrnet_shiq" / "predictions"
    scores = []
    for i in range(min(200, len(ds))):
        s = ds[i]
        sc = s.specular_gt.abs().mean().item() if s.specular_gt is not None else 0
        scores.append((sc, i))
    scores.sort(reverse=True)
    rows, labs = [], []
    for k, idx in enumerate([i for _, i in scores[:n_rows]], 1):
        s = ds[idx]
        img = s.image.unsqueeze(0).to(device)
        pred = to_u8(model(img, use_refine=True, detach_physics=True)["pred"][0])
        gt = to_u8(s.image_clean)
        dhan = load_rgb(dhan_dir / f"{idx:06d}.png", PANEL) if (dhan_dir / f"{idx:06d}.png").exists() else pred
        tshr = load_rgb(tshr_dir / f"{idx:06d}.png", PANEL) if (tshr_dir / f"{idx:06d}.png").exists() else pred
        rows.append(
            [
                to_u8(img[0]),
                tshr,
                dhan,
                pred,
                gt,
                err_heat(pred, gt),
            ]
        )
        labs.append(f"#{k}")
    save(
        make_grid(
            rows,
            ["Input", "TSHRNet", "DHAN-SHR", "Ours", "Diffuse GT", "Err(Ours)"],
            labs,
            panel=140,
            ours_cols={3},
        ),
        "fig_shiq_qualitative.png",
    )


@torch.inference_mode()
def fig_external_zs_ft():
    """SSHR/NSH: Input | Ours-ZS | Ours-FT | GT | Error; PSD: ZS only."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pro(ckpt: Path, size: int):
        m = MVBRDFSHRPro(size, 48, 96).to(device)
        st = torch.load(ckpt, map_location=device, weights_only=False)
        m.load_state_dict(st.get("model", st), strict=False)
        m.eval()
        return m

    shiq_ckpt = OUT / "shiq_mask" / "best.pt"
    specs = [
        (
            "SSHR",
            SSHRAdapter(REPO / "data" / "raw" / "SSHR_extracted", 256, "test", True),
            shiq_ckpt,
            OUT / "sshr_finetuned_strict" / "best.pt",
            "fig_sshr_qualitative.png",
        ),
        (
            "NSH",
            NSHAdapter(REPO / "data" / "raw" / "NSH_extracted", 256, "test"),
            shiq_ckpt,
            OUT / "nsh_finetuned_strict" / "best.pt",
            "fig_nsh_qualitative.png",
        ),
    ]
    for name, adapter, zs_ckpt, ft_ckpt, outfile in specs:
        if not ft_ckpt.exists():
            ft_ckpt = zs_ckpt
        zs = load_pro(zs_ckpt, 256)
        ft = load_pro(ft_ckpt, 256)
        scores = []
        for i in range(min(60, len(adapter))):
            s = adapter[i]
            scores.append(((s.image - s.image_clean).abs().mean().item(), i))
        scores.sort(reverse=True)
        rows, labs = [], []
        for k, idx in enumerate([i for _, i in scores[:4]], 1):
            s = adapter[idx]
            x = s.image.unsqueeze(0).to(device)
            if x.shape[-1] != 256:
                x = F.interpolate(x, (256, 256), mode="bilinear", align_corners=False)
            gt = s.image_clean
            if gt.shape[-1] != 256:
                gt = F.interpolate(gt.unsqueeze(0), (256, 256), mode="bilinear", align_corners=False)[0]
            pz = to_u8(zs(x, use_refine=True, detach_physics=True)["pred"][0])
            pf = to_u8(ft(x, use_refine=True, detach_physics=True)["pred"][0])
            gtu = to_u8(gt)
            rows.append([to_u8(x[0]), pz, pf, gtu, err_heat(pf, gtu)])
            labs.append(f"#{k}")
        save(
            make_grid(
                rows,
                ["Input", "Ours-ZS", "Ours-FT", "GT", "Err(FT)"],
                labs,
                panel=148,
                ours_cols={1, 2},
            ),
            outfile,
        )

    # PSD: ZS vs domain FT on held-out groups (ft_test protocol)
    adapter = PSDAdapter(REPO / "data" / "raw" / "PSD_extracted", 256, "ft_test")
    zs = load_pro(shiq_ckpt, 256)
    ft_ckpt = OUT / "psd_finetuned_strict" / "best.pt"
    ft = load_pro(ft_ckpt if ft_ckpt.exists() else shiq_ckpt, 256)
    scores = []
    for i in range(min(60, len(adapter))):
        s = adapter[i]
        scores.append(((s.image - s.image_clean).abs().mean().item(), i))
    scores.sort(reverse=True)
    rows, labs = [], []
    for k, idx in enumerate([i for _, i in scores[:4]], 1):
        s = adapter[idx]
        x = s.image.unsqueeze(0).to(device)
        pz = to_u8(zs(x, use_refine=True, detach_physics=True)["pred"][0])
        pf = to_u8(ft(x, use_refine=True, detach_physics=True)["pred"][0])
        gt = to_u8(s.image_clean)
        rows.append([to_u8(x[0]), pz, pf, gt, err_heat(pf, gt)])
        labs.append(f"#{k}")
    save(
        make_grid(
            rows,
            ["Input", "Ours-ZS", "Ours-FT", "GT", "Err(FT)"],
            labs,
            panel=148,
            ours_cols={1, 2},
        ),
        "fig_psd_qualitative.png",
    )


def fig_bars(metrics: dict):
    """Publication-style bar charts (tight spacing, muted palette)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Soften Fig.16 (summary) style: pale teal Ours + cool gray reference.
    C_OURS = "#5BA3A8"       # faded teal (was too dark/abrupt)
    C_OURS_SOFT = "#8EC4C8"
    C_BASE = "#8FA4B0"       # soft slate
    C_LIT = "#C5CED6"        # pale gray for literature
    C_INPUT = "#B7C0C8"      # near-reference gray (not gold)
    C_HARD = "#D4A08A"       # soft terracotta
    C_ALL = "#7A93A6"
    C_FAIL = "#D2B0A0"
    C_REF = "#C5CED6"
    C_GRID = "#F0F2F4"
    C_INK = "#2C3640"
    C_MUTED = "#6B7785"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#94A3B0",
            "xtick.color": C_INK,
            "ytick.color": C_MUTED,
            "text.color": C_INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 220,
        }
    )

    def _style_axes(ax, ymax: float):
        ax.set_ylim(0, ymax)
        ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#94A3B0")
        ax.tick_params(axis="x", length=0, pad=4)
        ax.tick_params(axis="y", length=3, labelsize=9)
        ax.set_ylabel("PSNR (dB)", fontsize=10, color=C_MUTED, labelpad=6)

    def _annotate(ax, bars, values, fontsize=8.5):
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ax.get_ylim()[1] * 0.012,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=C_INK,
                clip_on=False,
            )

    def _save_fig(fig, outfile: str):
        path = FIG / outfile
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print("wrote", path)

    def chart_single(
        title: str,
        labels: list[str],
        values: list[float],
        outfile: str,
        ymax: float | None = None,
        ymin: float = 0.0,
        bar_colors: list[str] | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        n = len(labels)
        ymax = ymax or max(values) * 1.12
        # Match Fig.16 proportions: solid bars, small but readable gaps.
        width = 0.78
        step = 1.0
        x = np.arange(n, dtype=float) * step
        data_span = (n - 1) * step + width
        if figsize is None:
            figsize = (max(3.8, 0.55 * n + 1.3), 2.75)
        fig, ax = plt.subplots(figsize=figsize)
        colors = bar_colors or [C_BASE] * n
        bars = ax.bar(
            x,
            values,
            width=width,
            color=colors,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        _style_axes(ax, ymax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(x[0] - width * 0.65, x[-1] + width * 0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=11, color=C_INK, pad=5, loc="left")
        _annotate(ax, bars, values, fontsize=8)
        fig.tight_layout(pad=0.25)
        _save_fig(fig, outfile)

    def chart_grouped(
        title: str,
        labels: list[str],
        series: dict[str, list[float]],
        outfile: str,
        ymax: float | None = None,
        ymin: float = 0.0,
        series_colors: dict[str, str] | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        keys = list(series)
        n = len(labels)
        n_s = len(keys)
        ymax = ymax or max(v for vals in series.values() for v in vals) * 1.12
        # Fig.16 grouped style: bars abut inside group; modest gap between groups.
        bar_w = 0.38 if n_s == 2 else min(0.26, 0.72 / n_s)
        gap = 0.0
        cluster = n_s * bar_w + (n_s - 1) * gap
        step = cluster + 0.22
        x = np.arange(n, dtype=float) * step
        offsets = np.linspace(-cluster / 2 + bar_w / 2, cluster / 2 - bar_w / 2, n_s)
        default_colors = {
            "Ours": C_OURS,
            "Ours-FT": C_OURS,
            "Ours-ZS": C_OURS_SOFT,
            "Input": C_INPUT,
            "All": C_ALL,
            "Hard": C_HARD,
            "PSNR": C_OURS,
            "Reference": C_REF,
        }
        series_colors = series_colors or {}
        if figsize is None:
            figsize = (max(4.0, 0.72 * n + 1.4), 2.75)
        fig, ax = plt.subplots(figsize=figsize)
        for j, key in enumerate(keys):
            vals = series[key]
            bars = ax.bar(
                x + offsets[j],
                vals,
                width=bar_w,
                color=series_colors.get(key, default_colors.get(key, C_BASE)),
                edgecolor="white",
                linewidth=0.3,
                label=key,
                zorder=2,
            )
            _annotate(ax, bars, vals, fontsize=7.5)
        _style_axes(ax, ymax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(x[0] - cluster * 0.7, x[-1] + cluster * 0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=11, color=C_INK, pad=5, loc="left")
        ax.legend(
            frameon=False,
            fontsize=9,
            loc="upper right",
            ncol=min(3, n_s),
            handlelength=0.9,
            columnspacing=0.7,
            borderaxespad=0.1,
        )
        fig.tight_layout(pad=0.25)
        _save_fig(fig, outfile)

    def colorize_labels(labels: list[str]) -> list[str]:
        cols = []
        for lab in labels:
            low = lab.lower()
            # Highlight only the main "Ours / best" bars; keep the rest in one family.
            if lab in {"Ours", "Ours-FT", "Ours-Phys", "Phys", "Ours-Fuse", "Fuse"}:
                cols.append(C_OURS)
            elif "ours" in low or lab in {"Pro", "Analytic", "Refine"}:
                cols.append(C_OURS_SOFT)
            elif lab.endswith("*"):
                cols.append(C_LIT)
            elif lab in {"Inv", "Inv+Pro"}:
                cols.append(C_FAIL)
            elif "input" in low:
                cols.append(C_INPUT)
            else:
                cols.append(C_BASE)
        return cols

    sh = metrics["shiq"]
    lb = sh["local_baselines"]
    shiq_labels = ["TSHRNet*", "HRNet*", "Ours", "TSHRNet", "DHAN", "DHAN*", "JSHDR*", "DRM*"]
    shiq_vals = [
        25.58,
        30.23,
        sh["ours_psnr"],
        lb["tshrnet"]["psnr"],
        lb["dhan_shr"]["psnr"],
        33.81,
        34.13,
        34.50,
    ]
    chart_single(
        "SHIQ official test (local + literature*)",
        shiq_labels,
        shiq_vals,
        "fig_shiq_psnr_bars.png",
        ymax=38,
        ymin=22,
        bar_colors=colorize_labels(shiq_labels),
    )

    wv = metrics["world_v2"]
    chart_grouped(
        "world_benchmark_v2 (local; * = zero-shot)",
        ["TSHRNet*", "DHAN*", "Ours-ZS*", "Ours-FT"],
        {
            "All": [
                round(wv["tshrnet"]["test"]["psnr"], 2),
                round(wv["dhan_shr"]["test"]["psnr"], 2),
                round(wv["ours_zero_shot"]["test"]["psnr"], 2),
                round(wv["ours_finetuned"]["test"]["psnr"], 2),
            ],
            "Hard": [
                round(wv["tshrnet"]["hard_test"]["psnr"], 2),
                round(wv["dhan_shr"]["hard_test"]["psnr"], 2),
                round(wv["ours_zero_shot"]["hard_test"]["psnr"], 2),
                round(wv["ours_finetuned"]["hard_test"]["psnr"], 2),
            ],
        },
        "fig_world_v2_psnr_bars.png",
        ymax=42,
        series_colors={"All": C_ALL, "Hard": C_HARD},
    )

    h = metrics["holdout_0070"]
    b = h["report_baselines"]
    hold_labels = ["TSHRNet", "DHAN", "Input", "Ours (SI)", "Ours-Fuse", "Ours-Phys"]
    hold_vals = [
        b["tshrnet"]["psnr"],
        b["dhan_shr"]["psnr"],
        b["input"]["psnr"],
        b["single"]["psnr"],
        h["hybrid"]["psnr"],
        h["physical"]["psnr"],
    ]
    chart_single(
        "scene_0070 holdout (same 3 views)",
        hold_labels,
        hold_vals,
        "fig_holdout_psnr_bars.png",
        ymax=42,
        ymin=15,
        bar_colors=colorize_labels(hold_labels),
    )

    ext = metrics["external"]
    chart_grouped(
        "External datasets: Ours vs Input (local)",
        ["SSHR-ZS", "SSHR-FT", "NSH-ZS", "NSH-FT", "PSD-ZS", "PSD-FT"],
        {
            "Ours": [
                ext["sshr_zs"]["ours_psnr"],
                ext["sshr_ft"]["ours_psnr"],
                ext["nsh_zs"]["ours_psnr"],
                ext["nsh_ft"]["ours_psnr"],
                ext["psd_zs"]["ours_psnr"],
                ext["psd_ft"]["ours_psnr"],
            ],
            "Input": [
                ext["sshr_zs"]["input_psnr"],
                ext["sshr_ft"]["input_psnr"],
                ext["nsh_zs"]["input_psnr"],
                ext["nsh_ft"]["input_psnr"],
                ext["psd_zs"]["input_psnr"],
                ext["psd_ft"]["input_psnr"],
            ],
        },
        "fig_external_psnr.png",
        ymax=38,
        ymin=20,
        series_colors={"Ours": C_OURS, "Input": C_REF},
    )

    ab = metrics["ablation_0070"]["variants"]
    abl_labels = ["Inv", "Inv+Pro", "Refine", "Analytic", "Pro", "Fuse", "Phys"]
    abl_vals = [
        ab["inverse"]["psnr"],
        ab["inverse_pro"]["psnr"],
        ab["ours_refined"]["psnr"],
        ab["fusion_analytic"]["psnr"],
        ab["ours_pro"]["psnr"],
        ab["fusion_full"]["psnr"],
        ab["ours_phys"]["psnr"],
    ]
    chart_single(
        "scene_0070 World/Fusion ablation",
        abl_labels,
        abl_vals,
        "fig_inverse_ablation.png",
        ymax=42,
        bar_colors=colorize_labels(abl_labels),
    )

    m = metrics["mvs"]
    mvs_labels = ["DTU-tr", "DTU-ho", "BMVS-tr", "BMVS-ho"]
    mvs_vals = [
        round(m["dtu_train"]["full_psnr"], 2),
        round(m["dtu_holdout"]["full_psnr"], 2),
        round(m["bmvs_train"]["full_psnr"], 2),
        round(m["bmvs_holdout"]["full_psnr"], 2),
    ]
    chart_single(
        "DTU / BlendedMVS diagnostic NVS (not SHR)",
        mvs_labels,
        mvs_vals,
        "fig_mvs_nvs_bars.png",
        ymax=12,
        bar_colors=[C_BASE, C_HARD, C_BASE, C_HARD],
    )

    chart_grouped(
        "Summary across protocols (Ours vs reference)",
        ["SHIQ", "WB-v2", "Holdout", "SSHR-FT", "NSH-FT"],
        {
            "Ours": [
                sh["ours_psnr"],
                round(wv["ours_finetuned"]["test"]["psnr"], 2),
                h["physical"]["psnr"],
                ext["sshr_ft"]["ours_psnr"],
                ext["nsh_ft"]["ours_psnr"],
            ],
            "Reference": [
                34.50,
                round(wv["dhan_shr"]["test"]["psnr"], 2),
                b["single"]["psnr"],
                ext["sshr_ft"]["input_psnr"],
                ext["nsh_ft"]["input_psnr"],
            ],
        },
        "fig_summary_psnr.png",
        ymax=42,
        ymin=20,
        series_colors={"Ours": C_OURS, "Reference": C_REF},
    )


@torch.inference_mode()
def fig_teaser_results(metrics: dict, n_rows: int = 4):
    """Representative single-image outputs; measurements remain in manuscript tables."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = OUT / "shiq_mse" / "best.pt"
    if not ckpt.exists():
        ckpt = OUT / "shiq_mask" / "best.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = MVBRDFSHRPro(200, 48, 96).to(device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    ds = SHIQAdapter(REPO / "data" / "raw" / "SHIQ_extracted", 200, "test")
    scores = []
    for i in range(min(180, len(ds))):
        s = ds[i]
        sc = s.specular_gt.abs().mean().item() if s.specular_gt is not None else 0
        scores.append((sc, i))
    scores.sort(reverse=True)

    panel = 220
    pad = 22
    gap = 12
    header_h = 36
    n_c = 4
    w = pad * 2 + n_c * panel + (n_c - 1) * gap
    h = pad * 2 + header_h + n_rows * panel + (n_rows - 1) * gap
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    cols = ["Input", "Ours (single-image)", "Diffuse GT", "Absolute error"]
    for j, lab in enumerate(cols):
        x = pad + j * (panel + gap)
        color = OURS if lab == "Ours (single-image)" else INK
        tw = draw.textlength(lab, font=font(13))
        draw.text((x + (panel - tw) / 2, pad), lab, fill=color, font=font(13))

    for r, (_, idx) in enumerate(scores[:n_rows]):
        s = ds[idx]
        img = s.image.unsqueeze(0).to(device)
        pred = to_u8(model(img, use_refine=True, detach_physics=True)["pred"][0])
        gt = to_u8(s.image_clean)
        cells = [to_u8(img[0]), pred, gt, err_heat(pred, gt)]
        y = pad + header_h + r * (panel + gap)
        for j, arr in enumerate(cells):
            x = pad + j * (panel + gap)
            tile = Image.fromarray(arr).resize((panel, panel), Image.Resampling.BICUBIC)
            canvas.paste(tile, (x, y))
            outline = OURS if j == 1 else GRID
            draw.rectangle([x, y, x + panel - 1, y + panel - 1], outline=outline, width=3 if j == 1 else 1)
    save(canvas, "fig_teaser_results.png")


def main():
    metrics = collect_verified_metrics()
    fig_bars(metrics)
    try:
        fig_teaser_results(metrics, 4)
    except Exception as e:
        print("[warn] teaser", e)
    try:
        fig_world_v2(6)
    except Exception as e:
        print("[warn] world_v2", e)
    try:
        fig_holdout_compare()
    except Exception as e:
        print("[warn] holdout", e)
    try:
        fig_ours_advantage(metrics)
    except Exception as e:
        print("[warn] ours_advantage", e)
    try:
        fig_shiq(6)
    except Exception as e:
        print("[warn] shiq", e)
    try:
        fig_external_zs_ft()
    except Exception as e:
        print("[warn] external", e)
    print("done")


if __name__ == "__main__":
    main()
