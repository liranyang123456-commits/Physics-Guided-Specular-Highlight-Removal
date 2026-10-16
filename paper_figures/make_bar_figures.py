"""Bar-chart figures (fig_bars_core, fig_bars_generalization, fig_inverse_ablation).

Every local value is read from the archived metric JSON files under
``mvbrdf_shr/outputs`` so that the figures cannot drift from the tables;
literature values are the published numbers cited in the SHIQ table.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

C_OURS = "#5BA3A8"
C_OURS_SOFT = "#8EC4C8"
C_BASE = "#8FA4B0"
C_INPUT = "#C9D3DC"
C_FAIL = "#D2B0A0"
C_INK = "#2C3640"
C_MUTED = "#6B7785"
C_GRID = "#EEF1F4"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#94A3B0",
        "axes.linewidth": 0.8,
        "xtick.color": C_INK,
        "ytick.color": C_MUTED,
        "text.color": C_INK,
        "savefig.dpi": 220,
    }
)


PRED = ROOT / "data" / "world_benchmark_v2" / "predictions"


def load(path: str) -> dict:
    return json.loads((OUT / path).read_text(encoding="utf-8"))


def collect() -> dict:
    return {
        "shiq_ours": load("shiq_mse/shiq_results.json")["mvbrdf_pro"]["psnr"],
        "shiq_ours_valsel": load("shiq_leakage_fixed_s0/test_full/metrics.json")["ours_psnr"],
        "shiq_tshrnet": load("baselines/tshrnet_shiq/metrics.json")["psnr"],
        "shiq_dhan": load("baselines/dhan_shr_shiq/metrics.json")["psnr"],
        "world_v2": {k: json.loads((PRED / k / "metrics.json").read_text())["test"]["psnr"]
                     for k in ("tshrnet", "dhan_shr", "mvbrdf_shr_shiq_pretrained", "mvbrdf_shr_world_v2_finetuned")},
        "dhan_ft": load("dhan_world_v2_finetuned/metrics.json")["test"]["psnr"],
        "tshrnet_ft": load("tshrnet_world_v2_finetuned/metrics.json")["test"]["psnr"],
        "ablation": load("v2/ablation_scene_0070/ablation_metrics.json")["variants"],
        "strict": load("v2/ablation_strict_nossim_eval/ablation_metrics.json")["variants"],
        "bootstrap": load("v2/ablation_bootstrap_eval/ablation_metrics.json")["variants"],
        # scripts/eval_holdout_zero_shot.py
        "holdout_tshrnet": load("world_holdout_zero_shot.json")["tshrnet"]["psnr"],
        "holdout_dhan": load("world_holdout_zero_shot.json")["dhan_shr"]["psnr"],
    }


def style(ax, ymin: float, ymax: float, title: str) -> None:
    ax.set_ylim(ymin, ymax)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=3, labelsize=9)
    ax.set_ylabel("PSNR (dB)", color=C_MUTED, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)


def bars(ax, labels, values, colors, fmt="{:.2f}", fontsize=7.5, label_size=7.0) -> None:
    x = np.arange(len(labels))
    rects = ax.bar(x, values, width=0.72, color=colors, edgecolor="white", linewidth=0.4, zorder=2)
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for r, v in zip(rects, values):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012 * span, fmt.format(v),
                ha="center", va="bottom", fontsize=fontsize, fontweight="bold", color=C_INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=label_size, linespacing=1.1)


def fig_core(m: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 4.3), gridspec_kw={"width_ratios": [1.05, 0.85, 1.65]})

    labels = ["TSHRNet\n(lit.)", "Highlight-\nRNet (lit.)", "Ours\n(val-sel.)", "Ours\n(local)", "TSHRNet\n(local)",
              "DHAN-SHR\n(local)", "DHAN-SHR\n(lit.)", "JSHDR\n(lit.)", "Neural DRM\n(lit.)"]
    values = [25.58, 30.23, m["shiq_ours_valsel"], m["shiq_ours"], m["shiq_tshrnet"], m["shiq_dhan"], 33.81, 34.13,
              34.50]
    colors = [C_INPUT, C_INPUT, C_OURS_SOFT, C_OURS, C_BASE, C_BASE, C_INPUT, C_INPUT, C_INPUT]
    style(axes[0], 20, 37, "(a) SHIQ single-image test")
    bars(axes[0], labels, values, colors)

    wv = m["world_v2"]
    labels = ["TSHRNet\n(ZS)", "DHAN-SHR\n(ZS)", "Ours single\n(ZS)", "TSHRNet\n(FT)", "DHAN-SHR\n(FT)",
              "Ours single\n(FT)"]
    values = [wv["tshrnet"], wv["dhan_shr"], wv["mvbrdf_shr_shiq_pretrained"], m["tshrnet_ft"], m["dhan_ft"],
              wv["mvbrdf_shr_world_v2_finetuned"]]
    colors = [C_BASE, C_BASE, C_OURS_SOFT, C_BASE, C_BASE, C_OURS]
    style(axes[1], 10, 41, "(b) world_benchmark_v2 (240 frames)")
    bars(axes[1], labels, values, colors)

    ab = m["ablation"]
    labels = ["Input", "TSHRNet\n(ZS)", "DHAN-SHR\n(ZS)", "TSHRNet\n(FT)", "DHAN-SHR\n(FT)", "Ours\n(single)",
              "Ours-\nPhys", "Fixed\nblend", "Ours-\nFuse", "Fuse w/\nDHAN-SHR", "Refiner\n(strict)"]
    values = [ab["input"]["psnr"], m["holdout_tshrnet"], m["holdout_dhan"], ab["tshrnet_ft"]["psnr"],
              ab["dhan_ft"]["psnr"], ab["ours_pro"]["psnr"], ab["ours_phys"]["psnr"], ab["fixed_blend"]["psnr"],
              ab["fusion_full"]["psnr"], ab["dhan_gate"]["psnr"], m["strict"]["ours_refined"]["psnr"]]
    colors = [C_INPUT, C_BASE, C_BASE, C_BASE, C_BASE, C_OURS_SOFT, C_OURS, C_BASE, C_OURS, C_OURS, C_FAIL]
    style(axes[2], 15, 66, "(c) scene_0070 strict holdout (3 views)")
    bars(axes[2], labels, values, colors)

    fig.suptitle("Core SHR results across benchmarks", fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "fig_bars_core.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_generalization(m: dict) -> None:
    ext = {
        "sshr_zs": load("sshr_shiq_zero_shot/metrics.json"),
        "sshr_ft": load("sshr_finetuned_strict/test/metrics.json"),
        "nsh_zs": load("nsh_shiq_zero_shot/metrics.json"),
        "nsh_ft": load("nsh_finetuned_strict/test/metrics.json"),
        "psd_zs_ft_test": load("psd_finetuned_strict/zs_ft_test/metrics.json"),
        "psd_ft": load("psd_finetuned_strict/ft_test/metrics.json"),
    }
    mvs = {
        "DTU\ntrain": load("dtu_scan1_nossim/world_metrics_train.json")["summary"]["full_psnr"],
        "DTU\nholdout": load("dtu_scan1_nossim/world_metrics_holdout.json")["summary"]["full_psnr"],
        "BlendedMVS\ntrain": load("blendedmvs_57f8d9bb_nossim/world_metrics_train.json")["summary"]["full_psnr"],
        "BlendedMVS\nholdout": load("blendedmvs_57f8d9bb_nossim/world_metrics_holdout.json")["summary"]["full_psnr"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1.0]})

    ax = axes[0]
    names = ["SSHR", "NSH", "PSD (ft_test)"]
    series = {
        "Input": [ext[k]["input_psnr"] for k in ("sshr_ft", "nsh_ft", "psd_ft")],
        "Ours (zero-shot)": [ext[k]["ours_psnr"] for k in ("sshr_zs", "nsh_zs", "psd_zs_ft_test")],
        "Ours (fine-tuned)": [ext[k]["ours_psnr"] for k in ("sshr_ft", "nsh_ft", "psd_ft")],
    }
    colors = {"Input": C_INPUT, "Ours (zero-shot)": C_OURS_SOFT, "Ours (fine-tuned)": C_OURS}
    style(ax, 20, 38, "(a) Cross-dataset generalization")
    x = np.arange(len(names))
    w = 0.26
    for j, (key, vals) in enumerate(series.items()):
        rects = ax.bar(x + (j - 1) * w, vals, width=w, color=colors[key], edgecolor="white", linewidth=0.4,
                       label=key, zorder=2)
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.2, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))

    ax = axes[1]
    style(ax, 0, 20, "(b) Real MVS NVS diagnostic")
    bars(ax, list(mvs), list(mvs.values()), [C_BASE, C_FAIL, C_BASE, C_FAIL])
    ax.set_xlabel("full-radiance NVS, not an SHR score", fontsize=8, color=C_MUTED, style="italic")

    fig.suptitle("Generalization and diagnostics", fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(FIG / "fig_bars_generalization.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_ablation(m: dict) -> None:
    ab, st, bo = m["ablation"], m["strict"], m["bootstrap"]
    labels = ["Single-\nimage", "Analytic\ngate", "Fixed\nblend", "Bootstrap\ninit.", "Refiner\n(strict)", "Fusion\n(full)",
              "Physical", "Strict\nprotocol"]
    values = [ab["ours_pro"]["psnr"], ab["fusion_analytic"]["psnr"], ab["fixed_blend"]["psnr"], bo["ours_phys"]["psnr"],
              st["ours_refined"]["psnr"], ab["fusion_full"]["psnr"], ab["ours_phys"]["psnr"], st["ours_phys"]["psnr"]]
    colors = [C_OURS_SOFT, C_OURS_SOFT, C_BASE, C_FAIL, C_FAIL, C_OURS, C_OURS, C_OURS_SOFT]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    style(ax, 0, 66, "scene_0070 strict-holdout ablation")
    bars(ax, labels, values, colors, fontsize=8, label_size=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_inverse_ablation.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    m = collect()
    fig_core(m)
    fig_generalization(m)
    fig_ablation(m)
    print("wrote fig_bars_core.png, fig_bars_generalization.png, fig_inverse_ablation.png")


if __name__ == "__main__":
    main()
