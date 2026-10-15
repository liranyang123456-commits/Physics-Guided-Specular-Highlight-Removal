# Physics-Guided Specular Highlight Removal via World-Space 3D Gaussian BRDF Fields and Confidence-Gated Fusion

Code, evaluation scripts, and per-scene metric files for the paper of the same
title (Ranyang Li, Nan Wei, Zhipeng Lin, Wufeng Liu, Chao Fan, Junjun Pan),
under review at *Array* (manuscript ARRAY-D-26-05047).

MVBRDF-SHR removes highlights by rendering the diffuse lobe of a world-space
3D Gaussian BRDF field, falls back to a physics-guided single-image restorer
when multi-view input is unavailable, and combines the two with a
confidence-gated fusion module.

## Contents

| Path | Purpose |
|---|---|
| `mvbrdf_shr/` | Package: Gaussian BRDF field, renderer, world pipeline, fusion, single-image restorer, dataset adapters, metrics |
| `scripts/` | Data generation, training, fine-tuning of baselines, and evaluation scripts used in the paper |
| `configs/` | Scene and training configurations |
| `outputs/` | Metric JSON files behind every locally measured number in the paper |
| `data/world_benchmark_v2/manifest.json` | Frame list and provenance of the synthetic benchmark (images are regenerated, see below) |
| `data/world_benchmark_material/manifest.json` | The eight material-controlled scenes |
| `paper_figures/` | Scripts that regenerate the figures from the metric files and checkpoints |
| `tests/` | Unit tests (`python -m pytest -q tests/test_world_gaussian.py tests/test_pipeline.py`) |

Not included: third-party datasets, third-party baseline code, model
checkpoints (several hundred MB each), and rendered predictions.  Checkpoints
are available from the corresponding authors on request.

## Installation

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.13, PyTorch 2.11 (CUDA 12.8), and gsplat 1.5.3 on an
RTX 5090.  The PyTorch compositor in `mvbrdf_shr/world/renderer.py` is a
fallback when gsplat is unavailable.  On Windows, gsplat's JIT build needs the
MSVC/ninja toolchain; the renderer sets
`torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")` so
that a localized compiler banner does not break the build.

## Data

**Synthetic multi-view benchmark.** Regenerate it with the method's renderer
(60/10/20 train/validation/test scenes, 12 views each, 128x128, seed 2026):

```bash
python scripts/generate_world_benchmark.py --out data/world_benchmark_v2
python scripts/generate_material_scenes.py --verify   # re-renders scene_0070 and compares it with the generated data
python scripts/generate_material_scenes.py            # eight material-controlled scenes
```

The benchmark is rendered by the same renderer that the method uses, from
known Gaussian fields; it is a controlled probe, not an independent
measurement (see `gt_provenance` in `data/world_benchmark_v2/manifest.json`).

**Public datasets.** Download SHIQ, SSHR, NSH, PSD, DTU, and BlendedMVS from
their official sources and extract them to `data/raw/SHIQ_extracted`,
`data/raw/SSHR_extracted`, `data/raw/NSH_extracted`, `data/raw/PSD_extracted`,
`data/raw/DTU_extracted`, and `data/raw/BlendedMVS_extracted`
(`scripts/prepare_data.py` unpacks the archives).

**Baselines.** Clone [DHAN-SHR](https://github.com/CXH-Research/DHAN-SHR) and
[TSHRNet](https://github.com/fu123456/TSHRNet) into `baselines_ext/DHAN-SHR`
and `baselines_ext/TSHRNet`, together with their released weights
(`weights/model.pth` and `checkpoints_mix_SSHR_SHIQ_PSD/`).

## Reproducing the paper

Table numbers refer to the revised manuscript.

| Paper | Command | Output |
|---|---|---|
| Table 3, single-image branch on world_benchmark_v2 | `python scripts/train_flat_benchmark.py` | `outputs/world_benchmark_single/metrics.json` |
| Table 3, matched DHAN-SHR / TSHRNet fine-tuning | `python scripts/finetune_dhan_world_v2.py`, `python scripts/finetune_tshrnet_world_v2.py` | `outputs/{dhan,tshrnet}_world_v2_finetuned/metrics.json` |
| Tables 4, 5, 14, scene_0070 strict holdout | `python -m mvbrdf_shr.world.train --config configs/world/benchmark_scene_0070.yaml`; `python -m mvbrdf_shr.world.train_fusion --world-checkpoint outputs/world_benchmark_v2_scene_0070_strict/last.pt --single-checkpoint outputs/world_benchmark_single/best.pt --output outputs/world_benchmark_v2_scene_0070_hybrid`; `python scripts/eval_world_ablation.py --raster-backend gsplat`; `python scripts/eval_holdout_zero_shot.py` | `outputs/world_ablation_scene_0070/ablation_metrics.json`, `outputs/world_holdout_zero_shot.json` |
| Tables 6, 7, ten held-out scenes | `python scripts/run_multiscene_eval.py`; `python scripts/aggregate_world_holdout.py --output outputs/world_holdout_multiscene.json`; `python scripts/summarize_multiscene.py` | `outputs/world_holdout_multiscene.json`, `outputs/world_confidence_correlation.json` |
| Table 8, material-controlled scenes | `python scripts/run_material_study.py --launch`, then `--eval --summary` | `outputs/material_study/summary.json` |
| Table 9, prior sensitivity | `python scripts/run_sensitivity_reopt.py --axis <axis>`; `python scripts/eval_oracle_init.py` | `outputs/sensitivity_reopt/` |
| Table 10, SHIQ | `python scripts/train_shiq_pro.py`; baselines: `python scripts/eval_baseline_paired.py` | `outputs/shiq_mse/`, `outputs/shiq_leakage_fixed_s0/`, `outputs/baselines/` |
| Tables 11, 12, SSHR / NSH / PSD | `python scripts/train_paired_pro.py`; `python scripts/eval_paired_dataset.py` | `outputs/*_shiq_zero_shot/`, `outputs/*_finetuned_strict/` |
| Table 13, DTU / BlendedMVS diagnostic | `python -m mvbrdf_shr.world.train --config configs/world/dtu_scan1.yaml` (and `blendedmvs_57f8d9bb.yaml`) | `outputs/dtu_scan1/`, `outputs/blendedmvs_57f8d9bb/` |
| Bar charts | `python paper_figures/make_bar_figures.py` | `paper_figures/figures/` |

## Notes on evaluation

- **Rasterizer.** A scene must be evaluated with the rasterizer it was
  optimized with.  `scripts/check_raster_backend.py` identifies it from the
  gradient norm at the final parameters; scene_0070, DTU, and BlendedMVS used
  gsplat, and scenes 0071-0079 the PyTorch compositor
  (`outputs/world_raster_backend_check.json`).  Re-optimizing scenes 0071-0079
  with a working gsplat build gives slightly different numbers.
- **Metrics.** PSNR is computed per image over all pixels and RGB channels with
  peak 1 and then averaged.  SSIM uses an 11x11 uniform window with reflective
  padding and C1 = 0.01^2, C2 = 0.03^2 (`mvbrdf_shr/metrics.py`); literature
  SSIM values may use a Gaussian window and are not strictly comparable.
- **SHIQ checkpoint selection.** The SHIQ checkpoint used in the paper
  (`outputs/shiq_mse`, 32.61 dB) was selected on the SHIQ test set in earlier
  training runs.  The run in `outputs/shiq_leakage_fixed_s0` is selected on a
  500-image validation half (31.57 dB on the other 500 test images, 31.74 dB on
  all 1,000).  All other experiments select checkpoints on validation data
  disjoint from their test data.

## License

MIT, see `LICENSE`.  Third-party datasets and baselines keep their own
licenses.

## Citation

If you use this code, please cite the paper once it is published; until then:

```bibtex
@unpublished{li2026mvbrdfshr,
  title  = {Physics-Guided Specular Highlight Removal via World-Space {3D} {Gaussian} {BRDF} Fields and Confidence-Gated Fusion},
  author = {Li, Ranyang and Wei, Nan and Lin, Zhipeng and Liu, Wufeng and Fan, Chao and Pan, Junjun},
  note   = {Manuscript under review at Array},
  year   = {2026}
}
```
