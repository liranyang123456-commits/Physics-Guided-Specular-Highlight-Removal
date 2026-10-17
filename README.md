# Physics-Guided Specular Highlight Removal via World-Space 3D Gaussian BRDF Fields and Confidence-Gated Fusion

Code, evaluation scripts, and per-scene metric files for the paper of the same
title (Ranyang Li, Nan Wei, Zhipeng Lin, Wufeng Liu, Chao Fan, Junjun Pan),
under review at *Array* (manuscript ARRAY-D-26-05047).

MVBRDF-SHR removes highlights by rendering the diffuse lobe of a world-space
3D Gaussian BRDF field, falls back to a physics-guided single-image restorer
when multi-view input is unavailable, and combines the two with a
confidence-gated fusion module.

## Version note: corrected optimization recipe

The physical-branch results of the first submission were produced with a photometric loss that included a 7x7 SSIM term
(weight 0.2).  That term stalls the optimization of the world field (its gradient at initialization is 6.7-10.7 times that of
the L1 term; `outputs/inverse_diag/gradient_ratio.json`, `outputs/inverse_diag/*.json`).  Every physical-branch result of the
revised manuscript was recomputed with the corrected recipe (RGB L1 photometric loss and regularizers, learning rate 0.03,
1,500 steps, no diffuse/specular supervision, no albedo initialization) and lives in `outputs/v2/`.

The directories directly under `outputs/` that belong to the physical branch of the first submission
(`world_benchmark_v2_scene_*_strict`, `world_benchmark_v2_scene_0070_{inverse,inverse_pro,hybrid}`, `world_ablation_scene_*`,
`sensitivity_reopt`, `material_study`, `dtu_scan1`, `blendedmvs_57f8d9bb`, `world_holdout_multiscene.json`,
`world_confidence_correlation.json`, `world_multiscene_family.json`) are kept for transparency and are **superseded** by
`outputs/v2/` (and `outputs/dtu_scan1_nossim`, `outputs/blendedmvs_57f8d9bb_nossim`).  Single-image results (SHIQ, SSHR, NSH, PSD,
the matched fine-tuning of the baselines, ICTPolarReal) do not use the world field and are unaffected.

## Baselines with partial releases (revision)

DHAN-SHR and TSHRNet are fine-tuned under the matched protocol (`scripts/finetune_dhan_world_v2.py`,
`scripts/finetune_tshrnet_world_v2.py`).  The other three methods that report SHIQ numbers could not be, and we checked their
public releases in October 2026:

| Method | Public release | What was run | Files |
|---|---|---|---|
| Neural DRM (ICCV 2025) | no code or weights that we could find | nothing | - |
| JSHDR (CVPR 2021) | compiled CPython-3.6 modules for Linux and a checkpoint, no training code | zero-shot, CPU, Python 3.6 / PyTorch 1.5.1 in WSL, on the 240 benchmark test frames and the 1,000 SHIQ test images | `scripts/wsl_setup_jshdr.sh`, `scripts/export_jshdr_inputs.py`, `scripts/jshdr_batch.sh`, `scripts/jshdr_cpu_shim.py`, `scripts/eval_jshdr_zero_shot.py`, `outputs/jshdr_zero_shot/metrics.json` |
| HighlightRNet (ACM MM 2024) | no weights, no `options` package, the authors' data paths | an attempt to train the released architecture from scratch with the matched protocol did not converge (test 9.71 dB); not reported as a baseline | `scripts/train_hrnet_world_v2.py`, `outputs/hrnet_world_v2_trained/metrics.json` |

The JSHDR value on SHIQ (37.37 dB) is not used for comparison: the JSHDR paper trains on a random 12K of its 16K images, so the
checkpoint may have seen the official test images.

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
(`scripts/prepare_data.py` unpacks the archives).  The public sample of ICTPolarReal
(link on its [project page](https://jingyangcarl.github.io/ICTPolarReal/)) goes to
`data/raw/ICTPolarReal_sample/<object>/camXX/` (used for Table 19 only).

**Baselines.** Clone [DHAN-SHR](https://github.com/CXH-Research/DHAN-SHR) and
[TSHRNet](https://github.com/fu123456/TSHRNet) into `baselines_ext/DHAN-SHR`
and `baselines_ext/TSHRNet`, together with their released weights
(`weights/model.pth` and `checkpoints_mix_SSHR_SHIQ_PSD/`).

## Reproducing the paper

Table numbers refer to the revised manuscript.  Run the scripts from the repository root with `PYTHONPATH=.`.

| Paper | Command | Output |
|---|---|---|
| Table 3, single-image branch on world_benchmark_v2 | `python scripts/train_flat_benchmark.py` | `outputs/world_benchmark_single/metrics.json` |
| Table 3, matched DHAN-SHR / TSHRNet fine-tuning | `python scripts/finetune_dhan_world_v2.py`, `python scripts/finetune_tshrnet_world_v2.py`, then `python scripts/rescore_saved_predictions.py` (the fine-tuning scripts score floating-point outputs; Table 3 scores the saved 8-bit predictions of every row, and the script keeps the floating-point scores under `float_outputs`) | `outputs/{dhan,tshrnet}_world_v2_finetuned/metrics.json` |
| Table 3, raw-input row; every row split into the hard subset and the other 119 frames | `python scripts/eval_input_world_v2.py` | `data/world_benchmark_v2/predictions/input/metrics.json`, `outputs/v2/benchmark_by_subset.json` |
| Section 4.3, the three fine-tuned models scored in floating point and as truncated or rounded 8-bit images | `python scripts/eval_table3_consistent.py` | `outputs/v2/table3_consistent.json` |
| Tables 4, 5, scene_0070 strict holdout | `python scripts/make_v2_configs.py --steps 1500`; `python scripts/run_v2_world.py`; `python -m mvbrdf_shr.world.train_fusion --world-checkpoint outputs/v2/world_scene_0070/last.pt --single-checkpoint outputs/world_benchmark_single/best.pt --output outputs/v2/fusion_scene_0070`; `python scripts/run_v2_eval.py`; `python scripts/eval_world_split_v2.py`; `python scripts/eval_holdout_zero_shot.py` | `outputs/v2/ablation_scene_0070/ablation_metrics.json`, `outputs/v2/world_scene_0070/split_metrics.json`, `outputs/world_holdout_zero_shot.json` |
| Table 6, why the first configuration failed | `python scripts/diag_inverse.py --name <run> [--set KEY=VALUE] [--loss KEY=VALUE]` (the settings of every run are stored in its JSON); `python scripts/diag_gradient_ratio.py` | `outputs/inverse_diag/` |
| Table 6, ten-scene rows (the SSIM term on the other nine scenes) | `python scripts/diag_inverse.py --name ms_l1_00NN --config configs/world/v2/benchmark_scene_00NN_v2.yaml --set steps=600 --eval-every 600` and the same with `--loss photo_ssim=0.2` (`ms_ssim_00NN`), NN = 71..79 | `outputs/inverse_diag/ms_*.json` |
| Tables 7, 8, ten held-out scenes | `python scripts/run_v2_eval.py`; `python scripts/compare_v2.py` | `outputs/v2/ablation_scene_00NN/`, `outputs/v2/world_holdout_multiscene.json`, `outputs/v2/comparison_v1_v2.{json,md}` (old-versus-new comparison) |
| Table 9, material-controlled scenes | `python scripts/run_material_study_v2.py --launch`, then `--eval --summary` | `outputs/v2/material_study/summary.json` |
| Table 10, prior sensitivity | `python scripts/run_sensitivity_v2.py --axis <axis> --seed <k>`; `python scripts/eval_oracle_init_v2.py` | `outputs/v2/sensitivity/` |
| Section 4.4, the exact state is limited by the 8-bit truncation of the targets | `python scripts/check_quantization.py` | `outputs/v2/quantization_check.json` |
| Table 11, gate variants (cross-fitting, corruption) | `python scripts/train_crossfit_fields.py --fold <0,1,2>`; `python -m mvbrdf_shr.world.train_fusion_crossfit ...`; `python -m mvbrdf_shr.world.train_fusion_robust --p-clean 1.0 --seed <s> ...`; `python scripts/eval_sensitivity_gate.py --gate NAME=PATH ... --tag gates_all`; `python scripts/eval_gate_scenes.py --gate NAME=PATH ... --tag all`; `python scripts/summarize_gates.py`; `python scripts/explore_confidence.py` | `outputs/v2/gate_seeds.json`, `outputs/v2/gate_scenes_all.json`, `outputs/v2/sensitivity/gates_all.json`, `outputs/v2/confidence_exploration.json` |
| Table 12, training-view fit check | `python scripts/eval_train_fit.py`; `python scripts/eval_scene_check.py` | `outputs/v2/train_fit.json`, `outputs/v2/scene_check.json` |
| Tables 13, 14, estimating the lighting and the geometry (Section 4.6) | `python scripts/run_sensitivity_unfrozen.py --axis <axis> --unfreeze <none\|geometry\|lighting\|both> [--scene 00NN] [--geo-params ...] [--tag ...]` (every run is listed by its file name; Table 13 starts from the exact albedo like Table 10, Table 14 from the albedo-free initialization of the main results: `--init geometry --tag af`); `python scripts/summarize_estimation.py` | `outputs/v2/sensitivity_unfrozen/*.json`, `outputs/v2/estimation_summary.json` |
| Table 15, fields whose training views are reproduced but whose held-out views are not | `python scripts/eval_view_check.py` (after the runs above and `scripts/run_sensitivity_v2.py`); `python scripts/analyze_view_check.py` | `outputs/v2/view_check.json`, `outputs/v2/view_check_analysis.json` |
| Clean-room reproduction (Section 4.2) | fresh clone; `python scripts/generate_world_benchmark.py --out data/world_benchmark_v2`; `python scripts/make_v2_configs.py --steps 1500`; `python scripts/run_v2_world.py --scenes 70`; `python scripts/eval_world_split_v2.py` | `outputs/v2/reproduction.json` (the regenerated scenes are bit-identical to those of the paper; held-out diffuse PSNR of scene_0070 56.54 dB against 56.51 dB) |
| Table 16, SHIQ | `python scripts/train_shiq_pro.py`; baselines: `python scripts/eval_baseline_paired.py` | `outputs/shiq_mse/`, `outputs/shiq_leakage_fixed_s0/`, `outputs/baselines/` |
| Tables 17, 18, SSHR / NSH / PSD | `python scripts/train_paired_pro.py`; `python scripts/eval_paired_dataset.py` | `outputs/*_shiq_zero_shot/`, `outputs/*_finetuned_strict/` |
| Table 19, ICTPolarReal sample (real light-stage captures) | `python scripts/eval_ictpolarreal.py` (needs the public sample under `data/raw/ICTPolarReal_sample/<object>/camXX/`; it is not redistributed here) | `outputs/ictpolarreal_sample/metrics.json` |
| Table 20, DTU / BlendedMVS diagnostic | `python -m mvbrdf_shr.world.train --config configs/world/dtu_scan1_nossim.yaml` (and `blendedmvs_57f8d9bb_nossim.yaml`) | `outputs/dtu_scan1_nossim/`, `outputs/blendedmvs_57f8d9bb_nossim/` (first submission, with the SSIM term: `outputs/dtu_scan1/`, `outputs/blendedmvs_57f8d9bb/`) |
| Table 21, ablations | `python scripts/make_v2_ablation_configs.py`; train `configs/world/v2_ablation/strict_nossim.yaml` and `bootstrap.yaml`; evaluate each with `python scripts/eval_world_ablation.py --world-checkpoint ... --fusion-checkpoint outputs/v2/fusion_scene_0070/fusion.pt --output ... --raster-backend gsplat` | `outputs/v2/ablation_strict_nossim_eval/`, `outputs/v2/ablation_bootstrap_eval/` |
| Material-parameter errors (Section 4.4) | `python scripts/eval_material_error_v2.py` | `outputs/v2/material_error.json` |
| Diagnostic runs on the first configuration (Section 4.4) | `python scripts/diag_drift.py ...`; `python scripts/summarize_drift.py` | `outputs/drift_diag/` |
| Bar charts and figures | `python paper_figures/make_bar_figures.py` | `paper_figures/figures/` |

The training-view fit check (Section 3.7) is `mvbrdf_shr/world/fit_check.py`; unit tests: `python -m pytest -q tests/test_fit_check.py`.

## Notes on evaluation

- **Rasterizer.** A scene must be evaluated with the rasterizer it was
  optimized with.  `scripts/check_raster_backend.py` identifies it from the
  gradient norm at the final parameters; all runs of `outputs/v2/` and the real-MVS runs used gsplat.  The first-submission scenes 0071-0079 had
  used the PyTorch compositor (`outputs/world_raster_backend_check.json`).
- **Metrics.** PSNR is computed per image over all pixels and RGB channels with
  peak 1 and then averaged.  SSIM uses an 11x11 uniform window with reflective
  padding and C1 = 0.01^2, C2 = 0.03^2 (`mvbrdf_shr/metrics.py`); literature
  SSIM values may use a Gaussian window and are not strictly comparable.
- **SHIQ checkpoint selection.** The SHIQ checkpoint used in the paper
  (`outputs/shiq_mse`, 32.61 dB) was selected on the SHIQ test set in earlier
  training runs.  The run in `outputs/shiq_leakage_fixed_s0` is selected on a
  500-image validation half (31.56 dB on the other 500 test images, 31.74 dB on
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
