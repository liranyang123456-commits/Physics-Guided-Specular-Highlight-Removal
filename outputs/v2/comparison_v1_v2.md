# Old (v1: SSIM photometric term, lr 2e-3, 3000 steps) vs new (v2: RGB L1 only, lr 0.03, 1500 steps)

## 1. scene_0070, strict holdout (3 views {0,4,8}), mean over views

| variant | v1 PSNR | v1 SSIM | v2 PSNR | v2 SSIM | dPSNR |
|---|---|---|---|---|---|
| Input (no removal) | 30.98 | 0.9736 | 30.98 | 0.9736 | +0.00 |
| Ours-Phys | 36.82 | 0.9756 | 56.51 | 0.9995 | +19.69 |
| Ours single-image | 35.59 | 0.9851 | 35.59 | 0.9851 | +0.00 |
| Fixed blend (Phys+single) | 37.04 | 0.9849 | 41.41 | 0.9947 | +4.37 |
| Fusion, analytic gate only | 35.75 | 0.9834 | 37.68 | 0.9888 | +1.93 |
| Fusion, no correction | 36.31 | 0.9864 | 56.15 | 0.9998 | +19.84 |
| Ours-Fuse | 36.28 | 0.9865 | 56.11 | 0.9998 | +19.83 |
| DHAN-SHR (FT) | 37.91 | 0.9907 | 37.91 | 0.9907 | +0.00 |
| DHAN-SHR fixed blend | 37.71 | 0.9877 | 43.68 | 0.9973 | +5.97 |
| DHAN-SHR analytic gate | 37.79 | 0.9890 | 40.70 | 0.9945 | +2.91 |
| DHAN-SHR + learned gate | 38.91 | 0.9922 | 55.14 | 0.9997 | +16.22 |
| TSHRNet (FT) | 37.71 | 0.9872 | 37.71 | 0.9872 | -0.00 |

Per view (PSNR), v1 -> v2:
- Ours-Phys: v0: 25.11 -> 56.05, v4: 28.12 -> 56.89, v8: 57.24 -> 56.60
- Ours-Fuse: v0: 28.69 -> 54.76, v4: 36.20 -> 56.07, v8: 43.95 -> 57.50
- Fixed blend (Phys+single): v0: 29.48 -> 34.43, v4: 32.69 -> 40.84, v8: 48.95 -> 48.96
- DHAN-SHR + learned gate: v0: 33.78 -> 52.48, v4: 37.78 -> 56.82, v8: 45.17 -> 56.11

Repeated optimisation of scene_0070 (identical recipe, different random view order):
- v2 physical PSNR over 3 runs: 56.51, 56.56, 56.57 (mean 56.55, sd 0.03)
- v2 fused PSNR over 3 runs: 56.11, 56.34, 56.30

## 2. Ten scenes (30 v1 / 30 v2 held-out views)

| variant | v1 mean | v1 median | v1 IQR | v2 mean | v2 median | v2 IQR | v1 SSIM | v2 SSIM |
|---|---|---|---|---|---|---|---|---|
| Input (no removal) | 35.00 | 33.10 | [26.6, 42.3] | 35.00 | 33.10 | [26.6, 42.3] | 0.9754 | 0.9754 |
| Ours-Phys | 33.79 | 33.76 | [31.9, 35.4] | 53.91 | 55.17 | [52.7, 56.5] | 0.9769 | 0.9991 |
| Ours single-image | 36.87 | 35.33 | [29.4, 45.2] | 36.87 | 35.33 | [29.4, 45.2] | 0.9838 | 0.9838 |
| Fixed blend (Phys+single) | 36.49 | 37.24 | [32.4, 39.1] | 42.24 | 41.15 | [35.4, 48.9] | 0.9873 | 0.9946 |
| Fusion, analytic gate only | 34.98 | 36.00 | [31.6, 38.3] | 41.86 | 39.22 | [32.7, 51.4] | 0.9825 | 0.9898 |
| Fusion, no correction | 38.24 | 36.47 | [31.2, 46.2] | 53.91 | 55.15 | [51.7, 57.1] | 0.9864 | 0.9995 |
| Ours-Fuse | 38.27 | 36.47 | [31.2, 46.3] | 53.77 | 54.88 | [51.4, 57.0] | 0.9864 | 0.9995 |
| DHAN-SHR (FT) | 37.45 | 36.33 | [32.1, 43.7] | 37.45 | 36.33 | [32.1, 43.7] | 0.9872 | 0.9872 |
| DHAN-SHR fixed blend | 36.90 | 37.47 | [34.0, 38.9] | 42.75 | 42.18 | [38.1, 47.7] | 0.9888 | 0.9960 |
| DHAN-SHR analytic gate | 35.82 | 36.40 | [32.6, 38.3] | 43.15 | 42.53 | [34.6, 53.3] | 0.9857 | 0.9928 |
| DHAN-SHR + learned gate | 38.59 | 37.66 | [34.6, 45.2] | 53.53 | 54.81 | [50.7, 56.7] | 0.9897 | 0.9995 |
| TSHRNet (FT) | 35.59 | 35.23 | [30.8, 41.6] | 35.59 | 35.23 | [30.8, 41.6] | 0.9807 | 0.9807 |

## 3. Per-scene held-out PSNR (mean of 3 views), v1 -> v2

| scene | shape | Ours-Phys | Ours single-image | Fixed blend (Phys+single) | Ours-Fuse | DHAN-SHR (FT) | DHAN-SHR + learned gate |
|---|---|---|---|---|---|---|---|
| 0070 | sphere | 36.82 -> 56.51 | 35.59 -> 35.59 | 37.04 -> 41.41 | 36.28 -> 56.11 | 37.91 -> 37.91 | 38.91 -> 55.14 |
| 0071 | ellipsoid | 33.79 -> 55.12 | 28.33 -> 28.33 | 32.99 -> 34.30 | 31.37 -> 54.39 | 32.83 -> 32.83 | 34.71 -> 54.54 |
| 0072 | bumpy | 26.27 -> 54.18 | 30.75 -> 30.75 | 29.89 -> 36.70 | 31.72 -> 47.94 | 31.00 -> 31.00 | 32.78 -> 48.59 |
| 0073 | torus | 37.51 -> 51.07 | 48.37 -> 48.37 | 42.86 -> 51.70 | 48.59 -> 57.67 | 46.24 -> 46.24 | 45.24 -> 56.98 |
| 0074 | sphere | 34.43 -> 51.73 | 31.44 -> 31.44 | 35.22 -> 37.34 | 34.23 -> 54.90 | 33.60 -> 33.60 | 36.11 -> 53.95 |
| 0075 | ellipsoid | 33.61 -> 49.52 | 42.01 -> 42.01 | 38.45 -> 46.22 | 43.61 -> 52.07 | 41.60 -> 41.60 | 43.15 -> 51.69 |
| 0076 | bumpy | 31.45 -> 54.79 | 34.23 -> 34.23 | 34.34 -> 39.92 | 34.57 -> 49.30 | 34.26 -> 34.26 | 33.93 -> 49.78 |
| 0077 | torus | 36.59 -> 53.74 | 45.34 -> 45.34 | 41.76 -> 50.57 | 45.70 -> 55.46 | 41.82 -> 41.82 | 42.32 -> 54.98 |
| 0078 | sphere | 34.23 -> 56.52 | 35.01 -> 35.01 | 35.21 -> 40.81 | 36.88 -> 56.20 | 36.41 -> 36.41 | 38.31 -> 56.39 |
| 0079 | ellipsoid | 33.18 -> 55.94 | 37.60 -> 37.60 | 37.16 -> 43.39 | 39.77 -> 53.66 | 38.82 -> 38.82 | 40.41 -> 53.31 |
| mean | | 33.79 -> 53.91 | 36.87 -> 36.87 | 36.49 -> 42.24 | 38.27 -> 53.77 | 37.45 -> 37.45 | 38.59 -> 53.53 |

## 4. Counts over the ten scenes (PSNR / SSIM), v1 -> v2

- Ours-Fuse > Ours single-image: v1 10/10 PSNR, 9/10 SSIM;  v2 10/10 PSNR, 10/10 SSIM
- Ours-Fuse > Ours-Phys: v1 7/10 PSNR, 7/10 SSIM;  v2 4/10 PSNR, 8/10 SSIM
- Ours-Phys > Ours single-image: v1 3/10 PSNR, 3/10 SSIM;  v2 10/10 PSNR, 9/10 SSIM
- Ours-Fuse > Fixed blend (Phys+single): v1 7/10 PSNR, 6/10 SSIM;  v2 10/10 PSNR, 10/10 SSIM
- Ours-Fuse > Fusion, analytic gate only: v1 10/10 PSNR, 10/10 SSIM;  v2 10/10 PSNR, 10/10 SSIM
- Ours-Fuse > DHAN-SHR (FT): v1 8/10 PSNR, 5/10 SSIM;  v2 10/10 PSNR, 10/10 SSIM
- DHAN-SHR + learned gate > DHAN-SHR (FT): v1 8/10 PSNR, 10/10 SSIM;  v2 10/10 PSNR, 10/10 SSIM
- DHAN-SHR + learned gate > DHAN-SHR fixed blend: v1 7/10 PSNR, 6/10 SSIM;  v2 10/10 PSNR, 10/10 SSIM
- DHAN-SHR + learned gate > DHAN-SHR analytic gate: v1 10/10 PSNR, 10/10 SSIM;  v2 10/10 PSNR, 9/10 SSIM
- Ours-Phys > DHAN-SHR (FT): v1 2/10 PSNR, 1/10 SSIM;  v2 10/10 PSNR, 10/10 SSIM
- Ours-Phys > Input (no removal): v1 4/10 PSNR, 4/10 SSIM;  v2 9/10 PSNR, 9/10 SSIM

## 5. Shape families (mean PSNR), v1 -> v2

| family | n | Phys | single | Fuse |
|---|---|---|---|---|
| bumpy | 2 | 28.86 -> 54.49 | 32.49 -> 32.49 | 33.14 -> 48.62 |
| ellipsoid | 3 | 33.53 -> 53.53 | 35.98 -> 35.98 | 38.25 -> 53.37 |
| sphere | 3 | 35.16 -> 54.92 | 34.01 -> 34.01 | 35.80 -> 55.74 |
| torus | 2 | 37.05 -> 52.40 | 46.86 -> 46.86 | 47.15 -> 56.56 |

## 6. Confidence-error correlation (Pearson r between gate confidence and diffuse error, 30 views)

- v1: n=30, Pearson mean 0.700, median 0.868, min 0.029, positive 30/30; diffuse L1 low-conf 0.0226 vs high-conf 0.0063 (x3.58); low>high in 30/30
- v2: n=30, Pearson mean 0.219, median 0.064, min -0.069, positive 19/30; diffuse L1 low-conf 0.0018 vs high-conf 0.0011 (x1.67); low>high in 30/30

## 7. Material-controlled scenes (8 scenes), held-out PSNR (3 views), v1 -> v2

| scene | input | Ours-Phys | Ours single-image | Fixed blend (Phys+single) | Ours-Fuse | DHAN-SHR (FT) | DHAN-SHR + learned gate | TSHRNet (FT) |
|---|---|---|---|---|---|---|---|---|
| sphere_lowrough_lowmetal | 38.96 | 57.76 -> 56.44 | 35.32 -> 35.32 | 41.28 -> 41.26 | 41.11 -> 61.22 | 36.23 -> 36.23 | 41.72 -> 54.82 | 37.96 -> 37.96 |
| sphere_lowrough_highmetal | 35.01 | 57.57 -> 55.24 | 33.34 -> 33.34 | 39.32 -> 39.29 | 37.88 -> 59.88 | 35.24 -> 35.24 | 39.27 -> 55.01 | 35.26 -> 35.26 |
| sphere_highrough_lowmetal | 25.87 | 35.85 -> 56.40 | 30.41 -> 30.41 | 34.82 -> 36.33 | 32.86 -> 55.38 | 29.92 -> 29.92 | 32.54 -> 55.00 | 29.78 -> 29.78 |
| sphere_highrough_highmetal | 23.64 | 25.37 -> 52.58 | 26.48 -> 26.48 | 28.69 -> 32.43 | 27.05 -> 49.90 | 25.97 -> 25.97 | 26.54 -> 50.95 | 26.02 -> 26.02 |
| torus_lowrough_lowmetal | 41.47 | 29.32 -> 49.83 | 38.38 -> 38.38 | 33.66 -> 41.60 | 42.92 -> 53.42 | 41.06 -> 41.06 | 45.93 -> 52.14 | 36.69 -> 36.69 |
| torus_lowrough_highmetal | 37.66 | 26.85 -> 47.58 | 38.58 -> 38.58 | 31.12 -> 39.93 | 42.41 -> 50.37 | 39.71 -> 39.71 | 43.17 -> 49.32 | 36.22 -> 36.22 |
| torus_highrough_lowmetal | 28.83 | 25.96 -> 48.36 | 35.03 -> 35.03 | 29.63 -> 39.40 | 35.74 -> 45.01 | 32.97 -> 32.97 | 34.15 -> 45.56 | 33.58 -> 33.58 |
| torus_highrough_highmetal | 25.87 | 24.26 -> 46.59 | 29.34 -> 29.34 | 26.84 -> 34.83 | 30.03 -> 37.40 | 28.03 -> 28.03 | 28.82 -> 37.89 | 29.60 -> 29.60 |

Family means (sphere + torus), v1 -> v2:
| family | Ours-Phys | Ours single-image | Fixed blend (Phys+single) | Ours-Fuse | DHAN-SHR (FT) | DHAN-SHR + learned gate | TSHRNet (FT) |
|---|---|---|---|---|---|---|---|
| lowrough_lowmetal | 43.54 -> 53.13 | 36.85 -> 36.85 | 37.47 -> 41.43 | 42.02 -> 57.32 | 38.64 -> 38.64 | 43.83 -> 53.48 | 37.33 -> 37.33 |
| lowrough_highmetal | 42.21 -> 51.41 | 35.96 -> 35.96 | 35.22 -> 39.61 | 40.15 -> 55.12 | 37.48 -> 37.48 | 41.22 -> 52.17 | 35.74 -> 35.74 |
| highrough_lowmetal | 30.91 -> 52.38 | 32.72 -> 32.72 | 32.22 -> 37.86 | 34.30 -> 50.19 | 31.45 -> 31.45 | 33.35 -> 50.28 | 31.68 -> 31.68 |
| highrough_highmetal | 24.82 -> 49.59 | 27.91 -> 27.91 | 27.77 -> 33.63 | 28.54 -> 43.65 | 27.00 -> 27.00 | 27.68 -> 44.42 | 27.81 -> 27.81 |

Counts v1: {'fuse_gt_single': 8, 'fuse_gt_phys': 5, 'phys_gt_single': 3, 'fuse_gt_fixed': 4, 'dhan_gate_gt_dhan': 8, 'fuse_gt_dhan': 8, 'n': 8}
Counts v2: {'fuse_gt_single': 8, 'fuse_gt_phys': 4, 'phys_gt_single': 8, 'fuse_gt_fixed': 8, 'dhan_gate_gt_dhan': 8, 'fuse_gt_dhan': 8, 'n': 8}

## 8. Prior-sensitivity study (re-optimisation from perturbed priors), held-out PSNR, v1 -> v2

| axis | seed | Phys | Fuse | single | gate mean |
|---|---|---|---|---|---|
| clean | 0 | 36.77 -> 56.82 | 36.18 -> 56.44 | 35.59 -> 35.59 | 0.31 -> 0.38 |
| clean | 1 | 37.97 -> 56.79 | 36.40 -> 56.68 | 35.59 -> 35.59 | 0.33 -> 0.38 |
| clean | 2 | 39.41 -> 56.80 | 36.31 -> 56.77 | 35.59 -> 35.59 | 0.33 -> 0.38 |
| geo_mild | 0 | 13.89 -> 18.82 | 35.57 -> 29.58 | 35.59 -> 35.59 | 0.03 -> 0.14 |
| geo_severe | 0 | 12.33 -> 14.33 | 35.43 -> 30.42 | 35.59 -> 35.59 | 0.03 -> 0.08 |
| albedo15 | 0 | 19.81 -> 56.78 | 35.73 -> 55.79 | 35.59 -> 35.59 | 0.15 -> 0.38 |
| albedo30 | 0 | 17.00 -> 56.68 | 35.46 -> 54.26 | 35.59 -> 35.59 | 0.06 -> 0.38 |
| light | 0 | 14.19 -> 16.02 | 34.94 -> 29.53 | 35.59 -> 35.59 | 0.03 -> 0.11 |
| drop25 | 0 | 15.40 -> 19.54 | 35.68 -> 29.57 | 35.59 -> 35.59 | 0.03 -> 0.16 |
| drop50 | 0 | 13.84 -> 15.55 | 35.39 -> 31.00 | 35.59 -> 35.59 | 0.02 -> 0.10 |
| sparse6 | 0 | 34.68 -> 53.54 | 35.99 -> 56.01 | 35.59 -> 35.59 | 0.30 -> 0.36 |
| sparse3 | 0 | 42.48 -> 41.25 | 36.12 -> 55.71 | 35.59 -> 35.59 | 0.30 -> 0.32 |

## 9. Ablation rows of scene_0070 (3 held-out views), v1 -> corrected recipe

| row | v1 PSNR / SSIM | corrected PSNR / SSIM | corrected run |
|---|---|---|---|
| Ours-Phys, strict protocol (oracle albedo init, paired D/S supervision, refiner on) | 36.82 / 0.9756 | 57.34 / 0.9996 | strict config with only photo_ssim=0 |
| World refiner (mask-confined), strict protocol | 26.87 / 0.9499 | 50.05 / 0.9978 | strict config with only photo_ssim=0 |
| Ours-Phys, albedo-free RGB-only (v2 main recipe) | 12.89 / 0.5537 | 56.51 / 0.9995 | configs/world/v2 (v1 counterpart = 'Inverse (no albedo)' row) |
| Inverse + single-image pseudo-diffuse bootstrap | 13.59 / 0.5879 | 53.08 / 0.9993 | v2 recipe with pseudo-diffuse albedo init |
| Ours-Fuse (full), v2 world field + v2 gate | 36.28 / 0.9865 | 56.11 / 0.9998 | v2 gate |
| Fusion with the strict-protocol physical branch (v2 gate) | 36.28 / 0.9865 | 56.35 / 0.9998 | strict config with only photo_ssim=0 |

Note: the 'refiner' entry of the v2 bootstrap and v2 world runs is meaningless (those runs have no refiner).

## 10. Exact initialisation (0 optimisation steps) and real-MVS diagnostic

- exact initialisation, physical: 53.87 dB (state is recipe-independent) -> 53.87 dB;  fused: 36.64 (v1 gate) -> 55.37 dB (v2 gate)
- optimisation from the exact state, held-out physical PSNR: v1 36.77 dB (drops by 17.1 dB), strict config with photo_ssim=0: 57.34 dB (gains 3.5 dB); v2 RGB-only from exact albedo: see 'clean' rows above

| real MVS (full-radiance NVS PSNR / SSIM) | v1 train | v1 holdout | SSIM removed, other settings unchanged: train | holdout |
|---|---|---|---|---|
| DTU scan1 | 7.04 / 0.305 | 7.43 / 0.329 | 7.49 / 0.388 | 7.88 / 0.415 |
| BlendedMVS 57f8d9bb | 9.62 / 0.065 | 9.48 / 0.059 | 16.62 / 0.200 | 16.24 / 0.199 |

