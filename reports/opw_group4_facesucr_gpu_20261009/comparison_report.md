# Group 4: FacesUCR official full TEST

All TRAIN gallery=200; all TEST queries=2050. Primary ACC@1; full-gallery MAP.
Frozen parameters, main Eq19 for FlashOPW; original journal inverse/prior and <P,D> for OPW.
Common marginal policy from TRAIN; no TEST tuning, query dropping or best-k selection.

## selected

| Metric | MAP % | ACC@1 % | ACC@3 % | ACC@5 % | ACC@7 % | ACC@15 % | ACC@30 % | Unconverged pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flash-opw | 69.946 | 92.244 | 92.244 | 91.317 | 90.537 | 85.902 | 69.171 | 5 |
| opw | 66.576 | 89.707 | 89.805 | 89.756 | 88.146 | 83.610 | 63.805 | 0 |
| tlp | 70.432 | 92.341 | 92.634 | 91.415 | 90.927 | 86.829 | 70.098 | 0 |
| opw-kl | 66.503 | 89.659 | 89.659 | 89.610 | 88.098 | 83.756 | 63.854 | 0 |
| sinkhorn | 42.307 | 60.732 | 61.415 | 62.146 | 62.000 | 56.927 | 49.220 | 0 |
| tcot | 45.470 | 63.951 | 65.561 | 65.268 | 63.951 | 58.829 | 49.512 | 0 |
| dtw | 62.164 | 90.488 | 89.512 | 87.610 | 86.244 | 80.293 | 66.195 | N/A |
| ldtw | 62.164 | 90.488 | 89.512 | 87.610 | 86.244 | 80.293 | 66.195 | N/A |
| ndtw | 58.112 | 87.805 | 86.780 | 84.341 | 82.634 | 75.805 | 62.195 | N/A |
| soft-dtw | 68.544 | 92.585 | 92.000 | 90.829 | 89.610 | 83.317 | 70.683 | N/A |
| ot | 42.261 | 60.488 | 61.268 | 61.902 | 61.805 | 56.683 | 48.927 | N/A |
## preset

| Metric | MAP % | ACC@1 % | ACC@3 % | ACC@5 % | ACC@7 % | ACC@15 % | ACC@30 % | Unconverged pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flash-opw | 53.098 | 77.122 | 75.122 | 73.902 | 72.146 | 64.293 | 52.634 | 2 |
| opw | 71.108 | 92.829 | 92.878 | 92.244 | 91.415 | 86.390 | 73.561 | 333 |
| tlp | 70.432 | 92.341 | 92.634 | 91.415 | 90.927 | 86.829 | 70.098 | 0 |
| opw-kl | 71.110 | 92.829 | 92.878 | 92.244 | 91.415 | 86.390 | 73.512 | 322 |
| sinkhorn | 42.651 | 61.073 | 61.610 | 62.439 | 61.854 | 57.756 | 48.927 | 0 |
| tcot | 40.498 | 49.415 | 50.732 | 49.171 | 49.561 | 46.976 | 42.927 | 0 |
| dtw | 62.164 | 90.488 | 89.512 | 87.610 | 86.244 | 80.293 | 66.195 | N/A |
| ldtw | 62.164 | 90.488 | 89.512 | 87.610 | 86.244 | 80.293 | 66.195 | N/A |
| ndtw | 58.112 | 87.805 | 86.780 | 84.341 | 82.634 | 75.805 | 62.195 | N/A |
| soft-dtw | 64.757 | 91.366 | 90.098 | 88.683 | 87.317 | 82.488 | 67.268 | N/A |
| ot | 42.261 | 60.488 | 61.268 | 61.902 | 61.805 | 56.683 | 48.927 | N/A |

## Paired query comparisons: FlashOPW minus baseline

| Profile | Baseline | ACC@1 difference pp | MAP difference pp (95% CI) | Exact McNemar p | Holm p |
|---|---|---:|---:|---:|---:|
| selected | opw | 2.537 | 3.369 (3.008, 3.729) | 3.2777e-07 | 1.63885e-06 |
| selected | tlp | -0.098 | -0.486 (-0.615, -0.357) | 0.850554 | 1 |
| selected | opw-kl | 2.585 | 3.442 (3.080, 3.809) | 2.16585e-07 | 1.29951e-06 |
| selected | sinkhorn | 31.512 | 27.639 (26.636, 28.654) | 2.13341e-163 | 1.92007e-162 |
| selected | tcot | 28.293 | 24.475 (23.534, 25.437) | 3.84637e-140 | 3.0771e-139 |
| selected | dtw | 1.756 | 7.782 (6.975, 8.594) | 0.00642306 | 0.0256922 |
| selected | ldtw | 1.756 | 7.782 (6.975, 8.594) | 0.00642306 | 0.0256922 |
| selected | ndtw | 4.439 | 11.834 (10.961, 12.719) | 1.31342e-10 | 9.19397e-10 |
| selected | soft-dtw | -0.341 | 1.402 (0.650, 2.185) | 0.642569 | 1 |
| selected | ot | 31.756 | 27.685 (26.682, 28.698) | 8.05134e-165 | 8.05134e-164 |
| preset | opw | -15.707 | -18.010 (-18.830, -17.191) | 2.67932e-69 | 2.41139e-68 |
| preset | tlp | -15.220 | -17.334 (-18.093, -16.562) | 3.43445e-66 | 2.40411e-65 |
| preset | opw-kl | -15.707 | -18.012 (-18.831, -17.193) | 2.67932e-69 | 2.41139e-68 |
| preset | sinkhorn | 16.049 | 10.446 (9.800, 11.099) | 2.12467e-56 | 1.06234e-55 |
| preset | tcot | 27.707 | 12.600 (11.925, 13.293) | 5.44756e-129 | 5.44756e-128 |
| preset | dtw | -13.366 | -9.066 (-9.980, -8.151) | 1.24463e-46 | 3.73389e-46 |
| preset | ldtw | -13.366 | -9.066 (-9.980, -8.151) | 1.24463e-46 | 3.73389e-46 |
| preset | ndtw | -10.683 | -5.014 (-5.956, -4.074) | 2.65541e-28 | 2.65541e-28 |
| preset | soft-dtw | -14.244 | -11.659 (-12.561, -10.766) | 6.54014e-54 | 2.61606e-53 |
| preset | ot | 16.634 | 10.837 (10.184, 11.504) | 1.83617e-59 | 1.1017e-58 |

MAP CIs are unadjusted paired percentile query bootstrap intervals conditional on the fixed gallery/selection.
McNemar is two-sided exact at predeclared k=1; Holm correction covers all requested baselines within each profile.
Any unconverged entropic pairs make that method's quality comparison provisional; no post-TEST cap changes.
The artifact chooses one parameter set per metric jointly over three TRAIN splits; query CIs do not estimate retraining variability.
Recorded solve wall times include conversion, diagnostics and compilation; this is not the group5 speed benchmark.
