# Group 3: TRAIN validation only

ACC@1 is primary; MAP breaks ties; means and sample SD across three splits.
No TEST data or speed comparison. Presets use repo journal parameters with
the same residual policy as tuning, rather than journal fixed iteration counts.

## Preset parameters

| Metric | Candidate | ACC@1 % (SD) | MAP % (SD) | Capped pairs |
|---|---:|---:|---:|---:|
| flash-opw | 0 | 58.333 (5.455) | 58.464 (5.594) | 0 |
| dtw | 0 | 67.857 (7.143) | 65.688 (7.305) | 0 |
| ldtw | 0 | 67.857 (7.143) | 65.688 (7.305) | 0 |
| ndtw | 0 | 64.286 (6.186) | 61.321 (6.824) | 0 |
| soft-dtw | 0 | 72.619 (7.435) | 69.106 (7.646) | 0 |
| ot | 0 | 41.667 (4.124) | 47.484 (3.464) | 0 |
| sinkhorn | 0 | 39.286 (3.571) | 47.527 (3.019) | 0 |
| tlp | 0 | 80.952 (12.542) | 75.130 (8.109) | 0 |
| opw-kl | 0 | 77.381 (12.542) | 72.899 (8.192) | 1 |
| tcot | 0 | 35.714 (6.186) | 44.922 (4.825) | 0 |
| opw | 0 | 77.381 (12.542) | 72.899 (8.192) | 1 |

## All metrics tuned on TRAIN

| Metric | Candidate | ACC@1 % (SD) | MAP % (SD) | Capped pairs |
|---|---:|---:|---:|---:|
| flash-opw | 1 | 79.762 (14.434) | 74.405 (7.329) | 0 |
| dtw | 0 | 67.857 (7.143) | 65.688 (7.305) | 0 |
| ldtw | 0 | 67.857 (7.143) | 65.688 (7.305) | 0 |
| ndtw | 0 | 64.286 (6.186) | 61.321 (6.824) | 0 |
| soft-dtw | 4 | 78.571 (7.143) | 72.956 (5.695) | 0 |
| ot | 0 | 41.667 (4.124) | 47.484 (3.464) | 0 |
| sinkhorn | 3 | 41.667 (2.062) | 47.268 (4.137) | 0 |
| tlp | 0 | 80.952 (12.542) | 75.130 (8.109) | 0 |
| opw-kl | 4 | 76.190 (13.521) | 71.376 (7.876) | 0 |
| tcot | 3 | 46.429 (9.449) | 49.580 (5.957) | 0 |
| opw | 5 | 76.190 (13.521) | 71.397 (7.897) | 0 |
