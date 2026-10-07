# FlashOPW (main) versus original OPW (journal)

Frozen three-split TRAIN pilot; original method definitions and native scores.
Journal inverse penalty is exact; its perpendicular Gaussian prior is preserved.
No sigma conversion, retuning, TEST evaluation or speed benchmark.

| Profile | Mode | Method | ACC@1 % (SD) | MAP % (SD) | Infeasible pairs |
|---|---|---|---:|---:|---:|
| preset | residual_stop | flash-opw | 58.333 (5.455) | 58.464 (5.594) | 0 |
| preset | residual_stop | opw | 77.381 (12.542) | 72.899 (8.192) | 1 |
| selected | residual_stop | flash-opw | 79.762 (14.434) | 74.405 (7.329) | 0 |
| selected | residual_stop | opw | 76.190 (13.521) | 71.397 (7.897) | 0 |

Infeasible pairs exceed the common marginal L1 tolerance; retain them visibly.
Selected candidates were frozen separately for each metric before this check.
The scores have different definitions: raw score differences/ratios are not parity errors.
Prior units, Taylor approximation, epsilon, parameters and score can all change rankings.
Paired query outcomes and NN-index agreement are in summary.json; agreement is not required.
TRAIN estimates after tuning do not establish superiority on the official TEST split.
