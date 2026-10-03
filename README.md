# FlashSinkhorn / FlashOPW CPU reference

This workspace contains a CPU-only research implementation derived from the
three local papers:

* `FlashSinkhorn.pdf`: tiled online-logsumexp Sinkhorn;
* `main (2).pdf`: affine/Taylor reduction of OPW to an extra feature dimension;
* `OWD_journal.pdf`: exact OPW objective and NM/k-NN evaluation protocol.

## Install

The current environment already provides PyTorch, NumPy, pandas, matplotlib
and psutil.  From the repository root, run:

```powershell
python -m pytest -q
```

## 1. FlashSinkhorn toy benchmark

```powershell
python experiments/toy_flashsinkhorn.py --sizes 32,64,128,256
```

This compares the CPU tiled implementation with dense PyTorch Sinkhorn and,
when installed, GeomLoss.  It writes runtime/memory CSV data and a PNG plot to
`outputs/`.

Optional paper baselines:

```powershell
pip install geomloss ott-jax
```

The code reports missing optional backends rather than silently substituting a
different algorithm.

## 2. OPW / FlashOPW experiment

```powershell
python experiments/opw_experiment.py
```

Outputs include:

* `opw_solver_benchmark.csv/.png`: exact OPW versus affine FlashOPW runtime
  and approximate peak RSS;
* `opw_accuracy_report.csv`: spatial-distance, transport and marginal errors;
* `owd_style_classification.csv/.png`: NM accuracy/MAP, k-NN accuracy for
  k=1,3,5,7,15,30, and k-NN MAP on a reproducible toy sequence dataset.

## Important interpretation

`OPW-exact` uses the full effective cost

`D - lambda1/(1+F) + lambda2*(F/(2*sigma^2) + log(sigma*sqrt(2*pi)))`.

`FlashOPW-affine` uses `D + mu*F`, where

`mu = lambda1 + lambda2/(2*sigma^2)`.

Therefore FlashOPW is an approximation to the original OPW objective.  The
solver itself is Flash-style and CPU-tiled; it is not a Triton GPU kernel.

