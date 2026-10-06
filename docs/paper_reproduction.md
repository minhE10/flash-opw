# Paper reproduction and optimization checks

The mathematical data/settings remain uniform `[0,1]^d`, uniform weights,
full squared Euclidean cost, epsilon 0.1, 10 forward/backward iterations,
100 untimed HVP solve iterations and 50 CG steps with absolute Schur damping
1e-5. The measurement defaults remain 10 warmups and 50/30/20 repetitions.
No precision or accuracy threshold was lowered to improve a ratio.

References: [paper v3](https://arxiv.org/html/2602.03067v3),
[FlashSinkhorn source](https://github.com/ot-triton-lab/flash-sinkhorn),
[OTT-Hessian source](https://github.com/yexf308/OTT-Hessian).
The source checkouts are not established as the exact revisions used to
produce the paper tables. RTX 5080 results cannot establish A100 runtimes.

## Implemented changes

- A dedicated vector transport kernel reduces `weights * vector` directly,
  with online exponential rescaling. It handles signed vectors, transpose,
  partial tiles and single-column matrices without a padded matrix output.
- A source/target gradient kernel computes each score tile once and reuses
  it across all output feature blocks. It uses actual induced masses,
  including for unconverged potentials. Its feature accumulators can increase
  register pressure; offline compilation alone does not prove an improvement.
- Opt-in tuning searches a bounded set of row/key tiles, stages and warps,
  with unchanged dot-product precision. Compiler shared-memory metadata gates
  candidates before launch. Selected configurations, timings, registers and
  spills are recorded in `autotuning.json`; selection is cached in-process.
  Default fixed tiles and the generic transport/gradient paths remain available.
- Fixed-step CG no longer transfers residual diagnostics to the CPU when the
  caller does not request `return_info`. Diagnostics remain available outside
  measured sections.
- TF32x3 transport at small feature dimensions uses a 32-column output tile
  to fit the 64 KiB shared-memory limit. This changes tiling, not precision.

## OTT-Hessian baseline

`scripts/setup_ott_hessian.sh` clones the external baseline into ignored
`outputs/third_party/OTT-Hessian` at revision
`7eb189fe39982f587da935044480655b65939637`. No third-party code is vendored.
The adapter checks the normalized source checksum and refuses a different file.
The adapter calls its Lineax `HessianA` implementation (called
`HessianALineax` in the FlashSinkhorn reference tests).

The baseline is configured with the same shared coupling as Flash/KeOps.
OTT potentials include marginal logs: `f_OTT = epsilon*u + ||x||^2`,
`g_OTT = epsilon*v + ||y||^2`. Passing `result.f/g` directly would omit those
logs and compute a different plan. Geometry explicitly uses squared Euclidean
cost with scaling 1.0, and points/potentials stay dynamic JIT arguments.

Upstream adds `epsilon*tau2` to the Schur diagonal. The adapter passes
`tau2=damping/epsilon` to preserve the existing benchmark's absolute damping
1e-5. The adapter uses a guarded JAX CG with the same fixed-step arithmetic
and device-side breakdown conditions as Flash/KeOps for the 50-step budget.
The source's HessianA formulas and OTT transport geometry remain unchanged.
Upstream defaults instead use Lineax tolerances 1e-6 and may stop early;
forcing its tolerances to zero produced NaN in a uniform d=65 CPU fixture
after near-convergence. The fixed-step adapter avoids that failure without
changing the iteration budget or damping. It does not apply Lineax's residual
stabilization. This CG substitution is recorded in the legend, row detail,
environment and source provenance. Exact raw Lineax performance and exact
paper-table reproduction are not claimed. `load_hessian` also supports
positive tolerances when an upstream Lineax call is needed separately.

Missing dependencies/source produce a baseline failure. There is no automatic
fallback. `--jax-hvp-backend matrix-free` explicitly selects the previous
custom JAX implementation, and the plots label it accordingly. Neither baseline
is `linearize(grad(loss))`. The setup script downloads source only; benchmark
dependencies come from the isolated environment's `baselines` extra.

The custom matrix-free implementation evaluates source-by-target score tiles
in the same orientation and with the same FP32 rounding order for `P` and
`P.T`. Explicit highest matmul precision and
[JAX optimization barriers](https://docs.jax.dev/en/latest/_autosummary/jax.lax.optimization_barrier.html)
preserve these rounding points under JIT. Its CG uses the guarded fixed-step routine
shared with the OTT-Hessian adapter. These changes can affect custom JAX
timings; rerun that backend rather than mixing its old and new results.

## Server sequence

Pull the committed changes on the server before installing dependencies:

```bash
conda activate minh
cd ~/minh.nd/flash-opw
git pull --ff-only origin main
python -m pip install -e '.[dev,plots,baselines]'
bash scripts/setup_ott_hessian.sh
```

If you already applied the transfer patch, `git pull` may report conflicting
local changes. Keep those changes and inspect `git status` before proceeding;
do not apply the patch again or discard local work.

Run real GPU correctness before the long benchmark:

```bash
bash scripts/validate_paper_optimizations.sh 1 --cases 10000:64 20000:1024 --profile
```

GPU tests cover updates across dimensions/precisions/schedules, signed vector
transport/transpose, gradients with nonuniform and unconverged masses, wide
Hadamard transport, tuned kernels and a complete fixed-step HVP. JAX tests
exercise the actual external baseline, including its potential conversion,
damping conversion and the guarded fixed 50-step protocol. `--require-gpu` prevents
an all-skipped GPU run from being accepted as validation.

The ablation runner compares `generic`, `specialized`, `tuned` in this same
revision, keeping tensors and cached HVP potentials identical. Forward+backward
includes 10 solve iterations for each mode. It reports all samples, dispersion,
relative L2 differences, actual marginal residuals and CG information. It exits
nonzero if the output comparison fails. Its baseline is the retained generic
transport route, not an independently timed checkout of an earlier commit.
HVP is omitted from dimensions above 128 by default; set `--hvp-max-d` to expand.
Profiler traces are separate untimed calls for the first case and can be viewed
in a Chrome/Perfetto trace viewer. `profile_*.txt` lists operator/device time;
traces help distinguish launches, CPU synchronization and kernel cost. Compiler
register/spill records help investigate gradient register pressure.

Then run the full protocol:

```bash
bash scripts/run_paper_benchmarks.sh 1 --autotune --block-m 64 --block-n 128 --diagnostics
```

Outputs include CSV/JSON, eight panels plus overview, environment and all samples,
`autotuning.json`, and `diagnostics.json`. Diagnostics are outside timing and
the environment records source hashes and GPU driver/clocks/temperature when
`nvidia-smi` is available. Nonfinite operation outputs keep their timing row
but are marked `nonfinite_output` and excluded from plots. Diagnostics
include Flash coupling residuals, Flash/KeOps CG information and JAX OTT-Hessian
HVP comparison against Flash using the same coupling. Diagnostic failures are
recorded separately from timing status. Forward/backward baseline parity is not
established just by these diagnostics. Ten fixed solve iterations need not
converge, so a residual is evidence to inspect rather than a reason to silently
increase the iteration budget for one method. Memory timing warms up first,
then resets the allocator peak, excluding compilation/tuning scratch.

Keep failed/OOM/estimated-memory-skipped rows. Separate runs with the custom
JAX implementation from OTT-Hessian runs. Tensorized still precomputes forward
cost outside timing. Compare paper ratios with their correct Flash denominator,
and never mix protocols or the short/full runs of the original log.

## Author code on the current experiments

The author comparison changes only the Flash implementation. It imports the
official `ot-triton-lab/flash-sinkhorn` checkout at
`75d48cc42d2efe8d4f654d91152ccf6f857c993f` directly from `torch-ext`, without
installing its similarly named distribution over this repository. This is a
pinned released source revision, not a verified identification of the exact
commit that produced every paper table. Tracked source modifications and a
different revision are rejected; module origin and all Python source hashes
are recorded in `environment.json`.

On the allocated GPU 1 in the existing `minh` environment:

```bash
git pull --ff-only origin main
conda activate minh
bash scripts/setup_author_reference.sh
bash scripts/setup_ott_hessian.sh
CUDA_VISIBLE_DEVICES=1 \
  FLASHOPW_AUTHOR_PATH=outputs/third_party/flash-sinkhorn-author \
  python -m pytest -q --require-gpu tests/test_author_reference.py
```

The author GPU checks exercise symmetric/alternating forward and source
gradient at d=3,65,1024 against an independent direct-distance FP64 oracle,
and raw author HVP at 12/50 CG iterations against a dense shared coupling.
If a check fails, retain its output and investigate before the long run;
the checks do not substitute local kernels or patch the author's CG.

After those checks pass, run the same eight panels and controls as the current
full experiment (uniform dataset, same per-case seed, epsilon 0.1, source-only
backward, 10/50/30/20 warmup/forward/backward/HVP repetitions):

```bash
bash scripts/run_author_benchmarks.sh 1 \
  --autotune --block-m 64 --block-n 128 --diagnostics \
  --output outputs/author_paper_20261006
python -m experiments.compare_author_runs \
  outputs/paper_20261005T090633.880402Z outputs/author_paper_20261006
```

Use a fresh `--output` directory if that name already exists. If your original
full run used different overrides, copy those exact overrides to the author
run. The comparison checks shared settings, available shared source hashes,
GPU and package versions before joining cases; it rejects mismatched runs.
`comparison_with_local.csv` contains both means/statuses and `local_over_author`:
a value above 1 means the author is faster for ms or uses less memory for MB.
Missing, failed, nonfinite and estimated-memory-skipped measurements have no
ratio. Use `diagnostics.json` for numerical differences, not just timings.

Forward/backward use the original author's `SamplesLoss`, fixed `n_iters=10`,
no epsilon scaling/debiasing/normalization/extrapolation and full squared cost.
Its symmetric solver retains a full initialization step before the 10 damped
updates; this differs from the local solver's 10 updates from zero. Its source
gradient uses target marginal `a` and conditional row means, which can differ
from the local gradient using actual masses after a finite solve. These are
original source behaviors, recorded rather than silently changed for parity.
`--diagnostics` reports forward/gradient relative L2 versus local outside timing.
Author tiles and autotune candidates remain those in the original code;
`--block-m/--block-n` control local shared HVP setup, not author solver tiles.

HVPs use exactly the current benchmark's cached local IEEE coupling (100 solve
iterations outside timing), converted to the author's OTT potentials. The
timed operation calls original author transport/HVP and original Python CG,
with no preconditioner, IEEE matmul, `max_cg_iter=50`, `rtol=atol=0` and
`tau2=1e-5/epsilon` to retain absolute Schur damping `1e-5`. Its original
exact-zero/breakdown exits and CPU synchronizations remain. Reported author
CG fields and HVP relative L2 versus local are saved outside timing.

KeOps, Tensorized and JAX retain the current harness implementations, including
the guarded OTT-Hessian JAX CG adapter. This run isolates **author Flash versus
local Flash on the current protocol**; it is not a run of every author's native
benchmark script and baseline unchanged. Native scripts generate different
data by default. Original source failures on RTX 5080 are recorded as failures,
without falling back to a local implementation.

## Local developer checks

```bash
python -m pytest -q
python scripts/interpret_kernels.py
python scripts/compile_kernels.py --arch 120
python scripts/compile_tuning.py --arch 120
```

The last three commands require Linux/WSL Triton. Interpreter and offline
compilation check masks, recurrence logic and compiler/resource acceptance.
They are not GPU correctness or measured speedups. See
[validation record](validation.md) for the checks actually performed.
