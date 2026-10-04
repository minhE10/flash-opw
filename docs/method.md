# Method and implementation map

Reference: Ye et al., **FlashSinkhorn: IO-Aware Entropic Optimal Transport on GPU**,
[arXiv:2602.03067v3](https://arxiv.org/html/2602.03067v3),
[ICML/PMLR page](https://proceedings.mlr.press/v306/ye26l.html).
This is an independent implementation, not a vendored or renamed official package.

## Conventions

Positive probability weights `a,b`; cost `C_ij = s ||x_i-y_j||²`, where `s=1`
by default (`s=0.5` is supported). Objective:

```text
min_P  <C,P> + eps * sum_ij [P_ij log(P_ij/(a_i b_j)) - P_ij + a_i b_j]
subject to P 1 = a, P^T 1 = b.
```

The dense reference uses ordinary potentials and direct `torch.cdist` distances:

```text
f <- -eps logsumexp_j(log b_j + (g_j-C_ij)/eps)
g <- -eps logsumexp_i(log a_i + (f_i-C_ij)/eps)
```

The GPU implementation stores fused, dimensionless shifted potentials:

```text
u = (f - s||x||²)/eps + log a
v = (g - s||y||²)/eps + log b
S_ij = 2s x_i.y_j/eps + v_j
u_i <- log a_i - logsumexp_j(S_ij)
v_j <- log b_j - logsumexp_i(2s y_j.x_i/eps + u_i)
log P_ij = 2s x_i.y_j/eps + u_i + v_j
```

Initialization corresponds to `f=g=0`, **not** `u=v=0`. Alternating updates
consume the newly computed `u`. The symmetric schedule uses both old potentials
and averages each old/new update with factor 1/2; writes go to separate buffers.

## Kernels

| Paper component | Implementation |
| --- | --- |
| Algorithm 1, alternating streamed LSE half-steps | `flashopw/triton_kernels.py::_update_kernel` |
| Algorithm 3, one-launch symmetric update | `flashopw/triton_kernels.py::_symmetric_update_kernel` |
| Algorithm 2, streamed `P V` | `flashopw/triton_kernels.py::_apply_kernel` |
| Adjoint `P^T V` | Same kernel, swapping source/target and potentials |
| Theorem 5 Hadamard transport | `flashopw/triton_kernels.py::_hadamard_apply_kernel` |
| Point gradients at convergence | `flashopw/transport.py::point_gradients` |
| Schur-CG HVP, equations (25)-(31) | `flashopw/differentiation.py::hessian_vector_product` |
| Analytic autograd wrapper | `flashopw/differentiation.py::sinkhorn_cost` |
| Optional KeOps HVP transport oracle | `flashopw/transport.py` (`backend="keops"`) |

Each query tile remains resident while key tiles stream. The running row maximum
and rescaled exponential sum prevent exponent overflow. We fold the marginal
logarithm into the potentials so each key tile loads a single bias. `tl.dot`
uses FP32 storage/accumulators; `tf32` is the paper-compatible forward/backward
default, `tf32x3` is the higher-accuracy option, and `ieee` is used for strict
FP32 checks and HVP. See [Triton's dot precision documentation](https://triton-lang.org/main/python-api/generated/triton.language.dot.html).

Tile arguments are upper bounds. For d <= 64, launches cap tiles at 32x64 with
two stages; for d > 64 they cap at 16x32 with one stage. This controls the extra
shared buffers required by high-accuracy dot products on consumer GPUs. Unlike
the paper's A100 run, this RTX-oriented implementation does not autotune tiles.
Offline compilation checks a 64 KiB shared-memory budget for the covered launch
configurations; this is separate from the global VRAM cap. Effective tiles are
recorded in results.

Transport application keeps an online rescaled weighted sum and restores its
scale using `exp(u + running_max)`. This includes the **actual row mass** even
before convergence. Simply returning `a * softmax(S) V` would be incorrect for
an unconverged iterate.

GPU storage is O((n+m)d + (n+m) + np) for inputs, potentials and a transport
output with p channels; no cost, kernel, score or transport matrix is written
to global memory. Computation remains quadratic in point count. Calling the
explicit diagnostic helper `materialize_plan` deliberately loses linear memory.

`point_gradients` returns `2s(r*x - P@y)` and `2s(c*y - P.T@x)`, using actual
masses r,c. These become the OT envelope gradients at convergence. They do not
differentiate through a fixed number of solver iterations.

`hessian_vector_product` implements the paper's implicit/explicit split. It
forms `R A`, solves the damped Schur complement with CG, applies `R^T`, and
adds `E A`. The coupling is accessed through streamed transport products and
one fused Hadamard-weighted transport; HVP work uses strict IEEE precision.
`sinkhorn_cost` exposes this as analytic backward and x-double-backward without
retaining the Sinkhorn iteration graph.

## Supported scope and limits

- Balanced OT, strictly positive weights summing to one, rectangular clouds,
  squared Euclidean cost, constant epsilon, alternating/symmetric updates.
- Triton: CUDA float32, feature dimensions 1..1024, unbatched clouds. Feature
  dimensions above 128 are accumulated in 128-wide dot-product chunks to keep
  register and shared-memory use bounded.
- Dense and tiled Torch oracle: CPU/CUDA float32/float64.
- Early stopping checks both marginal L1 residuals; benchmarks use fixed
  iterations and separately report convergence, so equal work is compared.
- HVP/double backward is implemented for the source support `x` with `y` held
  fixed. Double backward with respect to `y`, unbalanced OT, epsilon annealing,
  distributed execution and the paper's downstream tasks are not implemented.
- Input coordinates should be reasonably scaled and centered. Large shared
  offsets or very small epsilon can cause cancellation in shifted FP32
  potentials; test against float64 and increase precision/iterations as needed.

The purpose is a readable implementation of the FlashSinkhorn solver plus
toy comparisons, not a claim of reproducing the paper's A100 speedup numbers
on RTX 5080. Small 2D cases can be launch-bound and need not be faster.
