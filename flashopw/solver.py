"""Forward solvers for <C,P> + epsilon * KL(P | a tensor b).

C_ij = cost_scale * ||x_i-y_j||^2. Both marginals are positive probabilities.
The public API is intentionally forward-only; see point_gradients for the
envelope gradient at convergence. No graph through the iterations is retained.
"""

from dataclasses import dataclass
import math

import torch


@dataclass
class SinkhornResult:
    x: torch.Tensor
    y: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    epsilon: float
    cost_scale: float
    n_iters: int
    backend: str
    precision: str = "ieee"
    block_m: int = 32
    block_n: int = 64
    # log P_ij = 2 * cost_scale * x_i.y_j / epsilon + u_i + v_j.

    @property
    def f(self):
        return self.epsilon * (self.u - self.a.log()) + self.cost_scale * self.x.square().sum(1)

    @property
    def g(self):
        return self.epsilon * (self.v - self.b.log()) + self.cost_scale * self.y.square().sum(1)


def _prepare(x, y, a, b, epsilon, cost_scale, n_iters, schedule, tol, check_every):
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have shapes (n,d) and (m,d)")
    if min(*x.shape, *y.shape) < 1:
        raise ValueError("Empty point clouds/features are unsupported")
    if x.dtype not in (torch.float32, torch.float64) or y.dtype != x.dtype:
        raise ValueError("x and y must share float32 or float64 dtype")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError("cost_scale must be finite and positive")
    if not isinstance(n_iters, int) or n_iters < 1:
        raise ValueError("n_iters must be a positive integer")
    if not isinstance(check_every, int) or check_every < 1:
        raise ValueError("check_every must be a positive integer")
    if schedule not in ("alternating", "symmetric"):
        raise ValueError("schedule must be alternating or symmetric")
    if tol is not None and (not math.isfinite(tol) or tol <= 0):
        raise ValueError("tol must be None or finite and positive")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("Coordinates must be finite")

    def weights(w, n):
        if w is None:
            return x.new_full((n,), 1.0 / n)
        if w.shape != (n,) or w.device != x.device or w.dtype != x.dtype:
            raise ValueError("Weights must match shape, device and dtype of points")
        if not bool(torch.isfinite(w).all()) or not bool((w > 0).all()):
            raise ValueError("Weights must be finite and strictly positive; remove zero-weight points")
        if not torch.isclose(w.sum(), w.new_tensor(1.0), atol=1e-6, rtol=1e-6):
            raise ValueError("Weights must sum to 1 (no implicit normalization)")
        return w.detach().contiguous()

    return x.detach().contiguous(), y.detach().contiguous(), weights(a, len(x)), weights(b, len(y))


def _early_stop(result, tol):
    from .transport import apply_plan

    rows = apply_plan(result, torch.ones_like(result.b))
    cols = apply_plan(result, torch.ones_like(result.a), transpose=True)
    error = torch.maximum((rows - result.a).abs().sum(), (cols - result.b).abs().sum())
    return bool(error <= tol)


@torch.no_grad()
def sinkhorn_dense(x, y, *, a=None, b=None, epsilon=0.2, cost_scale=1.0,
                   n_iters=200, schedule="alternating", tol=None, check_every=20):
    """Standard stabilized Sinkhorn, materializing the n-by-m cost matrix.

    This independently implements unshifted log-domain updates, not the Triton
    recurrence. tol checks the maximum of the two marginal L1 residuals.
    """
    x, y, a, b = _prepare(x, y, a, b, epsilon, cost_scale, n_iters, schedule, tol, check_every)
    # Direct distance evaluation is an independent check on dot-product costs.
    cost = cost_scale * torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist").square()
    loga, logb = a.log(), b.log()
    f, g = torch.zeros_like(a), torch.zeros_like(b)
    xnorm, ynorm = cost_scale * x.square().sum(1), cost_scale * y.square().sum(1)
    result = SinkhornResult(x, y, a, b, f, g, epsilon, cost_scale, 0, "dense")
    for iteration in range(1, n_iters + 1):
        next_f = -epsilon * torch.logsumexp((g[None, :] - cost) / epsilon + logb[None, :], dim=1)
        for_g = next_f if schedule == "alternating" else f
        next_g = -epsilon * torch.logsumexp((for_g[:, None] - cost) / epsilon + loga[:, None], dim=0)
        if schedule == "symmetric":
            f, g = 0.5 * (f + next_f), 0.5 * (g + next_g)
        else:
            f, g = next_f, next_g
        if tol is not None and (iteration % check_every == 0 or iteration == n_iters):
            result.u, result.v = (f - xnorm) / epsilon + loga, (g - ynorm) / epsilon + logb
            result.n_iters = iteration
            if _early_stop(result, tol):
                break
    result.u, result.v = (f - xnorm) / epsilon + loga, (g - ynorm) / epsilon + logb
    result.n_iters = iteration
    return result


def _online_update(q, k, bias, logweights, scale, block_m, block_n):
    """Torch tiled mathematical oracle, NOT a GPU-performance substitute."""
    output = torch.empty_like(logweights)
    for start in range(0, len(q), block_m):
        qi = q[start:start + block_m]
        running = qi.new_full((len(qi),), -float("inf"))
        total = qi.new_zeros(len(qi))
        for j in range(0, len(k), block_n):
            scores = scale * (qi @ k[j:j + block_n].T) + bias[j:j + block_n]
            new_max = torch.maximum(running, scores.amax(1))
            total = total * torch.exp(running - new_max) + torch.exp(scores - new_max[:, None]).sum(1)
            running = new_max
        output[start:start + len(qi)] = logweights[start:start + len(qi)] - running - total.log()
    return output


@torch.no_grad()
def _sinkhorn_streaming(x, y, *, a, b, epsilon, cost_scale, n_iters, schedule,
                        tol, check_every, backend, precision, block_m, block_n):
    x, y, a, b = _prepare(x, y, a, b, epsilon, cost_scale, n_iters, schedule, tol, check_every)
    if not isinstance(block_m, int) or not isinstance(block_n, int) or min(block_m, block_n) < 1:
        raise ValueError("Tile sizes must be positive integers")
    if backend == "triton":
        if x.device.type != "cuda" or x.dtype != torch.float32:
            raise ValueError("FlashSinkhorn requires CUDA float32 inputs; use sinkhorn_online for a CPU oracle")
        if x.shape[1] > 1024:
            raise ValueError("This implementation supports feature dimensions 1..1024")
        if block_m not in (16, 32, 64) or block_n not in (32, 64, 128):
            raise ValueError("Triton tiles: block_m in {16,32,64}, block_n in {32,64,128}")
        if precision not in ("ieee", "tf32x3", "tf32"):
            raise ValueError("precision must be ieee, tf32x3 or tf32")
        try:
            from .triton_kernels import launch_config, update
        except ImportError as exc:
            raise RuntimeError("Triton is required: run in the server's Linux PyTorch CUDA environment") from exc
        block_m, block_n, _ = launch_config(x.shape[1], block_m, block_n)
    loga, logb = a.log(), b.log()
    u = loga - (cost_scale / epsilon) * x.square().sum(1)
    v = logb - (cost_scale / epsilon) * y.square().sum(1)
    unew, vnew = torch.empty_like(u), torch.empty_like(v)
    scale = 2.0 * cost_scale / epsilon
    result = SinkhornResult(x, y, a, b, u, v, epsilon, cost_scale, 0, backend, precision, block_m, block_n)
    for iteration in range(1, n_iters + 1):
        if backend == "triton":
            update(x, y, u, v, loga, unew, scale, schedule == "symmetric", precision, block_m, block_n)
            update(y, x, v, unew if schedule == "alternating" else u, logb, vnew,
                   scale, schedule == "symmetric", precision, block_m, block_n)
            u, unew, v, vnew = unew, u, vnew, v
        else:
            next_u = _online_update(x, y, v, loga, scale, block_m, block_n)
            next_v = _online_update(y, x, next_u if schedule == "alternating" else u,
                                    logb, scale, block_m, block_n)
            if schedule == "symmetric":
                u, v = 0.5 * (u + next_u), 0.5 * (v + next_v)
            else:
                u, v = next_u, next_v
        result.u, result.v, result.n_iters = u, v, iteration
        if tol is not None and (iteration % check_every == 0 or iteration == n_iters):
            if _early_stop(result, tol):
                break
    return result


def sinkhorn_flash(x, y, *, a=None, b=None, epsilon=0.2, cost_scale=1.0,
                   n_iters=200, schedule="alternating", tol=None, check_every=20,
                   precision="tf32x3", block_m=32, block_n=64):
    """Fused row-stationary Triton Sinkhorn; O((n+m)d) global memory.

    Defaults use FP32 storage/accumulation and tf32x3 dot products. ieee is the
    strict FP32 diagnostic mode. Tiles are upper bounds and are reduced for
    large feature dimensions. No autotuning or multi-GPU work is launched.
    """
    return _sinkhorn_streaming(x, y, a=a, b=b, epsilon=epsilon, cost_scale=cost_scale,
        n_iters=n_iters, schedule=schedule, tol=tol, check_every=check_every,
        backend="triton", precision=precision, block_m=block_m, block_n=block_n)


def sinkhorn_online(x, y, *, a=None, b=None, epsilon=0.2, cost_scale=1.0,
                    n_iters=200, schedule="alternating", tol=None, check_every=20,
                    block_m=32, block_n=64):
    """Portable Torch oracle for the streamed equations, also usable on CPU."""
    return _sinkhorn_streaming(x, y, a=a, b=b, epsilon=epsilon, cost_scale=cost_scale,
        n_iters=n_iters, schedule=schedule, tol=tol, check_every=check_every,
        backend="online", precision="ieee", block_m=block_m, block_n=block_n)
