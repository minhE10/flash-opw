"""CPU reference implementations of dense and Flash-style Sinkhorn.

The implementation follows the algebra used by FlashSinkhorn, but replaces
the Triton kernel with explicit PyTorch CPU tile loops.  It is intentionally
small and readable so that it can be used as a correctness reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SinkhornResult:
    """Result returned by a Sinkhorn solve."""

    distance: torch.Tensor
    f: torch.Tensor
    g: torch.Tensor
    transport: Optional[torch.Tensor]
    iterations: int
    marginal_error: float


def _prepare_weights(w: Optional[torch.Tensor], n: int, dtype: torch.dtype) -> torch.Tensor:
    if w is None:
        w = torch.full((n,), 1.0 / n, dtype=dtype)
    else:
        w = torch.as_tensor(w, dtype=dtype).flatten()
        if w.numel() != n:
            raise ValueError(f"weight length {w.numel()} does not match {n}")
        if torch.any(w <= 0):
            raise ValueError("Sinkhorn weights must be strictly positive")
        w = w / w.sum()
    return w


def _check_inputs(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must have shape [n, d] and [m, d]")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension")
    if not (x.device.type == "cpu" and y.device.type == "cpu"):
        raise ValueError("This reference implementation is CPU-only")
    if not x.is_floating_point():
        x = x.float()
    if not y.is_floating_point():
        y = y.float()
    return x, y


def _online_lse_dot(
    q: torch.Tensor,
    k: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
    row_block: int,
    col_block: int,
) -> torch.Tensor:
    """Compute row-wise logsumexp((q @ k.T + bias) / epsilon) by tiles."""

    n = q.shape[0]
    out = torch.empty(n, dtype=q.dtype)
    for i0 in range(0, n, row_block):
        i1 = min(i0 + row_block, n)
        qi = q[i0:i1]
        running_m = torch.full((i1 - i0,), -torch.inf, dtype=q.dtype)
        running_s = torch.zeros((i1 - i0,), dtype=q.dtype)
        for j0 in range(0, k.shape[0], col_block):
            j1 = min(j0 + col_block, k.shape[0])
            logits = (qi @ k[j0:j1].T + bias[j0:j1]) / epsilon
            tile_m = logits.max(dim=1).values
            new_m = torch.maximum(running_m, tile_m)
            running_s = (
                torch.exp(running_m - new_m) * running_s
                + torch.exp(logits - new_m[:, None]).sum(dim=1)
            )
            running_m = new_m
        out[i0:i1] = running_m + torch.log(running_s)
    return out


def _squared_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.square().sum(dim=1)[:, None] + y.square().sum(dim=1)[None, :] - 2 * x @ y.T).clamp_min(0)


def _stream_squared_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float,
    row_block: int,
    col_block: int,
) -> torch.Tensor:
    """Evaluate <P, ||x-y||^2> without materializing P."""

    total = torch.zeros((), dtype=x.dtype)
    for i0 in range(0, x.shape[0], row_block):
        i1 = min(i0 + row_block, x.shape[0])
        for j0 in range(0, y.shape[0], col_block):
            j1 = min(j0 + col_block, y.shape[0])
            c = _squared_cost(x[i0:i1], y[j0:j1])
            score = (2 * x[i0:i1] @ y[j0:j1].T + f_hat[i0:i1, None] + g_hat[j0:j1]) / epsilon
            p = torch.exp(score) * a[i0:i1, None] * b[j0:j1]
            total = total + (p * c).sum()
    return total


def _transport_from_shifted(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    q = 2 * x @ y.T
    score = (q + f_hat[:, None] + g_hat[None, :]) / epsilon
    return torch.exp(score) * a[:, None] * b[None, :]


def flash_sinkhorn_cpu(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    n_iters: int = 100,
    tol: Optional[float] = None,
    row_block: int = 128,
    col_block: int = 512,
    return_transport: bool = False,
) -> SinkhornResult:
    """Solve squared-Euclidean entropic OT with Flash-style CPU tiling.

    The score identity is
    ``||x-y||^2 = ||x||^2 + ||y||^2 - 2*x@y``.  The norm terms are absorbed
    into shifted dual potentials, leaving only tiled dot products and online
    log-sum-exp reductions in the iteration.
    """

    x, y = _check_inputs(x, y)
    dtype = torch.promote_types(x.dtype, y.dtype)
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    x, y = x.to(dtype), y.to(dtype)
    epsilon = float(epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    a = _prepare_weights(a, x.shape[0], dtype)
    b = _prepare_weights(b, y.shape[0], dtype)

    q = (2.0**0.5) * x
    k = (2.0**0.5) * y
    loga, logb = torch.log(a), torch.log(b)
    alpha, beta = x.square().sum(dim=1), y.square().sum(dim=1)
    # Match the usual dense initialization f=g=0.  Since the shifted
    # variables are f_hat=f-||x||^2 and g_hat=g-||y||^2, their corresponding
    # initial values are -alpha and -beta.
    f_hat = -alpha.clone()
    g_hat = -beta.clone()
    iterations = 0
    for it in range(int(n_iters)):
        old_f, old_g = f_hat, g_hat
        f_hat = -epsilon * _online_lse_dot(q, k, g_hat + epsilon * logb, epsilon, row_block, col_block)
        g_hat = -epsilon * _online_lse_dot(k, q, f_hat + epsilon * loga, epsilon, col_block, row_block)
        iterations = it + 1
        if tol is not None:
            change = max((f_hat - old_f).abs().max().item(), (g_hat - old_g).abs().max().item())
            if change <= tol:
                break

    transport = _transport_from_shifted(x, y, f_hat, g_hat, a, b, epsilon) if return_transport else None
    distance = (
        (transport * _squared_cost(x, y)).sum()
        if transport is not None
        else _stream_squared_cost(x, y, f_hat, g_hat, a, b, epsilon, row_block, col_block)
    )
    if transport is not None:
        marginal_error = max(
            (transport.sum(dim=1) - a).abs().max().item(),
            (transport.sum(dim=0) - b).abs().max().item(),
        )
    else:
        # A second streamed LSE gives the marginal residual without storing P.
        row_lse = _online_lse_dot(q, k, g_hat + epsilon * logb, epsilon, row_block, col_block)
        col_lse = _online_lse_dot(k, q, f_hat + epsilon * loga, epsilon, col_block, row_block)
        row_mass = a * torch.exp((f_hat / epsilon) + row_lse)
        col_mass = b * torch.exp((g_hat / epsilon) + col_lse)
        marginal_error = max((row_mass - a).abs().max().item(), (col_mass - b).abs().max().item())

    return SinkhornResult(
        distance=distance,
        f=f_hat + alpha,
        g=g_hat + beta,
        transport=transport,
        iterations=iterations,
        marginal_error=marginal_error,
    )


def dense_sinkhorn_cpu(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    n_iters: int = 100,
    tol: Optional[float] = None,
    return_transport: bool = False,
) -> SinkhornResult:
    """Dense PyTorch CPU baseline for squared-Euclidean entropic OT."""

    x, y = _check_inputs(x, y)
    dtype = torch.promote_types(x.dtype, y.dtype)
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    x, y = x.to(dtype), y.to(dtype)
    a = _prepare_weights(a, x.shape[0], dtype)
    b = _prepare_weights(b, y.shape[0], dtype)
    c = _squared_cost(x, y)
    loga, logb = torch.log(a), torch.log(b)
    f = torch.zeros(x.shape[0], dtype=dtype)
    g = torch.zeros(y.shape[0], dtype=dtype)
    for it in range(int(n_iters)):
        old_f, old_g = f, g
        f = -epsilon * torch.logsumexp((g[None, :] - c) / epsilon + logb[None, :], dim=1)
        g = -epsilon * torch.logsumexp((f[:, None] - c) / epsilon + loga[:, None], dim=0)
        if tol is not None and max((f - old_f).abs().max().item(), (g - old_g).abs().max().item()) <= tol:
            it += 1
            break
    log_plan = (f[:, None] + g[None, :] - c) / epsilon + loga[:, None] + logb[None, :]
    plan = torch.exp(log_plan) if return_transport else None
    distance = (plan * c).sum() if plan is not None else torch.sum(torch.exp(log_plan) * c)
    if plan is None:
        plan = torch.exp(log_plan)
    marginal_error = max((plan.sum(dim=1) - a).abs().max().item(), (plan.sum(dim=0) - b).abs().max().item())
    return SinkhornResult(
        distance=distance,
        f=f,
        g=g,
        transport=plan if return_transport else None,
        iterations=it + 1,
        marginal_error=marginal_error,
    )


def dense_sinkhorn_cost_matrix(
    cost: torch.Tensor,
    epsilon: float,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    n_iters: int = 100,
    tol: Optional[float] = None,
    return_transport: bool = True,
) -> SinkhornResult:
    """Generic dense log-domain Sinkhorn for an arbitrary cost matrix."""

    if cost.ndim != 2 or not cost.is_floating_point():
        raise ValueError("cost must be a floating-point matrix [n, m]")
    n, m = cost.shape
    dtype = cost.dtype
    a = _prepare_weights(a, n, dtype)
    b = _prepare_weights(b, m, dtype)
    loga, logb = torch.log(a), torch.log(b)
    f = torch.zeros(n, dtype=dtype)
    g = torch.zeros(m, dtype=dtype)
    for it in range(int(n_iters)):
        old_f, old_g = f, g
        f = -epsilon * torch.logsumexp((g[None, :] - cost) / epsilon + logb[None, :], dim=1)
        g = -epsilon * torch.logsumexp((f[:, None] - cost) / epsilon + loga[:, None], dim=0)
        if tol is not None and max((f - old_f).abs().max().item(), (g - old_g).abs().max().item()) <= tol:
            it += 1
            break
    log_plan = (f[:, None] + g[None, :] - cost) / epsilon + loga[:, None] + logb[None, :]
    plan = torch.exp(log_plan)
    return SinkhornResult(
        distance=(plan * cost).sum(),
        f=f,
        g=g,
        transport=plan if return_transport else None,
        iterations=it + 1,
        marginal_error=max((plan.sum(dim=1) - a).abs().max().item(), (plan.sum(dim=0) - b).abs().max().item()),
    )
