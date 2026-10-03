"""Order-Preserving Wasserstein (OPW) and affine FlashOPW CPU solvers.

The exact objective follows Eqs. (13)--(19) of ``OWD_journal.pdf``.  The
FlashOPW variant follows ``main (2).pdf``: ``1 / (1 + F)`` is approximated by
``1 - F`` and the resulting squared-Euclidean cost is embedded in one extra
feature dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .sinkhorn import (
    SinkhornResult,
    _prepare_weights,
    _squared_cost,
    dense_sinkhorn_cost_matrix,
    flash_sinkhorn_cpu,
)


@dataclass
class OPWResult:
    distance: torch.Tensor
    effective_distance: torch.Tensor
    transport: torch.Tensor
    f: torch.Tensor
    g: torch.Tensor
    iterations: int
    marginal_error: float
    approximation: bool


def _relative_positions(n: int, dtype: torch.dtype) -> torch.Tensor:
    # This is the convention used by main (2).pdf: i/N and j/M.
    return torch.arange(n, dtype=dtype) / float(n)


def opw_cost_terms(
    x: torch.Tensor,
    y: torch.Tensor,
    lambda1: float,
    lambda2: float,
    sigma: float,
    geometry: str = "main",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return spatial cost, temporal terms and exact effective OPW cost.

    ``geometry='main'`` uses F=(i/N-j/M)^2 as in main (2).pdf.  The optional
    ``geometry='owd'`` reproduces the normalized l^2 definition in the OWD
    journal paper, which is useful when comparing against its experiments.
    """

    if sigma <= 0 or lambda2 <= 0 or lambda1 < 0:
        raise ValueError("require sigma > 0, lambda2 > 0 and lambda1 >= 0")
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have shape [n, d] and [m, d]")
    dtype = torch.promote_types(x.dtype, y.dtype)
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    x, y = x.to(dtype), y.to(dtype)
    u = _relative_positions(x.shape[0], dtype)
    v = _relative_positions(y.shape[0], dtype)
    delta = u[:, None] - v[None, :]
    if geometry == "main":
        f_temporal = delta.square()
    elif geometry == "owd":
        f_temporal = delta.square() / (1.0 / x.shape[0] ** 2 + 1.0 / y.shape[0] ** 2)
    else:
        raise ValueError("geometry must be 'main' or 'owd'")
    e_inverse = 1.0 / (1.0 + delta.square())
    spatial = _squared_cost(x, y)
    q0 = lambda2 * torch.log(torch.as_tensor(sigma * (2.0 * torch.pi) ** 0.5, dtype=dtype)) - lambda1
    exact_effective = spatial - lambda1 * e_inverse + lambda2 * (
        f_temporal / (2.0 * sigma**2) + torch.log(torch.as_tensor(sigma * (2.0 * torch.pi) ** 0.5, dtype=dtype))
    )
    return spatial, f_temporal, e_inverse, exact_effective, q0


def _plan_from_potentials(
    cost: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return torch.exp((f[:, None] + g[None, :] - cost) / epsilon + torch.log(a)[:, None] + torch.log(b)[None, :])


def opw_exact_cpu(
    x: torch.Tensor,
    y: torch.Tensor,
    lambda1: float = 10.0,
    lambda2: float = 0.1,
    sigma: float = 1.0,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    n_iters: int = 100,
    tol: Optional[float] = None,
    geometry: str = "main",
) -> OPWResult:
    """Exact dense OPW reference, retaining the original spatial distance."""

    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    spatial, _, _, effective, _ = opw_cost_terms(x, y, lambda1, lambda2, sigma, geometry)
    aa = _prepare_weights(a, x.shape[0], effective.dtype)
    bb = _prepare_weights(b, y.shape[0], effective.dtype)
    solved: SinkhornResult = dense_sinkhorn_cost_matrix(
        effective, lambda2, aa, bb, n_iters=n_iters, tol=tol, return_transport=True
    )
    plan = solved.transport
    return OPWResult(
        distance=(plan * spatial).sum(),
        effective_distance=solved.distance,
        transport=plan,
        f=solved.f,
        g=solved.g,
        iterations=solved.iterations,
        marginal_error=solved.marginal_error,
        approximation=False,
    )


def _stream_plan_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float,
    lambda1: float,
    lambda2: float,
    sigma: float,
    geometry: str,
    row_block: int,
    col_block: int,
    solve_with_affine_cost: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute <T,D> and <T,C_eff> in tiles from dual potentials."""

    total_spatial = torch.zeros((), dtype=x.dtype)
    total_effective = torch.zeros((), dtype=x.dtype)
    row_mass = torch.zeros(x.shape[0], dtype=x.dtype)
    col_mass = torch.zeros(y.shape[0], dtype=x.dtype)
    dtype = x.dtype
    u = _relative_positions(x.shape[0], dtype)
    v = _relative_positions(y.shape[0], dtype)
    for i0 in range(0, x.shape[0], row_block):
        i1 = min(i0 + row_block, x.shape[0])
        for j0 in range(0, y.shape[0], col_block):
            j1 = min(j0 + col_block, y.shape[0])
            spatial = _squared_cost(x[i0:i1], y[j0:j1])
            delta = u[i0:i1, None] - v[None, j0:j1]
            f_temporal = delta.square() if geometry == "main" else delta.square() / (1.0 / x.shape[0] ** 2 + 1.0 / y.shape[0] ** 2)
            effective = spatial - lambda1 / (1.0 + delta.square()) + lambda2 * (
                f_temporal / (2.0 * sigma**2) + torch.log(torch.as_tensor(sigma * (2.0 * torch.pi) ** 0.5, dtype=dtype))
            )
            solve_cost = spatial + (lambda1 + lambda2 / (2.0 * sigma**2)) * f_temporal if solve_with_affine_cost else effective
            plan = torch.exp((f[i0:i1, None] + g[None, j0:j1] - solve_cost) / epsilon)
            plan = plan * a[i0:i1, None] * b[j0:j1]
            total_spatial = total_spatial + (plan * spatial).sum()
            total_effective = total_effective + (plan * effective).sum()
            row_mass[i0:i1] += plan.sum(dim=1)
            col_mass[j0:j1] += plan.sum(dim=0)
    error = max((row_mass - a).abs().max().item(), (col_mass - b).abs().max().item())
    return total_spatial, total_effective, error


def opw_flash_cpu(
    x: torch.Tensor,
    y: torch.Tensor,
    lambda1: float = 10.0,
    lambda2: float = 0.1,
    sigma: float = 1.0,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    n_iters: int = 100,
    tol: Optional[float] = None,
    row_block: int = 128,
    col_block: int = 512,
    geometry: str = "main",
    return_transport: bool = True,
) -> OPWResult:
    """Flash-style OPW using the affine approximation from main (2).pdf."""

    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    if not x.is_floating_point():
        x = x.float()
    if not y.is_floating_point():
        y = y.float()
    spatial, f_temporal, _, exact_effective, q0 = opw_cost_terms(x, y, lambda1, lambda2, sigma, geometry)
    mu = lambda2 / (2.0 * sigma**2) + lambda1
    u = _relative_positions(x.shape[0], x.dtype if x.is_floating_point() else torch.float32)
    v = _relative_positions(y.shape[0], y.dtype if y.is_floating_point() else torch.float32)
    extra_x = torch.sqrt(torch.as_tensor(mu, dtype=x.dtype)) * u[:, None]
    extra_y = torch.sqrt(torch.as_tensor(mu, dtype=y.dtype)) * v[:, None]
    x_aug = torch.cat([x, extra_x], dim=1)
    y_aug = torch.cat([y, extra_y], dim=1)
    aa = _prepare_weights(a, x.shape[0], x_aug.dtype)
    bb = _prepare_weights(b, y.shape[0], y_aug.dtype)
    solved = flash_sinkhorn_cpu(
        x_aug,
        y_aug,
        epsilon=lambda2,
        a=aa,
        b=bb,
        n_iters=n_iters,
        tol=tol,
        row_block=row_block,
        col_block=col_block,
        return_transport=return_transport,
    )
    if return_transport:
        plan = solved.transport
        distance = (plan * spatial).sum()
        effective_distance = (plan * exact_effective).sum()
        marginal_error = solved.marginal_error
    else:
        distance, effective_distance, marginal_error = _stream_plan_cost(
            x_aug[:, :-1],
            y_aug[:, :-1],
            solved.f,
            solved.g,
            aa,
            bb,
            lambda2,
            lambda1,
            lambda2,
            sigma,
            geometry,
            row_block,
            col_block,
            solve_with_affine_cost=True,
        )
        plan = torch.empty((0, 0), dtype=x_aug.dtype)
    return OPWResult(
        distance=distance,
        effective_distance=effective_distance,
        transport=plan,
        f=solved.f,
        g=solved.g,
        iterations=solved.iterations,
        marginal_error=marginal_error,
        approximation=True,
    )


def opw_approximation_error(
    x: torch.Tensor,
    y: torch.Tensor,
    lambda1: float,
    lambda2: float,
    sigma: float,
    geometry: str = "main",
) -> dict[str, float]:
    """Report the cost approximation error before solving Sinkhorn."""

    spatial, f_temporal, _, exact, q0 = opw_cost_terms(x, y, lambda1, lambda2, sigma, geometry)
    mu = lambda2 / (2.0 * sigma**2) + lambda1
    approx = spatial + mu * f_temporal
    # q0 is a global constant on the transport polytope and is intentionally
    # dropped in Eq. (7) of main (2).pdf.  Remove it before measuring the
    # Taylor approximation error; otherwise the report mostly measures a
    # harmless additive offset.
    exact_up_to_constant = exact - q0
    delta = approx - exact_up_to_constant
    return {
        "cost_abs_max": delta.abs().max().item(),
        "cost_abs_mean": delta.abs().mean().item(),
        "cost_relative_fro": (delta.norm() / exact.norm().clamp_min(1e-12)).item(),
    }
