"""Analytic first- and second-order differentiation for entropic OT.

The HVP follows Theorem 5 and Appendix F of Ye et al. (2026): it solves the
damped Schur complement with conjugate gradients and only accesses the
coupling through streaming transport applications.
"""

from dataclasses import dataclass
import math

import torch

from .solver import SinkhornResult, sinkhorn_dense, sinkhorn_flash, sinkhorn_online
from .transport import (apply_plan, apply_plan_hadamard, source_gradient,
                        target_gradient)


@dataclass(frozen=True)
class HVPInfo:
    """Diagnostics for the Schur-complement conjugate-gradient solve."""

    cg_converged: bool
    cg_iters: int
    cg_residual: float
    cg_initial_residual: float
    damping: float
    breakdown: bool = False


def regularized_ot_cost(result: SinkhornResult):
    """Return the regularized OT dual value induced by ``result``.

    The mass correction makes this the dual objective for arbitrary stored
    potentials. At a Sinkhorn fixed point the mass is one and the expression
    reduces to ``a @ f + b @ g``.
    """
    mass = apply_plan(result, torch.ones_like(result.b)).sum()
    return (result.a * result.f).sum() + (result.b * result.g).sum() + result.epsilon * (1 - mass)


def _conjugate_gradient(matvec, rhs, *, damping, max_iters, rtol, atol):
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    # With zero damping the Schur complement has null vector 1. The exact RHS
    # is orthogonal to it; projection prevents roundoff from exciting that mode.
    project = (lambda value: value - value.mean()) if damping == 0 else (lambda value: value)
    residual = project(residual)
    direction = residual.clone()
    residual_sq = torch.dot(residual, residual)
    initial = float(torch.sqrt(residual_sq))
    # rtol=atol=0 intentionally disables convergence-based early stopping.
    # This is the fixed-K CG protocol used by the paper's HVP benchmark.
    fixed_iterations = rtol == 0 and atol == 0
    threshold = -1.0 if fixed_iterations else max(float(atol), float(rtol) * initial)
    if initial == 0 or initial <= threshold:
        return solution, HVPInfo(True, 0, initial, initial, damping)

    converged = False
    breakdown = False
    iterations = 0
    for iterations in range(1, max_iters + 1):
        product = project(matvec(direction))
        curvature = torch.dot(direction, product)
        if not bool(torch.isfinite(curvature)) or float(curvature) <= 0:
            breakdown = True
            break
        step = residual_sq / curvature
        solution = solution + step * direction
        next_residual = project(residual - step * product)
        next_sq = torch.dot(next_residual, next_residual)
        norm = float(torch.sqrt(next_sq))
        if norm <= threshold:
            residual, residual_sq = next_residual, next_sq
            converged = True
            break
        beta = next_sq / residual_sq
        direction = next_residual + beta * direction
        residual, residual_sq = next_residual, next_sq
    final = float(torch.sqrt(residual_sq))
    return solution, HVPInfo(converged, iterations, final, initial, damping, breakdown)


@torch.no_grad()
def hessian_vector_product(result: SinkhornResult, vector, *, damping=1e-5,
                           max_cg_iters=50, cg_rtol=1e-6, cg_atol=0.0,
                           return_info=False):
    """Compute ``(d^2 OT_epsilon / dX^2) @ vector`` without storing ``P``.

    ``Y`` and the marginal weights are held fixed. The result is the paper's
    damped HVP when ``damping > 0``; use ``damping=0`` for a projected solve of
    the singular Schur system. Potentials should be converged before calling
    this routine. Internal transport products use strict IEEE FP32 on Triton,
    matching the paper's numerical recommendation for HVPs.
    """
    if vector.shape != result.x.shape:
        raise ValueError("vector must have the same shape as result.x")
    if vector.device != result.x.device or vector.dtype != result.x.dtype:
        raise ValueError("vector must share result.x dtype and device")
    if not math.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and nonnegative")
    if not isinstance(max_cg_iters, int) or max_cg_iters < 1:
        raise ValueError("max_cg_iters must be a positive integer")
    if not math.isfinite(cg_rtol) or cg_rtol < 0 or not math.isfinite(cg_atol) or cg_atol < 0:
        raise ValueError("cg_rtol and cg_atol must be finite and nonnegative")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("vector must be finite")

    precision = "ieee"
    ones_x = torch.ones_like(result.a)
    ones_y = torch.ones_like(result.b)

    def p(values):
        return apply_plan(result, values, precision=precision)

    def pt(values):
        return apply_plan(result, values, transpose=True, precision=precision)

    # The paper uses a,b at optimality. Using the actual induced masses makes
    # the same formula robust to the small residual left by a finite solve.
    row_mass = p(ones_y)
    col_mass = pt(ones_x)
    marginal_floor = max(torch.finfo(result.x.dtype).tiny, result.epsilon * 1e-10)
    diag_x = row_mass.clamp_min(marginal_floor)
    diag_y = col_mass.clamp_min(marginal_floor)

    gamma = 2.0 * result.cost_scale
    row_dot = (result.x * vector).sum(1)
    py = p(result.y)
    py_dot = (py * vector).sum(1)

    # Equation (29): r = R A.
    r1 = gamma * (row_mass * row_dot - py_dot)
    pt_vector = pt(vector)
    r2 = gamma * (pt(row_dot) - (pt_vector * result.y).sum(1))
    rhs = r2 - pt(r1 / diag_x)

    # Equation (30): damped Schur complement.
    def schur(value):
        return diag_y * value - pt(p(value) / diag_x) + damping * value

    w2, info = _conjugate_gradient(
        schur, rhs, damping=damping, max_iters=max_cg_iters,
        rtol=cg_rtol, atol=cg_atol,
    )
    pw2 = p(w2)
    w1 = (r1 - pw2) / diag_x

    # Equation (31): R.T w.
    p_w2_y = p(w2[:, None] * result.y)
    implicit = gamma * (
        (row_mass * w1)[:, None] * result.x
        - w1[:, None] * py
        + pw2[:, None] * result.x
        - p_w2_y
    )

    # Equations (27)-(28): E A. The only nonstandard primitive is B5.
    b1 = gamma * row_mass[:, None] * vector
    b2 = (row_mass * row_dot)[:, None] * result.x
    b3 = row_dot[:, None] * py
    b4 = py_dot[:, None] * result.x
    b5 = apply_plan_hadamard(
        result, vector, result.y, result.y, precision=precision,
    )
    explicit = b1 - (gamma * gamma / result.epsilon) * (b2 - b3 - b4 + b5)
    output = implicit / result.epsilon + explicit
    return (output, info) if return_info else output


@dataclass(frozen=True)
class _CostConfig:
    backend: str
    epsilon: float
    cost_scale: float
    n_iters: int
    schedule: str
    tol: float | None
    check_every: int
    precision: str
    block_m: int
    block_n: int
    hvp_damping: float
    hvp_max_cg_iters: int
    hvp_cg_rtol: float
    hvp_cg_atol: float


class _AnalyticGradient(torch.autograd.Function):
    """First-order envelope gradient whose x-double-backward is the HVP."""

    @staticmethod
    def forward(ctx, x, y, result, grad_scale, config):
        ctx.set_materialize_grads(False)
        ctx.result = result
        ctx.config = config
        ctx.save_for_backward(grad_scale)
        gx = source_gradient(result) * grad_scale if x.requires_grad else None
        gy = target_gradient(result) * grad_scale if y.requires_grad else None
        return gx, gy

    @staticmethod
    def backward(ctx, grad_grad_x, grad_grad_y):
        if grad_grad_y is not None:
            raise NotImplementedError("double backward is implemented only with respect to x")
        out_x = None
        if grad_grad_x is not None:
            (grad_scale,) = ctx.saved_tensors
            config = ctx.config
            out_x = hessian_vector_product(
                ctx.result, grad_grad_x,
                damping=config.hvp_damping,
                max_cg_iters=config.hvp_max_cg_iters,
                cg_rtol=config.hvp_cg_rtol,
                cg_atol=config.hvp_cg_atol,
            ) * grad_scale
        return out_x, None, None, None, None


class _SinkhornCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, a, b, config):
        solver = {"flash": sinkhorn_flash, "dense": sinkhorn_dense, "online": sinkhorn_online}[config.backend]
        result = solver(
            x, y, a=a, b=b, epsilon=config.epsilon,
            cost_scale=config.cost_scale, n_iters=config.n_iters,
            schedule=config.schedule, tol=config.tol,
            check_every=config.check_every,
            **({"precision": config.precision, "block_m": config.block_m,
                "block_n": config.block_n} if config.backend == "flash" else
               {"block_m": config.block_m, "block_n": config.block_n}
               if config.backend == "online" else {}),
        )
        ctx.result = result
        ctx.config = config
        ctx.save_for_backward(x, y)
        # The benchmark/API loss follows the balanced dual convention used by
        # GeomLoss and OTT. Avoiding an extra mass evaluation also keeps the
        # forward+backward path to one streamed transport gradient application.
        return (result.a * result.f).sum() + (result.b * result.g).sum()

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        result = ctx.result
        gx = gy = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            gx_all, gy_all = _AnalyticGradient.apply(x, y, result, grad_output, ctx.config)
            gx = gx_all if ctx.needs_input_grad[0] else None
            gy = gy_all if ctx.needs_input_grad[1] else None
        grad_a = grad_output * result.f if ctx.needs_input_grad[2] else None
        grad_b = grad_output * result.g if ctx.needs_input_grad[3] else None
        return gx, gy, grad_a, grad_b, None


def sinkhorn_cost(x, y, *, a=None, b=None, epsilon=0.2, cost_scale=1.0,
                  n_iters=200, schedule="alternating", tol=None, check_every=20,
                  backend="flash", precision="tf32", block_m=32, block_n=64,
                  hvp_damping=1e-5, hvp_max_cg_iters=50,
                  hvp_cg_rtol=1e-6, hvp_cg_atol=0.0):
    """Differentiable regularized OT cost with analytic backward and x-HVP.

    The backward pass applies the converged coupling instead of differentiating
    through Sinkhorn iterations. With ``create_graph=True``, a second backward
    with respect to ``x`` invokes :func:`hessian_vector_product`. Double
    backward with respect to ``y`` is intentionally unsupported, as in the
    paper's released source implementation.
    """
    if backend not in ("flash", "dense", "online"):
        raise ValueError("backend must be flash, dense or online")
    if a is None:
        a = x.new_full((len(x),), 1.0 / len(x))
    if b is None:
        b = y.new_full((len(y),), 1.0 / len(y))
    config = _CostConfig(
        backend, epsilon, cost_scale, n_iters, schedule, tol, check_every,
        precision, block_m, block_n, hvp_damping, hvp_max_cg_iters,
        hvp_cg_rtol, hvp_cg_atol,
    )
    return _SinkhornCost.apply(x, y, a, b, config)


__all__ = ["HVPInfo", "hessian_vector_product", "regularized_ot_cost", "sinkhorn_cost"]
