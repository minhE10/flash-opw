"""Adapter for the author's external OTT-Hessian baseline, without vendoring it.

Source: https://github.com/yexf308/OTT-Hessian
Tested revision: 7eb189fe39982f587da935044480655b65939637.
The fixed-step adapter uses the same guarded CG as Flash/KeOps; upstream
Lineax's unguarded zero-tolerance solve can become NaN after near-convergence.
The caller converts absolute Schur damping to tau2. Every control is recorded.
"""

import hashlib
import importlib.util
from pathlib import Path
from typing import NamedTuple, Any

PINNED_REVISION = "7eb189fe39982f587da935044480655b65939637"
PINNED_SOURCE_SHA256 = "7cd3c27a14563e949bf2f35d5719173308df38f498975a6b950488e5cb1c5158"


class OTState(NamedTuple):
    geom: Any
    f: Any
    g: Any


class CGResult(NamedTuple):
    value: Any


def fixed_step_cg(matvec, rhs, max_steps):
    """Same fixed-budget, device-side breakdown guards as flashsinkhorn CG."""
    import jax.numpy as jnp
    from jax import lax

    residual_sq = jnp.dot(rhs, rhs)

    def step(_, state):
        solution, residual, direction, residual_sq, active = state
        product = matvec(direction)
        curvature = jnp.dot(direction, product)
        valid = active & jnp.isfinite(curvature) & (curvature > 0)
        safe_curvature = jnp.where(valid, curvature, 1.0)
        alpha = jnp.where(valid, residual_sq / safe_curvature, 0.0)
        solution = solution + alpha * direction
        next_residual = residual - alpha * product
        next_sq = jnp.dot(next_residual, next_residual)
        beta = jnp.where(valid, next_sq / jnp.where(residual_sq > 0, residual_sq, 1.0), 0.0)
        direction = next_residual + beta * direction
        active = valid & jnp.isfinite(next_sq) & (next_sq > 0)
        return solution, next_residual, direction, next_sq, active

    state = (jnp.zeros_like(rhs), rhs, rhs, residual_sq, residual_sq > 0)
    return lax.fori_loop(0, max_steps, step, state)[0]


def load_hessian(path, *, cg_rtol=0.0, cg_atol=0.0):
    path = Path(path).resolve()
    source = path / "SinkhornHessian.py" if path.is_dir() else path
    if not source.is_file():
        raise FileNotFoundError(
            f"OTT-Hessian source missing: {source}. Run scripts/setup_ott_hessian.sh "
            "or supply --ott-hessian-path; no matrix-free fallback is substituted.")
    content = source.read_bytes()
    normalized_hash = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
    if normalized_hash != PINNED_SOURCE_SHA256:
        raise RuntimeError(f"OTT-Hessian source differs from tested revision {PINNED_REVISION}; "
                           "use the pinned checkout from scripts/setup_ott_hessian.sh")
    spec = importlib.util.spec_from_file_location("_flashopw_ott_hessian", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The current upstream HessianA is the Lineax implementation. Older
    # FlashSinkhorn tests refer to it under the name HessianALineax.
    function = getattr(module, "HessianALineax", None) or getattr(module, "HessianA", None)
    if function is None or not hasattr(module, "lx"):
        raise RuntimeError("Expected an OTT-Hessian Lineax HessianA implementation")

    class ConfiguredLineax:
        def __getattr__(self, name):
            return getattr(lineax, name)

        def CG(self, *args, **kwargs):
            kwargs.update(rtol=cg_rtol, atol=cg_atol)
            return lineax.CG(*args, **kwargs)

        def linear_solve(self, operator, rhs, solver, **kwargs):
            if cg_rtol == 0 and cg_atol == 0:
                if kwargs.get("options"):
                    raise ValueError("Fixed-CG adapter does not accept preconditioning options")
                return CGResult(fixed_step_cg(operator.mv, rhs, solver.max_steps))
            return lineax.linear_solve(operator, rhs, solver, **kwargs)

    lineax = module.lx
    module.lx = ConfiguredLineax()
    provenance = dict(source=str(source), sha256=hashlib.sha256(content).hexdigest(),
                      normalized_sha256=normalized_hash, pinned_revision=PINNED_REVISION,
                      function=function.__name__, cg_rtol=cg_rtol, cg_atol=cg_atol,
                      cg_backend="guarded fixed-step JAX CG" if cg_rtol == cg_atol == 0 else "upstream Lineax CG",
                      damping="upstream S + epsilon*tau2*I; tau2 = schur_damping/epsilon")
    return function, provenance


def state_from_shifted(x, y, u, v, *, epsilon, batch_size=256):
    """OTT potentials include marginal logs, unlike flashsinkhorn result.f/g."""
    import jax.numpy as jnp
    from ott.geometry import costs, pointcloud

    geom = pointcloud.PointCloud(x, y, cost_fn=costs.SqEuclidean(),
                                epsilon=epsilon, scale_cost=1.0, batch_size=batch_size)
    return OTState(geom, epsilon*u + jnp.sum(x*x, axis=1),
                   epsilon*v + jnp.sum(y*y, axis=1))
