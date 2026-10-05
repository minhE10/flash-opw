"""Transport applications and diagnostics, with no dense plan by default."""

import os
import torch


@torch.no_grad()
def materialize_plan(result, *, max_entries=4_194_304):
    """Explicit diagnostic helper for small problems only; quadratic storage."""
    if len(result.x) * len(result.y) > max_entries:
        raise ValueError("Dense plan exceeds max_entries; use apply_plan")
    scores = (2 * result.cost_scale / result.epsilon) * (result.x @ result.y.T)
    return (scores + result.u[:, None] + result.v[None, :]).exp()


@torch.no_grad()
def apply_plan(result, values, *, transpose=False, precision=None):
    """P @ values (or P.T @ values), including actual marginal correction.

    Works for nonconverged potentials too; does not silently replace row masses
    by the prescribed marginals. Vectors and multi-column values are supported.
    """
    q, k, u, v = result.x, result.y, result.u, result.v
    if transpose:
        q, k, u, v = k, q, v, u
    if values.ndim not in (1, 2) or values.shape[0] != len(k):
        raise ValueError("values must have shape (number of keys,) or (number of keys,p)")
    if values.dtype != q.dtype or values.device != q.device:
        raise ValueError("values must share the points' dtype and device")
    vector = values.ndim == 1
    values = values.reshape(len(k), -1).contiguous()
    if values.shape[1] == 0:
        raise ValueError("values must have at least one column")
    scale = 2 * result.cost_scale / result.epsilon
    if result.backend == "triton":
        from .triton_kernels import apply
        output = apply(q, k, u, v, values, scale, precision or result.precision,
                       result.block_m, result.block_n)
    elif result.backend == "keops":
        from pykeops.torch import LazyTensor
        q_i = LazyTensor(q[:, None, :])
        k_j = LazyTensor(k[None, :, :])
        u_i = LazyTensor(u[:, None, None])
        v_j = LazyTensor(v[None, :, None])
        values_j = LazyTensor(values[None, :, :])
        weights = (scale * (q_i * k_j).sum(-1) + u_i + v_j).exp()
        output = (weights * values_j).sum(dim=1)
    elif result.backend == "dense":
        p = materialize_plan(result, max_entries=len(q) * len(k))
        output = (p.T if transpose else p) @ values
    else:
        output = q.new_zeros((len(q), values.shape[1]))
        for i in range(0, len(q), result.block_m):
            qi = q[i:i + result.block_m]
            for j in range(0, len(k), result.block_n):
                logp = scale * (qi @ k[j:j + result.block_n].T)
                logp = logp + u[i:i + len(qi), None] + v[None, j:j + result.block_n]
                output[i:i + len(qi)] += logp.exp() @ values[j:j + result.block_n]
    return output[:, 0] if vector else output


@torch.no_grad()
def apply_plan_hadamard(result, left, right, values, *, precision=None):
    """Apply ``(P * (left @ right.T)) @ values`` without storing ``P``.

    This is the Hadamard-weighted transport primitive in Theorem 5 of
    Ye et al. (2026). ``left`` and ``right`` share a factor dimension while
    ``values`` may be a vector or a matrix.
    """
    n, m = len(result.x), len(result.y)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != n or right.shape[0] != m:
        raise ValueError("left and right must have shapes (n,r) and (m,r)")
    if left.shape[1] != right.shape[1] or left.shape[1] < 1:
        raise ValueError("left and right must share a positive factor dimension")
    if values.ndim not in (1, 2) or values.shape[0] != m:
        raise ValueError("values must have shape (m,) or (m,p)")
    tensors = (left, right, values)
    if any(value.device != result.x.device or value.dtype != result.x.dtype for value in tensors):
        raise ValueError("left, right and values must share the result dtype and device")
    vector = values.ndim == 1
    left, right = left.contiguous(), right.contiguous()
    values = values.reshape(m, -1).contiguous()
    if values.shape[1] == 0:
        raise ValueError("values must have at least one column")

    if result.backend == "triton":
        from .triton_kernels import hadamard_apply
        output = hadamard_apply(
            result.x, result.y, result.u, result.v, left, right, values,
            2 * result.cost_scale / result.epsilon, precision or result.precision,
            result.block_m, result.block_n,
        )
    elif result.backend == "keops":
        from pykeops.torch import LazyTensor
        q_i = LazyTensor(result.x[:, None, :])
        k_j = LazyTensor(result.y[None, :, :])
        u_i = LazyTensor(result.u[:, None, None])
        v_j = LazyTensor(result.v[None, :, None])
        left_i = LazyTensor(left[:, None, :])
        right_j = LazyTensor(right[None, :, :])
        values_j = LazyTensor(values[None, :, :])
        weights = ((2 * result.cost_scale / result.epsilon) * (q_i * k_j).sum(-1)
                   + u_i + v_j).exp()
        factors = (left_i * right_j).sum(-1)
        output = (weights * factors * values_j).sum(dim=1)
    elif result.backend == "dense":
        plan = materialize_plan(result, max_entries=n * m)
        output = (plan * (left @ right.T)) @ values
    else:
        output = result.x.new_zeros((n, values.shape[1]))
        scale = 2 * result.cost_scale / result.epsilon
        for i in range(0, n, result.block_m):
            qi = result.x[i:i + result.block_m]
            li = left[i:i + result.block_m]
            for j in range(0, m, result.block_n):
                kj = result.y[j:j + result.block_n]
                logp = scale * (qi @ kj.T)
                logp = logp + result.u[i:i + len(qi), None] + result.v[None, j:j + len(kj)]
                factors = li @ right[j:j + len(kj)].T
                output[i:i + len(qi)] += (logp.exp() * factors) @ values[j:j + len(kj)]
    return output[:, 0] if vector else output


@torch.no_grad()
def diagnostics(result):
    """Report objectives and actual marginal errors, not just potential changes.

    primal is the objective of the current, possibly infeasible coupling. The
    signed primal-minus-dual is NOT a certified gap until marginals converge.
    """
    rows_and_py = apply_plan(result, torch.cat((torch.ones_like(result.b[:, None]), result.y), 1))
    rows, py = rows_and_py[:, 0], rows_and_py[:, 1:]
    cols = apply_plan(result, torch.ones_like(result.a), transpose=True)
    mass = rows.sum()
    cost = result.cost_scale * (
        (rows * result.x.square().sum(1)).sum()
        + (cols * result.y.square().sum(1)).sum() - 2 * (result.x * py).sum())
    f, g = result.f, result.g
    primal = (rows * f).sum() + (cols * g).sum() + result.epsilon * (1 - mass)
    dual = (result.a * f).sum() + (result.b * g).sum() + result.epsilon * (1 - mass)
    metrics = {
        "transport_cost": cost, "regularized_primal": primal, "dual": dual,
        "primal_minus_dual": primal - dual, "mass": mass,
        "row_l1": (rows - result.a).abs().sum(),
        "col_l1": (cols - result.b).abs().sum(),
    }
    return {name: float(value) for name, value in metrics.items()}


@torch.no_grad()
def source_gradient(result):
    """Envelope gradient with respect to the source points only."""
    if result.backend == "triton" and os.environ.get("FLASHOPW_GRADIENT_KERNEL", "1") == "1":
        from .triton_kernels import gradient
        return gradient(result.x, result.y, result.u, result.v,
                        2 * result.cost_scale / result.epsilon,
                        2 * result.cost_scale, result.precision,
                        result.block_m, result.block_n)
    row_py = apply_plan(result, torch.cat((torch.ones_like(result.b[:, None]), result.y), 1))
    return 2 * result.cost_scale * (row_py[:, :1] * result.x - row_py[:, 1:])


@torch.no_grad()
def target_gradient(result):
    """Envelope gradient with respect to the target points only."""
    if result.backend == "triton" and os.environ.get("FLASHOPW_GRADIENT_KERNEL", "1") == "1":
        from .triton_kernels import gradient
        return gradient(result.y, result.x, result.v, result.u,
                        2 * result.cost_scale / result.epsilon,
                        2 * result.cost_scale, result.precision,
                        result.block_m, result.block_n)
    col_px = apply_plan(result, torch.cat((torch.ones_like(result.a[:, None]), result.x), 1), transpose=True)
    return 2 * result.cost_scale * (col_px[:, :1] * result.y - col_px[:, 1:])


@torch.no_grad()
def point_gradients(result):
    """Envelope gradients using actual masses; exact for the OT optimum only
    after convergence. These are not derivatives through a finite iteration
    algorithm. This low-level helper is detached; use ``sinkhorn_cost`` for
    PyTorch backward and x-double-backward support.
    """
    return source_gradient(result), target_gradient(result)
