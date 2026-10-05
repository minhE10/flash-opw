"""Matrix-free JAX HVP for the paper's fixed-step Schur-CG comparison.

The ordinary JAX ``linearize(grad(ott_loss))`` route differentiates through a
large online Sinkhorn graph and can require O(n*m) compiler temporaries. This
module uses the same cached coupling and Schur operator as the Torch/KeOps
benchmark, with blockwise JAX transport applications. Inputs remain dynamic
JIT arguments so XLA does not constant-fold a large point cloud into the code.
"""


def hvp_from_shifted_potentials(x, y, u, v, direction, *, epsilon, damping,
                                cg_iters, block_rows=64, block_keys=256):
    """Return an HVP; ``u,v`` are the shifted log-plan potentials.

    The plan is ``exp(2*x@y.T/epsilon + u[:,None] + v[None,:])``. This matches
    the full squared-Euclidean cost convention used by ``flashopw``.
    """
    import jax.numpy as jnp
    from jax import lax

    n = x.shape[0]
    m = y.shape[0]
    scale = 2.0 / epsilon

    def transport(q, k, qbias, kbias, values, left=None, right=None):
        vector = values.ndim == 1
        if vector:
            values = values[:, None]
        p = values.shape[1]
        count_q = (q.shape[0] + block_rows - 1) // block_rows
        count_k = (k.shape[0] + block_keys - 1) // block_keys
        pad_q = count_q * block_rows - q.shape[0]
        pad_k = count_k * block_keys - k.shape[0]
        q = jnp.pad(q, ((0, pad_q), (0, 0)))
        k = jnp.pad(k, ((0, pad_k), (0, 0)))
        qbias = jnp.pad(qbias, (0, pad_q))
        kbias = jnp.pad(kbias, (0, pad_k), constant_values=-jnp.inf)
        values = jnp.pad(values, ((0, pad_k), (0, 0)))
        if left is not None:
            left = jnp.pad(left, ((0, pad_q), (0, 0)))
            right = jnp.pad(right, ((0, pad_k), (0, 0)))

        def row_block(_, row_index):
            offset = row_index * block_rows
            q_tile = lax.dynamic_slice_in_dim(q, offset, block_rows)
            u_tile = lax.dynamic_slice_in_dim(qbias, offset, block_rows)
            left_tile = (lax.dynamic_slice_in_dim(left, offset, block_rows)
                         if left is not None else None)

            def key_block(key_index, acc):
                key_offset = key_index * block_keys
                k_tile = lax.dynamic_slice_in_dim(k, key_offset, block_keys)
                v_tile = lax.dynamic_slice_in_dim(kbias, key_offset, block_keys)
                value_tile = lax.dynamic_slice_in_dim(values, key_offset, block_keys)
                logits = scale * (q_tile @ k_tile.T) + u_tile[:, None] + v_tile[None, :]
                weights = jnp.exp(logits)
                if left_tile is not None:
                    right_tile = lax.dynamic_slice_in_dim(right, key_offset, block_keys)
                    weights = weights * (left_tile @ right_tile.T)
                return acc + weights @ value_tile

            result = lax.fori_loop(0, count_k, key_block,
                                   jnp.zeros((block_rows, p), dtype=q.dtype))
            return None, result

        _, blocks = lax.scan(row_block, None, jnp.arange(count_q))
        output = blocks.reshape((-1, p))[:q.shape[0] - pad_q]
        return output[:, 0] if vector else output

    def apply(values):
        return transport(x, y, u, v, values)

    def apply_t(values):
        return transport(y, x, v, u, values)

    row_mass = apply(jnp.ones((m,), dtype=x.dtype))
    col_mass = apply_t(jnp.ones((n,), dtype=x.dtype))
    floor = max(jnp.finfo(x.dtype).tiny, epsilon * 1e-10)
    diag_x = jnp.maximum(row_mass, floor)
    diag_y = jnp.maximum(col_mass, floor)
    row_dot = jnp.sum(x * direction, axis=1)
    py = apply(y)
    py_dot = jnp.sum(py * direction, axis=1)
    r1 = 2.0 * (row_mass * row_dot - py_dot)
    r2 = 2.0 * (apply_t(row_dot) - jnp.sum(apply_t(direction) * y, axis=1))
    rhs = r2 - apply_t(r1 / diag_x)

    def schur(value):
        return diag_y * value - apply_t(apply(value) / diag_x) + damping * value

    initial = jnp.zeros_like(rhs)
    residual = rhs
    search = residual
    residual_sq = jnp.dot(residual, residual)

    def cg_step(_, state):
        solution, residual, search, residual_sq = state
        product = schur(search)
        curvature = jnp.dot(search, product)
        valid = jnp.isfinite(curvature) & (curvature > 0) & (residual_sq > 0)
        step = jnp.where(valid, residual_sq / jnp.where(valid, curvature, 1.0), 0.0)
        solution = solution + step * search
        next_residual = residual - step * product
        next_sq = jnp.dot(next_residual, next_residual)
        beta = jnp.where(valid, next_sq / jnp.where(residual_sq > 0, residual_sq, 1.0), 0.0)
        return solution, next_residual, next_residual + beta * search, next_sq

    w2, _, _, _ = lax.fori_loop(0, cg_iters, cg_step,
                               (initial, residual, search, residual_sq))
    pw2 = apply(w2)
    w1 = (r1 - pw2) / diag_x
    p_w2_y = apply(w2[:, None] * y)
    implicit = 2.0 * ((row_mass * w1)[:, None] * x - w1[:, None] * py
                      + pw2[:, None] * x - p_w2_y)
    b5 = transport(x, y, u, v, y, direction, y)
    explicit = (2.0 * row_mass[:, None] * direction
                - (4.0 / epsilon) * ((row_mass * row_dot)[:, None] * x
                                      - row_dot[:, None] * py
                                      - py_dot[:, None] * x + b5))
    return implicit / epsilon + explicit
