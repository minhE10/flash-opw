"""Matrix-free JAX HVP for the paper's fixed-step Schur-CG comparison.

The ordinary JAX ``linearize(grad(ott_loss))`` route differentiates through a
large online Sinkhorn graph and can require O(n*m) compiler temporaries. This
module uses the same cached coupling and Schur operator as the Torch/KeOps
benchmark, with blockwise JAX transport applications. Inputs remain dynamic
JIT arguments so XLA does not constant-fold a large point cloud into the code.
"""


def transport_from_shifted_potentials(x, y, u, v, values, *, epsilon,
                                      transpose=False, left=None, right=None,
                                      block_rows=64, block_keys=256):
    """Stream P @ values or P.T @ values using reciprocal FP32 score tiles.

    Both directions evaluate source-by-target dot products with the same tile
    shape and add source u before target v. Reversing the potential addition
    order can make P.T differ from the transpose of P in finite precision.
    """
    import jax.numpy as jnp
    from jax import lax

    q, k, qbias, kbias = (y, x, v, u) if transpose else (x, y, u, v)
    row_tile_size = block_keys if transpose else block_rows
    key_tile_size = block_rows if transpose else block_keys
    if transpose:
        left, right = right, left
    scale = 2.0 / epsilon

    vector = values.ndim == 1
    if vector:
        values = values[:, None]
    p = values.shape[1]
    count_q = (q.shape[0] + row_tile_size - 1) // row_tile_size
    count_k = (k.shape[0] + key_tile_size - 1) // key_tile_size
    pad_q = count_q * row_tile_size - q.shape[0]
    pad_k = count_k * key_tile_size - k.shape[0]
    q = jnp.pad(q, ((0, pad_q), (0, 0)))
    k = jnp.pad(k, ((0, pad_k), (0, 0)))
    qbias = jnp.pad(qbias, (0, pad_q))
    kbias = jnp.pad(kbias, (0, pad_k), constant_values=-jnp.inf)
    values = jnp.pad(values, ((0, pad_k), (0, 0)))
    if left is not None:
        left = jnp.pad(left, ((0, pad_q), (0, 0)))
        right = jnp.pad(right, ((0, pad_k), (0, 0)))

    def row_block(_, row_index):
        offset = row_index * row_tile_size
        q_tile = lax.dynamic_slice_in_dim(q, offset, row_tile_size)
        u_tile = lax.dynamic_slice_in_dim(qbias, offset, row_tile_size)
        left_tile = (lax.dynamic_slice_in_dim(left, offset, row_tile_size)
                     if left is not None else None)

        def key_block(key_index, acc):
            key_offset = key_index * key_tile_size
            k_tile = lax.dynamic_slice_in_dim(k, key_offset, key_tile_size)
            v_tile = lax.dynamic_slice_in_dim(kbias, key_offset, key_tile_size)
            value_tile = lax.dynamic_slice_in_dim(values, key_offset, key_tile_size)
            # Use the identical source-by-target score tile for P and P.T.
            # Barriers retain the FP32 rounding points of the dense reference;
            # otherwise GPU fusion may contract/reassociate cancelling terms.
            source_tile, target_tile = ((k_tile, q_tile) if transpose
                                        else (q_tile, k_tile))
            scores = jnp.matmul(source_tile, target_tile.T,
                                precision=lax.Precision.HIGHEST)
            scores = lax.optimization_barrier(scores)
            scores = lax.optimization_barrier(scale * scores)
            source_bias = v_tile if transpose else u_tile
            target_bias = u_tile if transpose else v_tile
            logits = lax.optimization_barrier(scores + source_bias[:, None])
            weights = jnp.exp(logits + target_bias[None, :])
            if transpose:
                weights = weights.T
            if left_tile is not None:
                right_tile = lax.dynamic_slice_in_dim(right, key_offset, key_tile_size)
                weights = weights * jnp.matmul(left_tile, right_tile.T,
                                                precision=lax.Precision.HIGHEST)
            return acc + jnp.matmul(weights, value_tile,
                                    precision=lax.Precision.HIGHEST)

        result = lax.fori_loop(0, count_k, key_block,
                               jnp.zeros((row_tile_size, p), dtype=q.dtype))
        return None, result

    _, blocks = lax.scan(row_block, None, jnp.arange(count_q))
    output = blocks.reshape((-1, p))[:q.shape[0] - pad_q]
    return output[:, 0] if vector else output


def hvp_from_shifted_potentials(x, y, u, v, direction, *, epsilon, damping,
                                cg_iters, block_rows=64, block_keys=256):
    """Return an HVP; ``u,v`` are the shifted log-plan potentials.

    The plan is ``exp(2*x@y.T/epsilon + u[:,None] + v[None,:])``. This matches
    the full squared-Euclidean cost convention used by ``flashopw``.
    """
    import jax.numpy as jnp

    n = x.shape[0]
    m = y.shape[0]

    def apply(values):
        return transport_from_shifted_potentials(
            x, y, u, v, values, epsilon=epsilon,
            block_rows=block_rows, block_keys=block_keys)

    def apply_t(values):
        return transport_from_shifted_potentials(
            x, y, u, v, values, epsilon=epsilon, transpose=True,
            block_rows=block_rows, block_keys=block_keys)

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

    from .ott_hessian import fixed_step_cg
    w2 = fixed_step_cg(schur, rhs, cg_iters)
    pw2 = apply(w2)
    w1 = (r1 - pw2) / diag_x
    p_w2_y = apply(w2[:, None] * y)
    implicit = 2.0 * ((row_mass * w1)[:, None] * x - w1[:, None] * py
                      + pw2[:, None] * x - p_w2_y)
    b5 = transport_from_shifted_potentials(
        x, y, u, v, y, epsilon=epsilon, left=direction, right=y,
        block_rows=block_rows, block_keys=block_keys)
    explicit = (2.0 * row_mass[:, None] * direction
                - (4.0 / epsilon) * ((row_mass * row_dot)[:, None] * x
                                      - row_dot[:, None] * py
                                      - py_dot[:, None] * x + b5))
    return implicit / epsilon + explicit
