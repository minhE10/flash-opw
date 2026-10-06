"""Original implementation of the streaming recurrences in Ye et al. (2026).

Only imported for CUDA execution. Each program keeps its query tile and online
statistics in SRAM/registers and streams the key tiles. No n*m HBM tensors.
"""

import triton
import triton.language as tl


def launch_config(d, block_m, block_n):
    """Treat requested tiles as upper bounds; fit RTX-class shared memory.

    Large feature padding plus tf32x3 needs extra shared buffers. Cap both
    tiles and pipelining instead of assuming datacenter-sized shared memory.
    """
    if d > 64:
        return min(block_m, 16), min(block_n, 32), 1
    return min(block_m, 32), min(block_n, 64), 2


def feature_block(d):
    """Return the per-dot feature tile; larger dimensions are accumulated in chunks."""
    return min(128, max(32, triton.next_power_of_2(d)))


def value_block(columns, d, precision=None):
    """Amortize the score/LSE pass over more output channels when possible.

    A separate program recomputes every score tile for each output block.
    Using 32 channels for P @ Y at d=1024 repeated that work 32 times;
    128 channels needs only eight passes while fitting RTX shared memory with
    the conservative 16x32 query/key tiles used for large dimensions.
    """
    if columns < 64 or (precision == "tf32x3" and d <= 64):
        return 32
    # The d<=64 launch uses 32x64 tiles, twice the score area of the wide-d
    # 16x32 launch. Keep its output tile smaller to avoid excessive residency.
    return 64 if d <= 64 else 128


def hadamard_launch_config(d, rank, block_m, block_n):
    """Use smaller tiles when the HVP kernel keeps two dot products live."""
    block_m, block_n, stages = launch_config(d, block_m, block_n)
    if max(d, rank) >= 64:
        return min(block_m, 16), min(block_n, 32), min(stages, 1)
    return block_m, block_n, stages


@triton.jit
def _update_kernel(Q, K, OLD, BIAS, LOGW, OUT,
                   N: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                   SCALE: tl.constexpr, SYMMETRIC: tl.constexpr,
                   PRECISION: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    running = tl.full((BM,), -float("inf"), tl.float32)
    total = tl.zeros((BM,), tl.float32)
    for start in range(tl.cdiv(M, BN)):
        cols = start * BN + tl.arange(0, BN)
        bias = tl.load(BIAS + cols, cols < M, other=0)
        scores = tl.zeros((BM, BN), tl.float32)
        for dim_start in range(0, D, BD):
            dims = dim_start + tl.arange(0, BD)
            q = tl.load(Q + rows[:, None] * D + dims[None, :],
                        (rows[:, None] < N) & (dims[None, :] < D), other=0)
            k = tl.load(K + cols[None, :] * D + dims[:, None],
                        (cols[None, :] < M) & (dims[:, None] < D), other=0)
            scores += tl.dot(q, k, input_precision=PRECISION)
        scores = scores * SCALE + bias[None, :]
        scores = tl.where(cols[None, :] < M, scores, -float("inf"))
        new_max = tl.maximum(running, tl.max(scores, 1))
        total = total * tl.exp(running - new_max) + tl.sum(tl.exp(scores - new_max[:, None]), 1)
        running = new_max
    value = tl.load(LOGW + rows, rows < N, other=0) - (running + tl.log(total))
    if SYMMETRIC:
        value = 0.5 * (tl.load(OLD + rows, rows < N, other=0) + value)
    tl.store(OUT + rows, value, rows < N)


@triton.jit
def _symmetric_update_kernel(Q, K, U, V, LOGA, LOGB, UOUT, VOUT,
                             N: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                             SCALE: tl.constexpr, PRECISION: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    """Compute both Jacobi/symmetric half-steps in one kernel launch.

    Programs ``[0, ceil(N/BM))`` update U by reducing rows of Q @ K.T;
    the remaining programs update V by reducing columns of the same logical
    score matrix. Both branches read the old U,V buffers, so there is no
    cross-program dependency and the two updates match equations (4)-(5).
    """
    pid = tl.program_id(0)
    u_blocks = tl.cdiv(N, BM)

    if pid < u_blocks:
        u_rows = pid * BM + tl.arange(0, BM)
        u_running = tl.full((BM,), -float("inf"), tl.float32)
        u_total = tl.zeros((BM,), tl.float32)
        for u_start in range(tl.cdiv(M, BN)):
            u_cols = u_start * BN + tl.arange(0, BN)
            u_bias = tl.load(V + u_cols, u_cols < M, other=0)
            u_scores = tl.zeros((BM, BN), tl.float32)
            for u_dim_start in range(0, D, BD):
                u_dims = u_dim_start + tl.arange(0, BD)
                u_q = tl.load(Q + u_rows[:, None] * D + u_dims[None, :],
                              (u_rows[:, None] < N) & (u_dims[None, :] < D), other=0)
                u_k = tl.load(K + u_cols[None, :] * D + u_dims[:, None],
                              (u_cols[None, :] < M) & (u_dims[:, None] < D), other=0)
                u_scores += tl.dot(u_q, u_k, input_precision=PRECISION)
            u_scores = u_scores * SCALE + u_bias[None, :]
            u_scores = tl.where(u_cols[None, :] < M, u_scores, -float("inf"))
            u_new_max = tl.maximum(u_running, tl.max(u_scores, 1))
            u_total = (u_total * tl.exp(u_running - u_new_max)
                       + tl.sum(tl.exp(u_scores - u_new_max[:, None]), 1))
            u_running = u_new_max
        u_candidate = tl.load(LOGA + u_rows, u_rows < N, other=0) - (u_running + tl.log(u_total))
        u_old = tl.load(U + u_rows, u_rows < N, other=0)
        tl.store(UOUT + u_rows, 0.5 * (u_old + u_candidate), u_rows < N)
    else:
        # V update. Keep the same BM x BN score-tile orientation and reduce
        # over source rows (axis 0), as in the paper's fused symmetric kernel.
        v_pid = pid - u_blocks
        v_cols = v_pid * BN + tl.arange(0, BN)
        v_running = tl.full((BN,), -float("inf"), tl.float32)
        v_total = tl.zeros((BN,), tl.float32)
        for v_start in range(tl.cdiv(N, BM)):
            v_rows = v_start * BM + tl.arange(0, BM)
            v_bias = tl.load(U + v_rows, v_rows < N, other=0)
            v_scores = tl.zeros((BM, BN), tl.float32)
            for v_dim_start in range(0, D, BD):
                v_dims = v_dim_start + tl.arange(0, BD)
                v_q = tl.load(Q + v_rows[:, None] * D + v_dims[None, :],
                              (v_rows[:, None] < N) & (v_dims[None, :] < D), other=0)
                v_k = tl.load(K + v_cols[None, :] * D + v_dims[:, None],
                              (v_cols[None, :] < M) & (v_dims[:, None] < D), other=0)
                v_scores += tl.dot(v_q, v_k, input_precision=PRECISION)
            v_scores = v_scores * SCALE + v_bias[:, None]
            v_scores = tl.where((v_rows[:, None] < N) & (v_cols[None, :] < M),
                                v_scores, -float("inf"))
            v_new_max = tl.maximum(v_running, tl.max(v_scores, 0))
            v_total = (v_total * tl.exp(v_running - v_new_max)
                       + tl.sum(tl.exp(v_scores - v_new_max[None, :]), 0))
            v_running = v_new_max
        v_candidate = tl.load(LOGB + v_cols, v_cols < M, other=0) - (v_running + tl.log(v_total))
        v_old = tl.load(V + v_cols, v_cols < M, other=0)
        tl.store(VOUT + v_cols, 0.5 * (v_old + v_candidate), v_cols < M)


@triton.jit
def _apply_kernel(Q, K, U, V, VALUES, OUT,
                  N: tl.constexpr, M: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
                  SCALE: tl.constexpr, PRECISION: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr, BP: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    features = tl.program_id(1) * BP + tl.arange(0, BP)
    running = tl.full((BM,), -float("inf"), tl.float32)
    acc = tl.zeros((BM, BP), tl.float32)
    for start in range(tl.cdiv(M, BN)):
        cols = start * BN + tl.arange(0, BN)
        bias = tl.load(V + cols, cols < M, other=0)
        scores = tl.zeros((BM, BN), tl.float32)
        for dim_start in range(0, D, BD):
            dims = dim_start + tl.arange(0, BD)
            q = tl.load(Q + rows[:, None] * D + dims[None, :],
                        (rows[:, None] < N) & (dims[None, :] < D), other=0)
            k = tl.load(K + cols[None, :] * D + dims[:, None],
                        (cols[None, :] < M) & (dims[:, None] < D), other=0)
            scores += tl.dot(q, k, input_precision=PRECISION)
        scores = scores * SCALE + bias[None, :]
        scores = tl.where(cols[None, :] < M, scores, -float("inf"))
        new_max = tl.maximum(running, tl.max(scores, 1))
        probs = tl.exp(scores - new_max[:, None])
        values = tl.load(VALUES + cols[:, None] * P + features[None, :],
                         (cols[:, None] < M) & (features[None, :] < P), other=0)
        acc = acc * tl.exp(running - new_max)[:, None]
        acc = acc + tl.dot(probs, values, input_precision=PRECISION)
        running = new_max
    logscale = tl.load(U + rows, rows < N, other=0) + running
    acc = acc * tl.exp(logscale)[:, None]
    tl.store(OUT + rows[:, None] * P + features[None, :], acc,
             (rows[:, None] < N) & (features[None, :] < P))


@triton.jit
def _apply_vector_kernel(Q, K, U, V, VALUES, OUT,
                         N: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                         SCALE: tl.constexpr, PRECISION: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    """Signed P @ vector with online rescaling and a direct row reduction."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    running = tl.full((BM,), -float("inf"), tl.float32)
    acc = tl.zeros((BM,), tl.float32)
    for start in range(tl.cdiv(M, BN)):
        cols = start * BN + tl.arange(0, BN)
        scores = tl.zeros((BM, BN), tl.float32)
        for dim_start in range(0, D, BD):
            dims = dim_start + tl.arange(0, BD)
            q = tl.load(Q + rows[:, None] * D + dims[None, :],
                        (rows[:, None] < N) & (dims[None, :] < D), other=0)
            k = tl.load(K + cols[None, :] * D + dims[:, None],
                        (cols[None, :] < M) & (dims[:, None] < D), other=0)
            scores += tl.dot(q, k, input_precision=PRECISION)
        bias = tl.load(V + cols, cols < M, other=0)
        scores = tl.where(cols[None, :] < M,
                          scores * SCALE + bias[None, :], -float("inf"))
        new_max = tl.maximum(running, tl.max(scores, 1))
        weights = tl.exp(scores - new_max[:, None])
        values = tl.load(VALUES + cols, cols < M, other=0)
        acc = acc * tl.exp(running - new_max) + tl.sum(weights * values[None, :], 1)
        running = new_max
    logscale = tl.load(U + rows, rows < N, other=0) + running
    tl.store(OUT + rows, acc * tl.exp(logscale), rows < N)


@triton.jit
def _gradient_kernel(Q, K, U, V, OUT,
                     N: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                     SCALE: tl.constexpr, GAMMA: tl.constexpr, PRECISION: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr):
    """Reuse each score tile across all feature blocks; retain actual masses.

    A tuple of feature accumulators stays live across the key loop. No global
    scratch or atomic writes, and no repeated score pass per output block.
    """
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    running = tl.full((BM,), -float("inf"), tl.float32)
    mass = tl.zeros((BM,), tl.float32)
    feature_blocks: tl.constexpr = (D + BG - 1) // BG
    accumulators = ()
    for feature in tl.static_range(feature_blocks):
        accumulators += (tl.zeros((BM, BG), tl.float32),)
    for start in range(tl.cdiv(M, BN)):
        cols = start * BN + tl.arange(0, BN)
        scores = tl.zeros((BM, BN), tl.float32)
        for dim_start in range(0, D, BD):
            dims = dim_start + tl.arange(0, BD)
            q = tl.load(Q + rows[:, None] * D + dims[None, :],
                        (rows[:, None] < N) & (dims[None, :] < D), other=0)
            k = tl.load(K + cols[None, :] * D + dims[:, None],
                        (cols[None, :] < M) & (dims[:, None] < D), other=0)
            scores += tl.dot(q, k, input_precision=PRECISION)
        bias = tl.load(V + cols, cols < M, other=0)
        scores = tl.where(cols[None, :] < M,
                          scores * SCALE + bias[None, :], -float("inf"))
        new_max = tl.maximum(running, tl.max(scores, 1))
        rescale = tl.exp(running - new_max)
        weights = tl.exp(scores - new_max[:, None])
        mass = mass * rescale + tl.sum(weights, 1)
        next_accumulators = ()
        for feature in tl.static_range(feature_blocks):
            dims = feature * BG + tl.arange(0, BG)
            values = tl.load(K + cols[:, None] * D + dims[None, :],
                             (cols[:, None] < M) & (dims[None, :] < D), other=0)
            next_accumulators += (accumulators[feature] * rescale[:, None]
                                  + tl.dot(weights, values, input_precision=PRECISION),)
        accumulators = next_accumulators
        running = new_max
    factor = GAMMA * tl.exp(tl.load(U + rows, rows < N, other=0) + running)
    for feature in tl.static_range(feature_blocks):
        dims = feature * BG + tl.arange(0, BG)
        mask = (rows[:, None] < N) & (dims[None, :] < D)
        points = tl.load(Q + rows[:, None] * D + dims[None, :], mask, other=0)
        gradient = factor[:, None] * (mass[:, None] * points - accumulators[feature])
        tl.store(OUT + rows[:, None] * D + dims[None, :], gradient, mask)


@triton.jit
def _hadamard_apply_kernel(Q, K, U, V, LEFT, RIGHT, VALUES, OUT,
                           N: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                           R: tl.constexpr, P: tl.constexpr,
                           SCALE: tl.constexpr, PRECISION: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
                           BR: tl.constexpr, BP: tl.constexpr):
    """Stream ``(P * (LEFT @ RIGHT.T)) @ VALUES``.

    The score normalization follows ``_apply_kernel``. The signed Hadamard
    factor is accumulated only after the stable exponential rescaling.
    """
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    features = tl.program_id(1) * BP + tl.arange(0, BP)
    running = tl.full((BM,), -float("inf"), tl.float32)
    acc = tl.zeros((BM, BP), tl.float32)
    for start in range(tl.cdiv(M, BN)):
        cols = start * BN + tl.arange(0, BN)
        bias = tl.load(V + cols, cols < M, other=0)
        scores = tl.zeros((BM, BN), tl.float32)
        for dim_start in range(0, D, BD):
            dims = dim_start + tl.arange(0, BD)
            q = tl.load(Q + rows[:, None] * D + dims[None, :],
                        (rows[:, None] < N) & (dims[None, :] < D), other=0)
            k = tl.load(K + cols[None, :] * D + dims[:, None],
                        (cols[None, :] < M) & (dims[:, None] < D), other=0)
            scores += tl.dot(q, k, input_precision=PRECISION)
        factors = tl.zeros((BM, BN), tl.float32)
        for rank_start in range(0, R, BR):
            ranks = rank_start + tl.arange(0, BR)
            left = tl.load(LEFT + rows[:, None] * R + ranks[None, :],
                           (rows[:, None] < N) & (ranks[None, :] < R), other=0)
            right = tl.load(RIGHT + cols[None, :] * R + ranks[:, None],
                            (cols[None, :] < M) & (ranks[:, None] < R), other=0)
            factors += tl.dot(left, right, input_precision="ieee")
        scores = scores * SCALE + bias[None, :]
        scores = tl.where(cols[None, :] < M, scores, -float("inf"))
        new_max = tl.maximum(running, tl.max(scores, 1))
        weighted = tl.exp(scores - new_max[:, None]) * factors
        values = tl.load(VALUES + cols[:, None] * P + features[None, :],
                         (cols[:, None] < M) & (features[None, :] < P), other=0)
        acc = acc * tl.exp(running - new_max)[:, None]
        acc = acc + tl.dot(weighted, values, input_precision="ieee")
        running = new_max
    logscale = tl.load(U + rows, rows < N, other=0) + running
    acc = acc * tl.exp(logscale)[:, None]
    tl.store(OUT + rows[:, None] * P + features[None, :], acc,
             (rows[:, None] < N) & (features[None, :] < P))


def update(q, k, old, bias, logw, out, scale, symmetric, precision, block_m, block_n):
    import torch
    from .kernel_tuning import launch

    bm, bn, stages = launch_config(q.shape[1], block_m, block_n)
    with torch.cuda.device(q.device):
        launch(_update_kernel, (q, k, old, bias, logw, out),
               dict(N=len(q), M=len(k), D=q.shape[1], SCALE=scale,
                    SYMMETRIC=symmetric, PRECISION=precision, BD=feature_block(q.shape[1])),
               lambda c: (triton.cdiv(len(q), c[0]),),
               (bm, bn, stages, 4), block_m, block_n)


def symmetric_update(q, k, u, v, loga, logb, uout, vout,
                     scale, precision, block_m, block_n):
    """Launch one fused symmetric step for both shifted potentials."""
    import torch
    from .kernel_tuning import launch

    bm, bn, stages = launch_config(q.shape[1], block_m, block_n)
    with torch.cuda.device(q.device):
        launch(_symmetric_update_kernel, (q, k, u, v, loga, logb, uout, vout),
               dict(N=len(q), M=len(k), D=q.shape[1], SCALE=scale,
                    PRECISION=precision, BD=feature_block(q.shape[1])),
               lambda c: (triton.cdiv(len(q), c[0]) + triton.cdiv(len(k), c[1]),),
               (bm, bn, stages, 4), block_m, block_n)


def apply(q, k, u, v, values, scale, precision, block_m, block_n):
    import torch
    import os
    from .kernel_tuning import launch

    bm, bn, stages = launch_config(q.shape[1], block_m, block_n)
    block_p = value_block(values.shape[1], q.shape[1], precision)
    out = torch.empty((len(q), values.shape[1]), device=q.device, dtype=q.dtype)
    constants = dict(N=len(q), M=len(k), D=q.shape[1], SCALE=scale,
                     PRECISION=precision, BD=feature_block(q.shape[1]))
    with torch.cuda.device(q.device):
        if values.shape[1] == 1 and os.environ.get("FLASHOPW_VECTOR_KERNEL", "1") == "1":
            launch(_apply_vector_kernel, (q, k, u, v, values, out), constants,
                   lambda c: (triton.cdiv(len(q), c[0]),),
                   (bm, bn, stages, 4), block_m, block_n)
            return out
        launch(_apply_kernel, (q, k, u, v, values, out),
               dict(constants, P=values.shape[1], BP=block_p),
               lambda c: (triton.cdiv(len(q), c[0]), triton.cdiv(values.shape[1], block_p)),
               (bm, bn, stages, 4), block_m, block_n)
    return out


def gradient(q, k, u, v, scale, gamma, precision, block_m, block_n):
    import torch
    from .kernel_tuning import launch

    bm, bn, stages = launch_config(q.shape[1], block_m, block_n)
    # Keeping all output features live can be register-heavy. Start with the
    # smallest row tile; tuning may select a larger one after compilation.
    bm = min(bm, 16)
    out = torch.empty_like(q)
    with torch.cuda.device(q.device):
        launch(_gradient_kernel, (q, k, u, v, out),
               dict(N=len(q), M=len(k), D=q.shape[1], SCALE=scale, GAMMA=gamma,
                    PRECISION=precision, BD=feature_block(q.shape[1]), BG=32),
               lambda c: (triton.cdiv(len(q), c[0]),),
               (bm, bn, 1, 4), block_m, block_n, gradient=True)
    return out


def hadamard_apply(q, k, u, v, left, right, values, scale, precision, block_m, block_n):
    import torch
    from .kernel_tuning import launch

    bm, bn, stages = hadamard_launch_config(
        q.shape[1], left.shape[1], block_m, block_n,
    )
    # This kernel also keeps a second feature dot product live. Cap its value
    # tile at 64 for wide inputs to preserve the 64 KiB shared-memory budget.
    block_p = (min(64, value_block(values.shape[1], q.shape[1], precision))
               if max(q.shape[1], left.shape[1]) >= 256 else
               value_block(values.shape[1], q.shape[1], precision))
    out = torch.empty((len(q), values.shape[1]), device=q.device, dtype=q.dtype)
    with torch.cuda.device(q.device):
        launch(_hadamard_apply_kernel, (q, k, u, v, left, right, values, out),
               dict(N=len(q), M=len(k), D=q.shape[1], R=left.shape[1], P=values.shape[1],
                    SCALE=scale, PRECISION=precision, BD=feature_block(q.shape[1]),
                    BR=feature_block(left.shape[1]), BP=block_p),
               lambda c: (triton.cdiv(len(q), c[0]), triton.cdiv(values.shape[1], block_p)),
               (bm, bn, stages, 4), block_m, block_n, hadamard=True)
    return out
