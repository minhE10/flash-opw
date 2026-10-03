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


@triton.jit
def _update_kernel(Q, K, OLD, BIAS, LOGW, OUT,
                   N: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                   SCALE: tl.constexpr, SYMMETRIC: tl.constexpr,
                   PRECISION: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    dims = tl.arange(0, BD)
    q = tl.load(Q + rows[:, None] * D + dims[None, :],
                (rows[:, None] < N) & (dims[None, :] < D), other=0)
    running = tl.full((BM,), -float("inf"), tl.float32)
    total = tl.zeros((BM,), tl.float32)
    for start in range(tl.cdiv(M, BN)):
        cols = start * BN + tl.arange(0, BN)
        k = tl.load(K + cols[None, :] * D + dims[:, None],
                    (cols[None, :] < M) & (dims[:, None] < D), other=0)
        bias = tl.load(BIAS + cols, cols < M, other=0)
        scores = tl.dot(q, k, input_precision=PRECISION) * SCALE + bias[None, :]
        scores = tl.where(cols[None, :] < M, scores, -float("inf"))
        new_max = tl.maximum(running, tl.max(scores, 1))
        total = total * tl.exp(running - new_max) + tl.sum(tl.exp(scores - new_max[:, None]), 1)
        running = new_max
    value = tl.load(LOGW + rows, rows < N, other=0) - (running + tl.log(total))
    if SYMMETRIC:
        value = 0.5 * (tl.load(OLD + rows, rows < N, other=0) + value)
    tl.store(OUT + rows, value, rows < N)


@triton.jit
def _apply_kernel(Q, K, U, V, VALUES, OUT,
                  N: tl.constexpr, M: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
                  SCALE: tl.constexpr, PRECISION: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr, BP: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    features = tl.program_id(1) * BP + tl.arange(0, BP)
    dims = tl.arange(0, BD)
    q = tl.load(Q + rows[:, None] * D + dims[None, :],
                (rows[:, None] < N) & (dims[None, :] < D), other=0)
    running = tl.full((BM,), -float("inf"), tl.float32)
    acc = tl.zeros((BM, BP), tl.float32)
    for start in range(tl.cdiv(M, BN)):
        cols = start * BN + tl.arange(0, BN)
        k = tl.load(K + cols[None, :] * D + dims[:, None],
                    (cols[None, :] < M) & (dims[:, None] < D), other=0)
        bias = tl.load(V + cols, cols < M, other=0)
        scores = tl.dot(q, k, input_precision=PRECISION) * SCALE + bias[None, :]
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


def update(q, k, old, bias, logw, out, scale, symmetric, precision, block_m, block_n):
    import torch

    block_m, block_n, stages = launch_config(q.shape[1], block_m, block_n)
    with torch.cuda.device(q.device):
        _update_kernel[(triton.cdiv(len(q), block_m),)](
            q, k, old, bias, logw, out, len(q), len(k), q.shape[1], scale,
            symmetric, precision, block_m, block_n, max(32, triton.next_power_of_2(q.shape[1])),
            num_warps=4, num_stages=stages, enable_fp_fusion=False)


def apply(q, k, u, v, values, scale, precision, block_m, block_n):
    import torch

    block_m, block_n, stages = launch_config(q.shape[1], block_m, block_n)
    out = torch.empty((len(q), values.shape[1]), device=q.device, dtype=q.dtype)
    with torch.cuda.device(q.device):
        _apply_kernel[(triton.cdiv(len(q), block_m), triton.cdiv(values.shape[1], 32))](
            q, k, u, v, values, out, len(q), len(k), q.shape[1], values.shape[1],
            scale, precision, block_m, block_n, max(32, triton.next_power_of_2(q.shape[1])), 32,
            num_warps=4, num_stages=stages, enable_fp_fusion=False)
    return out
