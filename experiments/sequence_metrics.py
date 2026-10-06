"""CPU reference distances from OWD_journal.pdf, Sections 3--5.

All costs here use the common squared-Euclidean spatial ground cost requested
in main (2).pdf. Fixed hyperparameters are a trial, not the journal's grid search.
"""

from functools import lru_cache
import math

import numpy as np
from numba import njit
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


JOURNAL_METRICS = ("dtw", "ldtw", "ndtw", "soft-dtw", "ot", "sinkhorn",
                   "tlp", "opw-kl", "tcot", "opw")


@njit(cache=True)
def dtw_cost_and_length(cost):
    """Boundary/continuity/monotonicity DTW; diagonal wins exact path ties."""
    n, m = cost.shape
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    prev_length = np.zeros(m + 1, dtype=np.int64)
    for i in range(n):
        current = np.full(m + 1, np.inf)
        current_length = np.zeros(m + 1, dtype=np.int64)
        for j in range(m):
            best, length = prev[j], prev_length[j]
            if prev[j+1] < best:
                best, length = prev[j+1], prev_length[j+1]
            if current[j] < best:
                best, length = current[j], current_length[j]
            current[j+1] = cost[i, j] + best
            current_length[j+1] = length + 1
        prev, prev_length = current, current_length
    return prev[m], prev_length[m]


@njit(cache=True)
def soft_dtw_cost(cost, gamma):
    n, m = cost.shape
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(n):
        current = np.full(m + 1, np.inf)
        for j in range(m):
            a, b, c = prev[j], prev[j+1], current[j]
            minimum = min(a, b, c)
            softmin = minimum - gamma * math.log(
                math.exp((minimum-a)/gamma) + math.exp((minimum-b)/gamma)
                + math.exp((minimum-c)/gamma))
            current[j+1] = cost[i, j] + softmin
        prev = current
    return prev[m]


def entropic_plan(cost, epsilon, n_iters, *, a=None, b=None, return_potentials=False):
    """Independent dense, unshifted FP64 log-Sinkhorn, f then g updates."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or min(cost.shape) < 1 or not np.isfinite(cost).all():
        raise ValueError("cost must be a finite nonempty matrix")
    if not math.isfinite(epsilon) or epsilon <= 0 or n_iters < 1:
        raise ValueError("positive epsilon and n_iters required")
    n, m = cost.shape
    a = np.full(n, 1/n) if a is None else np.asarray(a, dtype=np.float64)
    b = np.full(m, 1/m) if b is None else np.asarray(b, dtype=np.float64)
    if (a.shape != (n,) or b.shape != (m,) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or (a <= 0).any() or (b <= 0).any()
            or not np.isclose(a.sum(), 1) or not np.isclose(b.sum(), 1)):
        raise ValueError("a,b must be strictly positive probability weights")
    loga, logb = np.log(a), np.log(b)
    f, g = np.zeros(n), np.zeros(m)
    for _ in range(n_iters):
        f = -epsilon * logsumexp((g[None, :] - cost)/epsilon + logb, axis=1)
        g = -epsilon * logsumexp((f[:, None] - cost)/epsilon + loga[:, None], axis=0)
    plan = np.exp((f[:, None] + g[None, :] - cost)/epsilon + loga[:, None] + logb)
    return (plan, f, g, a, b) if return_potentials else plan


@lru_cache(maxsize=16)
def _transport_constraints(n, m):
    indices = np.arange(n*m)
    rows = np.concatenate((np.repeat(np.arange(n), m), n + np.tile(np.arange(m), n)))
    cols = np.concatenate((indices, indices))
    return coo_matrix((np.ones(2*n*m), (rows, cols)), shape=(n+m, n*m)).tocsr()


def exact_ot_cost(cost):
    """Uniform unregularized OT, sparse LP solved by HiGHS; failures propagate."""
    n, m = cost.shape
    output = linprog(cost.ravel(), A_eq=_transport_constraints(n, m),
                     b_eq=np.concatenate((np.full(n, 1/n), np.full(m, 1/m))),
                     bounds=(0, None), method="highs")
    if not output.success:
        raise RuntimeError(f"OT LP failed: {output.message}")
    return float(output.fun)


def reference_distance(metric, x, y, *, lambda1=50, lambda2=0.1, sigma=1,
                       epsilon=0.1, n_iters=20, sinkhorn_iters=100,
                       tcot_lambda=1, tlp_weight=50, soft_dtw_gamma=0.1,
                       cost_scale=1, opw_score="pdf-loss"):
    """Original journal OPW uses perpendicular distance, NOT PDF relative F.

    Dense affine OPW and exact-relative OPW are extra controls for the proposed
    modification. Only those use the main PDF's F=(i/N-j/M)^2.
    """
    spatial = cost_scale * cdist(x, y, metric="sqeuclidean")
    if metric in ("dtw", "ldtw", "ndtw"):
        value, steps = dtw_cost_and_length(spatial)
        return float(value / (len(x) if metric == "ldtw" else steps if metric == "ndtw" else 1))
    if metric == "soft-dtw":
        return float(soft_dtw_cost(spatial, soft_dtw_gamma))
    if metric == "ot":
        return exact_ot_cost(spatial)
    t = np.arange(1, len(x)+1)/len(x)
    s = np.arange(1, len(y)+1)/len(y)
    delta = t[:, None] - s[None, :]
    delta2 = delta**2
    evaluation = spatial
    iterations = n_iters
    if metric == "sinkhorn":
        cost, eps, iterations = spatial, epsilon, sinkhorn_iters
    elif metric == "tcot":
        cost = spatial * (1 + np.abs(delta))
        evaluation, eps, iterations = cost, 1/tcot_lambda, sinkhorn_iters
    elif metric == "tlp":
        cost = spatial + tlp_weight * delta2
        evaluation, eps, iterations = cost, epsilon, sinkhorn_iters
    elif metric in ("opw", "opw-kl", "opw-exact-relative"):
        prior_squared_distance = (delta2 if metric == "opw-exact-relative" else
                                  delta2/(1/len(x)**2 + 1/len(y)**2))
        inverse_moment = 1/(1 + delta2)
        cost = spatial - (0 if metric == "opw-kl" else lambda1) * inverse_moment
        cost = cost + lambda2 * (prior_squared_distance/(2*sigma**2) + math.log(sigma*math.sqrt(2*math.pi)))
        eps = lambda2
    elif metric == "affine-opw-dense":
        mu = lambda1 + lambda2/(2*sigma**2)
        cost = spatial + mu * delta2
        eps = lambda2
    else:
        raise ValueError(f"Unknown sequence distance: {metric}")
    if metric == "affine-opw-dense" and opw_score == "pdf-loss":
        _, f, g, a, b = entropic_plan(cost, eps, iterations, return_potentials=True)
        q0 = lambda2 * math.log(sigma*math.sqrt(2*math.pi)) - lambda1
        return float(a @ f + b @ g - lambda2 + q0)
    plan = entropic_plan(cost, eps, iterations)
    return float(np.sum(plan * evaluation))
