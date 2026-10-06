"""Dense Torch versions of entropic journal distances, for same-GPU timings.

These are transparent reference implementations, not the authors' repository.
The NumPy/SciPy implementation remains the independent FP64 correctness oracle.
"""

import math

import torch


ENTROPIC_METRICS = ("sinkhorn", "tlp", "opw-kl", "tcot", "opw")


@torch.no_grad()
def dense_tensor_distance(metric, x, y, *, lambda1=1., lambda2=.1, sigma=1.,
                          cost_scale=1., n_iters=20, sinkhorn_iters=100,
                          epsilon=.1, tlp_weight=50., tcot_lambda=1., return_diagnostics=False):
    if metric not in ENTROPIC_METRICS:
        raise ValueError("Dense Torch reference supports entropic journal metrics only")
    if min(n_iters, sinkhorn_iters) < 1:
        raise ValueError("Positive iteration counts required")
    n, m = len(x), len(y)
    spatial = cost_scale * torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist").square()
    t = torch.arange(1, n+1, dtype=x.dtype, device=x.device)/n
    s = torch.arange(1, m+1, dtype=x.dtype, device=x.device)/m
    delta = t[:, None]-s[None, :]
    delta2 = delta.square()
    evaluation = spatial
    if metric == "sinkhorn":
        cost, eps, iterations = spatial, epsilon, sinkhorn_iters
    elif metric == "tlp":
        cost = spatial + tlp_weight*delta2
        evaluation, eps, iterations = cost, epsilon, sinkhorn_iters
    elif metric == "tcot":
        cost = spatial * (1+delta.abs())
        evaluation, eps, iterations = cost, 1/tcot_lambda, sinkhorn_iters
    else:
        moment = 0. if metric == "opw-kl" else lambda1
        cost = spatial - moment/(1+delta2) + lambda2 * (
            delta2/((1/n**2+1/m**2)*2*sigma**2) + math.log(sigma*math.sqrt(2*math.pi)))
        eps, iterations = lambda2, n_iters
    f, g = x.new_zeros(n), y.new_zeros(m)
    for _ in range(iterations):
        f = -eps * torch.logsumexp((g[None, :]-cost)/eps-math.log(m), dim=1)
        g = -eps * torch.logsumexp((f[:, None]-cost)/eps-math.log(n), dim=0)
    plan = ((f[:, None]+g[None, :]-cost)/eps-math.log(n)-math.log(m)).exp()
    score = (plan*evaluation).sum()
    if not return_diagnostics:
        return score
    row_error = (plan.sum(1)-1/n).abs().sum()
    column_error = (plan.sum(0)-1/m).abs().sum()
    return score, row_error, column_error
