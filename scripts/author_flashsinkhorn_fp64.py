"""Independent small dense references: FP64 CPU, no author imports or CG.

GeomLoss potentials use P = a b exp((f+g-C)/eps). The HVP reference
differentiates the marginal equations and solves the full KKT system,
rather than using the author's Schur/CG or five-term HVP implementation.
"""
import torch


def plan(x, y, a, b, f, g, eps, cost_scale=1.0):
    cost = cost_scale * (x[:, None] - y[None, :]).square().sum(-1)
    return torch.exp(a.log()[:, None] + b.log()[None, :] + (f[:, None] + g[None, :] - cost) / eps)


def solve(x, y, a, b, eps, cost_scale=1.0, iterations=1000):
    """Fixed-budget log-domain alternating solve; caller must check residual.

    Inputs are already FP64. We preserve the actual FP32-rounded weights,
    including their tiny total-mass mismatch, instead of renormalizing them.
    """
    cost = cost_scale * (x[:, None] - y[None, :]).square().sum(-1)
    f, g = torch.zeros_like(a), torch.zeros_like(b)
    for _ in range(iterations):
        f = -eps * torch.logsumexp(b.log()[None, :] + (g[None, :] - cost) / eps, dim=1)
        g = -eps * torch.logsumexp(a.log()[:, None] + (f[:, None] - cost) / eps, dim=0)
    return f, g, plan(x, y, a, b, f, g, eps, cost_scale)


def direct_hvp(x, y, p, v, eps, cost_scale=1.0, tau2=1e-5):
    """Damped implicit HVP at a given plan; tau2 matches the author's damping.

    [diag(r), P; P.T, diag(c)+eps*tau2 I] [df;dg] = sums(P*dC).
    dP=P*(df+dg-dC)/eps. With tau2=0 we fix the last dg for the gauge.
    Damping is NOT part of the OT forward objective; tau2>0 gives a
    regularized implicit product, not an exact undamped Hessian.
    """
    n, m = p.shape
    r, c = p.sum(1), p.sum(0)
    delta = x[:, None] - y[None, :]
    dc = 2 * cost_scale * (delta * v[:, None]).sum(-1)
    matrix = torch.cat((torch.cat((torch.diag(r), p), 1),
                        torch.cat((p.T, torch.diag(c + eps * tau2)), 1)), 0)
    rhs = torch.cat(((p * dc).sum(1), (p * dc).sum(0)))
    if tau2 == 0:
        sol = torch.cat((torch.linalg.solve(matrix[:-1, :-1], rhs[:-1]), rhs.new_zeros(1)))
    else:
        sol = torch.linalg.solve(matrix, rhs)
    dp = p * (sol[:n, None] + sol[None, n:] - dc) / eps
    hvp = 2 * cost_scale * (r[:, None] * v + (dp[:, :, None] * delta).sum(1))
    relative_residual = torch.linalg.vector_norm(matrix @ sol - rhs) / torch.linalg.vector_norm(rhs).clamp_min(1e-30)
    return hvp, float(relative_residual)
