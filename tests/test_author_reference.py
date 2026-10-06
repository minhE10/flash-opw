"""Check the shared-coupling conversion and actual author API on CUDA."""

import os
from pathlib import Path

import pytest
import torch

from experiments.author_reference import load_author, ott_potentials
from flashopw import hessian_vector_product, sinkhorn_dense


@pytest.mark.parametrize("cost_scale", [0.5, 1.0, 2.0])
def test_author_potential_conversion_preserves_dense_plan(cost_scale):
    generator = torch.Generator().manual_seed(13)
    x = torch.rand((7, 3), dtype=torch.float64, generator=generator)
    y = torch.rand((11, 3), dtype=torch.float64, generator=generator)
    a = torch.rand(7, dtype=torch.float64, generator=generator)
    b = torch.rand(11, dtype=torch.float64, generator=generator)
    result = sinkhorn_dense(x, y, a=a / a.sum(), b=b / b.sum(),
                            epsilon=0.7, cost_scale=cost_scale, n_iters=100)
    f, g = ott_potentials(result)
    cost = cost_scale * torch.cdist(x, y).square()
    author_log_plan = (f[:, None] + g[None, :] - cost) / result.epsilon
    shifted_log_plan = (2 * cost_scale / result.epsilon * (x @ y.T)
                        + result.u[:, None] + result.v[None, :])
    torch.testing.assert_close(author_log_plan, shifted_log_plan, rtol=1e-12, atol=1e-12)


@pytest.fixture
def author():
    source = os.environ.get("FLASHOPW_AUTHOR_PATH")
    if not source:
        pytest.skip("Set FLASHOPW_AUTHOR_PATH to the pinned checkout to run author CUDA checks")
    return load_author(Path(source))[0]


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["symmetric", "alternating"])
@pytest.mark.parametrize("d", [3, 65, 1024])
def test_author_forward_backward_against_dense_original_schedule(author, backend, d):
    generator = torch.Generator(device="cuda").manual_seed(13)
    x = torch.rand((37, d), device="cuda", generator=generator)
    y = torch.rand((79, d), device="cuda", generator=generator)
    a, b = x.new_full((37,), 1 / 37), x.new_full((79,), 1 / 79)
    eps, iterations = 0.1, 10
    loss_fn = author.SamplesLoss(
        "sinkhorn", backend=backend, eps=eps, n_iters=iterations,
        use_epsilon_scaling=False, last_extrapolation=False,
        debias=False, normalize=False, half_cost=False, allow_tf32=False,
        autotune=True,
    )
    tracked = x.detach().requires_grad_(True)
    actual_loss = loss_fn(a, tracked, b, y)
    actual_grad = torch.autograd.grad(actual_loss, tracked)[0]

    # Independent direct-distance FP64 oracle follows the author's extra full
    # initialization in sym, then n_iters damped updates. No extrapolation.
    xd, yd = x.double(), y.double()
    cost = torch.cdist(xd, yd, compute_mode="donot_use_mm_for_euclid_dist").square()
    loga, logb = a.double().log(), b.double().log()
    f, g = torch.zeros_like(loga), torch.zeros_like(logb)

    def step(f, g):
        next_f = -eps * torch.logsumexp((g[None, :] - cost) / eps + logb, dim=1)
        for_g = next_f if backend == "alternating" else f
        next_g = -eps * torch.logsumexp((for_g[:, None] - cost) / eps + loga[:, None], dim=0)
        return next_f, next_g

    if backend == "symmetric":
        f, g = step(f, g)
    for _ in range(iterations):
        next_f, next_g = step(f, g)
        f, g = ((f + next_f) / 2, (g + next_g) / 2) if backend == "symmetric" else (next_f, next_g)
    expected_loss = (a.double() * f).sum() + (b.double() * g).sum()
    # The original gradient uses target marginal a and conditional row means,
    # rather than the unconverged coupling's actual row mass.
    conditional = torch.softmax((g[None, :] - cost) / eps + logb, dim=1)
    expected_grad = 2 * a.double()[:, None] * (xd - conditional @ yd)
    torch.testing.assert_close(actual_loss.double(), expected_loss, rtol=1e-3, atol=3e-5)
    torch.testing.assert_close(actual_grad.double(), expected_grad, rtol=3e-3, atol=3e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("n,m,d,eps", [(7, 11, 3, 0.7), (37, 79, 65, 0.1)])
@pytest.mark.parametrize("cg_iters", [12, 50])
def test_raw_author_hvp_matches_dense_shared_coupling(author, n, m, d, eps, cg_iters):
    generator = torch.Generator(device="cuda").manual_seed(13)
    x = torch.rand((n, d), device="cuda", generator=generator)
    y = torch.rand((m, d), device="cuda", generator=generator)
    result = sinkhorn_dense(x, y, epsilon=eps, n_iters=200)
    direction = torch.randn(x.shape, device="cuda", generator=generator)
    direction /= direction.norm()
    f, g = ott_potentials(result)
    actual, info = author.hvp_x_sqeuclid_from_potentials(
        x, y, f, g, direction, eps=eps, tau2=1e-5 / eps,
        max_cg_iter=cg_iters, cg_rtol=0, cg_atol=0,
        preconditioner="none", use_preconditioner=False,
        allow_tf32=False, autotune=True,
    )
    expected = hessian_vector_product(result, direction, damping=1e-5,
                                      max_cg_iters=cg_iters, cg_rtol=0, cg_atol=0)
    assert bool(torch.isfinite(actual).all()), info
    error = float((actual - expected).double().norm() / expected.double().norm())
    assert error < 3e-3, (error, info)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-5)
