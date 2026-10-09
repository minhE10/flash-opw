"""Validate the independent reference against derivatives of the OT solution."""
import pytest
import torch

from scripts.author_flashsinkhorn_fp64 import direct_hvp, solve


@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_direct_kkt_hvp_matches_finite_difference_of_converged_gradient(scale):
    rng = torch.Generator().manual_seed(19)
    x, y = [0.2 * torch.randn(k, 3, dtype=torch.float64, generator=rng) for k in (7, 5)]
    a, b = [torch.rand(k, dtype=torch.float64, generator=rng) + 0.1 for k in (7, 5)]
    a, b = a / a.sum(), b / b.sum()
    v = torch.randn(x.shape, dtype=x.dtype, generator=rng)
    _, _, p = solve(x, y, a, b, 0.7, scale, iterations=150)
    assert float((p.sum(1) - a).abs().sum()) < 1e-12
    expected, residual = direct_hvp(x, y, p, v, 0.7, scale, tau2=0)
    step = 1e-5
    gradients = []
    for xx in (x + step*v, x - step*v):
        _, _, pp = solve(xx, y, a, b, 0.7, scale, iterations=150)
        gradients.append(2*scale*(pp.sum(1)[:, None]*xx - pp@y))
    torch.testing.assert_close(expected, (gradients[0]-gradients[1])/(2*step), rtol=1e-7, atol=1e-9)
    assert residual < 1e-12


def test_damped_direct_product_is_linear_symmetric_and_has_small_system_residual():
    rng = torch.Generator().manual_seed(29)
    x, y = [torch.randn(k, 2, dtype=torch.float64, generator=rng)*0.2 for k in (9, 6)]
    a, b = torch.full((9,), 1/9, dtype=x.dtype), torch.full((6,), 1/6, dtype=x.dtype)
    _, _, p = solve(x, y, a, b, 0.7, iterations=100)
    u, v = [torch.randn(x.shape, dtype=x.dtype, generator=rng) for _ in range(2)]
    hu, _ = direct_hvp(x, y, p, u, 0.7)
    hv, residual = direct_hvp(x, y, p, v, 0.7)
    combined, _ = direct_hvp(x, y, p, 2*u-v, 0.7)
    torch.testing.assert_close(combined, 2*hu-hv, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close((u*hv).sum(), (v*hu).sum(), rtol=1e-10, atol=1e-10)
    assert residual < 1e-10
