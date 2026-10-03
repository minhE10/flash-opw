import torch

from flashuot.opw import opw_exact_cpu, opw_flash_cpu
from flashuot.sinkhorn import dense_sinkhorn_cpu, flash_sinkhorn_cpu


def test_flash_sinkhorn_matches_dense():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(9, 3, generator=g, dtype=torch.float64)
    y = torch.randn(7, 3, generator=g, dtype=torch.float64)
    dense = dense_sinkhorn_cpu(x, y, epsilon=0.7, n_iters=60, return_transport=True)
    flash = flash_sinkhorn_cpu(x, y, epsilon=0.7, n_iters=60, row_block=3, col_block=4, return_transport=True)
    assert torch.allclose(dense.transport, flash.transport, atol=1e-8, rtol=1e-6)
    assert abs(dense.distance.item() - flash.distance.item()) < 1e-8
    assert flash.marginal_error < 2e-6


def test_flashopw_produces_valid_marginals_and_is_close_when_delta_is_small():
    t = torch.linspace(0, 1, 12, dtype=torch.float64)
    x = torch.stack([t, torch.sin(t)], dim=1)
    y = torch.stack([t + 0.01, torch.sin(t + 0.01)], dim=1)
    exact = opw_exact_cpu(x, y, lambda1=1.0, lambda2=0.5, sigma=1.0, n_iters=80)
    flash = opw_flash_cpu(x, y, lambda1=1.0, lambda2=0.5, sigma=1.0, n_iters=80, return_transport=True)
    assert flash.marginal_error < 1e-6
    assert abs(exact.distance.item() - flash.distance.item()) < 0.2
