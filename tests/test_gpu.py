"""Run with CUDA_VISIBLE_DEVICES=<allocated GPU> python -m pytest --require-gpu."""

import pytest
import torch

from experiments.datasets import make_dataset
from flashopw import sinkhorn_dense, sinkhorn_flash, apply_plan, materialize_plan, diagnostics, point_gradients

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("d", [1, 2, 7, 32, 64, 129, 256])
@pytest.mark.parametrize("precision", ["ieee", "tf32x3"])
@pytest.mark.parametrize("schedule", ["alternating", "symmetric"])
def test_updates_vs_float64_reference(d, precision, schedule):
    x, y, a, b = make_dataset("gaussian", 37, 79, d, device="cuda", weighted=True)
    kw = dict(epsilon=0.17, n_iters=40, schedule=schedule)
    ref = sinkhorn_dense(x.double(), y.double(), a=a.double()/a.double().sum(), b=b.double()/b.double().sum(), **kw)
    result = sinkhorn_flash(x, y, a=a, b=b, precision=precision, **kw)
    torch.testing.assert_close(materialize_plan(result).double(), materialize_plan(ref), rtol=5e-4, atol=2e-7)
    torch.testing.assert_close(result.f.double(), ref.f, rtol=5e-4, atol=2e-5)
    torch.testing.assert_close(result.g.double(), ref.g, rtol=5e-4, atol=2e-5)


@pytest.mark.parametrize("iterations", [1, 150])
@pytest.mark.parametrize("precision", ["ieee", "tf32x3"])
def test_transport_actual_masses_and_gradients(iterations, precision):
    x, y, a, b = make_dataset("rings", 33, 73, 2, device="cuda", weighted=True)
    result = sinkhorn_flash(x, y, a=a, b=b, n_iters=iterations, epsilon=0.1, precision=precision)
    plan = materialize_plan(result)
    values = torch.linspace(-1, 2, len(y)*35, device="cuda").reshape(len(y), 35)
    torch.testing.assert_close(apply_plan(result, values), plan @ values, rtol=5e-4, atol=3e-7)
    torch.testing.assert_close(apply_plan(result, x, transpose=True), plan.T @ x, rtol=5e-4, atol=3e-7)
    torch.testing.assert_close(apply_plan(result, torch.ones_like(b)), plan.sum(1), rtol=5e-4, atol=3e-7)
    gx, gy = point_gradients(result)
    torch.testing.assert_close(gx, 2*(plan.sum(1)[:, None]*x-plan@y), rtol=5e-4, atol=3e-7)
    torch.testing.assert_close(gy, 2*(plan.sum(0)[:, None]*y-plan.T@x), rtol=5e-4, atol=3e-7)


@pytest.mark.parametrize("shape", [(1, 1), (1, 67), (63, 1), (65, 97)])
def test_masks_half_cost_noncontiguous_and_low_epsilon(shape):
    n, m = shape
    x, y, a, b = make_dataset("gaussian", n, m, 6, device="cuda", weighted=True)
    x, y = x[:, ::2], y[:, ::2]
    kw = dict(a=a, b=b, n_iters=50, cost_scale=0.5, epsilon=0.03)
    ref = sinkhorn_dense(x, y, **kw)
    result = sinkhorn_flash(x, y, block_m=16, block_n=32, **kw)
    torch.testing.assert_close(materialize_plan(result), materialize_plan(ref), rtol=8e-4, atol=2e-6)


def test_gpu_early_stopping():
    x, y, a, b = make_dataset("gaussian", 37, 71, 3, device="cuda")
    result = sinkhorn_flash(x, y, a=a, b=b, n_iters=300, epsilon=0.5, tol=1e-4, check_every=10)
    assert result.n_iters < 300
    stats = diagnostics(result)
    assert max(stats["row_l1"], stats["col_l1"]) < 1e-4
