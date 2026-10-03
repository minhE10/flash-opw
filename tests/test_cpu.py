import pytest
import torch

from experiments.datasets import make_dataset
from flashopw import (sinkhorn_dense, sinkhorn_flash, sinkhorn_online,
                     materialize_plan, apply_plan, diagnostics, point_gradients)


@pytest.mark.parametrize("schedule", ["alternating", "symmetric"])
@pytest.mark.parametrize("shape", [(1, 1, 1), (7, 13, 3), (33, 65, 17)])
@pytest.mark.parametrize("cost_scale", [0.5, 1.0])
def test_streamed_recurrence_against_independent_dense(schedule, shape, cost_scale):
    n, m, d = shape
    x, y, a, b = make_dataset("gaussian", n, m, d, weighted=True, dtype=torch.float64)
    kw = dict(a=a, b=b, n_iters=31, epsilon=0.13, cost_scale=cost_scale, schedule=schedule)
    ref, got = sinkhorn_dense(x, y, **kw), sinkhorn_online(x, y, block_m=7, block_n=11, **kw)
    torch.testing.assert_close(got.f, ref.f, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(got.g, ref.g, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(materialize_plan(got), materialize_plan(ref), atol=1e-12, rtol=1e-11)


def test_classical_scaling_reference_and_objective():
    x, y, a, b = make_dataset("gaussian", 19, 23, 2, weighted=True, dtype=torch.float64)
    epsilon = 0.3
    cost = ((x[:, None, :] - y[None, :, :]) ** 2).sum(2)
    kernel = (-cost / epsilon).exp()
    right = b.clone()
    for _ in range(200):
        left = a / (kernel @ right)
        right = b / (kernel.T @ left)
    expected = left[:, None] * kernel * right[None, :]
    result = sinkhorn_online(x, y, a=a, b=b, epsilon=epsilon, n_iters=200)
    plan = materialize_plan(result)
    torch.testing.assert_close(plan, expected, rtol=1e-10, atol=1e-12)
    kl = (plan * (plan.log() - a.log()[:, None] - b.log()[None, :]) - plan + a[:, None]*b).sum()
    diag = diagnostics(result)
    assert diag["regularized_primal"] == pytest.approx(float((plan * cost).sum() + epsilon * kl), abs=1e-12)
    assert diag["transport_cost"] == pytest.approx(float((plan * cost).sum()), abs=1e-12)
    assert max(diag["row_l1"], diag["col_l1"]) < 1e-12


@pytest.mark.parametrize("iterations", [1, 100])
def test_apply_transpose_signed_values_and_actual_masses(iterations):
    x, y, a, b = make_dataset("rings", 17, 29, 2, weighted=True, dtype=torch.float64)
    result = sinkhorn_online(x, y, a=a, b=b, epsilon=0.07, n_iters=iterations, block_n=9)
    plan = materialize_plan(result)
    v = torch.linspace(-2, 3, len(y) * 5, dtype=x.dtype).reshape(len(y), 5)
    torch.testing.assert_close(apply_plan(result, v), plan @ v)
    torch.testing.assert_close(apply_plan(result, x, transpose=True), plan.T @ x)
    torch.testing.assert_close(apply_plan(result, torch.ones_like(b)), plan.sum(1))
    if iterations == 1:
        assert float((plan.sum(1) - a).abs().sum()) > 1e-3


@pytest.mark.parametrize("solver", [sinkhorn_dense, sinkhorn_online])
def test_early_stopping(solver):
    x, y, a, b = make_dataset("gaussian", 11, 17, dtype=torch.float64)
    result = solver(x, y, a=a, b=b, n_iters=300, epsilon=0.5, tol=1e-9, check_every=5)
    assert result.n_iters < 300
    assert max(diagnostics(result)[k] for k in ("row_l1", "col_l1")) <= 1e-9


def test_small_epsilon_and_constant_cloud():
    x = torch.zeros(9, 3, dtype=torch.float64)
    y = torch.full((11, 3), 0.2, dtype=torch.float64)
    result = sinkhorn_online(x, y, epsilon=1e-4, n_iters=5)
    torch.testing.assert_close(materialize_plan(result), torch.full((9, 11), 1/99, dtype=x.dtype))
    assert diagnostics(result)["regularized_primal"] == pytest.approx(0.12, abs=1e-10)


def test_gradient_finite_difference():
    x, y, a, b = make_dataset("gaussian", 7, 9, dtype=torch.float64, weighted=True)
    kw = dict(a=a, b=b, epsilon=0.3, n_iters=200)
    result = sinkhorn_online(x, y, **kw)
    gx, gy = point_gradients(result)
    h = 1e-5
    for source, expected in ((True, gx[2, 1]), (False, gy[2, 1])):
        plus, minus = (x if source else y).clone(), (x if source else y).clone()
        plus[2, 1] += h
        minus[2, 1] -= h
        rp = sinkhorn_dense(plus, y, **kw) if source else sinkhorn_dense(x, plus, **kw)
        rm = sinkhorn_dense(minus, y, **kw) if source else sinkhorn_dense(x, minus, **kw)
        fd = (diagnostics(rp)["dual"] - diagnostics(rm)["dual"]) / (2*h)
        assert float(expected) == pytest.approx(fd, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("kwargs", [dict(epsilon=0), dict(epsilon=float("nan")), dict(n_iters=0),
    dict(schedule="invalid"), dict(tol=-1), dict(cost_scale=0), dict(block_m=0), dict(check_every=0)])
def test_invalid_parameters(kwargs):
    x = torch.zeros(2, 2)
    with pytest.raises(ValueError):
        sinkhorn_online(x, x, **kwargs)


def test_invalid_weights_and_cpu_triton():
    x = torch.zeros(2, 2)
    for a in (torch.tensor([0., 1.]), torch.tensor([1., 1.]), torch.tensor([float("nan"), 1.])):
        with pytest.raises(ValueError):
            sinkhorn_dense(x, x, a=a)
    with pytest.raises(ValueError, match="CUDA"):
        sinkhorn_flash(x, x)


def test_dataset_reproducibility_and_plan_guard():
    for name in ("gaussian", "mixture", "rings"):
        for p, q in zip(make_dataset(name, 19), make_dataset(name, 19)):
            assert torch.equal(p, q)
    x, y, *_ = make_dataset("gaussian", 5)
    result = sinkhorn_dense(x, y, n_iters=2)
    with pytest.raises(ValueError, match="max_entries"):
        materialize_plan(result, max_entries=24)
