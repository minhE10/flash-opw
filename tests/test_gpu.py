"""Run with CUDA_VISIBLE_DEVICES=<allocated GPU> python -m pytest --require-gpu."""

import pytest
import torch

from experiments.datasets import make_dataset
from flashsinkhorn import (sinkhorn_dense, sinkhorn_flash, apply_plan,
                     apply_plan_hadamard, materialize_plan, diagnostics,
                     point_gradients, hessian_vector_product, sinkhorn_cost)

pytestmark = pytest.mark.gpu


def _assert_relative_l1(actual, expected, *, limit, name):
    """Compare nonnegative plans without over-weighting tiny entries."""
    actual64 = actual.double()
    expected64 = expected.double()
    assert torch.isfinite(actual64).all(), f"{name} contains non-finite values"
    error = (actual64 - expected64).abs().sum()
    scale = expected64.abs().sum().clamp_min(torch.finfo(torch.float64).tiny)
    relative_error = float(error / scale)
    assert relative_error < limit, \
        f"{name} relative L1 error {relative_error:.3e} exceeds {limit:.3e}"


@pytest.mark.parametrize("d", [1, 2, 4, 7, 32, 64, 129, 256, 512, 1024])
@pytest.mark.parametrize("precision", ["ieee", "tf32", "tf32x3"])
@pytest.mark.parametrize("schedule", ["alternating", "symmetric"])
def test_updates_vs_float64_reference(d, precision, schedule):
    x, y, a, b = make_dataset("gaussian", 37, 79, d, device="cuda", weighted=True)
    kw = dict(epsilon=0.17, n_iters=40, schedule=schedule)
    ref = sinkhorn_dense(x.double(), y.double(), a=a.double()/a.double().sum(), b=b.double()/b.double().sum(), **kw)
    result = sinkhorn_flash(x, y, a=a, b=b, precision=precision, **kw)
    actual_plan = materialize_plan(result).double()
    reference_plan = materialize_plan(ref)
    if precision == "tf32":
        # TensorFloat-32 rounds dot-product inputs to a 10-bit mantissa.  A
        # per-entry comparison against an IEEE/float64 plan is ill-conditioned
        # for tiny transport entries, especially at small d.  Check the plan as
        # a probability measure and use the same 1e-3 absolute potential floor
        # as the upstream FlashSinkhorn TF32 parity tests.
        _assert_relative_l1(actual_plan, reference_plan, limit=1e-2,
                            name=f"{schedule} TF32 plan at d={d}")
        torch.testing.assert_close(result.f.double(), ref.f, rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(result.g.double(), ref.g, rtol=1e-2, atol=1e-3)
        return
    if schedule == "symmetric":
        # The fused g branch reduces score tiles along axis 0. Its summation
        # order differs from the dense oracle while remaining mathematically
        # equivalent, so individual tiny plan entries need an absolute floor.
        plan_rtol, plan_atol = 1e-3, 4e-6
        potential_rtol, potential_atol = 1e-3, 5e-5
    else:
        plan_rtol, plan_atol = 5e-4, 2e-7
        potential_rtol, potential_atol = 5e-4, 2e-5
    torch.testing.assert_close(actual_plan, reference_plan, rtol=plan_rtol, atol=plan_atol)
    torch.testing.assert_close(result.f.double(), ref.f,
                               rtol=potential_rtol, atol=potential_atol)
    torch.testing.assert_close(result.g.double(), ref.g,
                               rtol=potential_rtol, atol=potential_atol)


@pytest.mark.parametrize("iterations", [1, 150])
@pytest.mark.parametrize("precision", ["ieee", "tf32", "tf32x3"])
def test_transport_actual_masses_and_gradients(iterations, precision):
    x, y, a, b = make_dataset("rings", 33, 73, 2, device="cuda", weighted=True)
    result = sinkhorn_flash(x, y, a=a, b=b, n_iters=iterations, epsilon=0.1, precision=precision)
    plan = materialize_plan(result)
    values = torch.linspace(-1, 2, len(y)*35, device="cuda").reshape(len(y), 35)
    # The paper-compatible path uses TF32 for both score and value matrix
    # products.  The upstream implementation reserves strict dense parity for
    # allow_tf32=False and uses a 1e-3 absolute floor for streaming apply tests.
    rtol, atol = (1e-2, 1e-3) if precision == "tf32" else (5e-4, 3e-7)
    torch.testing.assert_close(apply_plan(result, values), plan @ values, rtol=rtol, atol=atol)
    torch.testing.assert_close(apply_plan(result, x, transpose=True), plan.T @ x, rtol=rtol, atol=atol)
    torch.testing.assert_close(apply_plan(result, torch.ones_like(b)), plan.sum(1), rtol=rtol, atol=atol)
    gx, gy = point_gradients(result)
    torch.testing.assert_close(gx, 2*(plan.sum(1)[:, None]*x-plan@y), rtol=rtol, atol=atol)
    torch.testing.assert_close(gy, 2*(plan.sum(0)[:, None]*y-plan.T@x), rtol=rtol, atol=atol)


@pytest.mark.parametrize("shape", [(1, 1), (1, 67), (63, 1), (65, 97)])
def test_masks_half_cost_noncontiguous_and_low_epsilon(shape):
    n, m = shape
    x, y, a, b = make_dataset("gaussian", n, m, 6, device="cuda", weighted=True)
    x, y = x[:, ::2], y[:, ::2]
    kw = dict(a=a, b=b, n_iters=50, cost_scale=0.5, epsilon=0.03)
    ref = sinkhorn_dense(x, y, **kw)
    result = sinkhorn_flash(x, y, block_m=16, block_n=32, precision="ieee", **kw)
    torch.testing.assert_close(materialize_plan(result), materialize_plan(ref), rtol=8e-4, atol=2e-6)


def test_gpu_early_stopping():
    x, y, a, b = make_dataset("gaussian", 37, 71, 3, device="cuda")
    result = sinkhorn_flash(x, y, a=a, b=b, n_iters=300, epsilon=0.5,
                            tol=1e-4, check_every=10, precision="ieee")
    assert result.n_iters < 300
    stats = diagnostics(result)
    assert max(stats["row_l1"], stats["col_l1"]) < 1e-4


def test_hadamard_transport_kernel():
    x, y, a, b = make_dataset("gaussian", 17, 29, 7, device="cuda", weighted=True)
    result = sinkhorn_flash(x, y, a=a, b=b, epsilon=0.5, n_iters=200, precision="ieee")
    plan = materialize_plan(result)
    left = torch.randn(17, 7, device="cuda")
    right = torch.randn(29, 7, device="cuda")
    values = torch.randn(29, 13, device="cuda")
    expected = (plan * (left @ right.T)) @ values
    torch.testing.assert_close(
        apply_plan_hadamard(result, left, right, values, precision="ieee"),
        expected, rtol=8e-4, atol=3e-6,
    )


@pytest.mark.parametrize("d", [64, 129, 256, 1024])
def test_wide_transport_and_hadamard_against_dense(d):
    """Exercise the output tiles used by wide gradients and HVPs."""
    x, y, a, b = make_dataset("gaussian", 17, 29, d, device="cuda", weighted=True)
    result = sinkhorn_flash(x, y, a=a, b=b, epsilon=0.7,
                            n_iters=20, precision="ieee")
    plan = materialize_plan(result)
    generator = torch.Generator(device="cuda").manual_seed(21)
    values = torch.randn((len(y), d), device="cuda", generator=generator)
    left = torch.randn((len(x), d), device="cuda", generator=generator) / d**0.5
    right = torch.randn((len(y), d), device="cuda", generator=generator) / d**0.5
    torch.testing.assert_close(apply_plan(result, values), plan @ values,
                               rtol=1e-3, atol=1e-5)
    expected = (plan * (left @ right.T)) @ values
    torch.testing.assert_close(
        apply_plan_hadamard(result, left, right, values, precision="ieee"),
        expected, rtol=2e-3, atol=2e-5,
    )


def test_flash_hvp_and_double_backward_against_dense():
    x, y, a, b = make_dataset("gaussian", 17, 23, 3, device="cuda", weighted=True)
    direction = torch.randn_like(x)
    kwargs = dict(a=a, b=b, epsilon=0.8, n_iters=300, tol=1e-5, check_every=10)
    flash = sinkhorn_flash(x, y, precision="ieee", **kwargs)
    dense = sinkhorn_dense(x, y, **kwargs)
    hvp_kwargs = dict(damping=1e-5, max_cg_iters=80, cg_rtol=1e-6)
    actual = hessian_vector_product(flash, direction, **hvp_kwargs)
    expected = hessian_vector_product(dense, direction, **hvp_kwargs)
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=2e-5)

    tracked_x = x.detach().requires_grad_(True)
    loss = sinkhorn_cost(tracked_x, y, precision="ieee", backend="flash",
                         hvp_damping=1e-5, hvp_max_cg_iters=80,
                         hvp_cg_rtol=1e-6, **kwargs)
    gradient = torch.autograd.grad(loss, tracked_x, create_graph=True)[0]
    auto_hvp = torch.autograd.grad((gradient * direction).sum(), tracked_x)[0]
    torch.testing.assert_close(auto_hvp, actual, rtol=8e-4, atol=3e-6)


@pytest.mark.parametrize("d", [1, 64, 129, 1024])
@pytest.mark.parametrize("shape", [(1, 1), (1, 67), (63, 1), (37, 79)])
def test_signed_vector_transport_and_transpose(d, shape):
    n, m = shape
    x, y, a, b = make_dataset("gaussian", n, m, d, device="cuda", weighted=True)
    result = sinkhorn_flash(x, y, a=a, b=b, epsilon=0.7,
                            n_iters=1, precision="ieee")
    plan = materialize_plan(result)
    for transpose in (False, True):
        count = n if transpose else m
        # A strided, signed vector exercises packing and cancellation.
        values = torch.linspace(-2, 1, count * 2, device="cuda")[::2]
        expected = (plan.T if transpose else plan) @ values
        actual = apply_plan(result, values, transpose=transpose)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-5)
        column = apply_plan(result, values[:, None], transpose=transpose)
        assert column.shape == (len(expected), 1)
        torch.testing.assert_close(column[:, 0], expected, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("d", [64, 129, 512, 1024])
@pytest.mark.parametrize("iterations", [1, 20])
@pytest.mark.parametrize("precision", ["ieee", "tf32", "tf32x3"])
def test_fused_gradient_with_actual_unconverged_masses(d, iterations, precision):
    x, y, a, b = make_dataset("gaussian", 17, 29, d, device="cuda", weighted=True)
    result = sinkhorn_flash(x, y, a=a, b=b, epsilon=0.7, cost_scale=0.5,
                            n_iters=iterations, precision=precision)
    plan = materialize_plan(result)
    gx, gy = point_gradients(result)
    rtol, atol = (1e-2, 1e-3) if precision == "tf32" else (2e-3, 2e-5)
    torch.testing.assert_close(gx, plan.sum(1)[:, None]*x-plan@y, rtol=rtol, atol=atol)
    torch.testing.assert_close(gy, plan.sum(0)[:, None]*y-plan.T@x, rtol=rtol, atol=atol)


def test_tuned_kernels_match_dense_and_fixed_step_hvp(monkeypatch):
    monkeypatch.setenv("FLASHOPW_AUTOTUNE", "1")
    x, y, a, b = make_dataset("gaussian", 17, 29, 65, device="cuda", weighted=True)
    result = sinkhorn_flash(x, y, a=a, b=b, epsilon=0.7,
                            n_iters=20, precision="ieee")
    from dataclasses import replace
    dense = replace(result, backend="dense")
    gx, gy = point_gradients(result)
    dx, dy = point_gradients(dense)
    torch.testing.assert_close(gx, dx, rtol=2e-3, atol=2e-5)
    torch.testing.assert_close(gy, dy, rtol=2e-3, atol=2e-5)
    direction = torch.randn_like(x)
    kwargs = dict(damping=1e-5, max_cg_iters=50, cg_rtol=0, cg_atol=0)
    torch.testing.assert_close(hessian_vector_product(result, direction, **kwargs),
                               hessian_vector_product(dense, direction, **kwargs),
                               rtol=5e-3, atol=3e-5)
    from flashsinkhorn.kernel_tuning import tuning_records
    records = tuning_records()
    assert records and all(r["selected"]["shared_bytes"] <= r["shared_budget_bytes"] for r in records)
