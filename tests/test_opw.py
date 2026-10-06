"""Independent cost, potential, Eq.19 and streamed distance checks."""

import math

import pytest
import torch

from flashopw import (OPWParameters, effective_cost, materialize_opw_plan,
                      opw_dense, opw_diagnostics, opw_distance, opw_flash,
                      opw_online, temporal_features)


def sample(n, m, d, *, dtype=torch.float64, device="cpu"):
    generator = torch.Generator().manual_seed(19)
    x = torch.rand(n, d, generator=generator, dtype=dtype)
    y = torch.rand(m, d, generator=generator, dtype=dtype)
    a, b = torch.rand(n, generator=generator, dtype=dtype) + 0.2, torch.rand(m, generator=generator, dtype=dtype) + 0.2
    return tuple(t.to(device) for t in (x, y, a/a.sum(), b/b.sum()))


@pytest.mark.parametrize("shape", [(1, 1, 1), (7, 11, 3), (17, 29, 65)])
def test_main_affine_cost_identity(shape):
    x, y, _, _ = sample(*shape)
    parameters = OPWParameters(10, 0.3, 0.7, 0.5)
    xa, ya = temporal_features(x, parameters), temporal_features(y, parameters)
    expanded = (xa[:, None, :] - ya[None, :, :]).square().sum(-1) + parameters.q0
    torch.testing.assert_close(expanded, effective_cost(x, y, parameters), rtol=1e-12, atol=1e-12)
    assert float(xa[0, -1]) == pytest.approx(math.sqrt(parameters.mu)/len(x))
    assert float(xa[-1, -1]) == pytest.approx(math.sqrt(parameters.mu))


@pytest.mark.parametrize("schedule", ["alternating", "symmetric"])
@pytest.mark.parametrize("iterations", [1, 20, 200])
def test_main_potentials_plan_loss_against_unshifted_cost(schedule, iterations):
    x, y, a, b = sample(7, 11, 3)
    result = opw_online(x, y, a=a, b=b, lambda1=2, lambda2=0.7, sigma=0.8,
                        n_iters=iterations, schedule=schedule, block_m=3, block_n=4)
    parameters = result.parameters
    cost = effective_cost(x, y, parameters) - parameters.q0
    # Independent unshifted recurrence from the explicit temporal cost matrix.
    f, g = torch.zeros_like(a), torch.zeros_like(b)
    for _ in range(iterations):
        new_f = -parameters.lambda2 * torch.logsumexp((g[None, :] - cost)/parameters.lambda2 + b.log(), dim=1)
        new_g = -parameters.lambda2 * torch.logsumexp(((new_f if schedule == "alternating" else f)[:, None] - cost)/parameters.lambda2 + a.log()[:, None], dim=0)
        f, g = ((f+new_f)/2, (g+new_g)/2) if schedule == "symmetric" else (new_f, new_g)
    plan = (a.log()[:, None] + b.log() + (f[:, None]+g[None, :]-cost)/parameters.lambda2).exp()
    torch.testing.assert_close(result.f, f, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(result.g, g, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(materialize_opw_plan(result), plan, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(result.loss, a @ f + b @ g - parameters.lambda2 + parameters.q0)
    spatial = torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist").square()
    torch.testing.assert_close(opw_distance(result), (plan*spatial).sum(), rtol=1e-11, atol=1e-12)
    stats = opw_diagnostics(result)
    entropy_primal = (plan * (cost + parameters.q0 + parameters.lambda2 * plan.log())).sum()
    assert stats["entropy_primal"] == pytest.approx(float(entropy_primal), abs=1e-11)
    if iterations == 200:
        assert max(stats["row_l1"], stats["col_l1"]) < 1e-9
        assert stats["entropy_primal"] == pytest.approx(stats["entropy_dual"], abs=1e-9)


def test_taylor_cost_error_is_not_silently_exact_opw():
    x, y, _, _ = sample(7, 11, 3)
    parameters = OPWParameters(10, 0.1, 1)
    t, s = torch.arange(1, 8, dtype=x.dtype)/7, torch.arange(1, 12, dtype=x.dtype)/11
    delta2 = (t[:, None]-s[None, :]).square()
    exact = (torch.cdist(x, y).square() - parameters.lambda1/(1+delta2)
             + parameters.lambda2 * delta2/(2*parameters.sigma**2)
             + parameters.lambda2 * math.log(parameters.sigma*math.sqrt(2*math.pi)))
    expected_error = parameters.lambda1 * delta2.square()/(1+delta2)
    torch.testing.assert_close(effective_cost(x, y, parameters)-exact, expected_error, rtol=1e-11, atol=1e-12)
    assert float(expected_error.max()) > 0


@pytest.mark.parametrize("kwargs", [dict(lambda1=-1), dict(lambda2=0), dict(sigma=0),
                                   dict(sigma=1e-200), dict(sigma=1e200), dict(cost_scale=float("nan"))])
def test_invalid_opw_parameters(kwargs):
    with pytest.raises(ValueError):
        OPWParameters(**kwargs)


@pytest.mark.gpu
@pytest.mark.parametrize("d", [1, 63, 64, 65, 390, 1023])
@pytest.mark.parametrize("schedule", ["alternating", "symmetric"])
def test_flash_opw_plan_loss_and_distance_against_dense(d, schedule):
    x, y, a, b = sample(17, 29, d, dtype=torch.float32, device="cuda")
    kwargs = dict(a=a, b=b, lambda1=10, lambda2=0.1, sigma=1,
                  n_iters=20, schedule=schedule)
    reference = opw_dense(x.double(), y.double(), a=a.double()/a.double().sum(),
                          b=b.double()/b.double().sum(), **{k:v for k,v in kwargs.items() if k not in ("a", "b")})
    actual = opw_flash(x, y, precision="ieee", **kwargs)
    plan, expected = materialize_opw_plan(actual).double(), materialize_opw_plan(reference)
    relative = float((plan-expected).norm()/expected.norm())
    assert relative < 5e-3, relative
    torch.testing.assert_close(actual.loss.double(), reference.loss, rtol=3e-3, atol=3e-4)
    torch.testing.assert_close(opw_distance(actual).double(), opw_distance(reference), rtol=5e-3, atol=3e-4)
