"""Parity on the available JAX device for the benchmark's streaming HVP."""

import pytest
import torch
import os
from pathlib import Path

from experiments.datasets import make_dataset
from experiments.jax_hvp import (hvp_from_shifted_potentials,
                                 transport_from_shifted_potentials)
from flashsinkhorn import hessian_vector_product, sinkhorn_dense


@pytest.mark.parametrize("n,m,d,distribution,epsilon", [
    (7, 11, 3, "gaussian", 0.7),
    (37, 79, 65, "gaussian", 0.7),
    (37, 79, 65, "uniform", 0.1),
])
@pytest.mark.parametrize("cg_iters", [12, 50])
def test_jax_matrix_free_hvp_matches_dense_schur(n, m, d, distribution, epsilon, cg_iters):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_default_matmul_precision", "highest")
    if distribution == "uniform":
        generator = torch.Generator().manual_seed(13)
        x = torch.rand((n, d), generator=generator)
        y = torch.rand((m, d), generator=generator)
        a = torch.full((n,), 1.0 / n)
        b = torch.full((m,), 1.0 / m)
    else:
        x, y, a, b = make_dataset("gaussian", n, m, d, weighted=True)
    result = sinkhorn_dense(x, y, a=a, b=b, epsilon=epsilon, n_iters=200)
    direction = torch.randn(x.shape, generator=torch.Generator().manual_seed(5))
    expected = hessian_vector_product(result, direction, damping=1e-5,
                                      max_cg_iters=cg_iters, cg_rtol=0, cg_atol=0)
    operation = jax.jit(hvp_from_shifted_potentials,
                        static_argnames=("epsilon", "damping", "cg_iters",
                                         "block_rows", "block_keys"))
    actual = operation(
        *(jnp.asarray(t.numpy()) for t in (x, y, result.u, result.v, direction)),
        epsilon=epsilon, damping=1e-5, cg_iters=cg_iters, block_rows=4, block_keys=8,
    )
    actual_torch = torch.from_numpy(jax.device_get(actual).copy())
    error = float((actual_torch-expected).double().norm() / expected.double().norm())
    assert error < 3e-3
    torch.testing.assert_close(actual_torch, expected,
                               rtol=3e-3, atol=3e-5)


@pytest.mark.parametrize("block_rows,block_keys", [(4, 8), (16, 32), (64, 256)])
def test_jax_transport_uses_identical_plan_in_both_directions(block_rows, block_keys):
    """Unequal tile sizes and partial tiles must preserve P.T exactly."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    generator = torch.Generator().manual_seed(13)
    x = torch.rand((37, 65), generator=generator)
    y = torch.rand((79, 65), generator=generator)
    result = sinkhorn_dense(x, y, epsilon=0.1, n_iters=200)
    operands = tuple(jnp.asarray(t.numpy()) for t in (x, y, result.u, result.v))
    operation = jax.jit(transport_from_shifted_potentials,
                        static_argnames=("epsilon", "transpose", "block_rows", "block_keys"))
    controls = dict(epsilon=0.1, block_rows=block_rows, block_keys=block_keys)
    # Per-operation precision must remain IEEE even under JAX's faster default.
    with jax.default_matmul_precision("default"):
        plan = operation(*operands, jnp.eye(len(y)), **controls)
        transposed = operation(*operands, jnp.eye(len(x)), transpose=True, **controls)
    plan = torch.from_numpy(jax.device_get(plan).copy())
    transposed = torch.from_numpy(jax.device_get(transposed).copy())
    torch.testing.assert_close(transposed, plan.T, rtol=0, atol=0)
    expected = (20 * (x @ y.T) + result.u[:, None] + result.v[None, :]).exp()
    torch.testing.assert_close(plan, expected, rtol=5e-4, atol=3e-8)


@pytest.mark.parametrize("n,m,d,epsilon", [(7, 11, 3, 0.1), (7, 11, 3, 0.7), (37, 79, 65, 0.1)])
@pytest.mark.parametrize("cg_iters", [12, 50])
def test_author_ott_hessian_matches_same_coupling(n, m, d, epsilon, cg_iters):
    """Exercise the real upstream function with the guarded fixed-step CG adapter."""
    source = Path(os.environ.get("FLASHOPW_OTT_HESSIAN_PATH", "outputs/third_party/OTT-Hessian"))
    if not (source / "SinkhornHessian.py").is_file():
        pytest.skip("External OTT-Hessian checkout unavailable; run setup_ott_hessian.sh")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_default_matmul_precision", "highest")
    pytest.importorskip("ott")
    from experiments.ott_hessian import load_hessian, state_from_shifted
    hessian, provenance = load_hessian(source, cg_rtol=0, cg_atol=0)
    if d > 3:
        generator = torch.Generator().manual_seed(13)
        x, y = torch.rand((n, d), generator=generator), torch.rand((m, d), generator=generator)
        a, b = torch.full((n,), 1/n), torch.full((m,), 1/m)
    else:
        x, y, a, b = make_dataset("gaussian", n, m, d, weighted=True)
    result = sinkhorn_dense(x, y, a=a, b=b, epsilon=epsilon, n_iters=200)
    direction = torch.randn(x.shape, generator=torch.Generator().manual_seed(5))
    direction /= direction.norm()
    operands = tuple(jnp.asarray(t.numpy()) for t in (x, y, result.u, result.v, direction))
    state = state_from_shifted(*operands[:4], epsilon=epsilon, batch_size=4)
    # OTT weights are encoded in its potentials. Check both axes before HVP.
    plan = (2*x@y.T/epsilon + result.u[:, None]+result.v[None, :]).exp()
    actual_mass = state.geom.apply_transport_from_potentials(state.f, state.g,
                                                            jnp.ones(len(y)), axis=1)
    torch.testing.assert_close(torch.from_numpy(jax.device_get(actual_mass).copy()),
                               plan.sum(1), rtol=5e-4, atol=3e-6)
    operation = jax.jit(hessian, static_argnames=("tau2", "iter"))
    actual = operation(operands[4], state, tau2=1e-5/epsilon, iter=cg_iters)
    expected = hessian_vector_product(result, direction, damping=1e-5,
                                      max_cg_iters=cg_iters, cg_rtol=0, cg_atol=0)
    assert provenance["cg_rtol"] == 0
    assert provenance["cg_backend"] == "guarded fixed-step JAX CG"
    actual_torch = torch.from_numpy(jax.device_get(actual).copy())
    error = float((actual_torch-expected).double().norm() / expected.double().norm())
    assert error < 3e-3
    torch.testing.assert_close(actual_torch, expected,
                               rtol=3e-3, atol=3e-5)
