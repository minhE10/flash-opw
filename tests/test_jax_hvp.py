"""CPU parity for the JAX streaming HVP used by the paper benchmark."""

import pytest
import torch

from experiments.datasets import make_dataset
from experiments.jax_hvp import hvp_from_shifted_potentials
from flashopw import hessian_vector_product, sinkhorn_dense


@pytest.mark.parametrize("n,m,d,distribution,epsilon", [
    (7, 11, 3, "gaussian", 0.7),
    (37, 79, 65, "gaussian", 0.7),
    (37, 79, 65, "uniform", 0.1),
])
def test_jax_matrix_free_hvp_matches_dense_schur(n, m, d, distribution, epsilon):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
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
                                      max_cg_iters=12, cg_rtol=0, cg_atol=0)
    operation = jax.jit(hvp_from_shifted_potentials,
                        static_argnames=("epsilon", "damping", "cg_iters",
                                         "block_rows", "block_keys"))
    actual = operation(
        *(jnp.asarray(t.numpy()) for t in (x, y, result.u, result.v, direction)),
        epsilon=epsilon, damping=1e-5, cg_iters=12, block_rows=4, block_keys=8,
    )
    torch.testing.assert_close(torch.from_numpy(jax.device_get(actual).copy()), expected,
                               rtol=3e-3, atol=3e-5)
