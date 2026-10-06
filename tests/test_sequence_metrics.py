import math

import numpy as np
import pytest
import torch

from experiments.sequence_metrics import (JOURNAL_METRICS, dtw_cost_and_length,
                                         entropic_plan, exact_ot_cost, reference_distance)
from flashopw import OPWParameters, opw_dense, opw_distance


@pytest.mark.parametrize("metric", JOURNAL_METRICS)
def test_all_journal_distances_on_one_frame(metric):
    assert reference_distance(metric, np.array([[0.]]), np.array([[2.]])) == pytest.approx(4)


def test_dtw_cost_and_distinct_normalizations():
    x, y = np.array([[0.], [2.]]), np.array([[0.], [1.], [2.]])
    assert reference_distance("dtw", x, y) == 1
    assert reference_distance("ldtw", x, y) == 0.5
    assert reference_distance("ndtw", x, y) == pytest.approx(1/3)
    assert reference_distance("soft-dtw", x, y, soft_dtw_gamma=1e-8) == pytest.approx(1, abs=1e-7)


def test_exact_uniform_ot_rectangular_against_analytic_transport():
    # Uniform 2->3 mass: cost[[0,1,4],[4,1,0]], optimal cost1/3.
    assert exact_ot_cost(np.array([[0., 1., 4.], [4., 1., 0.]])) == pytest.approx(1/3)


def test_affine_dense_reference_matches_main_loss_and_spatial_score():
    generator = np.random.default_rng(19)
    x, y = generator.random((7, 3)), generator.random((11, 3))
    result = opw_dense(torch.from_numpy(x), torch.from_numpy(y), lambda1=10, lambda2=0.3, sigma=0.7)
    settings = dict(lambda1=10, lambda2=0.3, sigma=0.7, n_iters=200)
    assert reference_distance("affine-opw-dense", x, y, **settings) == pytest.approx(float(result.loss), abs=1e-11)
    assert reference_distance("affine-opw-dense", x, y, opw_score="spatial", **settings) == pytest.approx(float(opw_distance(result)), abs=1e-11)


def test_entropy_reference_has_correct_marginals_and_constant_invariance():
    cost = np.array([[0., 0.3, 0.5], [0.4, 0.1, 0.2]])
    plan = entropic_plan(cost, 0.3, 200)
    np.testing.assert_allclose(plan.sum(1), [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(plan.sum(0), [1/3]*3, atol=1e-12)
    np.testing.assert_allclose(plan, entropic_plan(cost - 50, 0.3, 200), atol=1e-12)


def test_main_taylor_distance_is_order_sensitive():
    x, reversed_x = np.array([[0.], [1.]]), np.array([[1.], [0.]])
    a = reference_distance("affine-opw-dense", x, x)
    b = reference_distance("affine-opw-dense", x, reversed_x)
    assert b > a + 0.9
    assert exact_ot_cost((x - reversed_x.T)**2) == pytest.approx(0)
