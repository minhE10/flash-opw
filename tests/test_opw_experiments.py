import numpy as np
import pytest
import torch

from experiments.opw_parameters import dense_score_matrix, load_selection, atomic_json
from experiments.opw_scaling import estimated_dense_bytes, synthetic_pair
from experiments.opw_tune import candidate_grid, select_candidate, training_holdout
from experiments.sequence_data import load_training, training_fingerprint
from experiments.sequence_metrics import reference_distance
from experiments.sequence_metrics_torch import ENTROPIC_METRICS, dense_tensor_distance
from flashopw import OPWParameters, opw_dense, opw_distance, opw_flash


DEFAULT = dict(lambda1=1., lambda2=.1, sigma=1., cost_scale=1., n_iters=12)


def test_training_only_loader_needs_no_test_arrays_and_fingerprint_ignores_test(tmp_path):
    path = tmp_path/"training.npz"
    train = np.arange(24).reshape(4, 6, 1)
    labels = np.array([0, 0, 1, 1])
    np.savez(path, train_x=train, train_labels=labels)
    values, actual_labels, origin = load_training("custom", tmp_path, path)
    original = origin["training_sha256"]
    np.savez(path, train_x=train, train_labels=labels, test_x=np.full((2, 6, 1), np.nan), test_labels=[999, -1])
    assert load_training("custom", tmp_path, path)[2]["training_sha256"] == original
    assert training_fingerprint(values, actual_labels) == original
    values[0][0,0] += 1
    assert training_fingerprint(values, actual_labels) != original


def test_training_holdout_disjoint_reproducible_and_contains_every_class():
    labels = np.repeat(np.arange(3), [2, 7, 20])
    gi, vi = training_holdout(labels, 6, 5, 9)
    assert not np.intersect1d(gi, vi).size
    assert set(labels[gi]) == set(labels[vi]) == {0, 1, 2}
    again = training_holdout(labels, 6, 5, 9)
    np.testing.assert_array_equal(gi, again[0])
    np.testing.assert_array_equal(vi, again[1])
    with pytest.raises(ValueError, match="two training"):
        training_holdout(np.array([0, 1, 1]), 0, 0, 9)


def test_grid_preserves_default_and_removes_equivalent_coupling_candidates():
    grid = candidate_grid(DEFAULT, [1.05, 50, 50], [.1, .1, .3])
    assert grid[0] == DEFAULT
    pairs = [(round(OPWParameters(**{k:p[k] for k in ("lambda1", "lambda2", "sigma", "cost_scale")}).mu, 10), p["lambda2"]) for p in grid]
    assert len(pairs) == len(set(pairs)) == 4
    assert (50., .3) in pairs


def test_selection_uses_both_metrics_and_keeps_default_on_exact_ties():
    rows = [dict(ACC=.8, MAP=.7), dict(ACC=.8, MAP=.9), dict(ACC=.9, MAP=.5)]
    assert select_candidate(rows, "ACC") == 2
    assert select_candidate(rows, "MAP") == 1
    assert select_candidate([rows[0], rows[0]]) == 0


@pytest.mark.parametrize("score", ["pdf-loss", "spatial"])
@pytest.mark.parametrize("mu", [1.05, 430])
def test_batched_cost_oracle_matches_independent_ragged_numpy_reference(score, mu):
    rng = np.random.default_rng(7)
    queries = [rng.normal(size=(7, 3)), rng.normal(size=(5, 3))]
    gallery = [rng.normal(size=(8, 3)), rng.normal(size=(11, 3)), rng.normal(size=(8, 3))]
    p = candidate_grid(DEFAULT, [mu], [.1])[-1]
    distances, residuals = dense_score_matrix(queries, gallery, p, score=score, memory_mib=1)
    expected = np.array([[reference_distance("affine-opw-dense", x, y, **p, opw_score=score)
                          for y in gallery] for x in queries])
    np.testing.assert_allclose(distances, expected, rtol=1e-10, atol=1e-11)
    assert (residuals >= 0).all()
    with pytest.raises(ValueError, match="memory budget"):
        dense_score_matrix(queries, gallery, p, memory_mib=.0001)


def test_selection_artifact_rejects_changed_training_protocol_and_score(tmp_path):
    path = tmp_path/"selection.json"
    value = dict(schema="flashopw-training-selection-v1", protocol="stratified holdout within official training split only",
                 dataset="tiny", training_sha256="abc", parameters=DEFAULT, n_iters=12, score="pdf-loss")
    atomic_json(path, value)
    assert load_selection(path, dataset="tiny", training_sha256="abc")["parameters"] == DEFAULT
    for kwargs in (dict(dataset="other"), dict(training_sha256="def"), dict(score="spatial"), dict(n_iters=20)):
        with pytest.raises(ValueError, match="mismatch"):
            load_selection(path, **kwargs)


def test_scaling_inputs_reproducible_identical_after_fp32_transfer_and_ragged():
    x, y = synthetic_pair(1025, 1537, 13, 7)
    assert x.shape == (1025, 13) and y.shape == (1537, 13)
    np.testing.assert_array_equal(x, synthetic_pair(1025, 1537, 13, 7)[0])
    np.testing.assert_array_equal(x, x.astype(np.float32).astype(np.float64))
    assert estimated_dense_bytes(1025, 1537, "affine-opw-dense-gpu", 4) == 8*1025*1537*4


@pytest.mark.parametrize("metric", ENTROPIC_METRICS)
def test_torch_journal_cost_score_against_independent_numpy_reference(metric):
    x, y = synthetic_pair(7, 11, 3, 8)
    kwargs = dict(lambda1=2., lambda2=.3, sigma=.7, n_iters=12, sinkhorn_iters=15)
    expected = reference_distance(metric, x, y, **kwargs)
    actual, row_error, column_error = dense_tensor_distance(metric, torch.from_numpy(x), torch.from_numpy(y),
                                                           **kwargs, return_diagnostics=True)
    assert float(actual) == pytest.approx(expected, abs=1e-12)
    assert float(row_error) >= 0 and float(column_error) >= 0
    torch.testing.assert_close(actual, dense_tensor_distance(metric, torch.from_numpy(x), torch.from_numpy(y), **kwargs))


@pytest.mark.gpu
@pytest.mark.parametrize("metric", ENTROPIC_METRICS)
def test_gpu_journal_cost_score_against_numpy_fp64(metric):
    x, y = synthetic_pair(17, 29, 13, 8)
    expected = reference_distance(metric, x, y, lambda1=1., n_iters=20, sinkhorn_iters=20)
    actual = dense_tensor_distance(metric, torch.from_numpy(x).float().cuda(), torch.from_numpy(y).float().cuda(),
                                   n_iters=20, sinkhorn_iters=20)
    assert float(actual) == pytest.approx(expected, rel=3e-3, abs=3e-4)


@pytest.mark.parametrize("shape", [(257, 513, 13), (1025, 1537, 1)])
def test_long_rectangular_batched_oracle_against_dense_solver(shape):
    x, y = synthetic_pair(*shape, seed=19)
    p = dict(DEFAULT, n_iters=3)
    actual, residuals = dense_score_matrix([x], [y], p)
    expected = opw_dense(torch.from_numpy(x), torch.from_numpy(y), **p)
    assert actual[0,0] == pytest.approx(float(expected.loss), abs=1e-12)
    assert np.isfinite(residuals).all()


@pytest.mark.gpu
@pytest.mark.parametrize("shape,mu", [((257, 769, 13), 50), ((1025, 1537, 1), 430)])
def test_long_flash_tuned_cost_matches_fp64_dense(shape, mu):
    x, y = synthetic_pair(*shape, seed=19)
    p = dict(candidate_grid(DEFAULT, [mu], [.1])[-1], n_iters=20)
    expected = opw_dense(torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda(), **p)
    actual = opw_flash(torch.from_numpy(x).float().cuda(), torch.from_numpy(y).float().cuda(), **p, precision="ieee")
    torch.testing.assert_close(actual.loss.double(), expected.loss, rtol=3e-3, atol=3e-4)
    torch.testing.assert_close(opw_distance(actual).double(), opw_distance(expected), rtol=5e-3, atol=3e-4)
