import numpy as np
import pytest

from experiments.retrieval import evaluate_distances


def test_full_gallery_map_and_majority_accuracy_by_hand():
    # Relevant ranks: query0 ranks1,3 => AP5/6; query1 ranks2,4 => AP1/2.
    result = evaluate_distances([[0, 1, 2, 3], [0, 1, 2, 3]], [0, 1, 0, 1], [0, 1], [1, 3])
    assert result["average_precision"] == pytest.approx([5/6, 1/2])
    assert result["MAP"] == pytest.approx(2/3)
    assert result["ACC"] == {"1": 0.5, "3": 0.5}


def test_even_k_tie_uses_nearest_class_and_exact_distance_tie_is_stable():
    result = evaluate_distances([[0, 0]], ["b", "a"], ["b"], [2])
    assert result["predictions"]["2"] == ["b"]
    assert result["ACC"]["2"] == 1


def test_query_without_relevant_gallery_is_retained_with_zero_ap():
    result = evaluate_distances([[0, 1]], [0, 0], [1], [1])
    assert result["MAP"] == 0
    assert result["queries_without_relevant_gallery"] == 1


@pytest.mark.parametrize("distances,ks", [([[0, np.nan]], [1]), ([[0, np.inf]], [1]),
                                        ([[0, 1]], [3]), ([[0, 1]], [0])])
def test_incomplete_distances_or_invalid_k_cannot_be_scored(distances, ks):
    with pytest.raises(ValueError):
        evaluate_distances(distances, [0, 1], [0], ks)
