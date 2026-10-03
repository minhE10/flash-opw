import torch

from experiments.phase2_real import stratified_indices


def test_stratified_indices_are_reproducible_and_balanced():
    labels = torch.arange(10).repeat_interleave(7)
    first = stratified_indices(labels, 30, seed=7)
    second = stratified_indices(labels, 30, seed=7)
    assert torch.equal(first, second)
    selected = labels[first]
    assert torch.bincount(selected, minlength=10).tolist() == [3] * 10


def test_stratified_indices_distribute_remainder():
    labels = torch.arange(3).repeat_interleave(5)
    selected = labels[stratified_indices(labels, 8, seed=9)]
    assert sorted(torch.bincount(selected, minlength=3).tolist()) == [2, 3, 3]
