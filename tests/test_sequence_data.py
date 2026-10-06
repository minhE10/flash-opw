import numpy as np
import pytest

from experiments.sequence_data import balanced_subset, load_sequences, read_ts


def test_ts_parser_preserves_frame_order_channels_and_labels(tmp_path):
    path = tmp_path / "tiny.ts"
    path.write_text("@problemName tiny\n@timestamps false\n@data\n1,2,3:4,5,6:a\n7,8:9,10:b\n")
    sequences, labels = read_ts(path)
    np.testing.assert_equal(sequences[0], [[1,4], [2,5], [3,6]])
    np.testing.assert_equal(sequences[1], [[7,9], [8,10]])
    assert labels.tolist() == ["a", "b"]


def test_balanced_subset_is_reproducible_preserves_all_classes_and_split_order():
    labels = np.repeat(np.arange(3), [5, 8, 2])
    first = balanced_subset(labels, 8, 42)
    np.testing.assert_equal(first, balanced_subset(labels, 8, 42))
    assert np.diff(first).min() > 0
    assert np.bincount(labels[first]).tolist() == [3, 3, 2]
    with pytest.raises(ValueError):
        balanced_subset(labels, 2, 42)


def test_packed_npz_supports_variable_lengths_without_pickle(tmp_path):
    path = tmp_path / "sequences.npz"
    np.savez(path, train_values=[[1.], [2.], [3.]], train_offsets=[0, 1, 3], train_labels=[0, 1],
             test_values=[[4.], [5.]], test_offsets=[0, 2], test_labels=[1])
    train, train_labels, test, test_labels, manifest = load_sequences("custom", tmp_path, path)
    assert [len(x) for x in train] == [1, 2]
    assert [len(x) for x in test] == [2]
    assert manifest["features"] == 1
    assert manifest["original_train"] == 2


def test_invalid_packed_offsets_are_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, train_values=[[1.]], train_offsets=[0, 0, 1], train_labels=[0, 1],
             test_x=[[[2.]]], test_labels=[0])
    with pytest.raises(ValueError, match="offsets"):
        load_sequences("custom", tmp_path, path)
