import json

import pytest

from experiments.author_reference import AUTHOR_REVISION
from experiments.compare_author_runs import compare


def write_runs(tmp_path):
    paths = [tmp_path / "local", tmp_path / "author"]
    for path, implementation in zip(paths, ("local", "author")):
        path.mkdir()
        env = dict(args=dict(flash_implementation=implementation, epsilon=0.1, seed=0))
        if implementation == "author":
            env["author_source"] = dict(revision=AUTHOR_REVISION)
        (path / "environment.json").write_text(json.dumps(env))
        (path / "paper_results.json").write_text(json.dumps([
            dict(experiment="forward_n", n=10000, m=10000, d=64,
                 method="flash-sym", status="ok", mean=10 if implementation == "local" else 5, unit="ms"),
            dict(experiment="forward_n", n=10000, m=10000, d=64,
                 method="tensorized", status="skipped_memory", mean=None, unit="ms"),
        ]))
    return paths


def test_comparison_ratio_and_skips(tmp_path):
    rows = compare(*write_runs(tmp_path))
    assert rows[0]["local_over_author"] == 2
    assert rows[1]["local_over_author"] is None
    assert rows[1]["author_status"] == "skipped_memory"


def test_comparison_rejects_different_data_seed(tmp_path):
    paths = write_runs(tmp_path)
    path = paths[1] / "environment.json"
    env = json.loads(path.read_text())
    env["args"]["seed"] = 1
    path.write_text(json.dumps(env))
    with pytest.raises(ValueError, match="seed"):
        compare(*paths)


def test_comparison_rejects_duplicate_cases(tmp_path):
    paths = write_runs(tmp_path)
    path = paths[1] / "paper_results.json"
    rows = json.loads(path.read_text())
    rows.append(rows[0])
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="Duplicate"):
        compare(*paths)
