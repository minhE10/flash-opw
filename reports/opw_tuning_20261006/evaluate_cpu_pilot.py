"""Recheck the frozen CPU pilot from the repo root; does not tune on TEST."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.opw_parameters import atomic_json, dense_score_matrix, load_selection
from experiments.retrieval import evaluate_distances
from experiments.sequence_data import balanced_subset, load_sequences, training_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/opw_cpu_pilot_recheck")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    selected = load_selection(Path(__file__).with_name("selected_parameters.json"),
                              dataset="FacesUCR", score="pdf-loss", n_iters=200)
    train, labels, test, test_labels, origin = load_sequences("FacesUCR", ROOT/"data/opw")
    assert training_fingerprint(train, labels) == selected["training_sha256"]
    gi = balanced_subset(labels, 32, 20261007)
    qi = balanced_subset(test_labels, 28, 20261008)
    results = {}
    for name, p in (("default", selected["default_validation"]["parameters"]), ("selected", selected["parameters"])):
        scores, residuals = dense_score_matrix([test[i] for i in qi], [train[i] for i in gi], p)
        evaluation = evaluate_distances(scores, labels[gi], test_labels[qi], [1,3,5,7,15,30])
        results[name] = dict(parameters=p, evaluation=evaluation,
                             max_marginal_l1=float(residuals.max()))
        np.savez_compressed(args.output/f"test_{name}.npz", distances=scores, residuals=residuals,
                            gallery_indices=gi, query_indices=qi, gallery_labels=labels[gi], query_labels=test_labels[qi])
        print(name, evaluation["ACC"], evaluation["MAP"], flush=True)
    atomic_json(args.output/"test_comparison.json", dict(origin=origin, results=results,
                gallery_indices=gi.tolist(), query_indices=qi.tolist(), backend="CPU FP64 batched oracle"))


if __name__ == "__main__":
    main()
