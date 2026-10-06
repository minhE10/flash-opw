"""k-NN majority ACC and full-gallery MAP (journal Section 5.2)."""

from collections import Counter
import numpy as np


def evaluate_distances(distances, train_labels, test_labels, ks=(1, 3, 5, 7, 15, 30)):
    distances = np.asarray(distances, dtype=np.float64)
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    if (train_labels.ndim != 1 or test_labels.ndim != 1 or not len(train_labels)
            or not len(test_labels) or distances.shape != (len(test_labels), len(train_labels))):
        raise ValueError("distances must be (test queries, training gallery) with matching labels")
    if not np.isfinite(distances).all():
        raise ValueError("MAP/ACC requires every distance to be finite; failed pairs cannot be dropped")
    if not ks or any(isinstance(k, bool) or not isinstance(k, (int, np.integer))
                     or k < 1 or k > len(train_labels) for k in ks):
        raise ValueError("each k must be an integer in [1, gallery size]")
    ranking = np.argsort(distances, axis=1, kind="stable")
    ordered_labels = train_labels[ranking]
    relevant = ordered_labels == test_labels[:, None]
    counts = relevant.sum(1)
    precision = np.cumsum(relevant, axis=1) / np.arange(1, len(train_labels)+1)
    aps = np.divide((precision * relevant).sum(1), counts,
                    out=np.zeros(len(test_labels), dtype=np.float64), where=counts > 0)
    predictions, acc = {}, {}
    for k in ks:
        votes = []
        for labels in ordered_labels[:, :k]:
            counts_by_label = Counter(labels.tolist())
            largest = max(counts_by_label.values())
            # For even k and all other tied votes, nearest tied class wins.
            votes.append(next(label for label in labels.tolist() if counts_by_label[label] == largest))
        predictions[str(k)] = votes
        acc[str(k)] = float(np.mean(np.asarray(votes) == test_labels))
    return dict(MAP=float(aps.mean()), ACC=acc, average_precision=aps.tolist(),
                predictions=predictions, queries_without_relevant_gallery=int(np.count_nonzero(counts == 0)),
                distance_tie_rule="stable gallery index", vote_tie_rule="nearest tied class",
                map_scope="all selected training sequences; same class is relevant; zero-relevant AP=0")
