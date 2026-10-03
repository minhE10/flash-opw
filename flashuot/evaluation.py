"""Sequence classification and retrieval metrics used in the OWD journal paper."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Sequence

import torch

from .opw import opw_exact_cpu, opw_flash_cpu
from .sinkhorn import dense_sinkhorn_cpu


DistanceFn = Callable[[torch.Tensor, torch.Tensor], float]


def mean_average_precision(distances: torch.Tensor, query_labels: torch.Tensor, database_labels: torch.Tensor) -> float:
    """Mean AP for retrieval where smaller distance means higher rank."""

    aps = []
    for row, label in zip(distances, query_labels):
        order = torch.argsort(row)
        relevant = (database_labels[order] == label).to(torch.float64)
        total_relevant = int((database_labels == label).sum().item())
        if total_relevant == 0:
            continue
        cumulative = torch.cumsum(relevant, dim=0)
        precision = cumulative / torch.arange(1, row.numel() + 1, dtype=torch.float64)
        aps.append((precision * relevant).sum().item() / total_relevant)
    return float(sum(aps) / max(len(aps), 1))


def _majority_vote(labels: torch.Tensor) -> int:
    values, counts = torch.unique(labels, return_counts=True)
    return int(values[counts.argmax()].item())


def pairwise_distances(
    queries: Sequence[torch.Tensor],
    database: Sequence[torch.Tensor],
    distance_fn: DistanceFn,
) -> torch.Tensor:
    out = torch.empty((len(queries), len(database)), dtype=torch.float64)
    for i, query in enumerate(queries):
        for j, item in enumerate(database):
            out[i, j] = float(distance_fn(query, item))
    return out


def nearest_mean_indices(
    train: Sequence[torch.Tensor],
    labels: torch.Tensor,
    distance_fn: DistanceFn,
) -> dict[int, int]:
    """Return the medoid index for each class, matching OWD's NM definition."""

    prototypes: dict[int, int] = {}
    for label in torch.unique(labels).tolist():
        indices = [i for i, y in enumerate(labels.tolist()) if y == label]
        if len(indices) == 1:
            prototypes[int(label)] = indices[0]
            continue
        sums = []
        for i in indices:
            total = sum(distance_fn(train[i], train[j]) for j in indices)
            sums.append(total)
        prototypes[int(label)] = indices[int(torch.tensor(sums).argmin().item())]
    return prototypes


def evaluate_nm(
    train: Sequence[torch.Tensor],
    train_labels: torch.Tensor,
    test: Sequence[torch.Tensor],
    test_labels: torch.Tensor,
    distance_fn: DistanceFn,
) -> dict[str, float]:
    prototype_indices = nearest_mean_indices(train, train_labels, distance_fn)
    labels = sorted(prototype_indices)
    prototypes = [train[prototype_indices[label]] for label in labels]
    distances = pairwise_distances(test, prototypes, distance_fn)
    pred = torch.tensor([labels[i] for i in distances.argmin(dim=1).tolist()])
    prototype_labels = torch.tensor(labels)
    return {
        "accuracy": float((pred == test_labels).double().mean().item() * 100.0),
        "map": mean_average_precision(distances, test_labels, prototype_labels) * 100.0,
    }


def evaluate_knn(
    train: Sequence[torch.Tensor],
    train_labels: torch.Tensor,
    test: Sequence[torch.Tensor],
    test_labels: torch.Tensor,
    distance_fn: DistanceFn,
    ks: Iterable[int] = (1, 3, 5, 7, 15, 30),
) -> dict[str, float]:
    distances = pairwise_distances(test, train, distance_fn)
    result = {"map": mean_average_precision(distances, test_labels, train_labels) * 100.0}
    for k in ks:
        k_eff = min(int(k), len(train))
        predictions = []
        for row in distances:
            order = torch.argsort(row)[:k_eff]
            predictions.append(_majority_vote(train_labels[order]))
        result[f"accuracy_{k}"] = float((torch.tensor(predictions) == test_labels).double().mean().item() * 100.0)
    return result


def evaluate_methods(
    train: Sequence[torch.Tensor],
    train_labels: torch.Tensor,
    test: Sequence[torch.Tensor],
    test_labels: torch.Tensor,
    distance_fns: dict[str, DistanceFn],
    ks: Iterable[int] = (1, 3, 5, 7, 15, 30),
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name, distance_fn in distance_fns.items():
        nm = evaluate_nm(train, train_labels, test, test_labels, distance_fn)
        knn = evaluate_knn(train, train_labels, test, test_labels, distance_fn, ks)
        row: dict[str, float | str] = {"method": name, "nm_accuracy": nm["accuracy"], "nm_map": nm["map"]}
        row.update({f"knn_{key}": value for key, value in knn.items()})
        rows.append(row)
    return rows


def make_toy_sequence_dataset(
    n_train_per_class: int = 6,
    n_test_per_class: int = 3,
    n_classes: int = 3,
    length: int = 24,
    dim: int = 2,
    seed: int = 7,
) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Create small variable-length temporal trajectories for CPU experiments."""

    if dim < 2:
        raise ValueError("toy trajectories need dim >= 2")
    generator = torch.Generator().manual_seed(seed)
    train: list[torch.Tensor] = []
    test: list[torch.Tensor] = []
    train_labels: list[int] = []
    test_labels: list[int] = []
    for label in range(n_classes):
        for split, count, seqs, labels in (
            ("train", n_train_per_class, train, train_labels),
            ("test", n_test_per_class, test, test_labels),
        ):
            for _ in range(count):
                length_i = max(8, length + int(torch.randint(-3, 4, (), generator=generator).item()))
                t = torch.linspace(0.0, 1.0, length_i, dtype=torch.float64)
                phase = float(torch.randn((), generator=generator, dtype=torch.float64).item()) * 0.12
                warp = torch.clamp(t + phase * (t - 0.5) * 0.8, 0.0, 1.0)
                if label % 3 == 0:
                    base = torch.stack([warp, torch.sin(2 * torch.pi * warp)], dim=1)
                elif label % 3 == 1:
                    base = torch.stack([warp, torch.cos(2 * torch.pi * warp)], dim=1)
                else:
                    base = torch.stack([warp, torch.sin(4 * torch.pi * warp)], dim=1)
                noise = 0.025 * torch.randn(base.shape, generator=generator, dtype=torch.float64)
                seq = base + noise
                if dim > 2:
                    extra = 0.025 * torch.randn((length_i, dim - 2), generator=generator, dtype=torch.float64)
                    seq = torch.cat([seq, extra], dim=1)
                seqs.append(seq)
                labels.append(label)
    return train, torch.tensor(train_labels), test, torch.tensor(test_labels)


def make_distance_functions(
    lambda1: float = 10.0,
    lambda2: float = 0.1,
    sigma: float = 1.0,
    n_iters: int = 40,
    row_block: int = 32,
    col_block: int = 64,
) -> dict[str, DistanceFn]:
    """Convenience distance functions for the OWD-style toy evaluation."""

    def sinkhorn(x: torch.Tensor, y: torch.Tensor) -> float:
        return float(dense_sinkhorn_cpu(x, y, epsilon=lambda2, n_iters=n_iters).distance.item())

    def opw(x: torch.Tensor, y: torch.Tensor) -> float:
        return float(opw_exact_cpu(x, y, lambda1, lambda2, sigma, n_iters=n_iters).distance.item())

    def flash_opw(x: torch.Tensor, y: torch.Tensor) -> float:
        return float(
            opw_flash_cpu(
                x,
                y,
                lambda1,
                lambda2,
                sigma,
                n_iters=n_iters,
                row_block=row_block,
                col_block=col_block,
                return_transport=False,
            ).distance.item()
        )

    return {"Sinkhorn": sinkhorn, "OPW-exact": opw, "FlashOPW": flash_opw}

