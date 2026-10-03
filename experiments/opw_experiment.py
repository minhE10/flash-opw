"""Toy OPW-vs-FlashOPW experiment plus OWD-style NM/k-NN evaluation.

Run:
    python experiments/opw_experiment.py

The script deliberately compares two different objects:
* OPW-exact: the full effective cost from the OWD journal paper;
* FlashOPW: the affine/Taylor approximation in main (2).pdf.

The returned ``distance`` is always the original spatial cost <T, D>, as in
Eq. (13) of the OWD paper.  This makes the approximation comparison fair.
"""

from __future__ import annotations

import argparse
import gc
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import torch

from flashuot.evaluation import evaluate_methods, make_distance_functions, make_toy_sequence_dataset
from flashuot.opw import opw_approximation_error, opw_exact_cpu, opw_flash_cpu


def measure(fn):
    process = psutil.Process()
    gc.collect()
    before = process.memory_info().rss
    peak = before
    stop = threading.Event()

    def sample():
        nonlocal peak
        while not stop.is_set():
            try:
                peak = max(peak, process.memory_info().rss)
            except psutil.Error:
                return
            time.sleep(0.001)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = time.perf_counter()
    result = fn()
    seconds = time.perf_counter() - start
    stop.set()
    thread.join(timeout=1.0)
    peak = max(peak, process.memory_info().rss)
    return result, seconds, max(0.0, peak - before) / (1024**2)


def plot_pairwise(rows: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for method, group in rows.groupby("method"):
        group = group.sort_values("n")
        axes[0].plot(group["n"], group["time_seconds"], marker="o", label=method)
        axes[1].plot(group["n"], group["peak_rss_delta_mb"], marker="o", label=method)
    axes[0].set_title("OPW solver runtime (CPU)")
    axes[0].set_xlabel("sequence length n=m")
    axes[0].set_ylabel("seconds")
    axes[0].set_yscale("log")
    axes[1].set_title("Approximate peak RSS increase")
    axes[1].set_xlabel("sequence length n=m")
    axes[1].set_ylabel("MiB")
    axes[1].set_yscale("symlog", linthresh=1e-2)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_classification(df: pd.DataFrame, output: Path):
    metric_cols = ["nm_accuracy", "nm_map", "knn_accuracy_1", "knn_accuracy_5", "knn_map"]
    available = [x for x in metric_cols if x in df.columns]
    ax = df.set_index("method")[available].plot(kind="bar", figsize=(11, 4.8), rot=0)
    ax.set_title("OWD-style toy sequence evaluation")
    ax.set_ylabel("percent")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="16,24,32,48")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--lambda1", type=float, default=10.0)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    pair_rows = []
    for n in sizes:
        generator = torch.Generator().manual_seed(42 + n)
        x = torch.randn((n, args.dim), generator=generator, dtype=torch.float64)
        y = torch.randn((n, args.dim), generator=generator, dtype=torch.float64)
        methods = {
            "OPW-exact": lambda: opw_exact_cpu(
                x, y, args.lambda1, args.lambda2, args.sigma, n_iters=args.iters
            ),
            "FlashOPW-affine": lambda: opw_flash_cpu(
                x,
                y,
                args.lambda1,
                args.lambda2,
                args.sigma,
                n_iters=args.iters,
                row_block=64,
                col_block=128,
                return_transport=False,
            ),
        }
        for name, fn in methods.items():
            result, seconds, peak_mb = measure(fn)
            pair_rows.append(
                {
                    "method": name,
                    "n": n,
                    "time_seconds": seconds,
                    "peak_rss_delta_mb": peak_mb,
                    "distance": float(result.distance.item()),
                    "effective_distance": float(result.effective_distance.item()),
                    "marginal_error": result.marginal_error,
                }
            )
            print(f"{name:20s} n={n:4d} time={seconds:8.4f}s peak_delta={peak_mb:8.2f}MB")
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(output_dir / "opw_solver_benchmark.csv", index=False)
    plot_pairwise(pair_df, output_dir / "opw_solver_benchmark.png")

    # Accuracy/coupling comparison on a representative pair.
    x = torch.randn((32, args.dim), generator=torch.Generator().manual_seed(123), dtype=torch.float64)
    y = torch.randn((28, args.dim), generator=torch.Generator().manual_seed(456), dtype=torch.float64)
    exact = opw_exact_cpu(x, y, args.lambda1, args.lambda2, args.sigma, n_iters=args.iters)
    flash = opw_flash_cpu(x, y, args.lambda1, args.lambda2, args.sigma, n_iters=args.iters, return_transport=True)
    transport_rel_error = (flash.transport - exact.transport).norm() / exact.transport.norm().clamp_min(1e-12)
    report = pd.DataFrame(
        [
            {
                "exact_distance": exact.distance.item(),
                "flash_distance": flash.distance.item(),
                "distance_abs_error": abs(flash.distance.item() - exact.distance.item()),
                "distance_relative_error": abs(flash.distance.item() - exact.distance.item()) / max(abs(exact.distance.item()), 1e-12),
                "transport_relative_error": transport_rel_error.item(),
                "exact_marginal_error": exact.marginal_error,
                "flash_marginal_error": flash.marginal_error,
                **opw_approximation_error(x, y, args.lambda1, args.lambda2, args.sigma),
            }
        ]
    )
    report.to_csv(output_dir / "opw_accuracy_report.csv", index=False)
    print("\nOPW vs FlashOPW accuracy:")
    print(report.to_string(index=False))

    train, train_labels, test, test_labels = make_toy_sequence_dataset(seed=11)
    distance_fns = make_distance_functions(
        args.lambda1,
        args.lambda2,
        args.sigma,
        n_iters=min(args.iters, 30),
        row_block=32,
        col_block=64,
    )
    classification = pd.DataFrame(evaluate_methods(train, train_labels, test, test_labels, distance_fns))
    classification.to_csv(output_dir / "owd_style_classification.csv", index=False)
    plot_classification(classification, output_dir / "owd_style_classification.png")
    print("\nOWD-style NM/k-NN metrics:")
    print(classification.to_string(index=False))
    print(f"\nSaved all outputs under {output_dir.resolve()}")


if __name__ == "__main__":
    main()
