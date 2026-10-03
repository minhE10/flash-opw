"""Toy CPU benchmark for FlashSinkhorn and the baselines from its paper.

Run from the repository root:
    python experiments/toy_flashsinkhorn.py --sizes 32,64,128,256

Outputs:
    outputs/flashsinkhorn_toy.csv
    outputs/flashsinkhorn_toy.png

GeomLoss and OTT-JAX are included only when installed.  Missing optional
backends are reported in the console and never replaced with fabricated data.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import torch

from flashuot.sinkhorn import dense_sinkhorn_cpu, flash_sinkhorn_cpu


def _measure(fn):
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
    elapsed = time.perf_counter() - start
    stop.set()
    thread.join(timeout=1.0)
    peak = max(peak, process.memory_info().rss)
    return result, elapsed, max(0, peak - before) / (1024**2)


def _optional_methods(epsilon: float, n_iters: int):
    methods = {}
    if importlib.util.find_spec("geomloss"):
        from geomloss import SamplesLoss

        loss = SamplesLoss(loss="sinkhorn", p=2, blur=epsilon**0.5, debias=False, backend="tensorized")

        def geomloss(x, y):
            return loss(x, y)

        methods["GeomLoss-tensorized"] = geomloss
    else:
        print("[skip] GeomLoss not installed; install geomloss to include its tensorized baseline.")

    if importlib.util.find_spec("ott"):
        print("[info] OTT-JAX detected; add an OTT adapter here if benchmarking its installed API version.")
    else:
        print("[skip] OTT-JAX not installed; install ott-jax to benchmark that paper baseline.")
    return methods


def run(sizes: list[int], d: int, epsilon: float, n_iters: int, row_block: int, col_block: int):
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    rows = []
    optional = _optional_methods(epsilon, n_iters)
    for n in sizes:
        generator = torch.Generator().manual_seed(1000 + n)
        x = torch.randn((n, d), generator=generator, dtype=torch.float32)
        y = torch.randn((n, d), generator=generator, dtype=torch.float32)
        methods = {
            "FlashSinkhorn-CPU-tiled": lambda: flash_sinkhorn_cpu(
                x,
                y,
                epsilon=epsilon,
                n_iters=n_iters,
                row_block=row_block,
                col_block=col_block,
                return_transport=False,
            ),
            "PyTorch-dense": lambda: dense_sinkhorn_cpu(
                x, y, epsilon=epsilon, n_iters=n_iters, return_transport=False
            ),
        }
        methods.update(optional)
        for name, fn in methods.items():
            result, seconds, peak_mb = _measure(fn)
            rows.append(
                {
                    "method": name,
                    "n": n,
                    "d": d,
                    "epsilon": epsilon,
                    "iterations": result.iterations if hasattr(result, "iterations") else n_iters,
                    "distance": float(result.distance.item()) if hasattr(result, "distance") else float(result.item()),
                    "time_seconds": seconds,
                    "peak_rss_delta_mb": peak_mb,
                }
            )
            print(f"{name:28s} n={n:5d} time={seconds:8.4f}s peak_delta={peak_mb:8.2f}MB")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method, group in df.groupby("method"):
        group = group.sort_values("n")
        axes[0].plot(group["n"], group["time_seconds"], marker="o", label=method)
        axes[1].plot(group["n"], group["peak_rss_delta_mb"], marker="o", label=method)
    axes[0].set_title("Toy runtime (CPU)")
    axes[0].set_xlabel("number of points n=m")
    axes[0].set_ylabel("seconds")
    axes[0].set_yscale("log")
    axes[1].set_title("Approximate peak RSS increase")
    axes[1].set_xlabel("number of points n=m")
    axes[1].set_ylabel("MiB")
    axes[1].set_yscale("symlog", linthresh=1e-2)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="32,64,128,256")
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--row-block", type=int, default=64)
    parser.add_argument("--col-block", type=int, default=128)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    df = run(sizes, args.d, args.epsilon, args.iters, args.row_block, args.col_block)
    csv_path = output_dir / "flashsinkhorn_toy.csv"
    png_path = output_dir / "flashsinkhorn_toy.png"
    df.to_csv(csv_path, index=False)
    plot(df, png_path)
    print(f"saved {csv_path}")
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
