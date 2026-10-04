"""Compare FlashSinkhorn with GeomLoss/KeOps and OTT-JAX on real data.

The benchmark uses the cached MNIST/Fashion-MNIST ResNet18 features from
``phase2_real``. The n-sweep uses the full 512-dimensional representation;
the d-sweep uses deterministic leading feature slices, so it is an
illustration of dimension scaling on the same real dataset rather than a new
trained representation at every dimension.
"""

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import statistics
import time

import torch

from flashopw import sinkhorn_flash
from .phase2_real import load_embeddings, stratified_indices
from .runtime import configure, metadata


def _time_torch(fn, warmups, repeats, device):
    for _ in range(warmups):
        value = fn()
        torch.cuda.synchronize(device)
        del value
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
        del value
    return statistics.median(samples), samples


def _time_jax(fn, warmups, repeats, jax):
    for _ in range(warmups):
        fn().block_until_ready()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn().block_until_ready()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples), samples


def _make_geomloss_backend(name, epsilon):
    try:
        from geomloss import SamplesLoss
    except ImportError as exc:
        raise RuntimeError("install GeomLoss and PyKeOps with: python -m pip install -e '.[baselines]'") from exc
    loss = SamplesLoss(
        loss="sinkhorn", p=2, blur=math.sqrt(epsilon), scaling=0.9,
        debias=False, backend="online" if name == "keops" else "tensorized",
    )

    @torch.no_grad()
    def run(x, y):
        return loss(x, y)

    return run, "GeomLoss SamplesLoss; blur=sqrt(epsilon), scaling=0.9"


def _make_jax_backend(epsilon, iterations):
    try:
        import jax
        import jax.numpy as jnp
        from ott.geometry import pointcloud
        from ott.problems.linear import linear_problem
        from ott.solvers.linear import sinkhorn
    except ImportError as exc:
        raise RuntimeError("install JAX and OTT-JAX with: python -m pip install -e '.[baselines]'") from exc
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError(f"JAX sees no GPU; devices={jax.devices()}")
    solver = sinkhorn.Sinkhorn(
        threshold=0.0,
        inner_iterations=iterations,
        min_iterations=iterations,
        max_iterations=iterations,
        parallel_dual_updates=False,
    )

    def solve(x, y):
        geometry = pointcloud.PointCloud(x, y, epsilon=epsilon)
        problem = linear_problem.LinearProblem(geometry)
        return solver(problem).reg_ot_cost

    compiled = jax.jit(solve)

    def prepare(x, y):
        # Transfer before timing; only the JAX computation is benchmarked.
        return jax.device_put(jnp.asarray(x.detach().cpu().numpy())), jax.device_put(jnp.asarray(y.detach().cpu().numpy()))

    return compiled, prepare, _time_jax, jax, "OTT-JAX PointCloud + fixed Sinkhorn iterations"


def _plot(rows, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plots require matplotlib; install with: python -m pip install -e '.[plots]'") from exc

    successful = [row for row in rows if row["status"] == "ok"]
    colors = {"flash": "#d62728", "keops": "#1f77b4", "tensorized": "#2ca02c", "jax": "#9467bd"}
    labels = {"flash": "FlashSinkhorn", "keops": "KeOps", "tensorized": "Tensorized", "jax": "OTT-JAX"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for column, sweep, xlabel in ((0, "n", "n = m"), (1, "d", "feature dimension d")):
        selected = [row for row in successful if row["sweep"] == sweep]
        for backend in ("flash", "keops", "tensorized", "jax"):
            points = sorted((row for row in selected if row["backend"] == backend),
                            key=lambda row: row["axis_value"])
            if points:
                axes[0, column].plot([row["axis_value"] for row in points],
                                     [row["median_ms"] for row in points], "o-",
                                     color=colors[backend], label=labels[backend])
                axes[1, column].plot([row["axis_value"] for row in points],
                                     [row["speedup_vs_flash"] for row in points], "o-",
                                     color=colors[backend], label=labels[backend])
        axes[0, column].set(xlabel=xlabel, ylabel="runtime (ms)", title=f"Runtime vs {xlabel}")
        axes[1, column].set(xlabel=xlabel, ylabel="speedup vs FlashSinkhorn", title=f"Speedup vs {xlabel}")
        axes[0, column].set_yscale("log")
        axes[1, column].axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
        axes[0, column].grid(alpha=0.25, which="both")
        axes[1, column].grid(alpha=0.25)
    axes[0, 0].set_xscale("log")
    axes[0, 1].set_xscale("log", base=2)
    axes[1, 0].set_xscale("log")
    axes[1, 1].set_xscale("log", base=2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("MNIST → Fashion-MNIST real-data baseline comparison")
    fig.savefig(output / "paper_style_baselines.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sizes", type=int, nargs="+", default=[5000, 10000, 15000, 20000])
    parser.add_argument("--d-sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--d-sweep-n", type=int, default=5000)
    parser.add_argument("--target-ratio", type=float, default=1.0)
    parser.add_argument("--backends", nargs="+", choices=("flash", "keops", "tensorized", "jax"),
                        default=["flash", "keops", "tensorized", "jax"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--precision", choices=("ieee", "tf32x3", "tf32"), default="tf32")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    parser.add_argument("--max-tensorized-mib", type=float, default=6144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.n_sizes + args.d_sizes + [args.d_sweep_n]) < 10:
        parser.error("all sizes must be at least 10")
    if max(args.d_sizes) > 512:
        parser.error("d-sizes cannot exceed the cached ResNet18 feature dimension 512")

    device = configure("cuda", threads=2, memory_fraction=args.memory_fraction)
    max_samples = max(max(args.n_sizes), args.d_sweep_n,
                       round(max(args.n_sizes) * args.target_ratio))
    saved = load_embeddings(args.data_root, max_samples, args.batch_size, 0,
                            device, args.seed, args.cache or args.data_root / "phase2_mnist_fashion_resnet18.pt")
    out = args.output or Path("outputs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out.mkdir(parents=True, exist_ok=False)

    env = metadata(device)
    env["args"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    env["dataset"] = "MNIST -> Fashion-MNIST"
    env["features"] = saved["model"]
    env["note"] = "d-sweep uses leading slices of the cached 512D ResNet18 features"
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    backends = {}
    for backend in args.backends:
        try:
            if backend == "flash":
                backends[backend] = (lambda x, y: sinkhorn_flash(
                    x, y, epsilon=args.epsilon, n_iters=args.iters,
                    schedule="symmetric", precision=args.precision), None, _time_torch, device,
                    "FlashSinkhorn Triton")
            elif backend in ("keops", "tensorized"):
                runner, detail = _make_geomloss_backend(backend, args.epsilon)
                backends[backend] = (runner, None, _time_torch, device, detail)
            else:
                backends[backend] = _make_jax_backend(args.epsilon, args.iters)
        except RuntimeError as exc:
            print(f"{backend}: unavailable ({exc})", flush=True)
            backends[backend] = None

    rows = []

    def run_sweep(sweep, values, fixed_n, fixed_d):
        for axis_value in values:
            n = fixed_n if sweep == "d" else axis_value
            m = max(1, round(n * args.target_ratio))
            d = axis_value if sweep == "d" else fixed_d
            source_indices = stratified_indices(saved["source_labels"], n, seed=args.seed)
            target_indices = stratified_indices(saved["target_labels"], m, seed=args.seed + 1)
            x = saved["source"][source_indices, :d].to(device)
            y = saved["target"][target_indices, :d].to(device)
            prepared = {}
            for backend, bundle in backends.items():
                if bundle is None:
                    rows.append(dict(sweep=sweep, axis_value=axis_value, n=n, m=m, d=d,
                                     backend=backend, status="unavailable", median_ms=None,
                                     speedup_vs_flash=None, detail="optional dependency unavailable"))
                    continue
                if backend == "tensorized":
                    estimated = 10 * n * m * 4 / 2**20
                    if estimated > args.max_tensorized_mib:
                        rows.append(dict(sweep=sweep, axis_value=axis_value, n=n, m=m, d=d,
                                         backend=backend, status="skipped_memory", median_ms=None,
                                         speedup_vs_flash=None, detail=f"estimated {estimated:.1f} MiB"))
                        continue
                runner, prepare, timer, sync_context, detail = bundle
                if prepare is not None:
                    prepared[backend] = prepare(x, y)
                else:
                    prepared[backend] = (x, y)
                try:
                    value = prepared[backend]
                    median, samples = timer(lambda: runner(*value), args.warmups, args.repeats, sync_context)
                    rows.append(dict(sweep=sweep, axis_value=axis_value, n=n, m=m, d=d,
                                     backend=backend, status="ok", median_ms=median,
                                     speedup_vs_flash=None, detail=detail, samples_ms=samples))
                    print(f"{sweep}={axis_value:6d} {backend:10s} {median:10.3f} ms", flush=True)
                except Exception as exc:  # keep other baselines and the plots usable
                    rows.append(dict(sweep=sweep, axis_value=axis_value, n=n, m=m, d=d,
                                     backend=backend, status="failed", median_ms=None,
                                     speedup_vs_flash=None, detail=f"{type(exc).__name__}: {exc}"))
                    print(f"{sweep}={axis_value:6d} {backend:10s} FAILED: {exc}", flush=True)
            successful = [row for row in rows if row["sweep"] == sweep and row["axis_value"] == axis_value
                          and row["status"] == "ok"]
            flash_rows = [row for row in successful if row["backend"] == "flash"]
            if flash_rows:
                flash_ms = flash_rows[-1]["median_ms"]
                for row in successful:
                    row["speedup_vs_flash"] = flash_ms / row["median_ms"]
            del x, y, prepared
            gc.collect()
            torch.cuda.empty_cache()

    run_sweep("n", args.n_sizes, None, 512)
    run_sweep("d", args.d_sizes, args.d_sweep_n, None)
    fieldnames = ["sweep", "axis_value", "n", "m", "d", "backend", "status", "median_ms",
                  "speedup_vs_flash", "detail", "samples_ms"]
    with (out / "baseline_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, out)
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
