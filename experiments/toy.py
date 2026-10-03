"""Fixed-iteration comparisons on deterministic synthetic probability measures."""

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

from flashopw import (sinkhorn_dense, sinkhorn_flash, sinkhorn_online,
                     apply_plan, diagnostics, materialize_plan)
from .datasets import DATASETS, make_dataset
from .runtime import configure, metadata


def measure(fn, device, repeats):
    # Compile/warm up separately. No retained previous result pollutes the peak.
    warm = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    del warm
    samples, peaks = [], []
    result = None
    for _ in range(repeats):
        if result is not None:
            del result
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            base = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        result = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
        if device.type == "cuda":
            peaks.append((torch.cuda.max_memory_allocated(device) - base) / 2**20)
    return result, statistics.median(samples), max(peaks) if peaks else None, samples


def plot_results(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for dataset in sorted({row["dataset"] for row in rows}):
        for backend in sorted({row["backend"] for row in rows}):
            selected = [r for r in rows if r["dataset"] == dataset and r["backend"] == backend]
            ns = [r["n"] for r in selected]
            label = f"{dataset}/{backend}"
            axes[0].plot(ns, [r["median_ms"] for r in selected], "o-", label=label)
            axes[1].plot(ns, [max(r["row_l1"], r["col_l1"], 1e-12) for r in selected], "o-", label=label)
            if selected[0]["peak_extra_allocated_mib"] is not None:
                axes[2].plot(ns, [r["peak_extra_allocated_mib"] for r in selected], "o-", label=label)
    for ax, title, ylabel in zip(axes, ("Warm solve time", "Marginal residual", "Peak extra PyTorch allocation"),
                                ("milliseconds", "max marginal L1", "MiB")):
        ax.set(title=title, xlabel="source points n", ylabel=ylabel)
        ax.grid(alpha=0.25)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=6)
    if all(r["peak_extra_allocated_mib"] is None for r in rows):
        axes[2].text(0.5, 0.5, "GPU memory: not measured on CPU", ha="center", transform=axes[2].transAxes)
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


def plot_transport(result, out):
    if result.x.shape[1] != 2:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pv = apply_plan(result, result.y)
    rows = apply_plan(result, torch.ones_like(result.b))
    # Conditional barycenters use actual row masses for a finite iterate.
    bary = pv / rows.clamp_min(torch.finfo(rows.dtype).tiny)[:, None]
    x, y, bary = [t.cpu().numpy() for t in (result.x, result.y, bary)]
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(y[:, 0], y[:, 1], s=12, alpha=0.4, label="target")
    ax.scatter(x[:, 0], x[:, 1], s=12, alpha=0.5, label="source")
    step = max(1, len(x) // 80)
    ax.quiver(x[::step, 0], x[::step, 1], (bary-x)[::step, 0], (bary-x)[::step, 1],
              angles="xy", scale_units="xy", scale=1, width=0.003)
    ax.set(title=f"Conditional barycentric map ({result.backend})", aspect="equal")
    ax.legend()
    fig.savefig(out / "transport.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--target-ratio", type=float, default=1.0)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--cost-scale", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--schedule", choices=("alternating", "symmetric"), default="alternating")
    parser.add_argument("--precision", choices=("ieee", "tf32x3", "tf32"), default="tf32x3")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    parser.add_argument("--max-dense-mib", type=float, default=512)
    parser.add_argument("--agreement-tol", type=float, default=5e-4)
    parser.add_argument("--residual-tol", type=float, default=1e-3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.repeats < 1 or args.dim < 1 or args.iters < 1:
        parser.error("sizes, dim, iters and repeats must be positive")
    for key in ("target_ratio", "max_dense_mib", "agreement_tol", "residual_tol", "epsilon", "cost_scale"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be finite and positive")
    if "rings" in args.datasets and args.dim < 2:
        parser.error("rings requires --dim >= 2")
    device = configure(args.device, args.threads, args.memory_fraction)
    out = args.output or Path("outputs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    # Refuse to overwrite another experiment's files.
    out.mkdir(parents=True, exist_ok=False)
    env = metadata(device)
    env["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    env["timing"] = "median synchronized wall time; includes validation/setup; excludes JIT warmup and diagnostics"
    env["memory"] = "peak PyTorch allocated bytes above live-input baseline; excludes driver/JIT caches; not total VRAM"
    env["comparison"] = "triton vs dense" if device.type == "cuda" else "torch online oracle vs dense; NOT a Triton benchmark"
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(env["comparison"], flush=True)
    solver = sinkhorn_flash if device.type == "cuda" else sinkhorn_online
    rows, raw_times = [], []
    failures = []
    last = None
    for dataset in args.datasets:
        for n in args.sizes:
            m = max(1, round(n * args.target_ratio))
            # Conservative estimate of overlapping dense work tensors (10*n*m).
            estimated = 10 * n * m * 4 / 2**20
            limit = args.max_dense_mib
            if device.type == "cuda":
                free, total = torch.cuda.mem_get_info(device)
                limit = min(limit, 0.5 * free / 2**20, 0.8 * args.memory_fraction * total / 2**20)
            if estimated > limit:
                raise RuntimeError(f"Dense case {n}x{m} estimates {estimated:.1f} MiB > budget {limit:.1f} MiB; reduce --sizes")
            last = None
            x, y, a, b = make_dataset(dataset, n, m, args.dim, seed=args.seed, weighted=args.weighted, device=device)
            kwargs = dict(a=a, b=b, epsilon=args.epsilon, cost_scale=args.cost_scale,
                          n_iters=args.iters, schedule=args.schedule)
            dense, dense_ms, dense_mem, dense_times = measure(lambda: sinkhorn_dense(x, y, **kwargs), device, args.repeats)
            extra = {"precision": args.precision} if device.type == "cuda" else {}
            candidate, cand_ms, cand_mem, cand_times = measure(lambda: solver(x, y, **kwargs, **extra), device, args.repeats)
            # Gauge-invariant log-coupling difference without n*m allocation:
            # the coordinate dot term is shared; delta(log P)=delta u+delta v.
            du, dv = candidate.u - dense.u, candidate.v - dense.v
            logp_error = float(torch.maximum((du.min() + dv.min()).abs(), (du.max() + dv.max()).abs()))
            plan_error = None
            if n * m <= 1_048_576:
                plan_error = float((materialize_plan(candidate) - materialize_plan(dense)).abs().max())
            stats = [diagnostics(dense), diagnostics(candidate)]
            for result, ms, mem, times, diag in zip((dense, candidate), (dense_ms, cand_ms),
                                                   (dense_mem, cand_mem), (dense_times, cand_times), stats):
                row = dict(dataset=dataset, n=n, m=m, dim=args.dim, weighted=args.weighted,
                           epsilon=args.epsilon, cost_scale=args.cost_scale, schedule=args.schedule,
                           iterations=result.n_iters, backend=result.backend, precision=result.precision,
                           block_m=result.block_m, block_n=result.block_n,
                           median_ms=ms, peak_extra_allocated_mib=mem, speedup_vs_dense=dense_ms/ms,
                           max_log_plan_diff=logp_error, max_plan_diff=plan_error, **diag)
                rows.append(row)
                raw_times.append(dict(dataset=dataset, n=n, m=m, backend=result.backend, samples_ms=times))
                print(f"{dataset:8s} {n:5d}x{m:<5d} {result.backend:7s} {ms:9.3f} ms "
                      f"residual={max(diag['row_l1'], diag['col_l1']):.2e}", flush=True)
                if not all(math.isfinite(v) for v in diag.values()) or max(diag["row_l1"], diag["col_l1"]) > args.residual_tol:
                    failures.append(f"{dataset}/{n}/{result.backend}: nonfinite or unconverged marginals")
            if not math.isfinite(logp_error) or logp_error > args.agreement_tol:
                failures.append(f"{dataset}/{n}: log-plan disagreement {logp_error:.3g}")
            with (out / "results.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (out / "timings.json").write_text(json.dumps(raw_times, indent=2), encoding="utf-8")
            last = candidate
            del dense, candidate
    (out / "validation.json").write_text(json.dumps({"passed": not failures, "failures": failures}, indent=2), encoding="utf-8")
    if not args.no_plots:
        plot_results(rows, out)
        plot_transport(last, out)
    print(f"Results: {out.resolve()}")
    if failures:
        raise SystemExit("Validation failed; see validation.json. Increase iterations for residual failures; investigate numerical mismatches.")


if __name__ == "__main__":
    main()
