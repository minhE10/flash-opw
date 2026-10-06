"""Paired long-sequence speed tests: same-GPU affine control and journal CPUs.

Synthetic smooth trajectories test scaling, not journal classification accuracy.
CPU baselines run in subprocesses with a wall-clock deadline (warmup included).
"""

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from flashopw import OPWParameters, opw_dense, opw_diagnostics, opw_flash
from .opw_parameters import atomic_json, load_selection, source_hashes
from .runtime import configure, metadata
from .sequence_metrics import JOURNAL_METRICS, reference_distance
from .sequence_metrics_torch import ENTROPIC_METRICS, dense_tensor_distance


FIELDS = ("n", "m", "d", "seed", "metric", "backend", "status", "median_ms", "min_ms",
          "max_ms", "repeats", "score", "iterations", "peak_cuda_extra_bytes", "detail")
GPU_JOURNAL_METRICS = tuple(name+"-dense-gpu" for name in ENTROPIC_METRICS)


def synthetic_pair(n, m, d, seed):
    """Continuous curves with a time warp and noise, spatial scale ~1 for all d."""
    rng = np.random.default_rng(seed)
    frequency = rng.uniform(0.5, 3, d)
    phase = rng.uniform(-np.pi, np.pi, d)
    t = np.arange(1, n+1)/n
    s = (np.arange(1, m+1)/m)**1.15
    x = np.sin(2*np.pi*t[:, None]*frequency + phase)
    y = np.sin(2*np.pi*s[:, None]*frequency + phase)
    x += rng.normal(0, 0.03, x.shape)
    y += rng.normal(0, 0.03, y.shape)
    # Give CPU FP64 references exactly the values seen by CUDA FP32.
    return tuple(np.ascontiguousarray(v/math.sqrt(d), dtype=np.float32).astype(np.float64) for v in (x, y))


def estimated_dense_bytes(n, m, metric, itemsize=8):
    factor = 2 if metric in ("dtw", "ldtw", "ndtw", "soft-dtw") else 10 if metric == "ot" else 8
    return factor*n*m*itemsize


def _cpu_worker(pipe, x, y, metric, kwargs, repeats, threads):
    try:
        torch.set_num_threads(threads)
        operation = lambda: reference_distance(metric, x, y, **kwargs)
        operation()  # compilation and first LP setup excluded
        times, values = [], []
        for _ in range(repeats):
            started = time.perf_counter()
            values.append(operation())
            times.append(1000*(time.perf_counter()-started))
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite baseline score")
        pipe.send(dict(status="ok", times=times, score=values[-1]))
    except Exception:
        pipe.send(dict(status="failed", detail=traceback.format_exc(limit=2)))
    finally:
        pipe.close()


def timed_cpu(x, y, metric, kwargs, repeats, threads, timeout):
    """Kill only this benchmark's own child if it exceeds its declared deadline."""
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_cpu_worker, args=(child, x, y, metric, kwargs, repeats, threads))
    process.start()
    child.close()
    try:
        if parent.poll(timeout):
            try:
                result = parent.recv()
            except EOFError:
                result = dict(status="failed", detail=f"CPU worker exited without results ({process.exitcode})")
        else:
            result = dict(status="timeout", detail=f"CPU warmup + {repeats} repeats exceeded {timeout:g}s")
    finally:
        if process.is_alive():
            process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
    return result


def _write(output, rows, details):
    with (output/"timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output/"timings.json", rows)
    atomic_json(output/"checks.json", details)


def _plot(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for d in sorted({r["d"] for r in rows}):
        selected = [r for r in rows if r["d"] == d and r["status"] == "ok"]
        if not selected:
            continue
        fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for metric in dict.fromkeys(r["metric"] for r in selected):
            points = [r for r in selected if r["metric"] == metric]
            lengths = sorted({r["n"] for r in points})
            medians = [np.median([r["median_ms"] for r in points if r["n"] == n]) for n in lengths]
            axis.plot(lengths, medians, marker="o", label=metric+" ("+points[0]["backend"]+")")
        axis.set(xscale="log", yscale="log", xlabel="Query frames (n)", ylabel="Median wall time per pair (ms)",
                 title=f"Synthetic sequences, d={d}; completed measurements only")
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
        fig.savefig(output/f"scaling_d{d}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[1, 13])
    parser.add_argument("--length-ratio", type=float, default=1.25, help="m/n; use 1 for square cases")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--metrics", nargs="+", choices=["flash-opw", "affine-opw-dense-gpu", *GPU_JOURNAL_METRICS, *JOURNAL_METRICS],
                        default=["flash-opw", "affine-opw-dense-gpu", *GPU_JOURNAL_METRICS, *JOURNAL_METRICS])
    parser.add_argument("--flash-parameters", type=Path)
    parser.add_argument("--lambda1", type=float, default=1)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--journal-opw-iters", type=int, default=20)
    parser.add_argument("--sinkhorn-iters", type=int, default=100)
    parser.add_argument("--equal-entropic-iters", type=int, help="override all entropic metrics to this fixed iteration count")
    parser.add_argument("--max-cpu-mib", type=float, default=512)
    parser.add_argument("--max-gpu-dense-mib", type=float, default=256)
    parser.add_argument("--max-ot-entries", type=int, default=65536)
    parser.add_argument("--cpu-timeout", type=float, default=45)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.equal_entropic_iters is not None:
        args.iters = args.journal_opw_iters = args.sinkhorn_iters = args.equal_entropic_iters
    if min(*args.lengths, *args.dimensions, args.repeats, args.iters, args.journal_opw_iters,
           args.sinkhorn_iters, args.max_ot_entries) < 1 or max(args.dimensions) > 1023:
        parser.error("Positive sizes/iterations required, with spatial dimensions <=1023")
    if any(not math.isfinite(v) or v <= 0 for v in (args.length_ratio, args.max_cpu_mib,
            args.max_gpu_dense_mib, args.cpu_timeout)):
        parser.error("Finite positive ratio, budgets and timeout required")
    if any(len(v) != len(set(v)) for v in (args.lengths, args.dimensions, args.seeds, args.metrics)):
        parser.error("Lengths, dimensions, seeds and metrics must be distinct")
    baseline_parameters = dict(lambda1=args.lambda1, lambda2=args.lambda2, sigma=args.sigma,
                               cost_scale=1., n_iters=args.iters)
    OPWParameters(**{k:v for k,v in baseline_parameters.items() if k != "n_iters"})
    selection = load_selection(args.flash_parameters, score="pdf-loss", n_iters=args.iters) if args.flash_parameters else None
    parameters = selection["parameters"] if selection else baseline_parameters
    os.environ["FLASHOPW_AUTOTUNE"] = "0"
    device = configure(args.device, args.threads, args.memory_fraction)
    environment = metadata(device)
    settings = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items() if k not in ("output", "resume")}
    signature = dict(settings=settings, parameters=parameters, selection=selection, sources=source_hashes(),
                     packages=environment["packages"], torch=environment["torch"], python=environment["python"],
                     gpu=environment.get("gpu"), kernel_controls=environment["kernel_controls"])
    output = args.output
    output.mkdir(parents=True, exist_ok=args.resume)
    rows, details = [], []
    if args.resume:
        prior = json.loads((output/"environment.json").read_text(encoding="utf-8"))
        if prior["signature"] != signature:
            raise ValueError("Resume refused: code/settings/environment/selection changed")
        rows = json.loads((output/"timings.json").read_text(encoding="utf-8"))
        details = json.loads((output/"checks.json").read_text(encoding="utf-8"))
    else:
        environment.update(signature=signature, protocol=dict(
            data="synthetic continuous sine trajectories, time warp 1.15, noise .03, features divided by sqrt(d)",
            purpose="length/feature scaling; no classification ACC/MAP claim for synthetic data",
            timing="one warmup excluded; wall time for full pair including cost/features/solve/scalar read; resident inputs",
            baselines="journal CPU FP64 and five dense entropic journal GPU FP32 references; dense affine GPU is same-cost/same-iteration control",
            iterations="Flash and GPU affine use identical fixed iters; journal OPW20, other entropic100 by default",
            memory="estimated dense working arrays preflight; CUDA extra allocator peak excludes resident inputs/context",
            limits="resource skips and CPU timeouts are explicit rows; no fabricated timings or speedup",
            selection="Frozen training parameters used only for Flash and its affine control; journal baselines retain CLI settings"))
        atomic_json(output/"environment.json", environment)
        _write(output, rows, details)
    atomic_json(output/"run_state.json", dict(status="running"))
    completed = {(r["n"], r["m"], r["d"], r["seed"], r["metric"]) for r in rows}
    for n in args.lengths:
        m = max(1, int(round(n*args.length_ratio)))
        for d in args.dimensions:
            for seed in args.seeds:
                x, y = synthetic_pair(n, m, d, seed)
                input_hash = hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
                tensors = None
                if device.type == "cuda":
                    tensors = tuple(torch.as_tensor(v, dtype=torch.float32, device=device) for v in (x, y))
                for metric in args.metrics:
                    key = n, m, d, seed, metric
                    if key in completed:
                        continue
                    gpu = metric == "flash-opw" or metric.endswith("-dense-gpu")
                    affine = metric in ("flash-opw", "affine-opw-dense-gpu")
                    base_metric = metric.removesuffix("-dense-gpu")
                    iterations = args.iters if affine else (args.journal_opw_iters if base_metric in ("opw", "opw-kl") else
                                 args.sinkhorn_iters if base_metric in ("sinkhorn", "tlp", "tcot") else None)
                    row = dict.fromkeys(FIELDS)
                    row.update(n=n, m=m, d=d, seed=seed, metric=metric,
                               backend="CUDA FP32" if gpu else "CPU FP64", iterations=iterations,
                               repeats=0, status="skipped", detail="")
                    extra = dict(n=n, m=m, d=d, seed=seed, metric=metric, input_sha256=input_hash)
                    try:
                        if gpu:
                            if device.type != "cuda":
                                row["detail"] = "CUDA unavailable in explicit CPU mode; no substitute Flash timing"
                            else:
                                required = estimated_dense_bytes(n, m, metric, 4)
                                free, total = torch.cuda.mem_get_info()
                                available = min(free * .7, max(0, total*args.memory_fraction-torch.cuda.memory_allocated())*.7)
                                if metric.endswith("gpu") and required > min(args.max_gpu_dense_mib*1024**2, available):
                                    row["detail"] = f"Dense working estimate {required} bytes exceeds declared budget or currently available VRAM"
                                else:
                                    if affine:
                                        solver = opw_flash if metric == "flash-opw" else opw_dense
                                        kwargs = dict(parameters, **({"precision":"ieee"} if metric == "flash-opw" else {}))
                                        operation = lambda: solver(*tensors, **kwargs)
                                    else:
                                        kwargs = dict(baseline_parameters, cost_scale=parameters["cost_scale"],
                                                      n_iters=args.journal_opw_iters, sinkhorn_iters=args.sinkhorn_iters)
                                        operation = lambda: dense_tensor_distance(base_metric, *tensors, **kwargs)
                                    warmup = operation()
                                    del warmup
                                    torch.cuda.synchronize()
                                    baseline = torch.cuda.memory_allocated()
                                    torch.cuda.reset_peak_memory_stats()
                                    times, scores = [], []
                                    for _ in range(args.repeats):
                                        torch.cuda.synchronize()
                                        started = time.perf_counter()
                                        result = operation()
                                        scores.append(float(result.loss if affine else result))
                                        torch.cuda.synchronize()
                                        times.append(1000*(time.perf_counter()-started))
                                        del result
                                    if not np.isfinite(scores).all():
                                        raise FloatingPointError("Nonfinite GPU score")
                                    row.update(status="ok", score=scores[-1], median_ms=float(np.median(times)),
                                               min_ms=min(times), max_ms=max(times), repeats=args.repeats,
                                               peak_cuda_extra_bytes=max(0, torch.cuda.max_memory_allocated()-baseline))
                                    # Streamed diagnostics, excluded from timing and peak measurement.
                                    diagnostic = (operation() if affine else
                                                  dense_tensor_distance(base_metric, *tensors, **kwargs, return_diagnostics=True))
                                    stats = (opw_diagnostics(diagnostic) if affine else
                                             dict(row_l1=float(diagnostic[1]), col_l1=float(diagnostic[2])))
                                    extra.update(diagnostics=stats, times_ms=times)
                                    del diagnostic
                        elif metric == "ot" and n*m > args.max_ot_entries:
                            row["detail"] = f"Exact OT LP entries {n*m} exceed --max-ot-entries={args.max_ot_entries}"
                        elif estimated_dense_bytes(n, m, metric) > args.max_cpu_mib*1024**2:
                            row["detail"] = "CPU dense working estimate exceeds --max-cpu-mib"
                        else:
                            kwargs = dict(baseline_parameters, cost_scale=parameters["cost_scale"], n_iters=args.journal_opw_iters,
                                          sinkhorn_iters=args.sinkhorn_iters)
                            response = timed_cpu(x, y, metric, kwargs, args.repeats, args.threads, args.cpu_timeout)
                            row.update(status=response["status"], detail=response.get("detail", ""))
                            if response["status"] == "ok":
                                times = response["times"]
                                row.update(score=response["score"], median_ms=float(np.median(times)),
                                           min_ms=min(times), max_ms=max(times), repeats=args.repeats)
                                extra["times_ms"] = times
                    except Exception as exc:
                        row.update(status="failed", detail=f"{type(exc).__name__}: {exc}")
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    rows.append(row)
                    details.append(extra)
                    _write(output, rows, details)
                    print(f"{n}x{m} d={d} seed={seed} {metric}: {row['status']} {row['median_ms'] if row['median_ms'] is not None else row['detail']}", flush=True)
                del tensors
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    comparisons = []
    for flash in rows:
        if flash["metric"] != "flash-opw" or flash["status"] != "ok":
            continue
        for other in rows:
            if other["metric"] == "flash-opw" or other["status"] != "ok" or any(
                    flash[k] != other[k] for k in ("n", "m", "d", "seed")):
                continue
            item = {k:flash[k] for k in ("n", "m", "d", "seed")}
            item.update(compared_metric=other["metric"], wall_time_ratio=other["median_ms"]/flash["median_ms"],
                        interpretation="same GPU, same cost and iterations" if other["metric"] == "affine-opw-dense-gpu"
                        else "same GPU, different cost/score; check recorded iteration counts" if other["metric"].endswith("-dense-gpu")
                        else "CPU reference / GPU Flash wall time; different algorithm/backend/iteration count")
            if other["metric"] == "affine-opw-dense-gpu":
                item["score_abs_error"] = abs(flash["score"]-other["score"])
            comparisons.append(item)
    atomic_json(output/"comparisons.json", comparisons)
    _plot(output, rows)
    failed = sum(r["status"] == "failed" for r in rows)
    atomic_json(output/"run_state.json", dict(status="failed" if failed else "completed", failures=failed,
                skipped=sum(r["status"] == "skipped" for r in rows), timeouts=sum(r["status"] == "timeout" for r in rows)))
    print(f"Results: {output.resolve()}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
