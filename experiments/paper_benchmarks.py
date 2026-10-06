"""Reproduce the eight synthetic scaling panels from the FlashSinkhorn paper.

Every panel is saved as an individual PNG and all available panels are also
combined into ``00_overview.png``. Results are checkpointed after every method
and problem size, so a long run still leaves usable CSV/JSON/plots if a
third-party baseline fails or the job is interrupted.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import partial
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch

from flashopw import diagnostics, hessian_vector_product, sinkhorn_cost, sinkhorn_flash
from .runtime import configure, metadata


METHODS = ("flash-sym", "flash-alt", "keops", "tensorized", "jax")
PANEL_METHODS = {
    "memory_forward": ("flash-alt", "tensorized"),
    "memory_backward": ("flash-alt", "tensorized"),
    "hvp_n": ("flash-sym", "keops", "jax"),
    "hvp_d": ("flash-sym", "keops", "jax"),
}
STYLE = {
    "flash-sym": ("FlashSinkhorn (sym)", "#e67e22", "o"),
    "flash-alt": ("FlashSinkhorn (alt)", "#c0392b", "s"),
    "keops": ("KeOps", "#3498db", "^"),
    "tensorized": ("Tensorized", "#9b59b6", "d"),
    "jax": ("JAX", "#2ecc71", "v"),
}
PANELS = {
    "forward_n": ("Fwd: d=64", "n, m", "Time (ms)", "01_forward_n.png"),
    "forward_d": ("Fwd: n=m=20,000", "d", "Time (ms)", "02_forward_d.png"),
    "backward_n": ("Fwd+Bwd: d=64", "n, m", "Time (ms)", "03_forward_backward_n.png"),
    "backward_d": ("Fwd+Bwd: n=m=20,000", "d", "Time (ms)", "04_forward_backward_d.png"),
    "memory_forward": ("Fwd: Memory d=1024", "n, m", "Peak Memory (MB)", "05_forward_memory.png"),
    "memory_backward": ("Fwd+Bwd: Memory d=1024", "n, m", "Peak Memory (MB)", "06_forward_backward_memory.png"),
    "hvp_n": ("HVP: d=64", "n, m", "Time (ms)", "07_hvp_n.png"),
    "hvp_d": ("HVP: n=m=10,000", "d", "Time (ms)", "08_hvp_d.png"),
}
CSV_FIELDS = (
    "experiment", "axis_value", "n", "m", "d", "method", "status",
    "mean", "median", "std", "minimum", "maximum", "unit", "detail",
    "samples",
)


def _preload_cuda_libraries():
    """Expose NVRTC globally before importing PyKeOps."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        try:
            from torch.utils.cpp_extension import CUDA_HOME
            cuda_home = CUDA_HOME
        except Exception:
            return
    if not cuda_home:
        return
    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    os.environ.setdefault("CUDA_PATH", str(cuda_home))
    candidates = (Path(cuda_home) / "targets" / "x86_64-linux" / "lib",
                  Path(cuda_home) / "lib64")
    library_dir = next((path for path in candidates if path.is_dir()), None)
    if library_dir is None:
        return
    for pattern in ("libnvrtc.so*", "libnvrtc-builtins.so*", "libcudart.so*"):
        for library in sorted(library_dir.glob(pattern)):
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _stats(samples):
    return {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "minimum": min(samples),
        "maximum": max(samples),
    }


def _time_torch(fn, warmups, repeats):
    for _ in range(warmups):
        value = fn()
        del value
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del value
    return samples


def _time_jax(fn, warmups, repeats, jax):
    for _ in range(warmups):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _memory_torch(fn):
    # Compile/tune before resetting the allocator peak. Autotune's buffers
    # belong to setup, not to the operation's measured peak allocation.
    value = fn()
    del value
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    value = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e6
    del value
    return [float(peak)]


def _sqdist(x, y):
    x2 = x.square().sum(-1, keepdim=True)
    y2 = y.square().sum(-1, keepdim=True).transpose(-2, -1)
    return x2 + y2 - 2.0 * (x @ y.transpose(-2, -1))


def _diagnose_result(result, vector=None, args=None):
    """Run outside timed sections; report finite-solve residuals without retuning."""
    record = dict(coupling=diagnostics(result))
    if vector is not None:
        value, info = hessian_vector_product(
            result, vector, damping=args.hvp_damping,
            max_cg_iters=args.hvp_cg_iters, cg_rtol=0, cg_atol=0, return_info=True)
        record.update(cg=asdict(info), hvp_finite=bool(torch.isfinite(value).all()),
                      hvp_norm=float(value.norm()))
    return record


def _flash_operation(method, experiment, x, y, args):
    schedule = "symmetric" if method == "flash-sym" else "alternating"
    is_hvp = experiment.startswith("hvp")
    iterations = args.hvp_sinkhorn_iters if is_hvp else args.iters
    precision = "ieee" if is_hvp else args.precision
    common = dict(
        epsilon=args.epsilon, n_iters=iterations, schedule=schedule,
        precision=precision, block_m=args.block_m, block_n=args.block_n,
    )
    if experiment in ("forward_n", "forward_d", "memory_forward"):
        def forward():
            result = sinkhorn_flash(x, y, **common)
            return (result.a * result.f).sum() + (result.b * result.g).sum()

        forward.diagnostics = lambda: _diagnose_result(sinkhorn_flash(x, y, **common))
        return forward, "CUDA events; fixed Sinkhorn iterations + balanced dual cost"
    if experiment in ("backward_n", "backward_d", "memory_backward"):
        tracked_x = x.detach().requires_grad_(True)

        def forward_backward():
            loss = sinkhorn_cost(tracked_x, y, backend="flash", **common)
            return torch.autograd.grad(loss, tracked_x, create_graph=False)[0]

        forward_backward.diagnostics = lambda: _diagnose_result(sinkhorn_flash(x, y, **common))
        return forward_backward, "CUDA events; forward + analytic source gradient"
    # Potentials are setup state, matching a Hessian-vector product from an
    # already solved OT problem. Timing includes the full HVP transport terms
    # and fixed-K Schur CG, excluding the potential solve.
    result = sinkhorn_flash(x, y, **common)
    generator = torch.Generator(device=x.device).manual_seed(args.seed + 991)
    vector = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    vector /= vector.norm().clamp_min(torch.finfo(vector.dtype).tiny)

    def hvp():
        return hessian_vector_product(
            result, vector, damping=args.hvp_damping,
            max_cg_iters=args.hvp_cg_iters, cg_rtol=0.0, cg_atol=0.0,
        )

    hvp.diagnostics = lambda: _diagnose_result(result, vector, args)
    return hvp, f"CUDA events; cached potentials; fixed {args.hvp_cg_iters}-step CG"


def _geomloss_imports():
    # The paper environment used the legacy module layout. Newer GeomLoss
    # releases expose the same low-level routines without the prefix.
    try:
        from geomloss._legacy.sinkhorn_divergence import (  # type: ignore
            log_weights, sinkhorn_cost as geomloss_cost, sinkhorn_loop)
        from geomloss._legacy.sinkhorn_samples import (  # type: ignore
            lse_genred, softmin_online, softmin_tensorized)
    except ImportError:
        from geomloss.sinkhorn_divergence import (  # type: ignore
            log_weights, sinkhorn_cost as geomloss_cost, sinkhorn_loop)
        from geomloss.sinkhorn_samples import (  # type: ignore
            lse_genred, softmin_online, softmin_tensorized)
    return log_weights, geomloss_cost, sinkhorn_loop, lse_genred, softmin_online, softmin_tensorized


def _geomloss_operation(method, experiment, x, y, args):
    is_hvp = experiment.startswith("hvp")
    if is_hvp and method == "tensorized":
        raise NotImplementedError("the paper HVP panels omit the dense Tensorized baseline")
    if is_hvp:
        # The paper compares identical Schur-CG arithmetic and changes only the
        # transport primitive. Reuse one converged plan, then dispatch every
        # P/P.T/Hadamard reduction through PyKeOps LazyTensors.
        from pykeops.torch import LazyTensor  # noqa: F401
        base = sinkhorn_flash(
            x, y, epsilon=args.epsilon, n_iters=args.hvp_sinkhorn_iters,
            schedule="symmetric", precision="ieee",
            block_m=args.block_m, block_n=args.block_n,
        )
        result = replace(base, backend="keops", precision="ieee")
        generator = torch.Generator(device=x.device).manual_seed(args.seed + 991)
        vector = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        vector /= vector.norm().clamp_min(torch.finfo(vector.dtype).tiny)

        def hvp():
            return hessian_vector_product(
                result, vector, damping=args.hvp_damping,
                max_cg_iters=args.hvp_cg_iters, cg_rtol=0.0, cg_atol=0.0,
            )

        hvp.diagnostics = lambda: _diagnose_result(result, vector, args)
        return hvp, f"CUDA events; PyKeOps transport; fixed {args.hvp_cg_iters}-step Schur CG"
    backward = experiment in ("backward_n", "backward_d", "memory_backward")
    log_weights, geomloss_cost, sinkhorn_loop, lse_genred, softmin_online, softmin_tensorized = \
        _geomloss_imports()
    a = x.new_full((len(x),), 1.0 / len(x))
    b = y.new_full((len(y),), 1.0 / len(y))
    iterations = args.hvp_sinkhorn_iters if is_hvp else args.iters
    eps_list = [args.epsilon] * iterations
    tracked_x = x.detach().requires_grad_(backward)

    if method == "keops":
        from pykeops.torch import generic_logsumexp  # noqa: F401
        softmin = partial(softmin_online, log_conv=lse_genred("SqDist(X,Y)", x.shape[1]))
        a_log, b_log = log_weights(a), log_weights(b)

        def potentials(last_extrapolation):
            return sinkhorn_loop(
                softmin, a_log, b_log, None, None,
                (tracked_x, y.detach()), (y, tracked_x.detach()), eps_list,
                rho=None, debias=False, last_extrapolation=last_extrapolation,
            )
    else:
        softmin = partial(softmin_tensorized)
        a_log, b_log = log_weights(a).unsqueeze(0), log_weights(b).unsqueeze(0)
        fixed_cost = None if backward else _sqdist(tracked_x.unsqueeze(0), y.detach().unsqueeze(0))

        def potentials(last_extrapolation):
            cost = fixed_cost if fixed_cost is not None else _sqdist(tracked_x.unsqueeze(0), y.detach().unsqueeze(0))
            return sinkhorn_loop(
                softmin, a_log, b_log, None, None, cost, cost.transpose(-1, -2),
                eps_list, rho=None, debias=False,
                last_extrapolation=last_extrapolation,
            )

    if not backward:
        def forward():
            _, _, g_ab, f_ba = potentials(False)
            if method == "tensorized":
                g_ab, f_ba = g_ab.squeeze(0), f_ba.squeeze(0)
            return geomloss_cost(
                args.epsilon, None, a, b, None, None, g_ab, f_ba,
                batch=False, debias=False, potentials=False,
            )

        return forward, "CUDA events; low-level GeomLoss fixed-iteration loop + dual cost"

    def loss_function():
        _, _, g_ab, f_ba = potentials(True)
        if method == "tensorized":
            g_ab, f_ba = g_ab.squeeze(0), f_ba.squeeze(0)
        return geomloss_cost(
            args.epsilon, None, a, b, None, None, g_ab, f_ba,
            batch=False, debias=False, potentials=False,
        )

    if is_hvp:
        loss = loss_function()
        gradient = torch.autograd.grad(loss, tracked_x, create_graph=True)[0]
        generator = torch.Generator(device=x.device).manual_seed(args.seed + 991)
        vector = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        vector /= vector.norm().clamp_min(torch.finfo(vector.dtype).tiny)
        scalar = (gradient * vector).sum()

        def hvp():
            return torch.autograd.grad(scalar, tracked_x, retain_graph=True)[0]

        return hvp, "CUDA events; cached differentiable GeomLoss graph; source HVP"

    def forward_backward():
        loss = loss_function()
        return torch.autograd.grad(loss, tracked_x, create_graph=False)[0]

    return forward_backward, "CUDA events; fixed iterations; Danskin source gradient"


def _jax_operation(experiment, x, y, args):
    import jax
    import jax.numpy as jnp
    from jax import config as jax_config

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError(f"JAX sees no GPU: {jax.devices()}")
    is_hvp = experiment.startswith("hvp")
    iterations = args.hvp_sinkhorn_iters if is_hvp else args.iters
    jax_config.update("jax_default_matmul_precision", "highest" if is_hvp or args.precision == "ieee" else "default")
    if is_hvp:
        # Use the same converged coupling as the Torch and KeOps HVP paths.
        # All arrays/potentials are dynamic JIT inputs. Solve and conversion
        # are setup; timing includes the full HVP, including Schur-CG.
        base = sinkhorn_flash(
            x, y, epsilon=args.epsilon, n_iters=iterations,
            schedule="symmetric", precision="ieee",
            block_m=args.block_m, block_n=args.block_n,
        )
        generator = torch.Generator(device=x.device).manual_seed(args.seed + 991)
        direction = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        direction /= direction.norm().clamp_min(torch.finfo(direction.dtype).tiny)

        def to_jax(tensor):
            return jax.device_put(jnp.asarray(tensor.detach().cpu().numpy()))

        operands = tuple(to_jax(tensor) for tensor in
                         (x, y, base.u, base.v, direction))
        if args.jax_hvp_backend == "ott-hessian":
            from .ott_hessian import load_hessian, state_from_shifted
            hessian, provenance = load_hessian(args.ott_hessian_path, cg_rtol=0, cg_atol=0)
            state = state_from_shifted(*operands[:4], epsilon=args.epsilon,
                                       batch_size=args.jax_batch_size)
            operation = jax.jit(hessian, static_argnames=("tau2", "iter"))

            def run_ott_hvp():
                # Upstream adds epsilon*tau2 to the Schur diagonal. Preserve
                # the existing benchmark's absolute Schur damping exactly.
                return operation(operands[4], state,
                                 tau2=args.hvp_damping / args.epsilon,
                                 iter=args.hvp_cg_iters)

            detail = (f"JAX OTT-Hessian {provenance['function']}; "
                      f"source_sha256={provenance['sha256']}; shared potentials; "
                      f"guarded fixed-CG adapter, steps={args.hvp_cg_iters}; "
                      f"tau2={args.hvp_damping / args.epsilon:g}, "
                      f"Schur damping={args.hvp_damping:g}")
            def diagnose_ott():
                actual = torch.as_tensor(jax.device_get(run_ott_hvp()).copy(), device=x.device)
                expected = hessian_vector_product(base, direction, damping=args.hvp_damping,
                                                  max_cg_iters=args.hvp_cg_iters, cg_rtol=0, cg_atol=0)
                error = float((actual-expected).double().norm() / expected.double().norm().clamp_min(1e-30))
                record = _diagnose_result(base, direction, args)
                record.update(source=provenance, relative_l2_vs_flash=error,
                              baseline_finite=bool(torch.isfinite(actual).all()))
                return record
            run_ott_hvp.diagnostics = diagnose_ott
            return run_ott_hvp, jax, detail

        from .jax_hvp import hvp_from_shifted_potentials
        operation = jax.jit(
            hvp_from_shifted_potentials,
            static_argnames=("epsilon", "damping", "cg_iters", "block_rows", "block_keys"),
        )

        def run_hvp():
            return operation(*operands, epsilon=args.epsilon,
                             damping=args.hvp_damping, cg_iters=args.hvp_cg_iters,
                             block_rows=64, block_keys=args.jax_batch_size)

        return (run_hvp, jax,
                f"JAX matrix-free streaming; shared potentials; "
                f"fixed {args.hvp_cg_iters}-step Schur CG")

    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    xj = jax.device_put(jnp.asarray(x.detach().cpu().numpy()))
    yj = jax.device_put(jnp.asarray(y.detach().cpu().numpy()))
    aj = jnp.full((len(x),), 1.0 / len(x), dtype=jnp.float32)
    bj = jnp.full((len(y),), 1.0 / len(y), dtype=jnp.float32)
    solver = sinkhorn.Sinkhorn(
        threshold=-1.0, min_iterations=iterations, max_iterations=iterations,
        use_danskin=True,
    )

    def loss(source):
        geometry = pointcloud.PointCloud(
            source, yj, epsilon=args.epsilon, batch_size=args.jax_batch_size)
        problem = linear_problem.LinearProblem(geometry, a=aj, b=bj)
        return solver(problem).reg_ot_cost

    if experiment in ("forward_n", "forward_d"):
        operation = jax.jit(loss)
        return lambda: operation(xj), jax, "JAX wall clock; OTT online fixed-iteration solve"
    if experiment in ("backward_n", "backward_d"):
        operation = jax.jit(jax.grad(loss))
        return lambda: operation(xj), jax, "JAX wall clock; OTT Danskin forward + source gradient"
    if experiment.startswith("memory"):
        raise NotImplementedError("JAX peak allocation is not exposed through the PyTorch allocator")
    raise ValueError(f"unsupported JAX experiment: {experiment}")


def _tensorized_estimate_mib(n, m, backward):
    # Conservative: cost plus Sinkhorn/autograd temporaries. This guard avoids
    # deliberately launching a known-OOM configuration on shared hardware.
    multiplier = 7 if backward else 5
    return multiplier * n * m * 4 / 2**20


def _write_results(output, rows):
    with (output / "paper_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output / "paper_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _format_n(value):
    return f"{value // 1000}k" if value >= 1000 and value % 1000 == 0 else str(value)


def _draw_panel(axis, experiment, rows, show_title=True):
    title, xlabel, ylabel, _ = PANELS[experiment]
    selected = [row for row in rows if row["experiment"] == experiment and row["status"] == "ok"]
    for method in METHODS:
        points = sorted((row for row in selected if row["method"] == method), key=lambda row: row["axis_value"])
        if not points:
            continue
        label, color, marker = STYLE[method]
        if method.startswith("flash") and "author source" in points[0]["detail"]:
            label += " (author)"
        if experiment.startswith("hvp") and method == "jax":
            label = ("JAX (OTT-Hessian, fixed CG)" if "OTT-Hessian" in points[0]["detail"]
                     else "JAX (matrix-free)")
        if experiment.startswith("memory") and len(points) >= 2:
            log_x = [math.log(row["axis_value"]) for row in points]
            log_y = [math.log(row["mean"]) for row in points if row["mean"] > 0]
            if len(log_y) == len(log_x):
                mean_x, mean_y = statistics.fmean(log_x), statistics.fmean(log_y)
                denominator = sum((value - mean_x) ** 2 for value in log_x)
                if denominator:
                    slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(log_x, log_y)) / denominator
                    label = f"{label}  O(n^{slope:.2f})"
        axis.plot([row["axis_value"] for row in points], [row["mean"] for row in points],
                  marker=marker, color=color, linewidth=1.8, markersize=5, label=label)
    if show_title:
        axis.set_title(title, fontsize=10, fontweight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_yscale("log")
    if experiment.endswith("_d"):
        axis.set_xscale("log", base=2)
    axis.grid(True, which="both", alpha=0.28)
    if not selected:
        axis.text(0.5, 0.5, "No successful measurements", ha="center", va="center", transform=axis.transAxes)
    if not experiment.endswith("_d"):
        ticks = sorted({row["axis_value"] for row in selected})
        if 1 < len(ticks) <= 10:
            axis.set_xticks(ticks, [_format_n(value) for value in ticks])


def _plot(output, rows, experiments):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for experiment in experiments:
        fig, axis = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
        _draw_panel(axis, experiment, rows)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=8)
        fig.savefig(output / PANELS[experiment][3], dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for axis, experiment in zip(axes.flat, PANELS):
        if experiment in experiments:
            _draw_panel(axis, experiment, rows)
        else:
            axis.axis("off")
    handles = [plt.Line2D([], [], color=STYLE[m][1], marker=STYLE[m][2], label=STYLE[m][0]) for m in METHODS]
    if any("author source" in row["detail"] for row in rows):
        for handle, method in zip(handles, METHODS):
            if method.startswith("flash"):
                handle.set_label(STYLE[method][0] + " (author)")
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("FlashSinkhorn: IO-Aware Entropic Optimal Transport", fontweight="bold", y=1.03)
    fig.savefig(output / "00_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _panel_points(experiment, args):
    if experiment in ("forward_n", "backward_n", "memory_forward", "memory_backward"):
        return [(value, value, value, 64 if not experiment.startswith("memory") else 1024)
                for value in sorted(args.n_sizes, reverse=True)]
    if experiment in ("forward_d", "backward_d"):
        return [(value, args.dimension_n, args.dimension_n, value)
                for value in sorted(args.d_sizes, reverse=True)]
    if experiment == "hvp_n":
        return [(value, value, value, args.hvp_fixed_d)
                for value in sorted(args.hvp_n_sizes, reverse=True)]
    return [(value, args.hvp_dimension_n, args.hvp_dimension_n, value)
            for value in sorted(args.hvp_d_sizes, reverse=True)]


def _record_failure(rows, experiment, axis_value, n, m, d, method, status, detail):
    rows.append(dict(experiment=experiment, axis_value=axis_value, n=n, m=m, d=d,
                     method=method, status=status, mean=None, median=None, std=None,
                     minimum=None, maximum=None, unit="MB" if experiment.startswith("memory") else "ms",
                     detail=detail, samples=[]))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=tuple(PANELS), default=list(PANELS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--flash-implementation", choices=("local", "author"), default="local",
                        help="use untouched pinned author kernels/API for Flash methods; other baselines stay unchanged")
    parser.add_argument("--author-path", type=Path,
                        default=Path("outputs/third_party/flash-sinkhorn-author"))
    parser.add_argument("--n-sizes", type=int, nargs="+", default=[5000, 10000, 20000, 30000, 40000, 50000])
    parser.add_argument("--d-sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--hvp-n-sizes", type=int, nargs="+",
                        default=[5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000])
    parser.add_argument("--hvp-d-sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--dimension-n", type=int, default=20000)
    parser.add_argument("--hvp-dimension-n", type=int, default=10000)
    parser.add_argument("--hvp-fixed-d", type=int, default=64)
    parser.add_argument("--hvp-baseline-max-n", type=int, default=10000,
                        help="JAX plotting limit for the HVP n sweep")
    parser.add_argument("--hvp-keops-max-n", type=int, default=50000,
                        help="KeOps plotting limit for the HVP n sweep")
    parser.add_argument("--hvp-baseline-max-d", type=int, default=128,
                        help="paper plotting range for KeOps/JAX in the HVP d sweep")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--hvp-sinkhorn-iters", type=int, default=100)
    parser.add_argument("--hvp-cg-iters", type=int, default=50)
    parser.add_argument("--hvp-damping", type=float, default=1e-5)
    parser.add_argument("--precision", choices=("ieee", "tf32x3", "tf32"), default="tf32")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--forward-repeats", type=int, default=50)
    parser.add_argument("--backward-repeats", type=int, default=30)
    parser.add_argument("--hvp-repeats", type=int, default=20)
    parser.add_argument("--memory-fraction", type=float, default=0.80)
    parser.add_argument("--max-tensorized-mib", type=float, default=12000)
    parser.add_argument("--jax-batch-size", type=int, default=256)
    parser.add_argument("--jax-hvp-backend", choices=("ott-hessian", "matrix-free"),
                        default="ott-hessian")
    parser.add_argument("--ott-hessian-path", type=Path,
                        default=Path("outputs/third_party/OTT-Hessian"))
    parser.add_argument("--autotune", action="store_true",
                        help="tune bounded tile/stage candidates during warmup on the allocated GPU")
    parser.add_argument("--diagnostics", action="store_true",
                        help="write coupling residuals and HVP CG/parity diagnostics outside timing")
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    all_sizes = args.n_sizes + args.d_sizes + args.hvp_n_sizes + args.hvp_d_sizes
    if min(all_sizes + [args.dimension_n, args.hvp_dimension_n, args.hvp_fixed_d,
                        args.hvp_baseline_max_n, args.hvp_keops_max_n,
                        args.hvp_baseline_max_d]) < 1:
        parser.error("sizes must be positive")
    if max(args.d_sizes + args.hvp_d_sizes + [args.hvp_fixed_d]) > 1024:
        parser.error("this FlashSinkhorn implementation supports d <= 1024")
    if min(args.warmups, args.forward_repeats, args.backward_repeats, args.hvp_repeats) < 1:
        parser.error("warmups and repetitions must be positive")
    if args.flash_implementation == "author" and args.precision == "tf32x3":
        parser.error("the pinned author API supports ieee/tf32, not tf32x3")
    return args


def main():
    args = _parse_args()
    os.environ["FLASHOPW_AUTOTUNE"] = "1" if args.autotune else "0"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    _preload_cuda_libraries()
    device = configure("cuda", threads=2, memory_fraction=args.memory_fraction)
    author_provenance = None
    if args.flash_implementation == "author":
        from .author_reference import load_author
        _, author_provenance = load_author(args.author_path)
    prefix = "author_paper" if author_provenance else "paper"
    output = args.output or Path("outputs") / datetime.now(timezone.utc).strftime(prefix + "_%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=False)
    environment = metadata(device)
    environment["args"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    environment["author_source"] = author_provenance
    environment["protocol"] = {
        "data": "independent uniform [0,1]^d point clouds; uniform marginals",
        "cost": "full squared Euclidean",
        "timing": "CUDA events for PyTorch/KeOps; synchronized wall clock for JAX",
        "order": "large to small",
        "backward": "source x only",
        "hvp": "strict FP32; damping=1e-5; fixed 50-step CG by default",
        "memory": "total PyTorch peak allocated, including live inputs",
        "precision": "TF32 forward/backward; strict FP32 HVP by default",
        "panel_methods": PANEL_METHODS,
        "hvp_baseline_range": {
            "jax_max_n": args.hvp_baseline_max_n,
            "keops_max_n": args.hvp_keops_max_n,
            "max_d": args.hvp_baseline_max_d,
        },
        "jax_hvp": args.jax_hvp_backend,
        "jax_hvp_controls": ("upstream HessianA/OTT geometry; shared coupling; guarded fixed-step CG adapter; tau2=Schur damping/epsilon"
                             if args.jax_hvp_backend == "ott-hessian" else "custom fixed-step Schur-CG"),
        "tensorized_forward": "dense squared-distance matrix precomputed and cached outside timing, matching the official benchmark",
        "tiles": "bounded GPU autotuning in warmup" if args.autotune else "fixed RTX-safe upper bounds",
    }
    if author_provenance:
        environment["protocol"]["hvp"] = "strict FP32; shared coupling; original author CG with configured iteration cap"
        environment["protocol"]["tiles"] = "author native configuration/autotuning; local tile controls apply to shared HVP setup"
        environment["protocol"]["author_flash"] = {
            "forward_backward": "author SamplesLoss; fixed n_iters parameter; original solver initialization retained; no extrapolation",
            "hvp": "author hvp_x_sqeuclid_from_potentials and unmodified CG; shared local IEEE coupling solved outside timing",
            "damping": "author tau2=absolute Schur damping/epsilon",
            "cg": "max_cg_iter from args; rtol=atol=0; author's exact-zero/breakdown exits retained",
            "tiles": "author autotune/configuration, not the local bounded candidate set",
            "baselines": "KeOps, Tensorized and JAX are the existing harness paths, including guarded JAX CG",
        }
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    rows = []
    validation = []

    for experiment in args.experiments:
        print(f"\n=== {PANELS[experiment][0]} ===", flush=True)
        for axis_value, n, m, d in _panel_points(experiment, args):
            generator = torch.Generator(device=device).manual_seed(args.seed + n * 3 + d * 1009)
            x = torch.rand((n, d), device=device, dtype=torch.float32, generator=generator)
            y = torch.rand((m, d), device=device, dtype=torch.float32, generator=generator)
            for method in args.methods:
                operation = None
                applicable = PANEL_METHODS.get(experiment, METHODS)
                if method not in applicable:
                    _record_failure(rows, experiment, axis_value, n, m, d, method,
                                    "not_applicable", "method omitted from this panel in the paper")
                    _write_results(output, rows)
                    continue
                if (experiment == "hvp_n" and method == "keops" and
                        n > args.hvp_keops_max_n):
                    _record_failure(rows, experiment, axis_value, n, m, d, method,
                                    "outside_paper_range", "KeOps omitted beyond configured HVP n range")
                    _write_results(output, rows)
                    continue
                if (experiment == "hvp_n" and method == "jax" and
                        n > args.hvp_baseline_max_n):
                    _record_failure(rows, experiment, axis_value, n, m, d, method,
                                    "outside_paper_range", "baseline omitted beyond the paper's HVP n range")
                    _write_results(output, rows)
                    continue
                if (experiment == "hvp_d" and method in ("keops", "jax") and
                        d > args.hvp_baseline_max_d):
                    _record_failure(rows, experiment, axis_value, n, m, d, method,
                                    "outside_paper_range", "baseline omitted beyond the paper's HVP d range")
                    _write_results(output, rows)
                    continue
                backward = experiment in ("backward_n", "backward_d", "memory_backward")
                if method == "tensorized":
                    estimate = _tensorized_estimate_mib(n, m, backward)
                    if estimate > args.max_tensorized_mib:
                        _record_failure(rows, experiment, axis_value, n, m, d, method,
                                        "skipped_memory", f"conservative estimate {estimate:.1f} MiB exceeds limit")
                        print(f"{axis_value:6d} {method:12s} SKIP memory estimate={estimate:.1f} MB", flush=True)
                        _write_results(output, rows)
                        continue
                try:
                    strict_fp32 = experiment.startswith("hvp") or args.precision == "ieee"
                    torch.backends.cuda.matmul.allow_tf32 = not strict_fp32
                    torch.set_float32_matmul_precision("highest" if strict_fp32 else "high")
                    if method.startswith("flash"):
                        if args.flash_implementation == "author":
                            from .author_reference import author_operation
                            operation, detail = author_operation(method, experiment, x, y, args)
                        else:
                            operation, detail = _flash_operation(method, experiment, x, y, args)
                        timing_kind = "torch"
                    elif method in ("keops", "tensorized"):
                        operation, detail = _geomloss_operation(method, experiment, x, y, args)
                        timing_kind = "torch"
                    else:
                        operation, jax, detail = _jax_operation(experiment, x, y, args)
                        timing_kind = "jax"

                    if experiment.startswith("memory"):
                        samples = _memory_torch(operation)
                        unit = "MB"
                    else:
                        repeats = (args.hvp_repeats if experiment.startswith("hvp") else
                                   args.backward_repeats if experiment.startswith("backward") else
                                   args.forward_repeats)
                        samples = (_time_torch(operation, args.warmups, repeats) if timing_kind == "torch"
                                   else _time_jax(operation, args.warmups, repeats, jax))
                        unit = "ms"
                    # A runtime is not a valid curve point if its output is
                    # nonfinite. Probe outside timing without changing counts.
                    probe = operation()
                    if timing_kind == "torch":
                        finite = bool(torch.isfinite(probe).all())
                    else:
                        import jax.numpy as jnp
                        finite = bool(jax.device_get(jnp.isfinite(probe).all()))
                    del probe
                    row = dict(experiment=experiment, axis_value=axis_value, n=n, m=m, d=d,
                               method=method, status="ok" if finite else "nonfinite_output", unit=unit, detail=detail,
                               samples=samples, **_stats(samples))
                    rows.append(row)
                    print(f"{axis_value:6d} {method:12s} {row['mean']:10.3f} {unit}", flush=True)
                    if not finite:
                        print(f"{axis_value:6d} {method:12s} NONFINITE output; excluded from plots", flush=True)
                    if args.diagnostics and hasattr(operation, "diagnostics"):
                        try:
                            diagnostic = operation.diagnostics()
                        except Exception as exc:
                            diagnostic = dict(error=f"{type(exc).__name__}: {exc}")
                        validation.append(dict(experiment=experiment, n=n, m=m, d=d,
                                               method=method, **diagnostic))
                except torch.cuda.OutOfMemoryError as exc:
                    _record_failure(rows, experiment, axis_value, n, m, d, method, "oom", str(exc))
                    print(f"{axis_value:6d} {method:12s} OOM", flush=True)
                except Exception as exc:
                    _record_failure(rows, experiment, axis_value, n, m, d, method, "failed",
                                    f"{type(exc).__name__}: {exc}")
                    print(f"{axis_value:6d} {method:12s} FAILED: {type(exc).__name__}: {exc}", flush=True)
                finally:
                    _write_results(output, rows)
                    from flashopw.kernel_tuning import tuning_records
                    (output / "autotuning.json").write_text(
                        json.dumps(tuning_records(), indent=2), encoding="utf-8")
                    (output / "diagnostics.json").write_text(
                        json.dumps(validation, indent=2), encoding="utf-8")
                    operation = None
                    if method == "jax":
                        try:
                            import jax
                            jax.clear_caches()
                        except ImportError:
                            pass
                    gc.collect()
                    torch.cuda.empty_cache()
            del x, y
            gc.collect()
            torch.cuda.empty_cache()
            _plot(output, rows, args.experiments)
        _plot(output, rows, args.experiments)

    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
