"""Same-input transport ablations, complete HVP timings and optional CUDA traces.

Generic uses the retained matrix transport/gradient route. Specialized changes
only those primitives. Tuned also searches tile/stage choices during warmup.
This compares implementations in this revision, not separate git revisions.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch

from flashsinkhorn import apply_plan, diagnostics, hessian_vector_product, sinkhorn_cost, sinkhorn_flash
from flashsinkhorn.transport import source_gradient
from flashsinkhorn.kernel_tuning import tuning_records
from .paper_benchmarks import _stats, _time_torch, _write_results
from .runtime import configure, metadata


@contextmanager
def kernel_mode(mode):
    settings = dict(FLASHOPW_VECTOR_KERNEL="0" if mode == "generic" else "1",
                    FLASHOPW_GRADIENT_KERNEL="0" if mode == "generic" else "1",
                    FLASHOPW_AUTOTUNE="1" if mode == "tuned" else "0")
    previous = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["10000:64", "20000:1024", "50000:64"])
    parser.add_argument("--operations", nargs="+", choices=("vector", "gradient", "forward_backward", "hvp"),
                        default=["vector", "gradient", "forward_backward", "hvp"])
    parser.add_argument("--hvp-max-d", type=int, default=128)
    parser.add_argument("--modes", nargs="+", choices=("generic", "specialized", "tuned"),
                        default=["generic", "specialized", "tuned"])
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = []
    for value in args.cases:
        try:
            n, d = map(int, value.split(":"))
        except ValueError:
            parser.error("cases must be n:d, for example 10000:64")
        if n < 1 or not 1 <= d <= 1024:
            parser.error("n must be positive and 1 <= d <= 1024")
        cases.append((n, d))
    if min(args.warmups, args.repeats) < 1:
        parser.error("warmups and repeats must be positive")
    device = configure("cuda", memory_fraction=0.8)
    output = args.output or Path("outputs") / datetime.now(timezone.utc).strftime("kernel_%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=False)
    environment = metadata(device)
    environment.update(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                       protocol="uniform points/weights; eps=.1; F+B=10; HVP setup=100, fixed CG=50, Schur damping=1e-5",
                       comparison="generic vs specialized vs tuned transport in one revision; same tensors")
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    rows, validation = [], []
    success = True
    for n, d in cases:
        generator = torch.Generator(device=device).manual_seed(args.seed+n*3+d*1009)
        x = torch.rand((n, d), device=device, generator=generator)
        y = torch.rand((n, d), device=device, generator=generator)
        vector = torch.randn((n,), device=device, generator=generator)
        direction = torch.randn((n, d), device=device, generator=generator)
        direction /= direction.norm()
        with kernel_mode("specialized"):
            # Setup once, IEEE coupling shared across modes in all HVPs.
            result = sinkhorn_flash(x, y, epsilon=0.1, n_iters=100, schedule="symmetric", precision="ieee")
            gradient_result = sinkhorn_flash(x, y, epsilon=0.1, n_iters=10,
                                             schedule="symmetric", precision="tf32")
        tracked_x = x.detach().requires_grad_(True)

        def forward_backward():
            loss = sinkhorn_cost(tracked_x, y, epsilon=0.1, n_iters=10,
                                 schedule="symmetric", precision="tf32",
                                 block_m=args.block_m, block_n=args.block_n)
            return torch.autograd.grad(loss, tracked_x)[0]

        operations = dict(vector=lambda: apply_plan(result, vector, precision="ieee"),
                          gradient=lambda: source_gradient(gradient_result),
                          forward_backward=forward_backward,
                          hvp=lambda: hessian_vector_product(result, direction, damping=1e-5,
                                                              max_cg_iters=50, cg_rtol=0, cg_atol=0))
        # Requested upper bounds are retained so tuned streaming primitives
        # can consider larger tiles without changing any potentials.
        result.block_m = gradient_result.block_m = args.block_m
        result.block_n = gradient_result.block_n = args.block_n
        for name in args.operations:
            if name == "hvp" and d > args.hvp_max_d:
                continue
            operation = operations[name]
            strict = name in ("vector", "hvp")
            torch.backends.cuda.matmul.allow_tf32 = not strict
            torch.set_float32_matmul_precision("highest" if strict else "high")
            with kernel_mode("generic"):
                reference = operation().detach()
            for mode in args.modes:
                with kernel_mode(mode):
                    actual = operation()
                    error = float((actual-reference).double().norm() / reference.double().norm().clamp_min(1e-30))
                    limit = 5e-3 if strict else 1e-2
                    passed = bool(torch.isfinite(actual).all()) and error <= limit
                    success &= passed
                    validation.append(dict(n=n, d=d, operation=name, mode=mode,
                                           relative_l2=error, limit=limit, passed=passed))
                    del actual
                    samples = _time_torch(operation, args.warmups, args.repeats)
                    rows.append(dict(experiment=name, axis_value=n, n=n, m=n, d=d,
                                     method=mode, status="ok" if passed else "failed_parity", unit="ms",
                                     detail="CUDA events; shared tensors; compile/tune excluded", samples=samples,
                                     **_stats(samples)))
                    print(f"n={n} d={d} {name} {mode}: {rows[-1]['mean']:.3f} ms; relL2={error:.3g}", flush=True)
                    if args.profile and (n, d) == cases[0]:
                        # Profile after warmup, separately from timing samples.
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA],
                                                    record_shapes=True) as prof:
                            with torch.profiler.record_function(f"{name}/{mode}"):
                                profiled = operation()
                                torch.cuda.synchronize()
                        del profiled
                        prof.export_chrome_trace(str(output / f"trace_{name}_{mode}.json"))
                        (output / f"profile_{name}_{mode}.txt").write_text(
                            prof.key_averages().table(sort_by="self_device_time_total", row_limit=30), encoding="utf-8")
                    _write_results(output, rows)
                    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
                    (output / "autotuning.json").write_text(json.dumps(tuning_records(), indent=2), encoding="utf-8")
            del reference
        with kernel_mode("specialized"):
            record = dict(n=n, d=d, diagnostics=diagnostics(result))
            if "hvp" in args.operations and d <= args.hvp_max_d:
                _, info = hessian_vector_product(result, direction, damping=1e-5, max_cg_iters=50,
                                                 cg_rtol=0, cg_atol=0, return_info=True)
                record["cg"] = asdict(info)
            validation.append(record)
        (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(f"Results: {output.resolve()}")
    if not success:
        raise SystemExit("Numerical comparison failed; do not interpret these timings as validated speedups")


if __name__ == "__main__":
    main()
