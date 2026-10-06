"""Bounded, opt-in GPU tuning. Compilation/tuning belongs in warmup, not timing.

No imports of CUDA or Triton at module import time. Each candidate uses the
same precision and mathematical kernel; compiler metadata gates shared memory.
"""

import os

_CACHE = {}
_RECORDS = []


def tuning_enabled():
    return os.environ.get("FLASHOPW_AUTOTUNE", "0") == "1"


def tuning_records():
    return list(_RECORDS)


def candidate_configs(baseline, max_m, max_n, *, gradient=False, hadamard=False):
    """Small search space within the caller's requested tile upper bounds."""
    candidates = [baseline]
    for bm, bn, stages, warps in (
        (16, 32, 1, 4), (16, 64, 2, 4), (32, 32, 2, 4),
        (32, 64, 2, 4), (32, 64, 3, 4), (32, 128, 2, 8),
        (64, 64, 2, 8), (64, 128, 3, 8),
    ):
        if bm > max_m or bn > max_n:
            continue
        if gradient and bm > 32:
            continue
        if hadamard and (bm > 32 or bn > 64):
            continue
        config = (bm, bn, stages, warps)
        if config not in candidates:
            candidates.append(config)
    return candidates


def launch(kernel, pointers, constants, grid, baseline, max_m, max_n,
           *, gradient=False, hadamard=False):
    """Compile and measure once per device/kernel/shape/precision signature."""
    def kwargs(config):
        bm, bn, stages, warps = config
        return dict(constants, BM=bm, BN=bn, num_stages=stages,
                    num_warps=warps, enable_fp_fusion=False)

    if not tuning_enabled():
        return kernel[grid(baseline)](*pointers, **kwargs(baseline))

    import torch
    import triton.testing
    from triton.runtime.errors import OutOfResources
    from triton.compiler.errors import CompilationError

    device = pointers[0].device
    props = torch.cuda.get_device_properties(device)
    identity = (device.index, props.name, props.major, props.minor)
    key = (identity, kernel.__name__, tuple(sorted(constants.items())), max_m, max_n)
    if key not in _CACHE:
        # RTX 5080 has much less shared memory than A100. Use the lesser of
        # its reported per-block opt-in budget and our validated 64 KiB limit.
        budget = min(65536, getattr(props, "shared_memory_per_block_optin",
                                    getattr(props, "shared_memory_per_block", 65536)))
        trials = []
        for config in candidate_configs(baseline, max_m, max_n,
                                        gradient=gradient, hadamard=hadamard):
            trial = dict(config=list(config))
            try:
                compiled = kernel.warmup(*pointers, **kwargs(config), grid=grid(config))
                trial.update(shared_bytes=compiled.metadata.shared,
                             registers=getattr(compiled, "n_regs", None),
                             spills=getattr(compiled, "n_spills", None))
                if compiled.metadata.shared > budget:
                    trial["status"] = "shared_memory_limit"
                else:
                    # No mutation of inputs: every candidate overwrites OUT.
                    trial["ms"] = float(triton.testing.do_bench(
                        lambda: kernel[grid(config)](*pointers, **kwargs(config)),
                        warmup=5, rep=20))
                    trial.update(registers=getattr(compiled, "n_regs", None),
                                 spills=getattr(compiled, "n_spills", None))
                    trial["status"] = "ok"
            except (OutOfResources, CompilationError, torch.cuda.OutOfMemoryError) as exc:
                # Resource failures are recorded; a baseline compiler failure
                # must remain visible rather than being silently suppressed.
                trial.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
                if config == baseline:
                    raise
            trials.append(trial)
        usable = [trial for trial in trials if trial["status"] == "ok"]
        if not usable:
            raise RuntimeError(f"No {kernel.__name__} candidate fits {budget} shared bytes")
        chosen = min(usable, key=lambda trial: trial["ms"])
        _CACHE[key] = tuple(chosen["config"])
        _RECORDS.append(dict(kernel=kernel.__name__, device=list(identity),
                             constants=constants, shared_budget_bytes=budget,
                             selected=chosen, candidates=trials))
    config = _CACHE[key]
    return kernel[grid(config)](*pointers, **kwargs(config))
