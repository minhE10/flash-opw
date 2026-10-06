"""Compile autotune candidates offline; report rejected resource configurations.

Linux Triton only. This checks compilation/resources, not execution or speed.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", type=int, default=120)
    parser.add_argument("--dims", nargs="+", type=int, default=[64, 1024])
    parser.add_argument("--precisions", nargs="+", choices=("ieee", "tf32", "tf32x3"), default=["ieee", "tf32"])
    parser.add_argument("--output", type=Path, default=Path("outputs/offline-tuning.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("TRITON_CACHE_DIR", str(root / ".triton-cache"))
    import triton
    from triton.compiler import ASTSource
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources
    from triton.backends.compiler import GPUTarget
    kernels = load("tuning_kernels", root / "flashsinkhorn/triton_kernels.py")
    tuning = load("offline_tuning", root / "flashsinkhorn/kernel_tuning.py")
    records = []
    for d in args.dims:
        for precision in args.precisions:
            bm, bn, stages = kernels.launch_config(d, 64, 128)
            hm, hn, hs = kernels.hadamard_launch_config(d, d, 64, 128)
            bp = kernels.value_block(d+1, d, precision)
            variants = [
                (kernels._update_kernel, ("Q", "K", "OLD", "BIAS", "LOGW", "OUT"),
                 dict(SYMMETRIC=False), (bm, bn, stages, 4), {}),
                (kernels._symmetric_update_kernel, ("Q", "K", "U", "V", "LOGA", "LOGB", "UOUT", "VOUT"),
                 {}, (bm, bn, stages, 4), {}),
                (kernels._apply_vector_kernel, ("Q", "K", "U", "V", "VALUES", "OUT"),
                 {}, (bm, bn, stages, 4), {}),
                (kernels._apply_kernel, ("Q", "K", "U", "V", "VALUES", "OUT"),
                 dict(P=d+1, BP=bp), (bm, bn, stages, 4), {}),
                (kernels._gradient_kernel, ("Q", "K", "U", "V", "OUT"),
                 dict(GAMMA=2.0, BG=32), (16, bn, 1, 4), dict(gradient=True)),
                (kernels._hadamard_apply_kernel, ("Q", "K", "U", "V", "LEFT", "RIGHT", "VALUES", "OUT"),
                 dict(R=d, P=d+1, BR=kernels.feature_block(d), BP=min(64, bp) if d >= 256 else bp),
                 (hm, hn, hs, 4), dict(hadamard=True)),
            ]
            for fn, pointers, extra, baseline, flags in variants:
                for config in tuning.candidate_configs(baseline, 64, 128, **flags):
                    cm, cn, cs, cw = config
                    constants = dict(N=37, M=79, D=d, SCALE=20.0, PRECISION=precision,
                                     BM=cm, BN=cn, BD=kernels.feature_block(d), **extra)
                    record = dict(kernel=fn.__name__, dim=d, precision=precision, config=list(config))
                    try:
                        source = ASTSource(fn, signature={name: "*fp32" for name in pointers}, constexprs=constants)
                        compiled = triton.compile(source, target=GPUTarget("cuda", args.arch, 32),
                                                  options=dict(num_warps=cw, num_stages=cs, enable_fp_fusion=False))
                        record.update(shared_bytes=compiled.metadata.shared,
                                      binary_bytes=len(compiled.asm["cubin"]),
                                      status="accepted" if compiled.metadata.shared <= 65536 else "shared_memory_limit")
                    except (OutOfResources, CompilationError) as exc:
                        record.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
                    if config == baseline and record["status"] != "accepted":
                        raise RuntimeError(f"Baseline candidate failed: {record}")
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(dict(arch=args.arch, triton=triton.__version__,
                        validation="offline compilation only, NOT GPU execution", candidates=records), indent=2), encoding="utf-8")
    accepted = sum(r["status"] == "accepted" for r in records)
    print(f"PASS: {accepted}/{len(records)} candidates fit 64 KiB; others will not launch")


if __name__ == "__main__":
    main()
