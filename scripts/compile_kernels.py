"""Offline compilation only: Linux Triton required, no torch/GPU/driver needed.

This checks code generation, not GPU execution, numerical accuracy or speed.
Example: python scripts/compile_kernels.py --arch 120
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", type=int, default=120)
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 7, 64, 65, 129, 256, 512])
    parser.add_argument("--output", type=Path, default=Path("outputs/offline-compile.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("TRITON_CACHE_DIR", str(root / ".triton-cache"))
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget

    # Load just the kernels; avoid importing the torch-based package __init__.
    spec = importlib.util.spec_from_file_location("offline_kernels", root / "flashopw/triton_kernels.py")
    kernels = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kernels
    spec.loader.exec_module(kernels)
    records = []
    for d in args.dims:
        configurations = sorted({kernels.launch_config(d, bm, bn) for bm, bn in ((16, 32), (32, 64))})
        for bm, bn, stages in configurations:
            for precision in ("ieee", "tf32x3"):
                base = dict(N=37, M=79, D=d, SCALE=10.0, PRECISION=precision,
                            BM=bm, BN=bn, BD=kernels.feature_block(d))
                variants = [
                    (kernels._update_kernel, ("Q", "K", "OLD", "BIAS", "LOGW", "OUT"), dict(SYMMETRIC=False)),
                    (kernels._update_kernel, ("Q", "K", "OLD", "BIAS", "LOGW", "OUT"), dict(SYMMETRIC=True)),
                    (kernels._apply_kernel, ("Q", "K", "U", "V", "VALUES", "OUT"), dict(P=35, BP=32)),
                ]
                for fn, pointers, extra in variants:
                    source = ASTSource(fn, signature={name: "*fp32" for name in pointers}, constexprs={**base, **extra})
                    compiled = triton.compile(source, target=GPUTarget("cuda", args.arch, 32),
                        options=dict(num_warps=4, num_stages=stages, enable_fp_fusion=False))
                    assert len(compiled.asm["cubin"]) > 0
                    assert compiled.metadata.shared <= 65536, "Shared-memory budget exceeded"
                    record = dict(kernel=fn.__name__, dim=d, block_m=bm, block_n=bn,
                                  precision=precision, stages=stages, shared_bytes=compiled.metadata.shared,
                                  binary_bytes=len(compiled.asm["cubin"]), **extra)
                    records.append(record)
                    print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(triton=triton.__version__, arch=args.arch,
        validation="offline compilation only; NOT executed on a GPU", kernels=records), indent=2), encoding="utf-8")
    print(f"PASS: compiled {len(records)} variants for sm_{args.arch}")


if __name__ == "__main__":
    main()
