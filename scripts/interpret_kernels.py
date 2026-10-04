"""Run the actual Triton kernel bodies on NumPy buffers with Triton's interpreter.

Linux + triton + numpy; no torch or GPU required. This is NOT CUDA execution:
it checks indexing, masks and recurrence logic, not hardware rounding or speed.
Uses the interpreter's internal TensorHandle API (tested version in validation.md).
"""

import importlib.util
import os
from pathlib import Path
import sys

os.environ["TRITON_INTERPRET"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import triton
import triton.language as tl
from triton.runtime.interpreter import TensorHandle


def pointer(array):
    # NumPy owns the storage throughout each synchronous interpreted launch.
    assert array.dtype == np.float32 and array.flags.c_contiguous
    ty = tl.pointer_type(tl.float32)
    return tl.tensor(TensorHandle(np.array([array.ctypes.data], dtype=np.uint64), ty), ty)


def lse(values, axis):
    maximum = values.max(axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


def main():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("interpreted_kernels", root / "flashopw/triton_kernels.py")
    kernels = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kernels
    spec.loader.exec_module(kernels)
    rng = np.random.default_rng(42)
    count, max_error = 0, 0.0
    for n, m, d in ((1, 1, 1), (1, 67, 2), (63, 1, 7), (37, 79, 7),
                    (33, 65, 64), (17, 35, 129), (5, 9, 512), (3, 5, 1024)):
        for symmetric in (False, True):
            for eps in (0.03, 0.2):
                q = (rng.normal(size=(n, d)) * 0.15 / np.sqrt(d)).astype(np.float32)
                k = (rng.normal(size=(m, d)) * 0.15 / np.sqrt(d) + 0.1).astype(np.float32)
                a, b = rng.random(n) + 0.2, rng.random(m) + 0.2
                a, b = (a/a.sum()).astype(np.float32), (b/b.sum()).astype(np.float32)
                loga, logb = np.log(a), np.log(b)
                u, v = loga - (q*q).sum(1)/eps, logb - (k*k).sum(1)/eps
                unew, vnew = np.empty_like(u), np.empty_like(v)
                f, g = np.zeros(n), np.zeros(m)
                cost = ((q.astype(np.float64)[:, None] - k.astype(np.float64)[None, :])**2).sum(2)
                bm, bn, bd = 16, 32, kernels.feature_block(d)

                def update(x, y, old, bias, logw, out):
                    kernels._update_kernel[(triton.cdiv(len(x), bm),)](
                        *map(pointer, (x, y, old, bias, logw, out)), len(x), len(y), d,
                        2/eps, False, "ieee", bm, bn, bd)

                def symmetric_update():
                    grid = (triton.cdiv(n, bm) + triton.cdiv(m, bn),)
                    kernels._symmetric_update_kernel[grid](
                        *map(pointer, (q, k, u, v, loga, logb, unew, vnew)),
                        n, m, d, 2/eps, "ieee", bm, bn, bd,
                    )

                for _ in range(11):
                    if symmetric:
                        symmetric_update()
                    else:
                        update(q, k, u, v, loga, unew)
                        update(k, q, v, unew, logb, vnew)
                    u, unew, v, vnew = unew, u, vnew, v
                    nf = -eps * lse((g[None, :] - cost)/eps + np.log(b.astype(np.float64)), 1)
                    fg = f if symmetric else nf
                    ng = -eps * lse((fg[:, None] - cost)/eps + np.log(a.astype(np.float64))[:, None], 0)
                    f, g = ((f+nf)/2, (g+ng)/2) if symmetric else (nf, ng)
                actual = np.exp((2/eps)*(q@k.T) + u[:, None] + v[None, :])
                expected = a[:, None]*b[None, :]*np.exp((f[:, None]+g[None, :]-cost)/eps)
                np.testing.assert_allclose(actual, expected, rtol=8e-4, atol=3e-6)
                max_error = max(max_error, float(np.abs(actual-expected).max()))
                for transpose in (False, True):
                    x, y, left, right = (k, q, v, u) if transpose else (q, k, u, v)
                    values = rng.normal(size=(len(y), 35)).astype(np.float32)
                    out = np.empty((len(x), 35), dtype=np.float32)
                    kernels._apply_kernel[(triton.cdiv(len(x), bm), 2)](
                        *map(pointer, (x, y, left, right, values, out)), len(x), len(y), d, 35,
                        2/eps, "ieee", bm, bn, bd, 32)
                    np.testing.assert_allclose(out, (expected.T if transpose else expected)@values,
                                               rtol=8e-4, atol=3e-6)
                left = rng.normal(size=(n, d)).astype(np.float32)
                right = rng.normal(size=(m, d)).astype(np.float32)
                values = rng.normal(size=(m, 35)).astype(np.float32)
                out = np.empty((n, 35), dtype=np.float32)
                kernels._hadamard_apply_kernel[(triton.cdiv(n, bm), 2)](
                    *map(pointer, (q, k, u, v, left, right, values, out)),
                    n, m, d, d, 35, 2/eps, "ieee", bm, bn, bd,
                    kernels.feature_block(d), 32)
                np.testing.assert_allclose(
                    out, (expected * (left@right.T))@values,
                    rtol=1e-3, atol=5e-6,
                )
                count += 1
    print(f"PASS: {count} solver, {count*2} transport/adjoint and {count} Hadamard cases; "
          f"max plan error={max_error:.3g}")
    print(f"Triton {triton.__version__}; CPU interpreter, NOT GPU execution.")


if __name__ == "__main__":
    main()
