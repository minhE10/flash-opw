"""Small on-server compilation/correctness check, including transport kernels."""

import argparse
import json
import torch

from flashopw import sinkhorn_dense, sinkhorn_flash, apply_plan, diagnostics, materialize_plan
from .datasets import make_dataset
from .runtime import configure, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    args = parser.parse_args()
    device = configure("cuda", memory_fraction=args.memory_fraction)
    print(json.dumps(metadata(device), indent=2))
    x, y, a, b = make_dataset("gaussian", 37, 79, d=5, weighted=True, device=device)
    expected = sinkhorn_dense(x.double(), y.double(), a=a.double() / a.double().sum(),
                              b=b.double() / b.double().sum(), n_iters=100)
    expected_plan = materialize_plan(expected)
    for precision in ("ieee", "tf32", "tf32x3"):
        print(f"Checking precision={precision}...", flush=True)
        actual = sinkhorn_flash(x, y, a=a, b=b, n_iters=100, precision=precision)
        plan = materialize_plan(actual)
        rtol, atol = (5e-3, 5e-6) if precision == "tf32" else (5e-4, 2e-7)
        difference = (plan.double() - expected_plan).abs()
        print(f"  plan max_abs={float(difference.max()):.3e} l1={float(difference.sum()):.3e}",
              flush=True)
        torch.testing.assert_close(
            plan.double(), expected_plan, rtol=rtol, atol=atol,
            msg=f"precision={precision}",
        )
        transport_atol = 1e-5 if precision == "tf32" else 2e-7
        torch.testing.assert_close(apply_plan(actual, y), plan @ y,
                                   rtol=rtol, atol=transport_atol)
        torch.testing.assert_close(apply_plan(actual, x, transpose=True), plan.T @ x,
                                   rtol=rtol, atol=transport_atol)
        stats = diagnostics(actual)
        residual_tol = 1e-3 if precision == "tf32" else 1e-4
        assert max(stats["row_l1"], stats["col_l1"]) < residual_tol, \
            f"precision={precision}, tolerance={residual_tol}, diagnostics={stats}"
        print(precision, json.dumps(stats, indent=2))
    print("PASS: Triton updates, P@V and P.T@V agree with the dense reference.")


if __name__ == "__main__":
    main()
