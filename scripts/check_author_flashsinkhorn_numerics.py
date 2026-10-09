"""Small independent GPU/CPU-FP64 checks, separate from the author's tests."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import torch

from author_flashsinkhorn_fp64 import direct_hvp, plan, solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite numerical results")
    implementation = args.implementation.resolve()
    sys.path.insert(0, str(implementation / "torch-ext"))
    import flash_sinkhorn
    from flash_sinkhorn.hvp import hvp_x_sqeuclid_from_potentials
    from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
        sinkhorn_flashstyle_alternating, sinkhorn_flashstyle_symmetric,
    )
    from flash_sinkhorn.kernels.apply_flash import apply_plan_mat_flashstyle
    assert implementation in Path(flash_sinkhorn.__file__).resolve().parents
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "One allocated CUDA GPU required"
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.45)
    # Generate on CPU so inputs do not depend on CUDA RNG/device architecture.
    generator = torch.Generator(device="cpu").manual_seed(20261009)
    report = {"status": "running", "package": str(flash_sinkhorn.__file__), "seed": 20261009,
              "scope": "Small balanced OT; FP32 IEEE GPU vs CPU FP64. No timing benchmark.",
              "settings": {"eps": 0.7, "forward_iterations": 500, "reference_iterations": 1000,
                           "tau2": 1e-5, "cg_rtol": 1e-6, "cg_atol": 1e-6, "max_cg_iter": 256,
                           "marginal_l1_limit": 1e-5, "plan_relative_l2_limit": 2e-4,
                           "hvp_relative_l2_limit": 5e-4, "direct_system_relative_residual_limit": 1e-9,
                           "apply_rtol": 2e-4, "apply_atol": 2e-6}, "cases": []}

    def save():
        def safe(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {k: safe(v) for k, v in value.items()}
            if isinstance(value, list):
                return [safe(v) for v in value]
            return value
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def error(actual, expected):
        return float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected).clamp_min(1e-30))

    save()
    for n, m, d in ((17, 23, 3), (32, 24, 16), (37, 29, 33)):
        x = 0.2 * torch.randn(n, d, generator=generator)
        y = 0.2 * torch.randn(m, d, generator=generator)
        a, b = torch.rand(n, generator=generator) + 0.2, torch.rand(m, generator=generator) + 0.2
        a, b = a / a.sum(), b / b.sum()
        v = torch.randn(n, d, generator=generator)
        # Upstream matrix-apply requires the output width to equal feature d.
        matrices = {1: torch.randn(m, d, generator=generator), 0: torch.randn(n, d, generator=generator)}
        xd, yd, ad, bd, vd = [t.double() for t in (x, y, a, b, v)]
        xg, yg, ag, bg, vg = [t.cuda() for t in (x, y, a, b, v)]
        for scale in (1.0, 0.5):
            _, _, ref_plan = solve(xd, yd, ad, bd, 0.7, scale)
            for mode, solver in (("symmetric", sinkhorn_flashstyle_symmetric),
                                 ("alternating", sinkhorn_flashstyle_alternating)):
                case = {"shape": [n, m, d], "mode": mode, "cost_scale": scale,
                        "allow_tf32": False, "use_exp2": d != 3, "status": "running"}
                report["cases"].append(case)
                print(f"[independent] {mode} shape={n,m,d} cost_scale={scale}", flush=True)
                try:
                    kwargs = {"use_epsilon_scaling": False, "last_extrapolation": False} if mode == "symmetric" else {}
                    f, g = solver(xg, yg, ag, bg, eps=0.7, n_iters=500, cost_scale=scale,
                                  allow_tf32=False, use_exp2=case["use_exp2"], autotune=False, **kwargs)
                    fd, gd = f.cpu().double(), g.cpu().double()
                    pg = plan(xd, yd, ad, bd, fd, gd, 0.7, scale)
                    case["reference_marginal_l1"] = max(float((ref_plan.sum(1)-ad).abs().sum()), float((ref_plan.sum(0)-bd).abs().sum()))
                    case["gpu_marginal_l1"] = max(float((pg.sum(1)-ad).abs().sum()), float((pg.sum(0)-bd).abs().sum()))
                    case["plan_relative_l2"] = error(pg, ref_plan)
                    assert case["reference_marginal_l1"] <= 1e-5, "Reference has not converged"
                    assert case["gpu_marginal_l1"] <= 1e-5, "GPU forward has not converged"
                    assert case["plan_relative_l2"] <= 2e-4, "Forward plan differs from independent solve"
                    shifted_f = f - scale * xg.square().sum(1)
                    shifted_g = g - scale * yg.square().sum(1)
                    case["apply_relative_l2"] = {}
                    for axis in (0, 1):
                        output = apply_plan_mat_flashstyle(xg, yg, shifted_f, shifted_g, ag.log(), bg.log(),
                                                          matrices[axis].cuda(), axis=axis, eps=0.7, cost_scale=scale,
                                                          allow_tf32=False, use_exp2=case["use_exp2"], autotune=False)
                        expected = (pg if axis else pg.T) @ matrices[axis].double()
                        case["apply_relative_l2"][str(axis)] = error(output.cpu().double(), expected)
                        torch.testing.assert_close(output.cpu().double(), expected, rtol=2e-4, atol=2e-6)
                    # Use the actual rounded OTT potentials consumed by the GPU HVP.
                    fo, go = f + 0.7 * ag.log(), g + 0.7 * bg.log()
                    cost = scale * (xd[:, None] - yd[None, :]).square().sum(-1)
                    hp = torch.exp((fo.cpu().double()[:, None] + go.cpu().double()[None, :] - cost) / 0.7)
                    expected_hvp, linear_residual = direct_hvp(xd, yd, hp, vd, 0.7, scale, tau2=1e-5)
                    actual_hvp, info = hvp_x_sqeuclid_from_potentials(
                        xg, yg, fo, go, vg, eps=0.7, cost_scale=scale, tau2=1e-5, max_cg_iter=256,
                        cg_rtol=1e-6, cg_atol=1e-6, use_preconditioner=False,
                        allow_tf32=False, use_exp2=case["use_exp2"], autotune=False)
                    case["cg"] = asdict(info)
                    case["direct_system_relative_residual"] = linear_residual
                    case["hvp_relative_l2"] = error(actual_hvp.cpu().double(), expected_hvp)
                    assert torch.isfinite(actual_hvp).all(), "Nonfinite HVP"
                    assert info.cg_converged and info.cg_residual <= max(1e-6, 1e-6 * info.cg_initial_residual), "CG not confirmed"
                    assert linear_residual <= 1e-9, "Independent linear solve residual too large"
                    assert case["hvp_relative_l2"] <= 5e-4, "HVP differs from independent direct solve"
                    case["status"] = "passed"
                except Exception as exc:
                    case["status"] = "failed"
                    case["error"] = f"{type(exc).__name__}: {exc}"
                    print(case["error"], flush=True)
                save()
    report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"]) else "failed"
    save()
    print(f"[independent] {report['status']}: {sum(c['status']=='passed' for c in report['cases'])}/12", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
