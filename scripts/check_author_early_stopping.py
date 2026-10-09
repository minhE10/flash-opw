"""Diagnose the exact early-stop fixture against independent CPU FP64 plans.

Warnings remain visible and are saved. A potential-change stop and a marginal
certificate are recorded separately. Author source/default precision is unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import warnings

import torch

try:
    from .author_flashsinkhorn_fp64 import plan
except ImportError:
    from author_flashsinkhorn_fp64 import plan


def marginal(p, a, b):
    return max(float((p.sum(1)-a).abs().sum()), float((p.sum(0)-b).abs().sum()))


def reference(x, y, a, b, eps, cap=64000, tolerance=1e-6, progress=None):
    """Independent alternating log-domain solve, checking the actual FP64 plan."""
    cost = (x[:, None]-y[None, :]).square().sum(-1)
    la, lb = a.log(), b.log()
    f, g = torch.zeros_like(a), torch.zeros_like(b)
    trace = []
    for count in range(1, cap+1):
        f = -eps*torch.logsumexp(lb[None, :]+(g[None, :]-cost)/eps, dim=1)
        g = -eps*torch.logsumexp(la[:, None]+(f[:, None]-cost)/eps, dim=0)
        if count % 100 == 0 or count == cap:
            p = plan(x, y, a, b, f, g, eps)
            residual = marginal(p, a, b)
            trace.append({"iterations":count, "marginal_l1":residual})
            if progress and (count % 1000 == 0 or residual <= tolerance):
                progress(count, residual)
            if residual <= tolerance:
                break
    return p, {"iterations":count, "marginal_l1":residual,
               "converged":residual <= tolerance, "tolerance":tolerance, "cap":cap, "trace":trace}


def symmetric_fixed(x, y, a, b, eps, budget):
    """Independent full init, Jacobi half updates, full final extrapolation."""
    c = (x[:, None]-y[None, :]).square().sum(-1)
    la, lb = a.log(), b.log()
    def update(f, g):
        return (-eps*torch.logsumexp(lb[None, :]+(g[None, :]-c)/eps, 1),
                -eps*torch.logsumexp(la[:, None]+(f[:, None]-c)/eps, 0))
    f, g = update(torch.zeros_like(a), torch.zeros_like(b))
    for _ in range(budget):
        ff, gg = update(f, g)
        f, g = (f+ff)*0.5, (g+gg)*0.5
    f, g = update(f, g)
    return plan(x, y, a, b, f, g, eps)


def relative(actual, expected):
    return float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected).clamp_min(1e-30))


def classify(cases, ref):
    if not ref["converged"]:
        return "reference_not_converged"
    if any(c.get("error") for c in cases):
        return "failed"
    valid = [c for c in cases if c["marginal_l1"] <= 1e-3 and c["plan_relative_l2"] <= 5e-3]
    if any(c["threshold"] == 1e-3 for c in valid):
        return "marginal_and_reference_confirmed"
    if valid:
        return "fixed_budget_confirmed_stop_not_certified"
    return "not_confirmed_at_cap"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.implementation.resolve()/"torch-ext"))
    import flash_sinkhorn
    from flash_sinkhorn.testing.test_geomloss_sinkhorn_triton import _rand_inputs
    from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import sinkhorn_flashstyle_symmetric
    assert args.implementation.resolve() in Path(flash_sinkhorn.__file__).resolve().parents
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.45)
    x, y, a, b = _rand_inputs(128, 128, 32, torch.device("cuda"))
    xd, yd, ad, bd = [t.cpu().double() for t in (x, y, a, b)]
    inputs = {name:t.tolist() for name,t in zip(("x", "y", "a", "b"), (xd, yd, ad, bd))}
    input_path = args.output/"inputs.json"
    input_path.write_text(json.dumps(inputs)+"\n", encoding="utf-8")
    report = {"status":"running", "package":str(flash_sinkhorn.__file__), "gpu":torch.cuda.get_device_name(),
              "settings":{"shape":[128,128,32], "seed":0, "input_rng":"CUDA author _rand_inputs",
                          "x_y_dtype":"float16", "weight_dtype":"float32", "eps":0.1,
                          "threshold":1e-3, "check_every":5, "budgets":[200,1000,4000,16000],
                          "allow_tf32":True, "use_exp2":True, "autotune":True, "last_extrapolation":True,
                          "reference_cap":64000, "reference_marginal_limit":1e-6,
                          "diagnostic_marginal_limit":1e-3, "diagnostic_plan_relative_l2_limit":5e-3},
              "input_sha256":hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "mass_mismatch":float(abs(ad.sum()-bd.sum())), "cases":[],
              "limitation":"Diagnostic limits do not modify upstream test tolerances; no benchmark."}
    def save():
        def safe(v):
            if isinstance(v, float) and not math.isfinite(v): return None
            if isinstance(v, dict): return {k:safe(x) for k,x in v.items()}
            if isinstance(v, list): return [safe(x) for x in v]
            return v
        (args.output/"early-stopping.json").write_text(json.dumps(safe(report), indent=2)+"\n", encoding="utf-8")
    save()
    print("[early-stop] Independent CPU FP64 reference on saved CUDA inputs", flush=True)
    ref_plan, report["reference"] = reference(xd,yd,ad,bd,0.1,
        progress=lambda k,r:print(f"[reference] {k} iterations; marginal L1={r:.6g}", flush=True))
    save()
    # Original fixed-budget comparator has the same 200 half updates plus init/final.
    dense200 = symmetric_fixed(xd,yd,ad,bd,0.1,200)
    for threshold in (None, 1e-3):
        budgets = (200,1000,4000,16000)
        for budget in budgets:
            case = {"threshold":threshold, "budget":budget}
            report["cases"].append(case)
            print(f"[early-stop] budget={budget}, threshold={threshold}", flush=True)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    f,g,steps = sinkhorn_flashstyle_symmetric(x,y,a,b,use_epsilon_scaling=False,
                        eps=0.1,n_iters=budget,threshold=threshold,check_every=5,return_n_iters=True)
                case["warnings"] = [{"category":w.category.__name__, "message":str(w.message)} for w in caught]
                for w in caught: print(f"{w.category.__name__}: {w.message}", flush=True)
                p = plan(xd,yd,ad,bd,f.cpu().double(),g.cpu().double(),0.1)
                case.update(updates=int(steps), finite=bool(torch.isfinite(p).all()),
                            marginal_l1=marginal(p,ad,bd), plan_relative_l2=relative(p,ref_plan),
                            potential_stop_confirmed=threshold is not None and not any(
                                "convergence to threshold=" in str(w.message) for w in caught))
                if budget == 200:
                    case["same_budget_fp64_plan_relative_l2"] = relative(p,dense200)
                case["marginal_confirmed"] = case["finite"] and case["marginal_l1"] <= 1e-3
                if not case["finite"]: case["error"] = "Nonfinite reconstructed plan"
                print(f"[early-stop] updates={steps}; residual={case['marginal_l1']:.6g}; "
                      f"plan relative L2={case['plan_relative_l2']:.6g}; stop={case['potential_stop_confirmed']}",flush=True)
            except Exception as exc:
                case["error"] = f"{type(exc).__name__}: {exc}"
                print(case["error"],flush=True)
            save()
    report["status"] = classify(report["cases"], report["reference"])
    save()
    print(f"[early-stop] {report['status']}",flush=True)
    return 0 if report["status"] == "marginal_and_reference_confirmed" else 3 if report["status"] == "fixed_budget_confirmed_stop_not_certified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
