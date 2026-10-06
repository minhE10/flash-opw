"""Group 1: same-iteration backend parity and fixed-parameter convergence.

The independent oracle solves explicit affine costs on CPU FP64. CPU mode
checks dense FP32 and small tiled FP64 cases; only CUDA mode checks Flash.
Iteration sweeps diagnose frozen parameters, never select using TEST labels.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from flashopw import OPWParameters, materialize_opw_plan, opw_dense, opw_diagnostics, opw_flash, opw_online
from flashsinkhorn import apply_plan
from .opw_parameters import atomic_json, load_selection, source_hashes
from .opw_scaling import synthetic_pair
from .opw_tune import training_holdout
from .runtime import configure, metadata
from .sequence_data import load_training


@torch.no_grad()
def reference_snapshots(xs, ys, parameters, checkpoints):
    """Uniform, explicit D+mu*F, unshifted FP64 updates; incremental checkpoints."""
    if not checkpoints or list(checkpoints) != sorted(set(checkpoints)) or checkpoints[0] < 1:
        raise ValueError("Checkpoints must be distinct increasing positive integers")
    x = torch.as_tensor(np.stack(xs), dtype=torch.float64)
    y = torch.as_tensor(np.stack(ys), dtype=torch.float64)
    p = OPWParameters(**parameters)
    batch, n, _ = x.shape
    m = y.shape[1]
    spatial = p.cost_scale * torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist").square()
    t = torch.arange(1, n+1, dtype=x.dtype)/n
    s = torch.arange(1, m+1, dtype=x.dtype)/m
    cost = spatial + p.mu * (t[:, None]-s[None, :]).square()
    f, g = x.new_zeros((batch, n)), y.new_zeros((batch, m))
    previous, eps = 0, p.lambda2
    for iteration in checkpoints:
        for _ in range(iteration-previous):
            f = -eps * torch.logsumexp((g[:, None, :]-cost)/eps-math.log(m), dim=2)
            g = -eps * torch.logsumexp((f[:, :, None]-cost)/eps-math.log(n), dim=1)
        previous = iteration
        plan = ((f[:, :, None]+g[:, None, :]-cost)/eps-math.log(n)-math.log(m)).exp()
        rows, columns = plan.sum(2), plan.sum(1)
        mass = rows.sum(1)
        entropy_f = f + eps*(-math.log(n)+.5)
        entropy_g = g + eps*(-math.log(m)+.5)
        primal = (rows*entropy_f).sum(1)+(columns*entropy_g).sum(1)-eps*mass+p.q0*mass
        dual = entropy_f.mean(1)+entropy_g.mean(1)-eps*mass+p.q0
        values = dict(score=f.mean(1)+g.mean(1)-eps+p.q0,
                      spatial=(plan*spatial).sum((1,2)), mass=mass,
                      row_l1=(rows-1/n).abs().sum(1), col_l1=(columns-1/m).abs().sum(1),
                      primal_minus_dual=primal-dual)
        if any(not bool(torch.isfinite(value).all()) for value in values.values()):
            raise FloatingPointError("Nonfinite independent reference")
        yield iteration, values, plan


def ranking_comparison(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[1] < 2:
        raise ValueError("Matching score matrices with at least two gallery items required")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise FloatingPointError("Cannot compare incomplete score matrices")
    ar = np.argsort(actual, axis=1, kind="stable")
    er = np.argsort(expected, axis=1, kind="stable")
    delta = np.abs(actual-expected)
    ordered = np.take_along_axis(expected, er[:, :2], axis=1)
    margin = ordered[:,1]-ordered[:,0]
    return dict(max_abs_score_error=float(delta.max()), median_abs_score_error=float(np.median(delta)),
                relative_l2_score_error=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1e-30)),
                nn_agreement=float(np.mean(ar[:,0] == er[:,0])),
                full_ranking_agreement=float(np.mean(np.all(ar == er, axis=1))),
                min_reference_nn_margin=float(margin.min()),
                ambiguous_queries_at_observed_error=int(np.sum(margin <= 2*delta.max(axis=1))))


@torch.no_grad()
def parity_check(x, y, p, iteration, reference, plan, backend, device, args):
    dtype = torch.float64 if backend == "tiled-fp64" else torch.float32
    tx, ty = (torch.as_tensor(v, dtype=dtype, device=device) for v in (x,y))
    solver = {"flash-fp32":opw_flash, "dense-fp32":opw_dense, "tiled-fp64":opw_online}[backend]
    result = solver(tx, ty, **p, n_iters=iteration, precision="ieee")
    stats = opw_diagnostics(result)
    actual_plan = materialize_opw_plan(result, max_entries=args.max_dense_entries).double().cpu()
    features = torch.cat((ty.new_ones((len(ty),1)), ty), dim=1)
    actual_apply = apply_plan(result.sinkhorn, features).double().cpu()
    expected_apply = plan @ features.double().cpu()
    plan_error = float((actual_plan-plan).norm()/plan.norm().clamp_min(1e-30))
    apply_error = float((actual_apply-expected_apply).norm()/expected_apply.norm().clamp_min(1e-30))
    expected_score, expected_spatial = float(reference["score"][0]), float(reference["spatial"][0])
    rtol, atol = (1e-10, 1e-11) if backend == "tiled-fp64" else (args.score_rtol, args.score_atol)
    plan_tolerance = 1e-10 if backend == "tiled-fp64" else args.plan_rtol
    passed = (math.isclose(stats["pdf_eq19"], expected_score, rel_tol=rtol, abs_tol=atol)
              and math.isclose(stats["spatial_transport_cost"], expected_spatial, rel_tol=rtol, abs_tol=atol)
              and plan_error <= plan_tolerance and apply_error <= plan_tolerance
              and all(math.isfinite(v) for v in stats.values()))
    return dict(backend=backend, iterations=iteration, status="passed" if passed else "failed",
                score_abs_error=abs(stats["pdf_eq19"]-expected_score),
                spatial_abs_error=abs(stats["spatial_transport_cost"]-expected_spatial),
                plan_relative_l2=plan_error, apply_relative_l2=apply_error,
                observed=stats, reference={k:float(v[0]) for k,v in reference.items()})


def build_reference_matrix(queries, gallery, p, checkpoints, batch_size, max_entries):
    keys = ("score", "spatial", "mass", "row_l1", "col_l1", "primal_minus_dual")
    shape = len(queries), len(gallery)
    snapshots = {it:{k:np.empty(shape) for k in keys} for it in checkpoints}
    groups = {}
    for i,x in enumerate(queries):
        for j,y in enumerate(gallery):
            if len(x)*len(y) > max_entries:
                raise ValueError("Training pair exceeds dense entry cap; lower dataset lengths for group1 oracle")
            groups.setdefault((x.shape,y.shape), []).append((i,j))
    for (xshape,yshape), pairs in groups.items():
        # Bound estimated FP64 working arrays to 128MiB even for ragged long pairs.
        size = min(batch_size, max(1, 128*1024**2//(8*8*xshape[0]*yshape[0])))
        for start in range(0,len(pairs),size):
            batch = pairs[start:start+size]
            xs, ys = [queries[i] for i,j in batch], [gallery[j] for i,j in batch]
            for iteration, values, _ in reference_snapshots(xs, ys, p, checkpoints):
                for k in keys:
                    for index,(i,j) in enumerate(batch):
                        snapshots[iteration][k][i,j] = float(values[k][index])
    return snapshots


def convergence_summary(snapshots, taus):
    iterations = sorted(snapshots)
    last = snapshots[iterations[-1]]
    rows = []
    previous = None
    for iteration in iterations:
        values = snapshots[iteration]
        residual = np.maximum(values["row_l1"], values["col_l1"])
        row = dict(iterations=iteration, max_marginal_l1=float(residual.max()),
                   median_marginal_l1=float(np.median(residual)),
                   p95_marginal_l1=float(np.quantile(residual,.95)),
                   max_mass_error=float(np.abs(values["mass"]-1).max()),
                   versus_last_checkpoint=ranking_comparison(values["score"],last["score"]),
                   versus_previous_checkpoint=None if previous is None else
                   ranking_comparison(values["score"],previous["score"]),
                   reached_fraction={str(tau):float(np.mean(residual <= tau)) for tau in taus})
        rows.append(row)
        previous = values
    first = {}
    for tau in taus:
        reached = np.zeros(last["score"].shape, dtype=np.int64)
        for iteration in iterations:
            residual = np.maximum(snapshots[iteration]["row_l1"],snapshots[iteration]["col_l1"])
            reached[(reached == 0) & (residual <= tau)] = iteration
        final_residual = np.maximum(last["row_l1"],last["col_l1"])
        first[str(tau)] = dict(never_reached_in_sweep=int(np.sum(reached == 0)),
                              not_satisfied_at_cap=int(np.sum(final_residual > tau)),
                              first_observed_checkpoint_counts={str(it):int(np.sum(reached == it)) for it in iterations},
                              limitation="First observed checkpoint, not exact stopping iteration; last checkpoint not assumed converged")
    return dict(checkpoints=rows, first_reached=first, cap=iterations[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="FacesUCR")
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/opw"))
    parser.add_argument("--flash-parameters", type=Path,
                        default=Path("reports/opw_tuning_20261006/selected_parameters.json"))
    parser.add_argument("--profiles", nargs="+", choices=("default","tuned","stress"), default=["default","tuned"])
    parser.add_argument("--gallery", type=int, default=16)
    parser.add_argument("--queries", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[20,100,200,500,1000,2000])
    parser.add_argument("--parity-iters", type=int, nargs="+", default=[20,200])
    parser.add_argument("--parity-shapes", nargs="+", default=["7:11:3","37:79:65","131:131:1","257:513:13","17:29:390","1025:1537:1"])
    parser.add_argument("--taus", type=float, nargs="+", default=[1e-3,1e-4])
    parser.add_argument("--reference-batch", type=int, default=16)
    parser.add_argument("--max-dense-entries", type=int, default=4_194_304)
    parser.add_argument("--score-rtol", type=float, default=3e-3)
    parser.add_argument("--score-atol", type=float, default=3e-4)
    parser.add_argument("--plan-rtol", type=float, default=5e-3)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=.45)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (min(args.gallery,args.queries) < 0 or min(args.reference_batch,args.max_dense_entries,
            *args.checkpoints,*args.parity_iters) < 1):
        parser.error("Positive iterations/batch/cap and nonnegative subset limits required")
    if args.checkpoints != sorted(set(args.checkpoints)) or args.parity_iters != sorted(set(args.parity_iters)):
        parser.error("Checkpoints and parity-iters must be distinct increasing integers")
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("Profiles must be distinct")
    if any(not math.isfinite(v) or v <= 0 for v in (*args.taus,args.score_rtol,args.score_atol,args.plan_rtol)):
        parser.error("Finite positive thresholds required")
    shapes = []
    for value in args.parity_shapes:
        try:
            n,m,d = map(int,value.split(":"))
        except ValueError:
            parser.error("Shapes must be n:m:d")
        if min(n,m,d) < 1 or d > 1023 or n*m > args.max_dense_entries:
            parser.error("Shape exceeds positive dimension/feature/dense limits")
        shapes.append((n,m,d))
    sequences, labels, origin = load_training(args.dataset,args.data_root,args.dataset_file)
    selection = load_selection(args.flash_parameters,dataset=args.dataset,
                               training_sha256=origin["training_sha256"],score="pdf-loss")
    gi,qi = training_holdout(labels,args.gallery,args.queries,args.seed)
    if len(gi) < 2:
        parser.error("Group1 ranking checks need at least two gallery samples")
    # Preserve original training fingerprint, then give CPU/GPU identical FP32 values.
    gallery = [sequences[i].astype(np.float32).astype(np.float64) for i in gi]
    queries = [sequences[i].astype(np.float32).astype(np.float64) for i in qi]
    pkeys = ("lambda1","lambda2","sigma","cost_scale")
    profiles = dict(default={k:selection["default_validation"]["parameters"][k] for k in pkeys},
                    tuned={k:selection["parameters"][k] for k in pkeys})
    profiles["stress"] = dict(profiles["tuned"],lambda1=1.,lambda2=.03,
                             sigma=math.sqrt(.03/(2*(430-1))))
    profiles = {name:profiles[name] for name in args.profiles}
    os.environ["FLASHOPW_AUTOTUNE"] = "0"
    device = configure(args.device,args.threads,args.memory_fraction)
    environment = metadata(device)
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ("output","resume")}
    sources = source_hashes()
    sources["experiments/opw_group1.py"] = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    input_sha = hashlib.sha256(b"".join(v.tobytes() for v in queries+gallery)).hexdigest()
    signature = dict(settings=settings,profiles=profiles,selection=selection,origin=origin,
                     gallery_indices=gi.tolist(),query_indices=qi.tolist(),input_sha256=input_sha,
                     sources=sources,torch=environment["torch"],packages=environment["packages"],
                     python=environment["python"],gpu=environment.get("gpu"),kernel_controls=environment["kernel_controls"])
    output = args.output
    output.mkdir(parents=True,exist_ok=args.resume)
    parity = []
    completed_reference = []
    if args.resume:
        prior = json.loads((output/"environment.json").read_text(encoding="utf-8"))
        if prior["signature"] != signature:
            raise ValueError("Resume refused: code/data/settings/environment/selection changed")
        parity = json.loads((output/"parity.json").read_text(encoding="utf-8"))
        completed_reference = json.loads((output/"reference_state.json").read_text(encoding="utf-8"))
    else:
        environment.update(signature=signature,protocol=dict(
            cost="explicit spatial squared distance + mu*(i/N-j/M)^2, main PDF; q0 restored only in score",
            input="original TRAIN fingerprint checked, then quantized to FP32 identically for all backends; TEST unopened",
            iterations="uniform, alternating f then g; diagnostic overrides frozen n_iters without mutating selection",
            cpu_scope="dense FP32 and small tiled FP64 parity, plus FP64 reference convergence; not Flash CUDA evidence",
            cuda_scope="Flash FP32 IEEE versus explicit CPU FP64 at identical iterations, including score-matrix parity",
            gap="primal_minus_dual not a certified gap when marginals infeasible",
            checkpoints="reference incremental; Flash restarted at each fixed checkpoint; not a speed benchmark",
            resume="parity cases, completed reference profiles and complete GPU query rows checkpointed"))
        atomic_json(output/"environment.json",environment)
        atomic_json(output/"parity.json",parity)
        atomic_json(output/"reference_state.json",completed_reference)
    atomic_json(output/"run_state.json",dict(status="running",flash_requested=device.type == "cuda"))
    print(f"Group1: TRAIN-only {len(qi)} queries x {len(gi)} gallery; {device}; profiles={list(profiles)}",flush=True)
    backend = "flash-fp32" if device.type == "cuda" else "dense-fp32"
    for name,p in profiles.items():
        cases = [(f"synthetic_{n}_{m}_{d}",*synthetic_pair(n,m,d,args.seed+index)) for index,(n,m,d) in enumerate(shapes)]
        cases += [(f"train_pair_{j}",queries[0],gallery[j]) for j in range(min(4,len(gallery)))]
        for case,x,y in cases:
            for iteration,ref,plan in reference_snapshots([x],[y],p,args.parity_iters):
                backends = [backend]
                if device.type == "cpu" and len(x)*len(y) <= 4096:
                    backends.append("tiled-fp64")
                for observed_backend in backends:
                    if any(r["profile"] == name and r["case"] == case and r["iterations"] == iteration
                           and r["backend"] == observed_backend for r in parity):
                        continue
                    row = parity_check(x,y,p,iteration,ref,plan[0],observed_backend,device,args)
                    row.update(profile=name,case=case,n=len(x),m=len(y),d=x.shape[1])
                    row["input_sha256"] = hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
                    parity.append(row)
                    atomic_json(output/"parity.json",parity)
                    print(f"{name} {case} {observed_backend} it={iteration}: {row['status']}, P relL2={row['plan_relative_l2']:.3g}",flush=True)
        if name not in completed_reference:
            reference = build_reference_matrix(queries,gallery,p,args.checkpoints,args.reference_batch,args.max_dense_entries)
            for iteration,values in reference.items():
                np.savez_compressed(output/f"{name}_reference_{iteration}.npz",**values,
                                    gallery_indices=gi,query_indices=qi)
            completed_reference.append(name)
            atomic_json(output/"reference_state.json",completed_reference)
        print(f"{name}: complete reference sweep to {args.checkpoints[-1]} iterations",flush=True)
    summaries = {}
    for name,p in profiles.items():
        reference = {}
        for iteration in args.checkpoints:
            with np.load(output/f"{name}_reference_{iteration}.npz",allow_pickle=False) as saved:
                reference[iteration] = {k:saved[k].copy() for k in
                    ("score","spatial","mass","row_l1","col_l1","primal_minus_dual")}
        summaries[name] = dict(parameters=p,mu=OPWParameters(**p).mu,
                               reference=convergence_summary(reference,args.taus))
        if device.type == "cuda":
            observed = {}
            parity_matrices = []
            tx = [torch.as_tensor(x,dtype=torch.float32,device=device) for x in queries]
            ty = [torch.as_tensor(y,dtype=torch.float32,device=device) for y in gallery]
            for iteration in args.checkpoints:
                cache = output/f"{name}_flash_{iteration}.npz"
                values = {k:np.full((len(tx),len(ty)),np.nan) for k in reference[iteration]}
                completed = 0
                if args.resume and cache.exists():
                    with np.load(cache,allow_pickle=False) as saved:
                        values = {k:saved[k].copy() for k in values}
                        completed = int(saved["completed_queries"])
                for i in range(completed,len(tx)):
                    for j in range(len(ty)):
                        result = opw_flash(tx[i],ty[j],**p,n_iters=iteration,precision="ieee")
                        stats = opw_diagnostics(result)
                        for k,key in (("score","pdf_eq19"),("spatial","spatial_transport_cost"),
                                      ("mass","mass"),("row_l1","row_l1"),("col_l1","col_l1"),
                                      ("primal_minus_dual","primal_minus_dual")):
                            values[k][i,j] = stats[key]
                    temporary = cache.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary,**values,completed_queries=i+1,gallery_indices=gi,query_indices=qi)
                    temporary.replace(cache)
                observed[iteration] = values
                comparison = ranking_comparison(values["score"],reference[iteration]["score"])
                passed = (np.allclose(values["score"],reference[iteration]["score"],rtol=args.score_rtol,atol=args.score_atol)
                          and np.allclose(values["spatial"],reference[iteration]["spatial"],rtol=args.score_rtol,atol=args.score_atol))
                parity_matrices.append(dict(iterations=iteration,status="passed" if passed else "failed",**comparison))
                print(f"{name} Flash matrix it={iteration}: NN agreement={comparison['nn_agreement']:.3%}",flush=True)
            summaries[name].update(flash=convergence_summary(observed,args.taus),matrix_parity=parity_matrices)
        atomic_json(output/"convergence.json",summaries)
    failed = sum(r["status"] == "failed" for r in parity)
    failed += sum(r["status"] == "failed" for s in summaries.values() for r in s.get("matrix_parity",[]))
    summary = dict(status="failed" if failed else "completed",parity_failures=failed,
                   flash_verification="executed" if device.type == "cuda" else "not_run_CPU_only",
                   queries=len(qi),gallery=len(gi),profiles=summaries,
                   limitation="Last checkpoint is not assumed converged; no TEST accuracy optimization or GPU timing claim")
    atomic_json(output/"summary.json",summary)
    atomic_json(output/"run_state.json",dict(status=summary["status"],parity_failures=failed,
                                            flash_verification=summary["flash_verification"]))
    fields = ("profile","backend","iterations","max_marginal_l1","median_marginal_l1","p95_marginal_l1",
              "reached_1e-3","reached_1e-4","score_drift_vs_last","nn_agreement_vs_last")
    with (output/"convergence.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for name,profile in summaries.items():
            for label in ("reference","flash"):
                for row in profile.get(label,{}).get("checkpoints",[]):
                    writer.writerow(dict(profile=name,backend="CPU FP64" if label == "reference" else "Flash CUDA FP32",
                        **{k:row[k] for k in ("iterations","max_marginal_l1","median_marginal_l1","p95_marginal_l1")},
                        **{"reached_1e-3":row["reached_fraction"].get("0.001"),"reached_1e-4":row["reached_fraction"].get("0.0001")},
                        score_drift_vs_last=row["versus_last_checkpoint"]["max_abs_score_error"],
                        nn_agreement_vs_last=row["versus_last_checkpoint"]["nn_agreement"]))
    print(f"Results: {output.resolve()}; parity_failures={failed}; Flash={summary['flash_verification']}",flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
