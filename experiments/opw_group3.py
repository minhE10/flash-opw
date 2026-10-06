"""TRAIN-only, three-split selection for all sequence metrics (experiment 3).

CPU uses explicit FP64 costs; CUDA uses Flash IEEE FP32 for the proposed
affine model and dense FP32 for entropic baselines. This is a quality/search
experiment, not a speed benchmark or the authors' implementation.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from flashopw import OPWParameters, opw_diagnostics, opw_flash
from .opw_parameters import atomic_json, source_hashes
from .opw_tune import training_holdout
from .retrieval import evaluate_distances
from .runtime import configure, metadata
from .sequence_data import load_training
from .sequence_metrics import JOURNAL_METRICS, reference_distance


METRICS = ("flash-opw",) + JOURNAL_METRICS
UNTUNABLE = ("dtw", "ldtw", "ndtw", "ot")
SCHEMA = "flashopw-all-metrics-selection-v1"
PROTOCOL = "three stratified holdouts within official TRAIN only"


def candidate_sets(dataset, budget=6):
    """Fixed nested joint search, declared before labels are evaluated.

    Equal candidate count for seven tunable methods, including the preset.
    The Flash and TLp sets cover the same (temporal coefficient, epsilon)
    pairs, ordered with each method's own preset first.
    """
    if not 2 <= budget <= 12:
        raise ValueError("Candidate budget must be between 2 and 12")
    lam = {"FacesUCR": 1., "FaceAll": 10.}.get(dataset, 50.)
    mu0 = lam + .05
    pairs = [(mu0, .1), (50., .1), (10., .3), (50., .3), (10., .03),
             (200., .1), (1.05, .3), (200., .3), (50., .03),
             (430., .03), (.1, .1), (1.05, .03)]
    flash = []
    for mu, eps in pairs:
        weight = lam if mu > lam else mu/2
        flash.append(dict(lambda1=weight, lambda2=eps,
                          sigma=math.sqrt(eps/(2*(mu-weight))), cost_scale=1.))
    # Keep the literal preset sigma=1, including its floating point spelling.
    flash[0] = dict(lambda1=lam, lambda2=.1, sigma=1., cost_scale=1.)
    opw = [(lam, .1, 1.), (lam, .1, 3.), (.1, .3, 1.), (10., .1, 3.),
           (50., .03, 5.), (lam, .3, 5.), (0., .1, 3.), (50., .3, 10.),
           (10., .03, 1.), (lam, 1., 1.), (50., .1, 1.), (lam, .03, 10.)]
    opw += list(itertools.product((.1, 1., 10., 50.), (.03, .1, .3), (1., 3., 5.)))
    opw = list(dict.fromkeys(opw))[:12]
    kl = [(.1, 1.), (.1, 3.), (.3, 1.), (.03, 5.), (.3, 5.), (.1, 5.),
          (.03, 1.), (1., 3.), (.3, 10.), (.1, 10.), (.01, 1.), (1., 1.)]
    eps = [.1, .03, .3, .01, 1., .003, 3., .001, 10., .0003, 30., .0001]
    sets = dict(
        **{"flash-opw": flash,
           "tlp": [dict(tlp_weight=mu, epsilon=e, cost_scale=1.)
                   for mu, e in [pairs[1], pairs[0], *pairs[2:]]],
           "opw": [dict(lambda1=l, lambda2=e, sigma=s, cost_scale=1.) for l,e,s in opw],
           "opw-kl": [dict(lambda1=0., lambda2=e, sigma=s, cost_scale=1.) for e,s in kl],
           "sinkhorn": [dict(epsilon=e, cost_scale=1.) for e in eps],
           "tcot": [dict(tcot_lambda=l, cost_scale=1.) for l in
                    (1., 10., .1, 3., .3, 30., .03, 100., .01, 300., .003, 1000.)],
           "soft-dtw": [dict(soft_dtw_gamma=e, cost_scale=1.) for e in eps]})
    sets.update({m: [dict(cost_scale=1.)] for m in UNTUNABLE})
    return {m: sets[m][:budget] for m in METRICS}


def cost_tensors(metric, xs, ys, parameters, *, device="cpu", dtype=torch.float64):
    """Uniform weights; center journal constants out of coupling computation."""
    x = torch.as_tensor(np.stack(xs), dtype=dtype, device=device)
    y = torch.as_tensor(np.stack(ys), dtype=dtype, device=device)
    n, m = x.shape[1], y.shape[1]
    spatial = parameters["cost_scale"] * torch.cdist(
        x, y, compute_mode="donot_use_mm_for_euclid_dist").square()
    delta = torch.arange(1,n+1,device=device,dtype=dtype)[:,None]/n - torch.arange(1,m+1,device=device,dtype=dtype)[None,:]/m
    F = delta.square()
    evaluation, q0 = spatial, 0.
    if metric == "flash-opw":
        p = OPWParameters(**parameters)
        cost, epsilon, q0 = spatial+p.mu*F, p.lambda2, p.q0
    elif metric == "sinkhorn":
        cost, epsilon = spatial, parameters["epsilon"]
    elif metric == "tlp":
        cost = spatial + parameters["tlp_weight"]*F
        evaluation, epsilon = cost, parameters["epsilon"]
    elif metric == "tcot":
        cost = spatial*(1+delta.abs())
        evaluation, epsilon = cost, 1/parameters["tcot_lambda"]
    elif metric in ("opw", "opw-kl"):
        weight = parameters["lambda1"] if metric == "opw" else 0.
        epsilon = parameters["lambda2"]
        cost = spatial + weight*F/(1+F) + epsilon*F/((1/n**2+1/m**2)*2*parameters["sigma"]**2)
    else:
        raise ValueError("An entropic metric is required")
    return cost, evaluation, epsilon, q0


@torch.no_grad()
def dense_batch(metric, xs, ys, parameters, policy, *, device="cpu", dtype=torch.float64):
    """Record each pair at its first passing checkpoint, retaining cap values.

    Converged rows are removed from the active batch. All pairs use the same
    f-then-g recurrence; the last cap checkpoint is checked even if off-grid.
    """
    cost, evaluation, eps, q0 = cost_tensors(metric,xs,ys,parameters,device=device,dtype=dtype)
    batch,n,m = cost.shape
    active = torch.arange(batch,device=device)
    f, g = cost.new_zeros(batch,n), cost.new_zeros(batch,m)
    saved = {k: np.empty(batch) for k in ("distances", "residuals", "iterations")}
    for iteration in range(1,policy["max_iters"]+1):
        f = -eps*torch.logsumexp((g[:,None,:]-cost)/eps-math.log(m),dim=2)
        g = -eps*torch.logsumexp((f[:,:,None]-cost)/eps-math.log(n),dim=1)
        if iteration % policy["check_every"] and iteration != policy["max_iters"]:
            continue
        plan = ((f[:,:,None]+g[:,None,:]-cost)/eps-math.log(n)-math.log(m)).exp()
        residual = torch.maximum((plan.sum(2)-1/n).abs().sum(1),(plan.sum(1)-1/m).abs().sum(1))
        done = residual <= policy["tau"]
        if iteration == policy["max_iters"]:
            done = torch.ones_like(done)
        if bool(done.any()):
            score = (f.mean(1)+g.mean(1)-eps+q0 if metric == "flash-opw"
                     else (plan*evaluation).sum((1,2)))
            ids = active[done].cpu().numpy()
            saved["distances"][ids] = score[done].double().cpu().numpy()
            saved["residuals"][ids] = residual[done].double().cpu().numpy()
            saved["iterations"][ids] = iteration
            keep = ~done
            if not bool(keep.any()):
                break
            active,f,g,cost,evaluation = active[keep],f[keep],g[keep],cost[keep],evaluation[keep]
    if any(not np.isfinite(v).all() for v in saved.values()):
        raise FloatingPointError("Nonfinite score or marginal residual; search aborted")
    return saved


@torch.no_grad()
def distance_matrix(metric, queries, gallery, parameters, policy, *, device, batch=16, memory_mib=128):
    shape = (len(queries),len(gallery))
    matrices = {k:np.zeros(shape) for k in ("distances","residuals","iterations")}
    if metric in UNTUNABLE or metric == "soft-dtw":
        for i,x in enumerate(queries):
            for j,y in enumerate(gallery):
                matrices["distances"][i,j] = reference_distance(metric,x,y,**parameters)
        return matrices
    if metric == "flash-opw" and device.type == "cuda":
        tensors_q = [torch.as_tensor(x,dtype=torch.float32,device=device) for x in queries]
        tensors_g = [torch.as_tensor(y,dtype=torch.float32,device=device) for y in gallery]
        for i,x in enumerate(tensors_q):
            for j,y in enumerate(tensors_g):
                result = opw_flash(x,y,**parameters,n_iters=policy["max_iters"],
                                   tol=policy["tau"],check_every=policy["check_every"],precision="ieee")
                diag = opw_diagnostics(result)
                matrices["distances"][i,j] = diag["pdf_eq19"]
                matrices["residuals"][i,j] = max(diag["row_l1"],diag["col_l1"])
                matrices["iterations"][i,j] = result.sinkhorn.n_iters
        return matrices
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    groups = {}
    for i,x in enumerate(queries):
        for j,y in enumerate(gallery):
            groups.setdefault((x.shape,y.shape),[]).append((i,j))
    for (xs,ys),pairs in groups.items():
        bytes_per_pair = 32*torch.empty((),dtype=dtype).element_size()*xs[0]*ys[0]
        if bytes_per_pair > memory_mib*1024**2:
            raise ValueError("Dense pair exceeds search memory budget")
        size = min(batch,max(1,memory_mib*1024**2//bytes_per_pair))
        for start in range(0,len(pairs),size):
            ids = pairs[start:start+size]
            values = dense_batch(metric,[queries[i] for i,j in ids],[gallery[j] for i,j in ids],
                                 parameters,policy,device=device,dtype=dtype)
            for k,v in values.items():
                for position,(i,j) in enumerate(ids):
                    matrices[k][i,j] = v[position]
    return matrices


def choose_candidate(rows):
    """Reject any candidate with capped/infeasible pairs on any split."""
    valid = [row for row in rows if row["eligible"]]
    if not valid:
        return None
    return max(valid,key=lambda row:(row["mean_ACC1"],row["mean_MAP"],-row["candidate"]))


def load_frozen(path, *, dataset=None, training_sha256=None):
    """All-metric artifact for group 4; separate from legacy Flash-only JSON."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("protocol") != PROTOCOL:
        raise ValueError("Unknown all-metric training selection")
    if value.get("status") != "completed" or value.get("test_used") is not False:
        raise ValueError("Selection incomplete or TEST isolation missing")
    for key,expected in (("dataset",dataset),("training_sha256",training_sha256)):
        if expected is not None and value.get(key) != expected:
            raise ValueError(f"Selection {key} mismatch")
    if len(set(value["seeds"])) != 3 or value["objective"] != ["mean_ACC1","mean_MAP","earliest_candidate"]:
        raise ValueError("Invalid selection protocol")
    if set(value["selected"]) != set(METRICS):
        raise ValueError("Incomplete metric selection")
    for metric,row in value["selected"].items():
        if not row["eligible"] or row["capped_pairs"] != 0:
            raise ValueError(f"Ineligible frozen candidate for {metric}")
        if metric == "flash-opw":
            OPWParameters(**row["parameters"])
    return value


def _csv(path, rows):
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _tables(path, presets, tuned):
    lines = ["# Group 3: TRAIN validation only", "",
             "ACC@1 is primary; MAP breaks ties; means and sample SD across three splits.",
             "No TEST data or speed comparison. Presets use repo journal parameters with",
             "the same residual policy as tuning, rather than journal fixed iteration counts.", ""]
    for title,rows in (("Preset parameters",presets),("All metrics tuned on TRAIN",tuned)):
        lines += [f"## {title}", "", "| Metric | Candidate | ACC@1 % (SD) | MAP % (SD) | Capped pairs |", "|---|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['metric']} | {row['candidate']} | {100*row['mean_ACC1']:.3f} ({100*row['sd_ACC1']:.3f}) | {100*row['mean_MAP']:.3f} ({100*row['sd_MAP']:.3f}) | {row['capped_pairs']} |")
        lines += [""]
    path.write_text("\n".join(lines),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",default="FacesUCR")
    parser.add_argument("--dataset-file",type=Path)
    parser.add_argument("--data-root",type=Path,default=Path("data/opw"))
    parser.add_argument("--gallery",type=int,default=16)
    parser.add_argument("--queries",type=int,default=14)
    parser.add_argument("--seeds",type=int,nargs=3,default=[20261006,20261007,20261008])
    parser.add_argument("--budget",type=int,default=6)
    parser.add_argument("--tau",type=float,default=1e-3)
    parser.add_argument("--max-iters",type=int,default=4000)
    parser.add_argument("--check-every",type=int,default=50)
    parser.add_argument("--batch",type=int,default=16)
    parser.add_argument("--memory-mib",type=int,default=128)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--memory-fraction",type=float,default=.45)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    if (len(set(args.seeds)) != 3 or min(args.seeds)<0 or min(args.gallery,args.queries)<0
            or min(args.max_iters,args.check_every,args.batch,args.memory_mib)<1
            or not math.isfinite(args.tau) or args.tau <= 0):
        parser.error("Three distinct nonnegative seeds and positive solver settings required")
    grids = candidate_sets(args.dataset,args.budget)
    sequences,labels,origin = load_training(args.dataset,args.data_root,args.dataset_file)
    splits = []
    for seed in args.seeds:
        gi,qi = training_holdout(labels,args.gallery,args.queries,seed)
        splits.append(dict(seed=seed,gallery_indices=gi.tolist(),query_indices=qi.tolist()))
    os.environ["FLASHOPW_AUTOTUNE"] = "0"
    device = configure(args.device,args.threads,args.memory_fraction)
    environment = metadata(device)
    sources = source_hashes()
    sources["experiments/opw_group3.py"] = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    policy = dict(tau=args.tau,max_iters=args.max_iters,check_every=args.check_every,schedule="f_then_g")
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ("output","resume")}
    signature = dict(settings=settings,grids=grids,splits=splits,origin=origin,sources=sources,
                     torch=environment["torch"],python=environment["python"],packages=environment["packages"],
                     gpu=environment.get("gpu"),kernel_controls=environment["kernel_controls"])
    output = args.output
    output.mkdir(parents=True,exist_ok=args.resume)
    (output/"matrices").mkdir(exist_ok=True)
    (output/"jobs").mkdir(exist_ok=True)
    if args.resume:
        if json.loads((output/"environment.json").read_text(encoding="utf-8"))["signature"] != signature:
            raise ValueError("Resume refused: code/data/settings/environment changed")
    else:
        environment.update(signature=signature,protocol=PROTOCOL,test_used=False,
                           input_precision="round TRAIN values to FP32, then feed identical values to FP64/FP32",
                           objective=["mean_ACC1","mean_MAP","earliest_candidate"],
                           preset_label="repo journal hyperparameter presets; common adaptive iteration policy",
                           search_timing="wall time including input conversion, solving, diagnostics and evaluation; compilation included",
                           candidate_eligibility="every pair on all three splits must reach both L1 marginals <= tau",
                           backends="CPU FP64 explicit control; CUDA Flash IEEE FP32 + dense FP32 entropic references; DTW/SoftDTW/LP CPU")
        atomic_json(output/"environment.json",environment)
        atomic_json(output/"candidate_manifest.json",dict(protocol=PROTOCOL,objective=environment["objective"],
                    solver_policy=policy,candidates=grids,splits=splits,test_used=False))
    total_jobs = sum(map(len,grids.values()))*3
    completed,details = 0,[]
    print(f"Group3: TRAIN-only; seeds={args.seeds}; {args.queries} queries x {args.gallery} gallery per split; {device}; budget={args.budget}; tau={args.tau}",flush=True)
    for split in splits:
        gi,qi = np.array(split["gallery_indices"]),np.array(split["query_indices"])
        gallery = [sequences[i].astype(np.float32).astype(np.float64) for i in gi]
        queries = [sequences[i].astype(np.float32).astype(np.float64) for i in qi]
        for metric in METRICS:
            for candidate,p in enumerate(grids[metric]):
                key = f"seed{split['seed']}__{metric}__{candidate:02d}"
                jobfile = output/"jobs"/f"{key}.json"
                matrixfile = output/"matrices"/f"{key}.npz"
                if args.resume and jobfile.exists() and matrixfile.exists():
                    job = json.loads(jobfile.read_text(encoding="utf-8"))
                    if job["matrix_sha256"] != hashlib.sha256(matrixfile.read_bytes()).hexdigest():
                        raise ValueError("Cached matrix hash mismatch")
                    with np.load(matrixfile,allow_pickle=False) as cached:
                        matrices = {k:cached[k] for k in ("distances","residuals","iterations")}
                else:
                    started = time.perf_counter()
                    matrices = distance_matrix(metric,queries,gallery,p,policy,device=device,
                                               batch=args.batch,memory_mib=args.memory_mib)
                    if any(not np.isfinite(v).all() for v in matrices.values()):
                        raise FloatingPointError("Nonfinite matrix; search aborted")
                    evaluation = evaluate_distances(matrices["distances"],labels[gi],labels[qi],ks=(1,))
                    if evaluation["queries_without_relevant_gallery"]:
                        raise ValueError("Validation query without a relevant gallery item")
                    job = dict(metric=metric,candidate=candidate,seed=split["seed"],parameters=p,
                               MAP=evaluation["MAP"],ACC1=evaluation["ACC"]["1"],evaluation=evaluation,
                               capped_pairs=int(np.count_nonzero(matrices["residuals"]>args.tau)),
                               max_marginal_l1=float(matrices["residuals"].max()),
                               median_iterations=float(np.median(matrices["iterations"])),
                               max_iterations=int(matrices["iterations"].max()),
                               pairs=int(matrices["distances"].size),seconds=time.perf_counter()-started)
                    temporary = matrixfile.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary,**matrices,gallery_indices=gi,query_indices=qi,
                                        gallery_labels=labels[gi],query_labels=labels[qi])
                    temporary.replace(matrixfile)
                    job["matrix_sha256"] = hashlib.sha256(matrixfile.read_bytes()).hexdigest()
                    atomic_json(jobfile,job)
                # Re-evaluate cached scores; JSON is not an independent oracle.
                evaluation = evaluate_distances(matrices["distances"],labels[gi],labels[qi],ks=(1,))
                if evaluation != job["evaluation"] or job["parameters"] != p:
                    raise ValueError("Cached evaluation/parameters mismatch")
                details.append(job)
                completed += 1
                atomic_json(output/"run_state.json",dict(status="running",completed_jobs=completed,total_jobs=total_jobs))
                print(f"{completed}/{total_jobs} seed={split['seed']} {metric} c{candidate}: ACC1={100*job['ACC1']:.3f}% MAP={100*job['MAP']:.3f}% capped={job['capped_pairs']} ({job['seconds']:.1f}s)",flush=True)
    rows = []
    for metric in METRICS:
        for candidate,p in enumerate(grids[metric]):
            jobs = [j for j in details if j["metric"] == metric and j["candidate"] == candidate]
            capped = sum(j["capped_pairs"] for j in jobs)
            rows.append(dict(metric=metric,candidate=candidate,
                             mean_ACC1=float(np.mean([j["ACC1"] for j in jobs])),
                             sd_ACC1=float(np.std([j["ACC1"] for j in jobs],ddof=1)),
                             mean_MAP=float(np.mean([j["MAP"] for j in jobs])),
                             sd_MAP=float(np.std([j["MAP"] for j in jobs],ddof=1)),
                             capped_pairs=capped,eligible=capped==0,
                             max_marginal_l1=max(j["max_marginal_l1"] for j in jobs),
                             seconds=sum(j["seconds"] for j in jobs),parameters=p))
    selected = {m:choose_candidate([r for r in rows if r["metric"] == m]) for m in METRICS}
    failed = [m for m,row in selected.items() if row is None]
    presets = [r for r in rows if r["candidate"] == 0]
    tuned = [row for row in selected.values() if row is not None]
    _csv(output/"preset_results.csv",presets)
    _csv(output/"tuned_results.csv",tuned)
    _csv(output/"candidate_results.csv",rows)
    _tables(output/"comparison_report.md",presets,tuned)
    artifact = dict(schema=SCHEMA,status="failed" if failed else "completed",protocol=PROTOCOL,
                    dataset=args.dataset,training_sha256=origin["training_sha256"],test_used=False,
                    seeds=args.seeds,splits=splits,objective=["mean_ACC1","mean_MAP","earliest_candidate"],
                    solver_policy=policy,score_conventions={"flash-opw":"literal main Eq19", "tlp":"<P,D+wF>",
                    "tcot":"<P,D*(1+abs(delta))>","other_entropic":"<P,D>","soft-dtw":"raw soft DTW; no divergence"},
                    budget=args.budget,candidate_counts={m:len(g) for m,g in grids.items()},
                    grids=grids,selected=selected,presets={r["metric"]:r for r in presets},sources=sources,
                    search_seconds_by_metric={m:sum(r["seconds"] for r in rows if r["metric"]==m) for m in METRICS},
                    interpretation="pilot selection; training validation estimates are optimistic after search, not TEST generalization")
    atomic_json(output/"selected_all_metrics.json",artifact)
    atomic_json(output/"evaluations.json",details)
    atomic_json(output/"run_state.json",dict(status=artifact["status"],completed_jobs=completed,total_jobs=total_jobs,
                                            metrics_without_eligible_candidate=failed))
    if failed:
        raise RuntimeError(f"No eligible candidate for {failed}; inspect capped pairs before increasing a predeclared cap")
    print(f"Frozen all-metric selection: {output/'selected_all_metrics.json'}; TEST untouched",flush=True)


if __name__ == "__main__":
    main()
