"""TRAIN-only prior, score and Taylor ablations; main PDF stays the primary model.

CPU runs explicit FP64 costs. CUDA additionally runs affine Flash IEEE FP32
and dense FP32 exact controls. Diagnostics materialize small pilot couplings;
this experiment makes no streaming-memory or GPU speed claim.
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

from flashopw import OPWParameters, materialize_opw_plan, opw_flash
from .opw_group1 import ranking_comparison
from .opw_parameters import atomic_json, load_selection, source_hashes
from .opw_tune import training_holdout
from .retrieval import evaluate_distances
from .runtime import configure, metadata
from .sequence_data import load_training


KINDS = ("affine", "exact_relative", "exact_journal")
SCORES = ("dual_score", "spatial_score", "affine_score")
FIELDS = (*SCORES, "mass", "row_l1", "col_l1", "iterations", "converged",
          "weighted_taylor_gap", "mean_relative_time_squared", "mass_outside_025",
          "plan_relative_l2_vs_affine", "plan_relative_l2_vs_fp64")


def frozen_profiles(selection):
    keys = ("lambda1", "lambda2", "sigma", "cost_scale")
    profiles = dict(default={k:selection["default_validation"]["parameters"][k] for k in keys},
                    tuned={k:selection["parameters"][k] for k in keys})
    for p in profiles.values():
        OPWParameters(**p)
    if any(profiles["default"][k] != profiles["tuned"][k]
           for k in ("lambda1", "lambda2", "cost_scale")):
        raise ValueError("Prior-only ablation requires the same lambda1, epsilon and cost_scale; only sigma may differ")
    return profiles


def cost_components(xs, ys, parameters, kind, *, device="cpu", dtype=torch.float64):
    """Center all costs by q0; exact inverse uses the same relative E as main.

    Only exact_journal changes Gaussian F to perpendicular-distance squared.
    C_affine - C_exact_relative = lambda1*F^2/(1+F) >= 0 exactly.
    """
    if kind not in KINDS:
        raise ValueError("Unknown ablation cost")
    p = OPWParameters(**parameters)
    x = torch.as_tensor(np.stack(xs), dtype=dtype, device=device)
    y = torch.as_tensor(np.stack(ys), dtype=dtype, device=device)
    n, m = x.shape[1], y.shape[1]
    spatial = p.cost_scale*torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist").square()
    t = torch.arange(1,n+1,dtype=dtype,device=device)/n
    s = torch.arange(1,m+1,dtype=dtype,device=device)/m
    relative = (t[:,None]-s[None,:]).square()
    affine = spatial+p.mu*relative
    gap = p.lambda1*relative.square()/(1+relative)
    if kind == "affine":
        cost = affine
    else:
        prior = relative if kind == "exact_relative" else relative/(1/n**2+1/m**2)
        cost = spatial+p.lambda1*relative/(1+relative)+p.lambda2*prior/(2*p.sigma**2)
    return p, spatial, relative, affine, gap, cost


def plan_statistics(plan, f, g, components, iteration, tau):
    p, spatial, relative, affine, gap, _ = components
    n, m = plan.shape[-2:]
    rows, columns = plan.sum(-1), plan.sum(-2)
    row_l1 = (rows-1/n).abs().sum(-1)
    col_l1 = (columns-1/m).abs().sum(-1)
    values = dict(dual_score=f.mean(-1)+g.mean(-1)-p.lambda2+p.q0,
                  spatial_score=(plan*spatial).sum((-1,-2)),
                  affine_score=(plan*affine).sum((-1,-2)),
                  mass=plan.sum((-1,-2)), row_l1=row_l1, col_l1=col_l1,
                  iterations=torch.full_like(row_l1,iteration),
                  converged=(torch.maximum(row_l1,col_l1)<=tau).to(plan.dtype),
                  weighted_taylor_gap=(plan*gap).sum((-1,-2)),
                  mean_relative_time_squared=(plan*relative).sum((-1,-2)),
                  mass_outside_025=(plan*(relative>.25**2)).sum((-1,-2)))
    if any(not bool(torch.isfinite(value).all()) for value in values.values()):
        raise FloatingPointError("Nonfinite ablation score/marginal; failed pairs cannot be dropped")
    return values


@torch.no_grad()
def dense_sweep(xs, ys, parameters, kind, fixed_iters, cap, check_every, tau,
                *, device="cpu", dtype=torch.float64):
    """Incremental fixed checkpoints plus each pair's first observed tau stop.

    Fixed checkpoints continue the original zero-init recurrence even after a
    pair's stopping snapshot was captured. Capped pairs remain in evaluation.
    """
    if (not fixed_iters or list(fixed_iters)!=sorted(set(fixed_iters))
            or min(*fixed_iters,cap,check_every)<1 or cap<max(fixed_iters)
            or not math.isfinite(tau) or tau<=0):
        raise ValueError("Increasing positive checkpoints, cap >= checkpoints and positive stopping settings required")
    components = cost_components(xs,ys,parameters,kind,device=device,dtype=dtype)
    p,_,_,_,_,cost = components
    batch,n,m = cost.shape
    f, g = cost.new_zeros((batch,n)), cost.new_zeros((batch,m))
    stopped = torch.zeros(batch,dtype=torch.bool,device=cost.device)
    adaptive_values, adaptive_plan = None, None
    output = {}
    for it in range(1,cap+1):
        f = -p.lambda2*torch.logsumexp((g[:,None,:]-cost)/p.lambda2-math.log(m),dim=2)
        g = -p.lambda2*torch.logsumexp((f[:,:,None]-cost)/p.lambda2-math.log(n),dim=1)
        if it in fixed_iters or it%check_every==0 or it==cap:
            plan = ((f[:,:,None]+g[:,None,:]-cost)/p.lambda2-math.log(n)-math.log(m)).exp()
            values = plan_statistics(plan,f,g,components,it,tau)
            if it in fixed_iters:
                output[f"fixed_{it}"] = (values,plan)
            # Stop checks are exactly check_every and cap, not incidental fixed checkpoints.
            if it%check_every==0 or it==cap:
                if adaptive_plan is None:
                    adaptive_plan = torch.empty_like(plan)
                    adaptive_values = {k:torch.empty_like(v) for k,v in values.items()}
                take = ~stopped & ((values["converged"]>0) | (it==cap))
                adaptive_plan[take] = plan[take]
                for k,v in values.items():
                    adaptive_values[k][take] = v[take]
                stopped |= take
            if it>=max(fixed_iters) and bool(stopped.all()):
                break
    output["residual_stop"] = (adaptive_values,adaptive_plan)
    return output


@torch.no_grad()
def flash_sweep(xs,ys,parameters,fixed_iters,cap,check_every,tau,device):
    """Actual Flash solves; dense coupling recovery is diagnostic only."""
    output = {}
    for mode,it in [(f"fixed_{it}",it) for it in fixed_iters]+[("residual_stop",cap)]:
        columns, plans = {}, []
        for x,y in zip(xs,ys):
            tx,ty = (torch.as_tensor(v,dtype=torch.float32,device=device) for v in (x,y))
            result = opw_flash(tx,ty,**parameters,n_iters=it,precision="ieee",
                               tol=tau if mode=="residual_stop" else None,check_every=check_every)
            plan = materialize_opw_plan(result)[None]
            components = cost_components([x],[y],parameters,"affine",device=device,dtype=torch.float32)
            values = plan_statistics(plan,result.f[None],result.g[None],components,result.sinkhorn.n_iters,tau)
            values["dual_score"] = result.loss[None]
            for k,v in values.items():
                columns.setdefault(k,[]).append(v)
            plans.append(plan)
        output[mode] = ({k:torch.cat(v) for k,v in columns.items()},torch.cat(plans))
    return output


def summarize_matrices(matrices, models, modes, backends, gallery_labels, query_labels, ks):
    rows, evaluations, comparisons = [], {}, []
    for backend in backends:
        for model in models:
            for mode in modes:
                values = matrices[backend,model,mode]
                residual = np.maximum(values["row_l1"],values["col_l1"])
                for score in SCORES:
                    ev = evaluate_distances(values[score],gallery_labels,query_labels,ks)
                    evaluations[backend,model,mode,score] = ev
                    rows.append(dict(backend=backend,model=model,mode=mode,score=score,
                                     MAP=ev["MAP"],**{f"ACC_{k}":v for k,v in ev["ACC"].items()},
                                     converged_fraction=float(values["converged"].mean()),
                                     max_marginal_l1=float(residual.max()),
                                     p95_marginal_l1=float(np.quantile(residual,.95)),
                                     median_iterations=float(np.median(values["iterations"])),
                                     max_iterations=int(values["iterations"].max()),
                                     mean_weighted_taylor_gap=float(values["weighted_taylor_gap"].mean()),
                                     max_plan_relative_l2_vs_affine=float(values["plan_relative_l2_vs_affine"].max()),
                                     max_plan_relative_l2_vs_fp64=float(values["plan_relative_l2_vs_fp64"].max())))

        def contrast(name,left,right):
            le,re = evaluations[(backend,*left)],evaluations[(backend,*right)]
            lv,rv = matrices[(backend,*left[:2])],matrices[(backend,*right[:2])]
            comparison = ranking_comparison(rv[right[2]],lv[left[2]])
            comparisons.append(dict(backend=backend,contrast=name,left=list(left),right=list(right),
                delta_MAP_pp=100*(re["MAP"]-le["MAP"]),
                delta_ACC_1_pp=100*(re["ACC"]["1"]-le["ACC"]["1"]),
                nn_agreement=comparison["nn_agreement"],full_ranking_agreement=comparison["full_ranking_agreement"],
                both_satisfy_tau=bool(lv["converged"].all() and rv["converged"].all())))

        for mode in modes:
            for score in SCORES:
                contrast("prior_strength",("affine_default",mode,score),("affine_tuned",mode,score))
                for profile in ("default","tuned"):
                    contrast("Taylor_inverse",(f"affine_{profile}",mode,score),(f"exact_relative_{profile}",mode,score))
                    contrast("journal_prior_geometry",(f"exact_relative_{profile}",mode,score),(f"exact_journal_{profile}",mode,score))
            for model in models:
                for score in ("spatial_score","affine_score"):
                    contrast("score_same_coupling",(model,mode,"dual_score"),(model,mode,score))
        if "fixed_100" in modes and "fixed_200" in modes:
            contrast("TLp_score_at_200",("affine_tuned","fixed_200","dual_score"),
                     ("affine_tuned","fixed_200","affine_score"))
            contrast("TLp_iterations_100_to_200",("affine_tuned","fixed_100","affine_score"),
                     ("affine_tuned","fixed_200","affine_score"))
            contrast("TLp_iterations_200_to_residual",("affine_tuned","fixed_200","affine_score"),
                     ("affine_tuned","residual_stop","affine_score"))
    return rows,evaluations,comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",default="FacesUCR")
    parser.add_argument("--dataset-file",type=Path)
    parser.add_argument("--data-root",type=Path,default=Path("data/opw"))
    parser.add_argument("--flash-parameters",type=Path,default=Path("reports/opw_tuning_20261006/selected_parameters.json"))
    parser.add_argument("--gallery",type=int,default=16)
    parser.add_argument("--queries",type=int,default=14)
    parser.add_argument("--seed",type=int,default=20261006)
    parser.add_argument("--fixed-iters",type=int,nargs="+",default=[100,200])
    parser.add_argument("--tau",type=float,default=1e-3)
    parser.add_argument("--max-iters",type=int,default=4000)
    parser.add_argument("--check-every",type=int,default=50)
    parser.add_argument("--batch",type=int,default=16)
    parser.add_argument("--memory-mib",type=int,default=128)
    parser.add_argument("--max-dense-entries",type=int,default=4_194_304)
    parser.add_argument("--ks",type=int,nargs="+",default=[1,3,5,7,15])
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--memory-fraction",type=float,default=.45)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    if (min(args.gallery,args.queries)<0 or min(*args.fixed_iters,args.max_iters,args.check_every,
            args.batch,args.memory_mib,args.max_dense_entries)<1 or args.max_iters<max(args.fixed_iters)
            or args.fixed_iters!=sorted(set(args.fixed_iters)) or not math.isfinite(args.tau) or args.tau<=0):
        parser.error("Positive settings, distinct increasing checkpoints <= cap and finite positive tau required")
    sequences,labels,origin = load_training(args.dataset,args.data_root,args.dataset_file)
    selection = load_selection(args.flash_parameters,dataset=args.dataset,
                               training_sha256=origin["training_sha256"],score="pdf-loss")
    profiles = frozen_profiles(selection)
    gi,qi = training_holdout(labels,args.gallery,args.queries,args.seed)
    if len(gi)<2 or not args.ks or 1 not in args.ks or min(args.ks)<1 or max(args.ks)>len(gi):
        parser.error("At least two gallery items and valid ks including 1 required; use --ks for smaller galleries")
    gallery = [sequences[i].astype(np.float32).astype(np.float64) for i in gi]
    queries = [sequences[i].astype(np.float32).astype(np.float64) for i in qi]
    models = {f"{kind}_{name}":dict(kind=kind,parameters=p) for name,p in profiles.items() for kind in KINDS}
    groups = {}
    for i,x in enumerate(queries):
        for j,y in enumerate(gallery):
            if len(x)*len(y)>args.max_dense_entries or 8*32*len(x)*len(y)>args.memory_mib*1024**2:
                parser.error("Pair exceeds pilot dense/memory budget; group2 is not the long-sequence benchmark")
            groups.setdefault((x.shape,y.shape),[]).append((i,j))
    batches = []
    for (xs,ys),pairs in groups.items():
        size = min(args.batch,max(1,args.memory_mib*1024**2//(8*32*xs[0]*ys[0])))
        batches.extend(pairs[start:start+size] for start in range(0,len(pairs),size))
    os.environ["FLASHOPW_AUTOTUNE"] = "0"
    device = configure(args.device,args.threads,args.memory_fraction)
    environment = metadata(device)
    sources = source_hashes()
    for name in ("opw_group1.py","opw_group2.py"):
        sources[f"experiments/{name}"] = hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ("output","resume")}
    signature = dict(settings=settings,selection=selection,origin=origin,models=models,
                     gallery_indices=gi.tolist(),query_indices=qi.tolist(),sources=sources,
                     input_sha256=hashlib.sha256(b"".join(x.tobytes() for x in queries+gallery)).hexdigest(),
                     torch=environment["torch"],packages=environment["packages"],python=environment["python"],
                     gpu=environment.get("gpu"),kernel_controls=environment["kernel_controls"])
    output = args.output
    output.mkdir(parents=True,exist_ok=args.resume)
    (output/"chunks").mkdir(exist_ok=True)
    if args.resume:
        prior = json.loads((output/"environment.json").read_text(encoding="utf-8"))
        if prior["signature"]!=signature:
            raise ValueError("Resume refused: code/data/settings/environment/selection changed")
    else:
        environment.update(signature=signature,protocol=dict(
            scope="Exploratory TRAIN holdout reused from group1, not independent TEST quality or model selection",
            primary="main PDF affine model, dual_score is literal Eq19; no solver/default/selection mutation",
            exact="exact_relative restores 1/(1+F) only; exact_journal additionally uses F/(1/N^2+1/M^2) in Gaussian prior",
            scores="dual_score uses Eq19-style f/g expression for exact controls; spatial=<P,D>; affine=<P,D+mu*F> for ALL couplings",
            constants="All solvers center q0 out of cost and restore it only in dual score; uniform a,b; f then g",
            stopping="first observed check_every checkpoint satisfying BOTH L1 marginals; cap retained and labelled, no failed-pair exclusion",
            backends="CPU explicit FP64 reference; CUDA adds Flash IEEE FP32 affine and dense Torch FP32 exact controls",
            timing="not a runtime benchmark; dense diagnostic coupling recovery included",
            tlp="matched TLp is the SAME affine coupling, <P,D+mu*F>; preset identity only if mu=50 and epsilon=.1",
            score_reuse="all three scores from ONE coupling per model/pair/mode, no score-specific re-solve"))
        atomic_json(output/"environment.json",environment)
    atomic_json(output/"run_state.json",dict(status="running",completed_chunks=0,total_chunks=len(batches)))
    backends = ["CPU_FP64"]+(["GPU_FP32"] if device.type=="cuda" else [])
    modes = [f"fixed_{it}" for it in args.fixed_iters]+["residual_stop"]
    matrices = {(b,model,mode):{k:np.empty((len(qi),len(gi))) for k in FIELDS}
                for b in backends for model in models for mode in modes}
    failures = []
    print(f"Group2: TRAIN-only {len(qi)} queries x {len(gi)} gallery; {device}; 6 costs x 3 scores; tau={args.tau}, cap={args.max_iters}",flush=True)
    for index,pairs in enumerate(batches):
        cache = output/"chunks"/f"{index:04d}.npz"
        if not (args.resume and cache.exists()):
            xs,ys = [queries[i] for i,j in pairs],[gallery[j] for i,j in pairs]
            payload = {}
            for name,p in profiles.items():
                affine_ref,affine_gpu = None,None
                for kind in KINDS:
                    model = f"{kind}_{name}"
                    ref = dense_sweep(xs,ys,p,kind,args.fixed_iters,args.max_iters,args.check_every,args.tau)
                    if kind=="affine":
                        affine_ref=ref
                    gpu = None
                    if device.type=="cuda":
                        gpu = (flash_sweep(xs,ys,p,args.fixed_iters,args.max_iters,args.check_every,args.tau,device)
                               if kind=="affine" else dense_sweep(xs,ys,p,kind,args.fixed_iters,args.max_iters,
                               args.check_every,args.tau,device=device,dtype=torch.float32))
                        if kind=="affine":
                            affine_gpu=gpu
                    for backend,sweep,affine in [("CPU_FP64",ref,affine_ref)]+([("GPU_FP32",gpu,affine_gpu)] if gpu else []):
                        for mode,(values,plan) in sweep.items():
                            fp64 = ref[mode][1].to(plan.device,dtype=torch.float64)
                            current = plan.double()
                            values["plan_relative_l2_vs_affine"] = (plan-affine[mode][1]).flatten(1).norm(dim=1)/affine[mode][1].flatten(1).norm(dim=1).clamp_min(1e-30)
                            values["plan_relative_l2_vs_fp64"] = (current-fp64).flatten(1).norm(dim=1)/fp64.flatten(1).norm(dim=1).clamp_min(1e-30)
                            for k,v in values.items():
                                payload[f"{backend}__{model}__{mode}__{k}"] = v.double().cpu().numpy()
            payload["pairs"] = np.asarray(pairs)
            temporary = cache.with_suffix(".tmp.npz")
            np.savez_compressed(temporary,**payload)
            temporary.replace(cache)
        with np.load(cache,allow_pickle=False) as saved:
            if not np.array_equal(saved["pairs"],pairs):
                raise ValueError("Cached chunk pair indices mismatch")
            for backend in backends:
                for model in models:
                    for mode in modes:
                        for k in FIELDS:
                            values = saved[f"{backend}__{model}__{mode}__{k}"]
                            if not np.isfinite(values).all():
                                raise FloatingPointError("Nonfinite cached chunk")
                            for pos,(i,j) in enumerate(pairs):
                                matrices[backend,model,mode][k][i,j]=values[pos]
        atomic_json(output/"run_state.json",dict(status="running",completed_chunks=index+1,total_chunks=len(batches)))
        print(f"Group2: chunk {index+1}/{len(batches)} complete ({len(pairs)} pairs, all six models)",flush=True)
    rows,evaluations,contrasts = summarize_matrices(matrices,models,modes,backends,labels[gi],labels[qi],args.ks)
    for (backend,model,mode),values in matrices.items():
        np.savez_compressed(output/f"{backend}_{model}_{mode}.npz",**values,
                            gallery_indices=gi,query_indices=qi,gallery_labels=labels[gi],query_labels=labels[qi])
        if backend=="GPU_FP32" and mode!="residual_stop":
            ref=matrices["CPU_FP64",model,mode]
            if (any(not np.allclose(values[k],ref[k],rtol=3e-3,atol=3e-4) for k in SCORES)
                    or values["plan_relative_l2_vs_fp64"].max()>5e-3):
                failures.append(dict(model=model,mode=mode))
    atomic_json(output/"evaluations.json",{"__".join(key):value for key,value in evaluations.items()})
    atomic_json(output/"contrasts.json",contrasts)
    with (output/"ablation_results.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    capped={"__".join(key):int(np.count_nonzero(values["converged"]==0))
            for key,values in matrices.items() if key[-1]=="residual_stop"}
    tuned=OPWParameters(**profiles["tuned"])
    summary=dict(status="failed" if failures else "completed",numerical_parity_failures=failures,
        flash_verification="executed" if device.type=="cuda" else "not_run_CPU_only",
        queries=len(qi),gallery=len(gi),tau=args.tau,cap=args.max_iters,capped_pairs=capped,results=rows,
        tlp_bridge=dict(mu=tuned.mu,epsilon=tuned.lambda2,
                        matches_legacy_weight50_epsilon01=math.isclose(tuned.mu,50,abs_tol=1e-12) and tuned.lambda2==.1,
                        score="affine_tuned/affine_score is matched TLp; SAME coupling, fixed_100 vs fixed_200 isolates iterations"),
        limitations="TRAIN exploratory pilot, reused tuning holdout; capped controls are provisional; exact models are dense; no TEST tuning or speed claim")
    atomic_json(output/"summary.json",summary)
    atomic_json(output/"run_state.json",dict(status=summary["status"],completed_chunks=len(batches),
        numerical_parity_failures=len(failures),capped_pairs=capped,flash_verification=summary["flash_verification"]))
    for row in rows:
        if row["mode"]=="residual_stop":
            print(f"{row['backend']} {row['model']} {row['score']}: MAP={row['MAP']:.3%}, ACC1={row['ACC_1']:.3%}, converged={row['converged_fraction']:.3%}",flush=True)
    print(f"Results: {output.resolve()}; numerical_parity_failures={len(failures)}; capped_pairs={capped}",flush=True)
    if failures:
        raise SystemExit(1)


if __name__=="__main__":
    main()
