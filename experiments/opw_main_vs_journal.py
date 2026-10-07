"""Compare main.pdf FlashOPW with the original journal OPW on frozen TRAIN splits.

Keep each method's native prior, parameters and distance. This is a model
comparison, not an equivalence test or an isolated Taylor ablation.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
import torch

from flashopw import OPWParameters
from .opw_group3 import _csv, cost_tensors, distance_matrix, load_frozen
from .opw_group3_audit import audit
from .opw_parameters import atomic_json, source_hashes
from .retrieval import evaluate_distances
from .runtime import configure, metadata
from .sequence_data import load_training
from .sequence_metrics import entropic_plan


METHODS = ("flash-opw", "opw")
SCORES = {"flash-opw": "main Eq19: a.f + b.g - lambda2 + q0",
          "opw": "journal Eq13: <P,D>"}


def literal_cost(metric, x, y, parameters):
    """Independent NumPy transcription, including the paper's constant.

    Journal F is perpendicular distance squared; main F is relative time
    squared. Sigma is used unchanged in the units of its own paper.
    """
    p = OPWParameters(**parameters)
    n, m = len(x), len(y)
    D = p.cost_scale * cdist(x, y, metric="sqeuclidean")
    delta = np.arange(1, n+1)[:, None]/n - np.arange(1, m+1)[None, :]/m
    relative = delta**2
    if metric == "flash-opw":
        cost = D + p.mu*relative + p.q0
    elif metric == "opw":
        perpendicular = relative/(1/n**2 + 1/m**2)
        cost = D - p.lambda1/(1+relative) + p.lambda2*(
            perpendicular/(2*p.sigma**2) + math.log(p.sigma*math.sqrt(2*math.pi)))
    else:
        raise ValueError("Only main FlashOPW and original journal OPW are compared")
    return cost, D, p


def oracle_check(metric, x, y, parameters, iterations, score):
    """Check centered implementation and score against literal FP64 equations.

    Main's Eq19 is evaluated with potentials for its centered augmented cost,
    then q0 is restored. Journal's score uses only the spatial ground cost.
    """
    literal, D, p = literal_cost(metric, x, y, parameters)
    centered, _, eps, _ = cost_tensors(metric, [x], [y], parameters)
    cost = centered[0].numpy()
    np.testing.assert_allclose(cost+p.q0, literal, rtol=1e-12, atol=1e-11)
    if metric == "opw":
        P = entropic_plan(literal, eps, iterations)
        expected = float(np.sum(P*D))
    else:
        _, f, g, a, b = entropic_plan(cost, eps, iterations, return_potentials=True)
        expected = float(a@f + b@g - eps + p.q0)
    np.testing.assert_allclose(score, expected, rtol=3e-3, atol=3e-4)
    return dict(expected_fp64=expected, actual=float(score),
                absolute_error=abs(float(score)-expected), iterations=int(iterations),
                literal_cost_check="passed", score_check="passed")


def summarize(rows):
    means = []
    for profile in dict.fromkeys(r["profile"] for r in rows):
        for mode in dict.fromkeys(r["mode"] for r in rows):
            for metric in METHODS:
                subset = [r for r in rows if (r["profile"],r["mode"],r["metric"]) == (profile,mode,metric)]
                means.append(dict(profile=profile,mode=mode,metric=metric,score=SCORES[metric],
                    mean_ACC1=float(np.mean([r["ACC1"] for r in subset])),
                    sd_ACC1=float(np.std([r["ACC1"] for r in subset],ddof=1)),
                    mean_MAP=float(np.mean([r["MAP"] for r in subset])),
                    sd_MAP=float(np.std([r["MAP"] for r in subset],ddof=1)),
                    infeasible_pairs=sum(r["infeasible_pairs"] for r in subset)))
    lines = ["# FlashOPW (main) versus original OPW (journal)", "",
             "Frozen three-split TRAIN pilot; original method definitions and native scores.",
             "Journal inverse penalty is exact; its perpendicular Gaussian prior is preserved.",
             "No sigma conversion, retuning, TEST evaluation or speed benchmark.", "",
             "| Profile | Mode | Method | ACC@1 % (SD) | MAP % (SD) | Infeasible pairs |",
             "|---|---|---|---:|---:|---:|"]
    for row in means:
        lines.append(f"| {row['profile']} | {row['mode']} | {row['metric']} | "
                     f"{100*row['mean_ACC1']:.3f} ({100*row['sd_ACC1']:.3f}) | "
                     f"{100*row['mean_MAP']:.3f} ({100*row['sd_MAP']:.3f}) | {row['infeasible_pairs']} |")
    lines += ["", "Infeasible pairs exceed the common marginal L1 tolerance; retain them visibly.",
              "Selected candidates were frozen separately for each metric before this check.",
              "The scores have different definitions: raw score differences/ratios are not parity errors.",
              "Prior units, Taylor approximation, epsilon, parameters and score can all change rankings.",
              "Paired query outcomes and NN-index agreement are in summary.json; agreement is not required.",
              "TRAIN estimates after tuning do not establish superiority on the official TEST split.", ""]
    return means, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection",type=Path,default=Path("reports/opw_group3_gpu_20261007/selected_all_metrics.json"))
    parser.add_argument("--data-root",type=Path,default=Path("data/opw"))
    parser.add_argument("--dataset-file",type=Path)
    parser.add_argument("--profiles",choices=("preset","selected"),nargs="+",default=["preset","selected"])
    parser.add_argument("--reuse-group3",type=Path,help="Verify and reuse completed group3 GPU/CPU matrices")
    parser.add_argument("--fixed-iters",type=int,help="Add a fixed-iteration check to a fresh solve")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--memory-fraction",type=float,default=.45)
    parser.add_argument("--batch",type=int,default=16)
    parser.add_argument("--memory-mib",type=int,default=128)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    if min(args.batch,args.memory_mib,args.threads) < 1 or len(set(args.profiles)) != len(args.profiles):
        parser.error("Positive resources and unique profiles required")
    if args.fixed_iters is not None and (args.fixed_iters < 1 or args.reuse_group3):
        parser.error("--fixed-iters requires a fresh solve and a positive iteration count")
    selection = load_frozen(args.selection)
    sequences,labels,origin = load_training(selection["dataset"],args.data_root,args.dataset_file)
    if selection["training_sha256"] != origin["training_sha256"]:
        raise ValueError("Frozen TRAIN data mismatch")
    sources = source_hashes()
    for name in ("opw_group3.py", "opw_main_vs_journal.py"):
        path = Path(__file__).with_name(name)
        sources[f"experiments/{name}"] = hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    if any(sources.get(k) != v for k,v in selection["sources"].items()):
        raise ValueError("Frozen solver sources changed; comparison requires an explicit new selection")
    policy = selection["solver_policy"]
    if policy["schedule"] != "f_then_g":
        raise ValueError("Unsupported frozen iteration schedule")
    os.environ["FLASHOPW_AUTOTUNE"] = "0"
    device = configure("cpu" if args.reuse_group3 else args.device,args.threads,args.memory_fraction)
    environment = metadata(device)
    imported_audit = None
    if args.reuse_group3:
        prior = load_frozen(args.reuse_group3/"selected_all_metrics.json")
        if prior != selection:
            raise ValueError("Reused run does not match frozen selection")
        imported_audit = audit(args.reuse_group3,data_root=args.data_root,dataset_file=args.dataset_file)
        environment["reused_environment"] = json.loads((args.reuse_group3/"environment.json").read_text(encoding="utf-8"))
    signature = dict(selection_sha256=hashlib.sha256(args.selection.read_bytes()).hexdigest(),
                     sources=sources,profiles=args.profiles,policy=policy,fixed_iters=args.fixed_iters,
                     reuse_group3=str(args.reuse_group3),origin=origin,device=str(device),
                     batch=args.batch,memory_mib=args.memory_mib,threads=args.threads,
                     memory_fraction=args.memory_fraction,torch=environment["torch"],
                     gpu=environment.get("gpu"),packages=environment["packages"])
    args.output.mkdir(parents=True,exist_ok=args.resume)
    (args.output/"matrices").mkdir(exist_ok=True)
    (args.output/"jobs").mkdir(exist_ok=True)
    if args.resume:
        old = json.loads((args.output/"environment.json").read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise ValueError("Resume refused: data/code/parameters/environment changed")
    else:
        atomic_json(args.output/"environment.json",dict(environment,signature=signature,test_used=False))
    rows,comparisons,oracles = [],[],[]
    for split in selection["splits"]:
        gi,qi = np.array(split["gallery_indices"]),np.array(split["query_indices"])
        if len(gi) == 0 or len(qi) == 0 or set(gi)&set(qi):
            raise ValueError("Nonempty disjoint TRAIN gallery/query required")
        gallery = [sequences[i].astype(np.float32).astype(np.float64) for i in gi]
        queries = [sequences[i].astype(np.float32).astype(np.float64) for i in qi]
        for profile in args.profiles:
            modes = ["residual_stop"] + ([f"fixed_{args.fixed_iters}"] if args.fixed_iters else [])
            for mode in modes:
                evaluations,matrices = {},{}
                for metric in METHODS:
                    frozen = selection["presets" if profile == "preset" else "selected"][metric]
                    p,candidate = frozen["parameters"],frozen["candidate"]
                    if p != selection["grids"][metric][candidate]:
                        raise ValueError("Frozen parameters/candidate mismatch")
                    OPWParameters(**p)
                    key = f"seed{split['seed']}__{profile}__{metric}__{mode}"
                    path = args.output/"matrices"/f"{key}.npz"
                    jobpath = args.output/"jobs"/f"{key}.json"
                    if args.reuse_group3:
                        oldkey = f"seed{split['seed']}__{metric}__{candidate:02d}"
                        path = args.reuse_group3/"matrices"/f"{oldkey}.npz"
                        jobpath = args.reuse_group3/"jobs"/f"{oldkey}.json"
                    if args.reuse_group3 or (args.resume and path.exists() and jobpath.exists()):
                        job = json.loads(jobpath.read_text(encoding="utf-8"))
                        if hashlib.sha256(path.read_bytes()).hexdigest() != job["matrix_sha256"] or job["parameters"] != p:
                            raise ValueError("Cached matrix hash/parameters mismatch")
                        with np.load(path,allow_pickle=False) as saved:
                            for field,expected in (("gallery_indices",gi),("query_indices",qi),
                                                   ("gallery_labels",labels[gi]),("query_labels",labels[qi])):
                                if not np.array_equal(saved[field],expected):
                                    raise ValueError("Cached split indices/labels mismatch")
                            values = {k:saved[k] for k in ("distances","residuals","iterations")}
                    else:
                        solve_policy = policy if mode == "residual_stop" else dict(
                            tau=1e-30,max_iters=args.fixed_iters,check_every=args.fixed_iters,schedule="f_then_g")
                        values = distance_matrix(metric,queries,gallery,p,solve_policy,device=device,
                                                 batch=args.batch,memory_mib=args.memory_mib)
                        np.savez_compressed(path,**values,gallery_indices=gi,query_indices=qi,
                                            gallery_labels=labels[gi],query_labels=labels[qi])
                        atomic_json(jobpath,dict(metric=metric,seed=split["seed"],parameters=p,
                            matrix_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    if any(v.shape != (len(qi),len(gi)) or not np.isfinite(v).all() for v in values.values()):
                        raise ValueError("Invalid comparison matrices")
                    iterations,residuals = values["iterations"],values["residuals"]
                    cap = policy["max_iters"] if mode == "residual_stop" else args.fixed_iters
                    if ((residuals<0).any() or (iterations<1).any() or (iterations>cap).any()
                            or np.any(iterations != np.floor(iterations))
                            or (mode != "residual_stop" and np.any(iterations != cap))):
                        raise ValueError("Invalid residuals/iterations")
                    evaluation = evaluate_distances(values["distances"],labels[gi],labels[qi],ks=(1,))
                    if evaluation["queries_without_relevant_gallery"]:
                        raise ValueError("Query has no relevant gallery sample")
                    evaluations[metric],matrices[metric] = evaluation,values
                    rows.append(dict(seed=split["seed"],profile=profile,mode=mode,metric=metric,
                        candidate=candidate,parameters=json.dumps(p,sort_keys=True),score=SCORES[metric],
                        ACC1=evaluation["ACC"]["1"],MAP=evaluation["MAP"],
                        infeasible_pairs=int(np.count_nonzero(residuals>policy["tau"])),
                        max_marginal_l1=float(residuals.max()),median_iterations=float(np.median(iterations)),
                        max_iterations=int(iterations.max()),matrix_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    # One fixed-iteration literal FP64 score check per matrix.
                    oracle = oracle_check(metric,queries[0],gallery[0],p,int(iterations[0,0]),values["distances"][0,0])
                    oracles.append(dict(seed=split["seed"],profile=profile,mode=mode,metric=metric,**oracle))
                    print(f"{key}: ACC1={100*evaluation['ACC']['1']:.3f}% MAP={100*evaluation['MAP']:.3f}% infeasible={rows[-1]['infeasible_pairs']}",flush=True)
                flash,original = [evaluations[m] for m in METHODS]
                fc = np.asarray(flash["predictions"]["1"]) == labels[qi].astype(str)
                oc = np.asarray(original["predictions"]["1"]) == labels[qi].astype(str)
                comparisons.append(dict(seed=split["seed"],profile=profile,mode=mode,
                    delta_ACC1_pp=100*(flash["ACC"]["1"]-original["ACC"]["1"]),
                    delta_MAP_pp=100*(flash["MAP"]-original["MAP"]),
                    both_correct=int(np.sum(fc&oc)),flash_only_correct=int(np.sum(fc&~oc)),
                    journal_only_correct=int(np.sum(~fc&oc)),both_wrong=int(np.sum(~fc&~oc)),
                    nn_index_agreement=float(np.mean(np.argmin(matrices["flash-opw"]["distances"],axis=1)
                                                     == np.argmin(matrices["opw"]["distances"],axis=1)))))
    means,report = summarize(rows)
    _csv(args.output/"split_results.csv",rows)
    _csv(args.output/"mean_results.csv",means)
    atomic_json(args.output/"summary.json",dict(status="completed",test_used=False,
        comparison="main FlashOPW versus original journal OPW; no sigma conversion",scores=SCORES,
        policy=policy,rows=rows,means=means,paired_comparisons=comparisons,formula_and_score_checks=oracles,
        reused_group3_audit=({k:imported_audit[k] for k in
            ("status","jobs_verified","metrics_verified","checks","limits")} if imported_audit else None),
        selection_sha256=signature["selection_sha256"],
        interpretation="TRAIN pilot after selection; different priors, parameters and scores; not Taylor-only causality"))
    (args.output/"comparison_report.md").write_text(report,encoding="utf-8")
    print(f"Comparison: {args.output/'comparison_report.md'}",flush=True)


if __name__ == "__main__":
    main()
