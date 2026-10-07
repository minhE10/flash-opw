"""Official full TEST evaluation with frozen all-metric TRAIN parameters.

Chunked, resumable distances; full-gallery MAP, predeclared k-NN ACC and
paired query statistics. GPU operations reuse the audited group3 solvers.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .opw_group3 import METRICS, UNTUNABLE, _csv, distance_matrix, load_frozen
from .opw_parameters import atomic_json, source_hashes
from .paired_statistics import holm_adjust, paired_query_statistics
from .retrieval import evaluate_distances
from .runtime import configure, metadata
from .sequence_data import load_sequences, training_fingerprint


DEFAULT_METRICS = ("flash-opw","opw","tlp","opw-kl","sinkhorn","tcot",
                   "dtw","ldtw","ndtw","soft-dtw","ot")
CPU_METRICS = (*UNTUNABLE,"soft-dtw")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configurations(selection, profiles, metrics):
    """Reuse identical preset/selected parameters within the same metric."""
    unique = {}
    for profile in profiles:
        for metric in metrics:
            row = selection["selected" if profile == "selected" else "presets"][metric]
            p = row["parameters"]
            if p != selection["grids"][metric][row["candidate"]]:
                raise ValueError("Frozen candidate/parameters mismatch")
            digest = hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
            key = f"{metric}__{digest}"
            unique.setdefault(key,dict(key=key,metric=metric,parameters=p,aliases=[]))["aliases"].append(
                dict(profile=profile,candidate=row["candidate"]))
    return list(unique.values())


def load_chunk(path, jobpath, *, identity, gallery_labels, query_labels, policy, entropic):
    job = json.loads(jobpath.read_text(encoding="utf-8"))
    if job["identity"] != identity or sha256(path) != job["matrix_sha256"]:
        raise ValueError("Cached chunk identity/hash mismatch")
    with np.load(path,allow_pickle=False) as saved:
        for field,expected in (("gallery_indices",np.arange(len(gallery_labels))),
                               ("query_indices",np.arange(identity["start"],identity["stop"])),
                               ("gallery_labels",gallery_labels),("query_labels",query_labels)):
            if not np.array_equal(saved[field],expected):
                raise ValueError(f"Cached chunk {field} mismatch")
        arrays = {k:saved[k] for k in ("distances","residuals","iterations")}
    shape = (len(query_labels),len(gallery_labels))
    if any(v.shape != shape or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Invalid cached chunk arrays")
    residuals,iterations = arrays["residuals"],arrays["iterations"]
    if (residuals<0).any() or (iterations != np.floor(iterations)).any():
        raise ValueError("Invalid cached residuals/iterations")
    if entropic:
        cap,every = policy["max_iters"],policy["check_every"]
        if ((iterations<1).any() or (iterations>cap).any()
                or np.any((iterations != cap)&(iterations % every != 0))
                or np.any((residuals>policy["tau"])&(iterations != cap))):
            raise ValueError("Invalid cached stopping checkpoints")
    elif np.any(residuals != 0) or np.any(iterations != 0):
        raise ValueError("Non-entropic methods must use zero diagnostic placeholders")
    return arrays,job


def save_chunk(path, jobpath, arrays, identity, gallery_labels, query_labels, seconds):
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise FloatingPointError("Nonfinite distances/residuals; no query may be dropped")
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,**arrays,gallery_indices=np.arange(len(gallery_labels)),
        query_indices=np.arange(identity["start"],identity["stop"]),gallery_labels=gallery_labels,query_labels=query_labels)
    temporary.replace(path)
    atomic_json(jobpath,dict(identity=identity,matrix_sha256=sha256(path),solve_wall_seconds=seconds))


def analyze(evaluations, labels, profiles, metrics, *, resamples, seed):
    rows = []
    for profile in profiles:
        family = []
        for metric in metrics:
            if metric == "flash-opw":
                continue
            stats = paired_query_statistics(evaluations[f"{profile}__flash-opw"],
                evaluations[f"{profile}__{metric}"],labels,resamples=resamples,seed=seed)
            family.append(dict(profile=profile,left="flash-opw",right=metric,
                               priority="primary" if metric in ("opw","tlp") else "secondary",**stats))
        corrected = holm_adjust([r["mcnemar_exact_two_sided_p"] for r in family])
        for row,p in zip(family,corrected):
            row["mcnemar_holm_p"] = p
            row["holm_family_size"] = len(family)
        rows.extend(family)
    return rows


def result_rows(config, evaluation, matrices, policy, *, device_type, seconds, ks):
    metric,p = config["metric"],config["parameters"]
    queries,gallery = matrices["distances"].shape
    entropic = metric not in CPU_METRICS
    unconverged = int(np.count_nonzero(matrices["residuals"]>policy["tau"])) if entropic else 0
    return [dict(profile=alias["profile"],metric=metric,candidate=alias["candidate"],k=k,
        ACC=evaluation["ACC"][str(k)],MAP=evaluation["MAP"],gallery=gallery,queries=queries,
        queries_without_relevant_gallery=evaluation["queries_without_relevant_gallery"],
        entropic=entropic,unconverged_pairs=unconverged,
        max_marginal_l1=float(matrices["residuals"].max()) if entropic else None,
        median_iterations=float(np.median(matrices["iterations"])) if entropic else None,
        max_iterations=int(matrices["iterations"].max()) if entropic else None,
        backend="CPU FP64" if device_type == "cpu" or not entropic else ("Flash IEEE FP32" if metric == "flash-opw" else "dense FP32"),
        solve_wall_seconds=seconds,parameters=json.dumps(p,sort_keys=True))
        for alias in config["aliases"] for k in ks]


def report(rows, comparisons, *, dataset, profiles, ks, gallery, queries):
    lines = [f"# Group 4: {dataset} official full TEST", "",
        f"All TRAIN gallery={gallery}; all TEST queries={queries}. Primary ACC@1; full-gallery MAP.",
        "Frozen parameters, main Eq19 for FlashOPW; original journal inverse/prior and <P,D> for OPW.",
        "Common marginal policy from TRAIN; no TEST tuning, query dropping or best-k selection.", ""]
    for profile in profiles:
        lines += [f"## {profile}", "", "| Metric | MAP % | "+" | ".join(f"ACC@{k} %" for k in ks)+" | Unconverged pairs |",
                  "|---|---:|"+"---:|"*len(ks)+"---:|"]
        for metric in dict.fromkeys(r["metric"] for r in rows):
            subset = [r for r in rows if r["profile"] == profile and r["metric"] == metric]
            values = {r["k"]:r for r in subset}
            row = subset[0]
            lines.append(f"| {metric} | {100*row['MAP']:.3f} | "+" | ".join(f"{100*values[k]['ACC']:.3f}" for k in ks)+f" | {row['unconverged_pairs'] if row['entropic'] else 'N/A'} |")
    lines += ["", "## Paired query comparisons: FlashOPW minus baseline", "",
        "| Profile | Baseline | ACC@1 difference pp | MAP difference pp (95% CI) | Exact McNemar p | Holm p |",
        "|---|---|---:|---:|---:|---:|"]
    for r in comparisons:
        lo,hi = r["map_difference_ci95_pp"]
        lines.append(f"| {r['profile']} | {r['right']} | {r['delta_ACC1_pp']:.3f} | {r['delta_MAP_pp']:.3f} ({lo:.3f}, {hi:.3f}) | {r['mcnemar_exact_two_sided_p']:.6g} | {r['mcnemar_holm_p']:.6g} |")
    lines += ["", "MAP CIs are unadjusted paired percentile query bootstrap intervals conditional on the fixed gallery/selection.",
        "McNemar is two-sided exact at predeclared k=1; Holm correction covers all requested baselines within each profile.",
        "Any unconverged entropic pairs make that method's quality comparison provisional; no post-TEST cap changes.",
        "The artifact chooses one parameter set per metric jointly over three TRAIN splits; query CIs do not estimate retraining variability.",
        "Recorded solve wall times include conversion, diagnostics and compilation; this is not the group5 speed benchmark.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection",type=Path,default=Path("reports/opw_group3_gpu_20261007/selected_all_metrics.json"))
    parser.add_argument("--data-root",type=Path,default=Path("data/opw"))
    parser.add_argument("--dataset-file",type=Path)
    parser.add_argument("--profiles",nargs="+",choices=("selected","preset"),default=["selected","preset"])
    parser.add_argument("--metrics",nargs="+",choices=METRICS,default=list(DEFAULT_METRICS))
    parser.add_argument("--ks",type=int,nargs="+",default=[1,3,5,7,15,30])
    parser.add_argument("--query-chunk",type=int,default=4)
    parser.add_argument("--batch",type=int,default=16)
    parser.add_argument("--memory-mib",type=int,default=128)
    parser.add_argument("--bootstrap-resamples",type=int,default=10000)
    parser.add_argument("--bootstrap-seed",type=int,default=20261007)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--memory-fraction",type=float,default=.45)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    if (min(args.query_chunk,args.batch,args.memory_mib,args.threads)<1 or args.bootstrap_resamples<100
            or args.bootstrap_seed<0 or 1 not in args.ks or "flash-opw" not in args.metrics
            or any(len(set(v)) != len(v) for v in (args.ks,args.metrics,args.profiles))):
        parser.error("Positive resources, unique settings, ACC@1, FlashOPW and >=100 bootstrap resamples required")
    selection = load_frozen(args.selection)
    configs = configurations(selection,args.profiles,args.metrics)
    # Validate frozen solver code and TRAIN identity before opening TEST.
    sources = source_hashes()
    for name in ("opw_group3.py","opw_group4.py","paired_statistics.py"):
        path = Path(__file__).with_name(name)
        sources[f"experiments/{name}"] = hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    if any(sources.get(k) != v for k,v in selection["sources"].items()):
        raise ValueError("Frozen numerical sources changed; no silent TEST retuning")
    from .sequence_data import load_training
    _,_,training = load_training(selection["dataset"],args.data_root,args.dataset_file)
    if training["training_sha256"] != selection["training_sha256"]:
        raise ValueError("Frozen TRAIN fingerprint mismatch")
    train,train_labels,test,test_labels,origin = load_sequences(selection["dataset"],args.data_root,args.dataset_file)
    if training_fingerprint(train,train_labels) != selection["training_sha256"]:
        raise ValueError("TRAIN data changed while loading TEST")
    if min(args.ks)<1 or max(args.ks)>len(train) or len(test)<2:
        parser.error("k must fit full gallery and statistics require >=2 TEST queries")
    policy = selection["solver_policy"]
    if (policy["schedule"] != "f_then_g" or policy["tau"]<=0
            or not np.isfinite(policy["tau"]) or min(policy["max_iters"],policy["check_every"])<1):
        raise ValueError("Invalid frozen convergence policy")
    os.environ["FLASHOPW_AUTOTUNE"] = "0"
    device = configure(args.device,args.threads,args.memory_fraction)
    environment = metadata(device)
    origin.update(training_sha256=selection["training_sha256"],test_sha256=training_fingerprint(test,test_labels),
        gallery_indices=list(range(len(train))),query_indices=list(range(len(test))),
        input_precision="round all values to FP32; identical rounded inputs to FP64 CPU and FP32 GPU")
    train,test = [[x.astype(np.float32).astype(np.float64) for x in data] for data in (train,test)]
    signature = dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ("output","resume")},
        sources=sources,selection_sha256=sha256(args.selection),configs=configs,policy=policy,origin=origin,
        torch=environment["torch"],python=environment["python"],packages=environment["packages"],
        gpu=environment.get("gpu"),torch_cuda=environment["torch_cuda"],kernel_controls=environment["kernel_controls"])
    args.output.mkdir(parents=True,exist_ok=args.resume)
    for name in ("chunks","matrices"):
        (args.output/name).mkdir(exist_ok=True)
    if args.resume:
        old = json.loads((args.output/"environment.json").read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise ValueError("Resume refused: data/code/selection/settings/environment changed")
    else:
        atomic_json(args.output/"environment.json",dict(environment,signature=signature,
            protocol="official TRAIN gallery / full official TEST queries; frozen TRAIN parameters",
            primary_comparisons=["flash-opw vs opw","flash-opw vs tlp"],primary_k=1,test_used_for_selection=False,
            stopping_policy=policy,backends="CUDA Flash IEEE FP32 and dense FP32; DTW/SoftDTW/OT CPU FP64; all CPU on --device cpu"))
        atomic_json(args.output/"data_manifest.json",origin)
        atomic_json(args.output/"frozen_selection.json",selection)
    rows,evaluations,matrix_hashes,quality = [],{},{},{}
    total = len(configs)*((len(test)+args.query_chunk-1)//args.query_chunk)
    completed = 0
    atomic_json(args.output/"run_state.json",dict(status="running",completed_chunks=0,total_chunks=total))
    print(f"Group4: {selection['dataset']} full TRAIN={len(train)}, TEST={len(test)}; {len(configs)} unique configurations; {device}; frozen tau={policy['tau']}",flush=True)
    try:
        for config in configs:
            metric,p,key = config["metric"],config["parameters"],config["key"]
            entropic = metric not in CPU_METRICS
            matrices = {k:np.empty((len(test),len(train))) for k in ("distances","residuals","iterations")}
            seconds = 0.
            for start in range(0,len(test),args.query_chunk):
                stop = min(len(test),start+args.query_chunk)
                identity = dict(metric=metric,parameters=p,start=start,stop=stop)
                path = args.output/"chunks"/f"{key}__q{start:06d}.npz"
                jobpath = path.with_suffix(".json")
                if not (args.resume and path.exists() and jobpath.exists()):
                    began = time.perf_counter()
                    arrays = distance_matrix(metric,test[start:stop],train,p,policy,device=device,
                                             batch=args.batch,memory_mib=args.memory_mib)
                    save_chunk(path,jobpath,arrays,identity,train_labels,test_labels[start:stop],time.perf_counter()-began)
                arrays,job = load_chunk(path,jobpath,identity=identity,gallery_labels=train_labels,
                    query_labels=test_labels[start:stop],policy=policy,entropic=entropic)
                for field in matrices:
                    matrices[field][start:stop] = arrays[field]
                seconds += job["solve_wall_seconds"]
                completed += 1
                atomic_json(args.output/"run_state.json",dict(status="running",metric=metric,queries_done=stop,
                    completed_chunks=completed,total_chunks=total))
                print(f"{metric}: {stop}/{len(test)} queries; chunks {completed}/{total}",flush=True)
            matrixfile = args.output/"matrices"/f"{key}.npz"
            temporary = matrixfile.with_suffix(".tmp.npz")
            np.savez_compressed(temporary,**matrices,gallery_indices=np.arange(len(train)),query_indices=np.arange(len(test)),
                gallery_labels=train_labels,query_labels=test_labels)
            temporary.replace(matrixfile)
            evaluation = evaluate_distances(matrices["distances"],train_labels,test_labels,args.ks)
            unconverged = int(np.count_nonzero(matrices["residuals"]>policy["tau"])) if entropic else 0
            for alias in config["aliases"]:
                evalkey = f"{alias['profile']}__{metric}"
                evaluations[evalkey] = evaluation
                matrix_hashes[evalkey] = dict(file=f"matrices/{matrixfile.name}",sha256=sha256(matrixfile))
                quality[evalkey] = dict(unconverged_pairs=unconverged,entropic=entropic,pairs=len(train)*len(test))
            rows.extend(result_rows(config,evaluation,matrices,policy,device_type=device.type,seconds=seconds,ks=args.ks))
            _csv(args.output/"results.csv",rows)
            atomic_json(args.output/"evaluations.json",evaluations)
            print(f"{metric} complete: ACC1={100*evaluation['ACC']['1']:.3f}% MAP={100*evaluation['MAP']:.3f}% unconverged={unconverged}",flush=True)
        comparisons = analyze(evaluations,test_labels,args.profiles,args.metrics,
                              resamples=args.bootstrap_resamples,seed=args.bootstrap_seed)
        for row in comparisons:
            row["convergence_valid"] = all(quality[f"{row['profile']}__{m}"]["unconverged_pairs"] == 0 for m in ("flash-opw",row["right"]))
        atomic_json(args.output/"paired_statistics.json",comparisons)
        if comparisons:
            _csv(args.output/"paired_statistics.csv",comparisons)
        status = "completed_with_nonconvergence" if any(q["unconverged_pairs"] for q in quality.values()) else "completed"
        atomic_json(args.output/"summary.json",dict(status=status,dataset=selection["dataset"],gallery=len(train),queries=len(test),
            test_used_for_selection=False,primary_k=1,ks=args.ks,profiles=args.profiles,metrics=args.metrics,
            selection_sha256=sha256(args.selection),sources=sources,policy=policy,quality=quality,matrix_hashes=matrix_hashes,
            completed_chunks=completed,total_chunks=total,
            limits="One aggregate TRAIN selection; query uncertainty conditional on fixed gallery; no native long-sequence or speed claim"))
        (args.output/"comparison_report.md").write_text(report(rows,comparisons,dataset=selection["dataset"],
            profiles=args.profiles,ks=args.ks,gallery=len(train),queries=len(test)),encoding="utf-8")
        atomic_json(args.output/"run_state.json",dict(status=status,completed_chunks=completed,total_chunks=total))
    except BaseException as exc:
        atomic_json(args.output/"run_state.json",dict(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",
            completed_chunks=completed,total_chunks=total,detail=f"{type(exc).__name__}: {exc}"))
        raise
    print(f"Group4 {status}: {args.output/'comparison_report.md'}",flush=True)


if __name__ == "__main__":
    main()
