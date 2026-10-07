"""Audit complete group4 TEST artifacts without executing distance solvers."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .opw_group3 import load_frozen
from .opw_group4 import CPU_METRICS, analyze, configurations, load_chunk, report, result_rows, sha256
from .opw_parameters import atomic_json, source_hashes
from .retrieval import evaluate_distances
from .sequence_data import load_sequences, training_fingerprint


def audit(output, *, data_root=Path("data/opw"), dataset_file=None):
    output = Path(output)
    environment = json.loads((output/"environment.json").read_text(encoding="utf-8"))
    signature = environment["signature"]
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    state = json.loads((output/"run_state.json").read_text(encoding="utf-8"))
    selection = load_frozen(output/"frozen_selection.json")
    if (state["status"] not in ("completed","completed_with_nonconvergence")
            or summary["status"] != state["status"] or environment["test_used_for_selection"] is not False
            or summary["test_used_for_selection"] is not False):
        raise ValueError("Incomplete run or TEST selection isolation missing")
    # The runner's selection hash includes its original formatting; the copied
    # frozen selection is written canonically, so compare its semantic identity.
    settings = signature["settings"]
    profiles,metrics,ks = settings["profiles"],settings["metrics"],settings["ks"]
    configs = configurations(selection,profiles,metrics)
    if configs != signature["configs"] or selection["solver_policy"] != signature["policy"]:
        raise ValueError("Frozen configurations/policy mismatch")
    sources = source_hashes()
    for name in ("opw_group3.py","opw_group4.py","paired_statistics.py"):
        path = Path(__file__).with_name(name)
        sources[f"experiments/{name}"] = hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    if sources != signature["sources"] or sources != summary["sources"]:
        raise ValueError("Audit numerical sources mismatch")
    if any(sources.get(k) != v for k,v in selection["sources"].items()):
        raise ValueError("Frozen numerical sources mismatch")
    train,tl,test,ql,origin = load_sequences(selection["dataset"],data_root,dataset_file)
    origin = signature["origin"]
    if (training_fingerprint(train,tl) != selection["training_sha256"]
            or training_fingerprint(test,ql) != origin["test_sha256"]
            or origin["gallery_indices"] != list(range(len(train)))
            or origin["query_indices"] != list(range(len(test)))):
        raise ValueError("Full TRAIN/TEST identity mismatch")
    policy = selection["solver_policy"]
    evaluations,rows,quality,hashes = {},[],{},{}
    chunks = 0
    for config in configs:
        metric,key = config["metric"],config["key"]
        entropic = metric not in CPU_METRICS
        arrays = {k:np.empty((len(test),len(train))) for k in ("distances","residuals","iterations")}
        seconds = 0.
        for start in range(0,len(test),settings["query_chunk"]):
            stop = min(len(test),start+settings["query_chunk"])
            path = output/"chunks"/f"{key}__q{start:06d}.npz"
            chunk,job = load_chunk(path,path.with_suffix(".json"),
                identity=dict(metric=metric,parameters=config["parameters"],start=start,stop=stop),
                gallery_labels=tl,query_labels=ql[start:stop],policy=policy,entropic=entropic)
            for field in arrays:
                arrays[field][start:stop] = chunk[field]
            seconds += job["solve_wall_seconds"]
            chunks += 1
        matrixfile = output/"matrices"/f"{key}.npz"
        with np.load(matrixfile,allow_pickle=False) as saved:
            for field,expected in {**arrays,"gallery_indices":np.arange(len(train)),"query_indices":np.arange(len(test)),
                                   "gallery_labels":tl,"query_labels":ql}.items():
                if not np.array_equal(saved[field],expected):
                    raise ValueError("Complete matrix differs from audited chunks")
        ev = evaluate_distances(arrays["distances"],tl,ql,ks)
        unconverged = int(np.count_nonzero(arrays["residuals"]>policy["tau"])) if entropic else 0
        for alias in config["aliases"]:
            name = f"{alias['profile']}__{metric}"
            evaluations[name] = ev
            hashes[name] = dict(file=f"matrices/{matrixfile.name}",sha256=sha256(matrixfile))
            quality[name] = dict(unconverged_pairs=unconverged,entropic=entropic,pairs=len(train)*len(test))
        rows.extend(result_rows(config,ev,arrays,policy,device_type=settings["device"],seconds=seconds,ks=ks))
    comparisons = analyze(evaluations,ql,profiles,metrics,resamples=settings["bootstrap_resamples"],seed=settings["bootstrap_seed"])
    for row in comparisons:
        row["convergence_valid"] = all(quality[f"{row['profile']}__{m}"]["unconverged_pairs"] == 0 for m in ("flash-opw",row["right"]))
    expected_status = "completed_with_nonconvergence" if any(q["unconverged_pairs"] for q in quality.values()) else "completed"
    if (summary["matrix_hashes"] != hashes or summary["quality"] != quality or summary["status"] != expected_status
            or summary["gallery"] != len(train) or summary["queries"] != len(test)
            or summary["profiles"] != profiles or summary["metrics"] != metrics or summary["ks"] != ks
            or summary["primary_k"] != 1 or summary["policy"] != policy
            or summary["selection_sha256"] != signature["selection_sha256"]
            or any(s[k] != chunks for s in (summary,state) for k in ("completed_chunks","total_chunks"))):
        raise ValueError("Recomputed summary/status mismatch")
    for filename,expected in (("evaluations.json",evaluations),("paired_statistics.json",comparisons)):
        if json.loads((output/filename).read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Recomputed {filename} mismatch")
    for filename,expected in (("results.csv",rows),("paired_statistics.csv",comparisons)):
        if not expected:
            continue
        with (output/filename).open(newline="",encoding="utf-8") as stream:
            stored = list(csv.DictReader(stream))
        if stored != [{k:"" if v is None else str(v) for k,v in row.items()} for row in expected]:
            raise ValueError(f"Recomputed {filename} mismatch")
    expected_report = report(rows,comparisons,dataset=selection["dataset"],profiles=profiles,ks=ks,gallery=len(train),queries=len(test))
    if (output/"comparison_report.md").read_text(encoding="utf-8") != expected_report:
        raise ValueError("Recomputed report mismatch")
    return dict(status="passed",chunks_verified=chunks,unique_configurations=len(configs),
        gallery=len(train),queries=len(test),method_profile_evaluations=len(evaluations),
        full_test=True,test_used_for_selection=False,paired_comparisons_verified=len(comparisons),
        convergence_status=expected_status,
        checks=["frozen sources and parameters", "full TRAIN/TEST values and indices", "chunk hashes/checkpoints",
                "complete matrices versus chunks", "per-query AP and k-NN predictions", "MAP/ACC and residual diagnostics",
                "paired bootstrap / exact McNemar / Holm", "CSV tables and report"],
        limits="Artifact readback only; does not independently re-solve every coupling or measure GPU performance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--data-root",type=Path,default=Path("data/opw"))
    parser.add_argument("--dataset-file",type=Path)
    args = parser.parse_args()
    value = audit(args.output,data_root=args.data_root,dataset_file=args.dataset_file)
    atomic_json(args.output/"audit.json",value)
    print(json.dumps(value,indent=2))


if __name__ == "__main__":
    main()
