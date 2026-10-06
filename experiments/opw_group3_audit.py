"""Recompute group-three quality/selection from NPZ matrices without re-solving."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .opw_group3 import METRICS, PROTOCOL, candidate_sets, load_frozen
from .opw_parameters import atomic_json
from .retrieval import evaluate_distances
from .sequence_data import load_training


def audit(output, *, data_root=Path("data/opw"), dataset_file=None):
    output = Path(output)
    selection = load_frozen(output/"selected_all_metrics.json")
    manifest = json.loads((output/"candidate_manifest.json").read_text(encoding="utf-8"))
    environment = json.loads((output/"environment.json").read_text(encoding="utf-8"))
    state = json.loads((output/"run_state.json").read_text(encoding="utf-8"))
    if state["status"] != "completed" or manifest["test_used"] is not False:
        raise ValueError("Incomplete run or TEST isolation missing")
    grids = candidate_sets(selection["dataset"],selection["budget"])
    if (selection["grids"] != grids or manifest["candidates"] != grids
            or manifest["protocol"] != PROTOCOL or manifest["splits"] != selection["splits"]
            or manifest["solver_policy"] != selection["solver_policy"]
            or manifest.get("accuracy_tie_decimals") != selection.get("accuracy_tie_decimals")
            or environment["signature"]["grids"] != grids):
        raise ValueError("Manifest/selection/declared search mismatch")
    _,labels,origin = load_training(selection["dataset"],data_root,dataset_file)
    if origin["training_sha256"] != selection["training_sha256"]:
        raise ValueError("Training data mismatch")
    tau = selection["solver_policy"]["tau"]
    cap = selection["solver_policy"]["max_iters"]
    every = selection["solver_policy"]["check_every"]
    jobs = []
    matrix_hashes = {}
    for split in selection["splits"]:
        gi,qi = np.array(split["gallery_indices"]),np.array(split["query_indices"])
        if set(gi) & set(qi):
            raise ValueError("TRAIN gallery/query overlap")
        for metric in METRICS:
            for candidate,p in enumerate(grids[metric]):
                key = f"seed{split['seed']}__{metric}__{candidate:02d}"
                job = json.loads((output/"jobs"/f"{key}.json").read_text(encoding="utf-8"))
                path = output/"matrices"/f"{key}.npz"
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if (digest != job["matrix_sha256"] or job["parameters"] != p
                        or job["metric"] != metric or job["candidate"] != candidate
                        or job["seed"] != split["seed"]):
                    raise ValueError(f"Job identity/hash mismatch: {key}")
                matrix_hashes[key] = digest
                with np.load(path,allow_pickle=False) as matrix:
                    for field,expected in (("gallery_indices",gi),("query_indices",qi),
                                           ("gallery_labels",labels[gi]),("query_labels",labels[qi])):
                        if not np.array_equal(matrix[field],expected):
                            raise ValueError(f"NPZ {field} mismatch: {key}")
                    distances,residuals,iterations = [matrix[k] for k in ("distances","residuals","iterations")]
                    if any(v.shape != (len(qi),len(gi)) or not np.isfinite(v).all()
                           for v in (distances,residuals,iterations)) or (residuals<0).any():
                        raise ValueError(f"Invalid matrices: {key}")
                    if ((iterations<0).any() or (iterations>cap).any()
                            or np.any(iterations != np.floor(iterations))
                            or np.any((iterations != cap) & (iterations % every != 0))):
                        raise ValueError(f"Invalid iteration checkpoints: {key}")
                    evaluation = evaluate_distances(distances,labels[gi],labels[qi],ks=(1,))
                    capped = int(np.count_nonzero(residuals>tau))
                    if (evaluation != job["evaluation"] or evaluation["MAP"] != job["MAP"]
                            or evaluation["ACC"]["1"] != job["ACC1"]
                            or capped != job["capped_pairs"]
                            or float(residuals.max()) != job["max_marginal_l1"]
                            or float(np.median(iterations)) != job["median_iterations"]
                            or int(iterations.max()) != job["max_iterations"]):
                        raise ValueError(f"Recomputed diagnostics/evaluation mismatch: {key}")
                jobs.append(job)
    if state["completed_jobs"] != len(jobs) or state["total_jobs"] != len(jobs):
        raise ValueError("Job count mismatch")
    # Rebuild means and eligibility from matrices before inspecting selected IDs.
    # This checks that the freeze followed the declared rule, not its TEST score.
    recomputed = {}
    all_rows, preset_rows, selected_rows = [], [], []
    for metric in METRICS:
        rows = []
        for candidate in range(len(grids[metric])):
            subset = [j for j in jobs if j["metric"] == metric and j["candidate"] == candidate]
            rows.append(dict(candidate=candidate,
                             mean_ACC1=float(np.mean([j["ACC1"] for j in subset])),
                             mean_MAP=float(np.mean([j["MAP"] for j in subset])),
                             sd_ACC1=float(np.std([j["ACC1"] for j in subset],ddof=1)),
                             sd_MAP=float(np.std([j["MAP"] for j in subset],ddof=1)),
                             capped_pairs=sum(j["capped_pairs"] for j in subset),
                             seconds=sum(j["seconds"] for j in subset),
                             max_marginal_l1=max(j["max_marginal_l1"] for j in subset)))
        valid = [r for r in rows if r["capped_pairs"] == 0]
        if not valid:
            raise ValueError(f"No eligible candidate: {metric}")
        decimals = selection.get("accuracy_tie_decimals")
        winner = sorted(valid,key=lambda r:(-(round(r["mean_ACC1"],decimals) if decimals is not None else r["mean_ACC1"]),
                                            -r["mean_MAP"],r["candidate"]))[0]
        for row in rows:
            row.update(metric=metric,eligible=row["capped_pairs"] == 0,
                       parameters=grids[metric][row["candidate"]])
        all_rows.extend(rows)
        preset_rows.append(rows[0])
        selected_rows.append(winner)
        for field,row in (("selected",winner),("presets",rows[0])):
            stored = selection[field][metric]
            if (any(stored[k] != row[k] for k in row) or stored["eligible"] != (row["capped_pairs"] == 0)
                    or stored["parameters"] != grids[metric][row["candidate"]]):
                raise ValueError(f"Frozen {field} mismatch: {metric}")
        recomputed[metric] = dict(selected_candidate=winner["candidate"],
                                 eligible_candidates=len(valid),attempted_candidates=len(rows))
    for name,expected in (("candidate_results.csv",all_rows),("preset_results.csv",preset_rows),
                          ("tuned_results.csv",selected_rows)):
        with (output/name).open(newline="",encoding="utf-8") as stream:
            stored = list(csv.DictReader(stream))
        if stored != [{k:str(v) for k,v in row.items()} for row in expected]:
            raise ValueError(f"Recomputed CSV mismatch: {name}")
    return dict(status="passed",jobs_verified=len(jobs),metrics_verified=len(METRICS),
                test_used=False,training_sha256=origin["training_sha256"],
                checks=["declared grids and equal tunable budgets", "TRAIN labels and indices", "NPZ hashes",
                        "AP, MAP and predictions for every job", "residuals, caps and iteration counters",
                        "three-split means/SD and ACC-first selection", "preset and selected parameters",
                        "all three result CSV tables"],
                selections=recomputed,matrix_sha256=matrix_hashes,
                limits="No independent re-solve or GPU numerical parity claim; CUDA tests and group1/2 supply solver verification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--data-root",type=Path,default=Path("data/opw"))
    parser.add_argument("--dataset-file",type=Path)
    args = parser.parse_args()
    result = audit(args.output,data_root=args.data_root,dataset_file=args.dataset_file)
    atomic_json(args.output/"audit.json",result)
    print(f"Audit {result['status']}: {result['jobs_verified']} matrices; all {result['metrics_verified']} metric selections verified")


if __name__ == "__main__":
    main()
