"""Select affine FlashOPW parameters using a training-only stratified holdout."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from flashopw import OPWParameters, opw_diagnostics, opw_distance, opw_flash
from .opw_parameters import atomic_json, dense_score_matrix, source_hashes
from .retrieval import evaluate_distances
from .runtime import configure, metadata
from .sequence_data import balanced_subset, load_training


def training_holdout(labels, gallery_limit, validation_limit, seed):
    """Reserve at least one validation and gallery sample per training class."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    gallery, validation = [], []
    for label in np.unique(labels):
        ids = rng.permutation(np.flatnonzero(labels == label))
        if len(ids) < 2:
            raise ValueError(f"Class {label!r} needs at least two training samples")
        count = max(1, min(len(ids)-1, int(math.ceil(0.25 * len(ids)))))
        validation.extend(ids[:count])
        gallery.extend(ids[count:])
    gallery, validation = np.sort(gallery), np.sort(validation)
    gallery = gallery[balanced_subset(labels[gallery], gallery_limit, seed+1)]
    validation = validation[balanced_subset(labels[validation], validation_limit, seed+2)]
    return gallery, validation


def candidate_grid(defaults, mus, epsilons):
    """Include the unchanged default first; do not search redundant lambda1/sigma."""
    candidates = [dict(defaults)]
    seen = {(round(OPWParameters(**{k:defaults[k] for k in ("lambda1", "lambda2", "sigma", "cost_scale")}).mu, 12),
             defaults["lambda2"])}
    for mu in mus:
        for eps in epsilons:
            if not math.isfinite(mu) or mu <= 0 or not math.isfinite(eps) or eps <= 0:
                raise ValueError("mu and epsilon candidates must be finite and positive")
            key = round(mu, 12), eps
            if key in seen:
                continue
            lambda1 = defaults["lambda1"] if mu > defaults["lambda1"] else mu/2
            sigma = math.sqrt(eps / (2 * (mu-lambda1)))
            candidates.append(dict(defaults, lambda1=lambda1, lambda2=eps, sigma=sigma))
            seen.add(key)
    return candidates


def select_candidate(rows, objective="ACC"):
    """Validation primary metric, other metric breaks ties; default wins exact ties."""
    if objective not in ("ACC", "MAP") or not rows:
        raise ValueError("Nonempty results and objective ACC/MAP required")
    secondary = "MAP" if objective == "ACC" else "ACC"
    return max(range(len(rows)), key=lambda i:(rows[i][objective], rows[i][secondary], -i))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="FacesUCR")
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/opw"))
    parser.add_argument("--gallery", type=int, default=32)
    parser.add_argument("--validation", type=int, default=28)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--mus", type=float, nargs="+", default=[0.1, 1.05, 10, 50, 200, 430])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    parser.add_argument("--lambda1", type=float)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1)
    parser.add_argument("--cost-scale", type=float, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--selection-k", type=int, default=1)
    parser.add_argument("--objective", choices=("ACC", "MAP"), default="ACC")
    parser.add_argument("--score", choices=("pdf-loss", "spatial"), default="pdf-loss")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.gallery, args.validation) < 0 or min(args.iters, args.selection_k) < 1:
        parser.error("Nonnegative subset limits, positive iterations and k required")
    default_lambda1 = {"FacesUCR": 1., "FaceAll": 10.}.get(args.dataset, 50.)
    defaults = dict(lambda1=default_lambda1 if args.lambda1 is None else args.lambda1,
                    lambda2=args.lambda2, sigma=args.sigma, cost_scale=args.cost_scale, n_iters=args.iters)
    candidates = candidate_grid(defaults, args.mus, args.epsilons)
    sequences, labels, origin = load_training(args.dataset, args.data_root, args.dataset_file)
    gi, vi = training_holdout(labels, args.gallery, args.validation, args.seed)
    if args.selection_k > len(gi):
        parser.error("selection-k exceeds training gallery")
    os.environ["FLASHOPW_AUTOTUNE"] = "0"
    device = configure(args.device, args.threads, args.memory_fraction)
    environment = metadata(device)
    settings = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items() if k not in ("output", "resume")}
    signature = dict(settings=settings, origin=origin, gallery_indices=gi.tolist(), validation_indices=vi.tolist(),
                     candidates=candidates, sources=source_hashes(), packages=environment["packages"],
                     torch=environment["torch"], python=environment["python"], gpu=environment.get("gpu"),
                     kernel_controls=environment["kernel_controls"])
    output = args.output
    output.mkdir(parents=True, exist_ok=args.resume)
    rows = []
    if args.resume:
        prior = json.loads((output/"search.json").read_text(encoding="utf-8"))
        if prior["signature"] != signature:
            raise ValueError("Resume refused: training data, code, environment or search settings changed")
        rows = prior["results"]
    atomic_json(output/"environment.json", environment)
    atomic_json(output/"search.json", dict(signature=signature, results=rows, status="running"))
    gallery, queries = [sequences[i] for i in gi], [sequences[i] for i in vi]
    print(f"Training only: gallery={len(gi)}, validation={len(vi)}, candidates={len(candidates)}, k={args.selection_k}", flush=True)
    if device.type == "cuda":
        tensors = ([torch.as_tensor(x, dtype=torch.float32, device=device) for x in queries],
                   [torch.as_tensor(x, dtype=torch.float32, device=device) for x in gallery])
    for index in range(len(rows), len(candidates)):
        p = candidates[index]
        started = time.perf_counter()
        if device.type == "cpu":
            distances, residuals = dense_score_matrix(queries, gallery, p, score=args.score)
            diagnostics = dict(max_marginal_l1=float(residuals.max()), pairs=residuals.size)
        else:
            def operation(i, j):
                return opw_flash(tensors[0][i], tensors[1][j], **p, precision="ieee")
            operation(0, 0)  # compile outside timing
            torch.cuda.synchronize()
            started = time.perf_counter()
            distances = np.empty((len(vi), len(gi)))
            for i in range(len(vi)):
                for j in range(len(gi)):
                    result = operation(i, j)
                    distances[i,j] = float(result.loss if args.score == "pdf-loss" else opw_distance(result))
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            stats = [opw_diagnostics(operation(0,j)) for j in range(min(8, len(gi)))]
            diagnostics = dict(max_marginal_l1=max(max(s["row_l1"], s["col_l1"]) for s in stats), pairs=len(stats))
        elapsed = elapsed if device.type == "cuda" else time.perf_counter()-started
        evaluation = evaluate_distances(distances, labels[gi], labels[vi], [args.selection_k])
        row = dict(candidate=index, parameters=p, mu=OPWParameters(**{k:p[k] for k in
                   ("lambda1", "lambda2", "sigma", "cost_scale")}).mu,
                   ACC=evaluation["ACC"][str(args.selection_k)], MAP=evaluation["MAP"],
                   seconds=elapsed, diagnostics=diagnostics)
        rows.append(row)
        np.savez_compressed(output/f"candidate_{index:03d}.npz", distances=distances,
                            gallery_indices=gi, validation_indices=vi)
        atomic_json(output/"search.json", dict(signature=signature, results=rows, status="running"))
        print(f"{index+1}/{len(candidates)} mu={row['mu']:g}, eps={p['lambda2']:g}: ACC={row['ACC']:.3%}, MAP={row['MAP']:.3%}, {elapsed:.1f}s", flush=True)
    selected = select_candidate(rows, args.objective)
    artifact = dict(schema="flashopw-training-selection-v1",
                    protocol="stratified holdout within official training split only", dataset=args.dataset,
                    training_sha256=origin["training_sha256"], parameters=rows[selected]["parameters"],
                    score=args.score, n_iters=args.iters, selection_k=args.selection_k, objective=args.objective,
                    selected_candidate=selected, validation=rows[selected], default_validation=rows[0],
                    gallery_indices=gi.tolist(), validation_indices=vi.tolist(), seed=args.seed,
                    backend="CUDA FP32 FlashOPW" if device.type == "cuda" else "CPU FP64 batched dense oracle",
                    search_source_sha256=signature["sources"],
                    limitation="Validation selection does not guarantee higher test ACC/MAP; fixed iterations may leave marginal error")
    atomic_json(output/"selected_parameters.json", artifact)
    atomic_json(output/"search.json", dict(signature=signature, results=rows, status="completed"))
    with (output/"validation.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["candidate", "mu", "lambda1", "lambda2", "sigma", "ACC", "MAP", "seconds", "max_marginal_l1", "diagnostic_pairs"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row[k] for k in ("candidate", "mu", "ACC", "MAP", "seconds")},
                             **{k:row["parameters"][k] for k in ("lambda1", "lambda2", "sigma")},
                             "max_marginal_l1":row["diagnostics"]["max_marginal_l1"], "diagnostic_pairs":row["diagnostics"]["pairs"]})
    print(f"Frozen candidate {selected}: {output/'selected_parameters.json'}", flush=True)


if __name__ == "__main__":
    main()
