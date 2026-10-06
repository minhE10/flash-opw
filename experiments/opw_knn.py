"""FlashOPW from main (2).pdf versus the ten journal sequence distances.

Default is a small, fixed-parameter trial on FacesUCR, with official train/test
splits and reproducible balanced subsets. Set subset limits to 0 for full splits.
FlashOPW's default score is the literal main PDF Eq.19, not journal <P,D>.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from flashopw import opw_diagnostics, opw_distance, opw_flash, opw_online
from .retrieval import evaluate_distances
from .runtime import configure, metadata
from .sequence_data import balanced_subset, load_sequences, training_fingerprint
from .sequence_metrics import JOURNAL_METRICS, reference_distance
from .opw_parameters import load_selection, source_hashes


METRICS = ("flash-opw", "affine-opw-dense", *JOURNAL_METRICS, "opw-exact-relative",
           "flash-opw-tuned", "affine-opw-dense-tuned")
FIELDS = ("dataset", "metric", "status", "k", "ACC", "MAP", "ACC_percent", "MAP_percent",
          "queries", "gallery", "backend", "distance_seconds", "detail")
DATASET_PARAMETERS = {"FacesUCR": (1.0, 0.1, 1.0), "FaceAll": (10.0, 0.1, 1.0)}


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _write_results(output, rows):
    with (output / "knn_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(output / "knn_results.json", rows)


def _source_hashes():
    return source_hashes()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["FacesUCR"])
    parser.add_argument("--dataset-file", type=Path, help="fixed or packed sequence NPZ; requires one dataset name")
    parser.add_argument("--data-root", type=Path, default=Path("data/opw"))
    parser.add_argument("--metrics", nargs="+", choices=METRICS,
                        default=["flash-opw", "affine-opw-dense", *JOURNAL_METRICS])
    parser.add_argument("--flash-parameters", type=Path,
                        help="frozen training selection; adds flash-opw-tuned, preserves baseline parameters")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 7, 15, 30])
    parser.add_argument("--max-train", type=int, default=64, help="0 means full official training split")
    parser.add_argument("--max-queries", type=int, default=32, help="0 means full official test split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda1", type=float, help="default: dataset fixed journal value, otherwise 50")
    parser.add_argument("--lambda2", type=float, help="default: 0.1")
    parser.add_argument("--sigma", type=float, help="default: 1")
    parser.add_argument("--cost-scale", type=float, default=1)
    parser.add_argument("--iters", type=int, default=200, help="fixed iterations for main FlashOPW and its affine dense control")
    parser.add_argument("--journal-opw-iters", type=int, default=20, help="fixed iterations for original OPW/OPW-KL journal baselines")
    parser.add_argument("--sinkhorn-iters", type=int, default=100)
    parser.add_argument("--ot-epsilon", type=float, default=0.1)
    parser.add_argument("--tcot-lambda", type=float, default=1)
    parser.add_argument("--tlp-weight", type=float, default=50)
    parser.add_argument("--soft-dtw-gamma", type=float, default=0.1)
    parser.add_argument("--opw-score", choices=("pdf-loss", "spatial"), default="pdf-loss")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("ieee", "tf32x3", "tf32"), default="ieee")
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--diagnostic-pairs", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="resume unchanged code, data, environment and settings")
    args = parser.parse_args()
    if args.flash_parameters and "flash-opw-tuned" not in args.metrics:
        args.metrics.append("flash-opw-tuned")
    if args.flash_parameters and len(args.datasets) != 1:
        parser.error("--flash-parameters requires exactly one dataset")
    if any(metric.endswith("-tuned") for metric in args.metrics) and not args.flash_parameters:
        parser.error("Tuned metrics require --flash-parameters")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) for name in args.datasets):
        parser.error("dataset names must contain only letters, digits, underscores and hyphens")
    if args.dataset_file and len(args.datasets) != 1:
        parser.error("--dataset-file requires exactly one dataset name")
    if any(name not in ("FacesUCR", "FaceAll") for name in args.datasets) and not args.dataset_file:
        parser.error("Other datasets require --dataset-file with provided sequence features and splits")
    if any(len(set(values)) != len(values) for values in (args.datasets, args.metrics, args.ks)):
        parser.error("datasets, metrics and ks must not contain duplicates")
    if min(args.max_train, args.max_queries, args.diagnostic_pairs) < 0:
        parser.error("subset limits and diagnostic-pairs must be nonnegative")
    if min(args.iters, args.journal_opw_iters, args.sinkhorn_iters, args.threads, *args.ks) < 1:
        parser.error("iterations, threads and ks must be positive")
    for name in ("lambda2", "sigma", "cost_scale", "ot_epsilon", "tcot_lambda", "soft_dtw_gamma"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{name} must be finite and positive")
    for name in ("lambda1", "tlp_weight"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"{name} must be finite and nonnegative")
    if args.resume and args.output is None:
        parser.error("--resume needs an explicit --output directory")
    return args


def _parameters(dataset, args):
    defaults = DATASET_PARAMETERS.get(dataset, (50, 0.1, 1))
    return dict(lambda1=defaults[0] if args.lambda1 is None else args.lambda1,
                lambda2=defaults[1] if args.lambda2 is None else args.lambda2,
                sigma=defaults[2] if args.sigma is None else args.sigma,
                cost_scale=args.cost_scale, n_iters=args.iters)


def _plot(output, rows):
    if not any(row["status"] == "ok" for row in rows):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for dataset in sorted({row["dataset"] for row in rows}):
        selected = [row for row in rows if row["dataset"] == dataset and row["status"] == "ok"]
        metrics = list(dict.fromkeys(row["metric"] for row in selected))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for metric in metrics:
            points = sorted((row for row in selected if row["metric"] == metric), key=lambda row:row["k"])
            axes[0].plot([row["k"] for row in points], [row["ACC_percent"] for row in points], marker="o", label=metric)
        axes[0].set(xlabel="k", ylabel="ACC (%)", title=dataset + ": k-NN")
        axes[0].grid(alpha=0.2)
        axes[0].legend(fontsize=7)
        axes[1].barh(metrics, [next(row["MAP_percent"] for row in selected if row["metric"] == metric) for metric in metrics])
        axes[1].set(xlabel="MAP (%)", title="Full selected-gallery retrieval")
        fig.savefig(output / f"{dataset}_map_acc.png", dpi=180)
        plt.close(fig)


def main():
    args = _parse_args()
    os.environ["FLASHOPW_AUTOTUNE"] = "1" if args.autotune else "0"
    device = configure(args.device, args.threads, args.memory_fraction)
    output = args.output or Path("outputs") / datetime.now(timezone.utc).strftime("opw_knn_%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=args.resume)
    if device.type == "cpu" and "flash-opw" in args.metrics:
        print("CPU mode: FlashOPW request uses the explicit tiled Torch oracle; results are labelled affine-opw-online, not GPU timings.", flush=True)
    parameters = {name: _parameters(name, args) for name in args.datasets}
    selection = (load_selection(args.flash_parameters, dataset=args.datasets[0],
                               score=args.opw_score, n_iters=args.iters)
                 if args.flash_parameters else None)
    if selection and selection["parameters"]["cost_scale"] != args.cost_scale:
        raise ValueError("Use the frozen cost_scale for all metrics via --cost-scale")
    settings = {key:str(value) if isinstance(value, Path) else value for key,value in vars(args).items()
                if key not in ("resume", "output")}
    environment = metadata(device)
    signature = dict(settings=settings, parameters=parameters, selection=selection, sources=_source_hashes(),
                     packages=environment["packages"], torch=environment["torch"],
                     torch_cuda=environment["torch_cuda"], gpu=environment.get("gpu"),
                     python=environment["python"], kernel_controls=environment["kernel_controls"])
    if args.resume:
        prior = json.loads((output / "environment.json").read_text(encoding="utf-8"))
        if prior["resume_signature"] != signature:
            raise ValueError("Resume refused: settings/source/environment changed; use a fresh output")
    else:
        environment.update(resume_signature=signature, args=settings, protocol=dict(
            flash_opw="main (2).pdf affine/Taylor; F=(i/N-j/M)^2; epsilon=lambda2",
            flash_opw_score="literal main Eq.19" if args.opw_score == "pdf-loss" else "spatial <P,D>",
            original_opw="journal Eq.12 perpendicular distance and exact inverse moment; spatial <P,D>",
            ground_cost="cost_scale * squared Euclidean for every distance",
            iterations="f then g, fixed count; main FlashOPW default200, journal OPW default20, Sinkhorn/TCOT/TLp default100",
            evaluation="majority k-NN ACC; full selected-gallery MAP independent of k; scores are fractions and percent",
            trial="fixed parameters and optional balanced subsets; no test-label hyperparameter search; not a claim of reproducing journal tables",
            baseline_backend="CPU FP64 dense references; FlashOPW CUDA FP32 or explicitly selected CPU oracle",
            timing="end-to-end distance matrix wall time after one warmup; excludes data loading, ranking and plots",
        ))
        if selection:
            environment["protocol"]["trial"] = "frozen training-only selection for flash-opw-tuned; default and journal metrics retain baseline parameters; not full journal reproduction"
        _atomic_json(output / "environment.json", environment)
    rows, failures = [], []
    all_evaluations, all_diagnostics, provenance = {}, {}, {}
    data_path = output / "data_manifest.json"
    prior_provenance = (json.loads(data_path.read_text(encoding="utf-8"))
                        if args.resume and data_path.is_file() else {})
    provenance.update(prior_provenance)
    _atomic_json(output / "run_state.json", dict(status="running", started=datetime.now(timezone.utc).isoformat()))
    for dataset in args.datasets:
        try:
            train, train_labels, test, test_labels, origin = load_sequences(dataset, args.data_root, args.dataset_file)
            if selection:
                load_selection(args.flash_parameters, dataset=dataset,
                               training_sha256=training_fingerprint(train, train_labels),
                               score=args.opw_score, n_iters=args.iters)
            ti = balanced_subset(train_labels, args.max_train, args.seed)
            qi = balanced_subset(test_labels, args.max_queries, args.seed + 1)
            if max(args.ks) > len(ti):
                raise ValueError(f"k={max(args.ks)} exceeds gallery={len(ti)}; change --ks or --max-train")
        except Exception as exc:
            _atomic_json(output / "run_state.json", dict(status="failed", dataset=dataset,
                         detail=f"{type(exc).__name__}: {exc}"))
            raise
        origin.update(train_indices=ti.tolist(), query_indices=qi.tolist())
        provenance[dataset] = origin
        if dataset in prior_provenance and prior_provenance[dataset] != origin:
            _atomic_json(output / "run_state.json", dict(status="failed", dataset=dataset,
                         detail="Resume refused: data bytes or selected split changed"))
            raise ValueError("Resume refused: data bytes or selected split changed")
        _atomic_json(data_path, provenance)
        train, test = [train[i] for i in ti], [test[i] for i in qi]
        train_labels, test_labels = train_labels[ti], test_labels[qi]
        tensors = None
        if any(metric.startswith("flash-opw") for metric in args.metrics):
            dtype = torch.float32 if device.type == "cuda" else torch.float64
            tensors = ([torch.as_tensor(x, dtype=dtype, device=device) for x in train],
                       [torch.as_tensor(x, dtype=dtype, device=device) for x in test])
        print(f"\n{dataset}: gallery={len(train)}, queries={len(test)}, parameters={parameters[dataset]}", flush=True)
        reference_kwargs = dict(**parameters[dataset], epsilon=args.ot_epsilon,
                                sinkhorn_iters=args.sinkhorn_iters, tcot_lambda=args.tcot_lambda,
                                tlp_weight=args.tlp_weight, soft_dtw_gamma=args.soft_dtw_gamma,
                                opw_score=args.opw_score)
        for metric in args.metrics:
            is_flash = metric.startswith("flash-opw")
            metric_parameters = selection["parameters"] if metric.endswith("-tuned") else parameters[dataset]
            metric_kwargs = dict(reference_kwargs, **metric_parameters)
            name = metric.replace("flash-opw", "affine-opw-online") if is_flash and device.type == "cpu" else metric
            key = dataset + "/" + name
            cache = output / f"{dataset}_{name}_distances.npz"
            distances = np.full((len(test), len(train)), np.nan)
            elapsed, completed = 0.0, 0
            diagnostics = []
            if args.resume and cache.is_file():
                with np.load(cache, allow_pickle=False) as saved:
                    distances = saved["distances"].copy()
                    elapsed, completed = float(saved["seconds"]), int(saved["completed_queries"])
                if distances.shape != (len(test), len(train)):
                    raise ValueError("Resume matrix shape mismatch")
                diagnostic_path = output / f"{dataset}_{name}_diagnostics.json"
                if diagnostic_path.is_file():
                    diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            backend = "CUDA FP32 FlashSinkhorn" if is_flash and device.type == "cuda" else (
                "CPU FP64 tiled Torch" if is_flash else "CPU FP64 reference")

            def operation(i, j):
                if not is_flash:
                    kwargs = dict(metric_kwargs)
                    if metric in ("opw", "opw-kl", "opw-exact-relative"):
                        kwargs["n_iters"] = args.journal_opw_iters
                    return reference_distance(metric.removesuffix("-tuned"), test[i], train[j], **kwargs)
                solver = opw_flash if device.type == "cuda" else opw_online
                result = solver(tensors[1][i], tensors[0][j], **metric_parameters, precision=args.precision)
                return float(result.loss if args.opw_score == "pdf-loss" else opw_distance(result))

            try:
                if completed < len(test):
                    operation(0, 0)  # compile/cache outside measured distance matrix
                for i in range(completed, len(test)):
                    started = time.perf_counter()
                    for j in range(len(train)):
                        distances[i, j] = operation(i, j)
                        if not math.isfinite(distances[i, j]):
                            raise FloatingPointError(f"Nonfinite distance at query={i}, gallery={j}")
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed += time.perf_counter() - started
                    # Row checkpoints: interrupted partial rows are recomputed.
                    temporary = cache.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, distances=distances, seconds=elapsed, completed_queries=i+1,
                                        train_labels=train_labels, test_labels=test_labels)
                    temporary.replace(cache)
                    if i == 0 and is_flash and args.diagnostic_pairs:
                        solver = opw_flash if device.type == "cuda" else opw_online
                        for j in range(min(args.diagnostic_pairs, len(train))):
                            result = solver(tensors[1][i], tensors[0][j], **metric_parameters, precision=args.precision)
                            expected = reference_distance("affine-opw-dense", test[i], train[j], **metric_kwargs)
                            diagnostics.append(dict(query=i, gallery=j, **opw_diagnostics(result),
                                                    score_abs_error=abs(distances[i,j]-expected)))
                        _atomic_json(output / f"{dataset}_{name}_diagnostics.json", diagnostics)
                    print(f"  {name}: {i+1}/{len(test)} queries ({elapsed:.2f}s distance time)", flush=True)
                evaluation = evaluate_distances(distances, train_labels, test_labels, args.ks)
                all_evaluations[key] = evaluation
                all_diagnostics[key] = diagnostics
                for k in args.ks:
                    acc, map_value = evaluation["ACC"][str(k)], evaluation["MAP"]
                    rows.append(dict(dataset=dataset, metric=name, status="ok", k=k,
                                     ACC=acc, MAP=map_value, ACC_percent=100*acc, MAP_percent=100*map_value,
                                     queries=len(test), gallery=len(train), backend=backend,
                                     distance_seconds=elapsed,
                                     detail="main Eq.19" if (is_flash or metric.startswith("affine-opw-dense")) and args.opw_score == "pdf-loss" else "journal-style ranking distance"))
                print(f"  {name}: MAP={100*evaluation['MAP']:.3f}%, ACC@1={100*evaluation['ACC'].get('1', float('nan')):.3f}%", flush=True)
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                failures.append(dict(dataset=dataset, metric=name, detail=detail))
                rows.append(dict(dataset=dataset, metric=name, status="failed", k=None,
                                 ACC=None, MAP=None, ACC_percent=None, MAP_percent=None,
                                 queries=len(test), gallery=len(train), backend=backend, distance_seconds=elapsed, detail=detail))
                print(f"  {name}: FAILED {detail}", flush=True)
            _write_results(output, rows)
            _atomic_json(output / "evaluations.json", all_evaluations)
            _atomic_json(output / "diagnostics.json", all_diagnostics)
            _atomic_json(output / "failures.json", failures)
            _atomic_json(output / "run_state.json", dict(status="running", dataset=dataset, last_metric=name))
            _plot(output, rows)
    # Compare complete affine score matrices, not just a few correctness cases.
    parity = []
    for dataset, suffix in ((dataset, suffix) for dataset in args.datasets for suffix in ("", "-tuned")):
        stream_name = "flash-opw" if device.type == "cuda" else "affine-opw-online"
        stream_file = output / f"{dataset}_{stream_name}{suffix}_distances.npz"
        dense_file = output / f"{dataset}_affine-opw-dense{suffix}_distances.npz"
        if stream_file.is_file() and dense_file.is_file():
            with np.load(stream_file) as a, np.load(dense_file) as b:
                actual, expected = a["distances"], b["distances"]
                if np.isfinite(actual).all() and np.isfinite(expected).all():
                    delta = actual - expected
                    parity.append(dict(dataset=dataset, variant="tuned" if suffix else "default", max_abs_error=float(np.abs(delta).max()),
                                       relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(expected), 1e-30)),
                                       nearest_neighbor_agreement=float(np.mean(np.argmin(actual, axis=1) == np.argmin(expected, axis=1)))))
    _atomic_json(output / "affine_parity.json", parity)
    _atomic_json(output / "run_state.json", dict(status="failed" if failures else "completed",
                                                finished=datetime.now(timezone.utc).isoformat(), failures=len(failures)))
    print(f"Results: {output.resolve()}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
