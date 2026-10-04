"""Phase 2 real-data benchmark on MNIST and Fashion-MNIST embeddings.

The paper uses ResNet18 penultimate-layer embeddings for its OTDD workload.
This script benchmarks the repository's supported squared-Euclidean solver on
the same feature representation. The label-augmented OTDD cost is intentionally
not included: the current kernel API accepts point features and scalar weights,
not a class-to-class cost lookup table.
"""

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import statistics

import torch

from flashopw import diagnostics, sinkhorn_dense, sinkhorn_flash
from .runtime import configure, metadata
from .toy import measure


def stratified_indices(labels, count, *, seed):
    """Return deterministic, approximately class-balanced indices."""
    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    classes = torch.unique(labels, sorted=True).tolist()
    if not classes or count < len(classes):
        raise ValueError("count must be at least the number of classes")
    if count > len(labels):
        raise ValueError("count cannot exceed the dataset size")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base, remainder = divmod(count, len(classes))
    chunks = []
    for position, class_id in enumerate(classes):
        candidates = torch.where(labels == class_id)[0]
        take = base + (position < remainder)
        if take > len(candidates):
            raise ValueError(f"class {class_id} has only {len(candidates)} samples")
        order = torch.randperm(len(candidates), generator=generator)[:take]
        chunks.append(candidates[order])
    return torch.cat(chunks)


def _make_preprocess(weights):
    transform = weights.transforms()

    def preprocess(image):
        # MNIST and Fashion-MNIST are grayscale; ImageNet ResNet weights expect RGB.
        return transform(image.convert("RGB"))

    return preprocess


def _extract_embeddings(dataset, indices, model, device, batch_size, workers):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    features, labels = [], []
    with torch.inference_mode():
        for images, batch_labels in loader:
            output = model(images.to(device, non_blocking=True)).float()
            features.append(output.cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(features), torch.cat(labels)


def load_embeddings(root, count, batch_size, workers, device, seed, cache):
    """Download datasets if needed, then cache deterministic ResNet18 features."""
    try:
        from torchvision import datasets, models
    except ImportError as exc:
        raise RuntimeError(
            "Phase 2 requires torchvision. Install it with: "
            "python -m pip install -e '.[phase2]'"
        ) from exc

    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        saved = torch.load(cache, map_location="cpu")
        required = {"source", "target", "source_labels", "target_labels", "model"}
        if required.issubset(saved) and saved["source"].shape[0] == count:
            return saved

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    weights = models.ResNet18_Weights.DEFAULT
    transform = _make_preprocess(weights)
    mnist = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    fashion = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    source_indices = stratified_indices(mnist.targets, count, seed=seed)
    target_indices = stratified_indices(fashion.targets, count, seed=seed + 1)

    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source, source_labels = _extract_embeddings(mnist, source_indices, model, device, batch_size, workers)
    target, target_labels = _extract_embeddings(fashion, target_indices, model, device, batch_size, workers)
    del model, mnist, fashion
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    saved = {
        "source": source.contiguous(),
        "target": target.contiguous(),
        "source_labels": source_labels,
        "target_labels": target_labels,
        "model": "ResNet18_Weights.DEFAULT penultimate layer",
    }
    torch.save(saved, cache)
    return saved


def _finite_metrics(stats):
    return all(math.isfinite(float(value)) for value in stats.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5000, 10000, 15000, 20000])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--schedule", choices=("alternating", "symmetric"), default="symmetric")
    parser.add_argument("--precision", choices=("ieee", "tf32x3", "tf32"), default="tf32")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    parser.add_argument("--max-dense-mib", type=float, default=6144)
    parser.add_argument("--agreement-tol", type=float, default=5e-2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.sizes) < 10 or args.batch_size < 1 or args.workers < 0 or args.iters < 1:
        parser.error("sizes must be >= 10; batch-size and iters must be positive; workers must be non-negative")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be non-negative and repeats must be positive")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("epsilon must be finite and positive")
    if not math.isfinite(args.max_dense_mib) or args.max_dense_mib <= 0:
        parser.error("max-dense-mib must be finite and positive")

    device = configure("cuda", args.threads, args.memory_fraction)
    max_size = max(args.sizes)
    cache = args.cache or args.data_root / f"phase2_mnist_fashion_resnet18_{max_size}.pt"
    saved = load_embeddings(args.data_root, max_size, args.batch_size, args.workers,
                            device, args.seed, cache)
    if saved["source"].shape[1] != 512 or saved["target"].shape[1] != 512:
        raise RuntimeError("Expected 512-dimensional ResNet18 penultimate-layer features")

    out = args.output or Path("outputs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out.mkdir(parents=True, exist_ok=False)
    env = metadata(device)
    env["args"] = {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()}
    env["dataset"] = "MNIST -> Fashion-MNIST"
    env["features"] = saved["model"]
    env["cost"] = "squared Euclidean only; label-augmented OTDD cost is not enabled"
    env["timing"] = "median synchronized wall time; excludes embedding extraction, warmups, and diagnostics"
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    fieldnames = ["dataset", "n", "m", "dim", "epsilon", "iterations", "schedule", "backend",
                  "precision", "median_ms", "peak_extra_allocated_mib", "speedup_vs_dense",
                  "transport_cost", "regularized_primal", "dual", "primal_minus_dual", "mass",
                  "row_l1", "col_l1", "max_log_plan_diff"]
    rows, raw_times, failures = [], [], []
    for size in args.sizes:
        source_indices = stratified_indices(saved["source_labels"], size, seed=args.seed)
        target_indices = stratified_indices(saved["target_labels"], size, seed=args.seed + 1)
        x = saved["source"][source_indices].to(device)
        y = saved["target"][target_indices].to(device)
        a = torch.full((size,), 1.0 / size, device=device)
        b = torch.full((size,), 1.0 / size, device=device)
        kwargs = dict(a=a, b=b, epsilon=args.epsilon, n_iters=args.iters, schedule=args.schedule)
        estimated = 10 * size * size * 4 / 2**20
        free, total = torch.cuda.mem_get_info(device)
        dense_limit = min(args.max_dense_mib, 0.5 * free / 2**20,
                          0.8 * args.memory_fraction * total / 2**20)
        run_dense = estimated <= dense_limit
        dense = None
        dense_ms = None
        if run_dense:
            dense, dense_ms, dense_mem, dense_samples = measure(
                lambda: sinkhorn_dense(x, y, **kwargs), device, args.repeats, args.warmups)
            dense_stats = diagnostics(dense)
            rows.append({"dataset": "mnist-fashion", "n": size, "m": size, "dim": 512,
                         "epsilon": args.epsilon, "iterations": dense.n_iters, "schedule": args.schedule,
                         "backend": dense.backend, "precision": dense.precision, "median_ms": dense_ms,
                         "peak_extra_allocated_mib": dense_mem, "speedup_vs_dense": 1.0,
                         **dense_stats, "max_log_plan_diff": 0.0})
            raw_times.append({"n": size, "backend": dense.backend, "samples_ms": dense_samples})
        else:
            print(f"{size}: dense skipped; estimate={estimated:.1f} MiB, limit={dense_limit:.1f} MiB", flush=True)

        candidate, candidate_ms, candidate_mem, candidate_samples = measure(
            lambda: sinkhorn_flash(x, y, **kwargs, precision=args.precision),
            device, args.repeats, args.warmups)
        candidate_stats = diagnostics(candidate)
        log_error = None
        speedup = None
        if dense is not None:
            du, dv = candidate.u - dense.u, candidate.v - dense.v
            log_error = float(torch.maximum((du.min() + dv.min()).abs(),
                                            (du.max() + dv.max()).abs()))
            speedup = dense_ms / candidate_ms
            if not math.isfinite(log_error) or log_error > args.agreement_tol:
                failures.append(f"{size}: log-plan disagreement {log_error:.3g}")
        if not _finite_metrics(candidate_stats):
            failures.append(f"{size}: non-finite FlashSinkhorn diagnostics")
        rows.append({"dataset": "mnist-fashion", "n": size, "m": size, "dim": 512,
                     "epsilon": args.epsilon, "iterations": candidate.n_iters, "schedule": args.schedule,
                     "backend": candidate.backend, "precision": candidate.precision, "median_ms": candidate_ms,
                     "peak_extra_allocated_mib": candidate_mem, "speedup_vs_dense": speedup,
                     **candidate_stats, "max_log_plan_diff": log_error})
        raw_times.append({"n": size, "backend": candidate.backend, "samples_ms": candidate_samples})
        print(f"mnist-fashion {size:5d}x{size:<5d} {candidate.backend:7s} {candidate_ms:9.3f} ms "
              f"residual={max(candidate_stats['row_l1'], candidate_stats['col_l1']):.2e}"
              + (f" speedup={speedup:.2f}x" if speedup is not None else " flash-only"), flush=True)
        del x, y, a, b, candidate, dense
        gc.collect()
        torch.cuda.empty_cache()

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out / "timings.json").write_text(json.dumps(raw_times, indent=2), encoding="utf-8")
    (out / "validation.json").write_text(json.dumps({"passed": not failures, "failures": failures}, indent=2),
                                           encoding="utf-8")
    print(f"Results: {out.resolve()}")
    if failures:
        raise SystemExit("Validation failed; see validation.json")


if __name__ == "__main__":
    main()
