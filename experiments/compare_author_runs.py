"""Compare local and author runs only after checking their shared protocol."""

import argparse
import csv
import json
from pathlib import Path

from .author_reference import AUTHOR_REVISION


SHARED_CONTROLS = (
    "experiments", "methods", "n_sizes", "d_sizes", "hvp_n_sizes", "hvp_d_sizes",
    "dimension_n", "hvp_dimension_n", "hvp_fixed_d", "epsilon", "iters",
    "hvp_sinkhorn_iters", "hvp_cg_iters", "hvp_damping", "precision", "seed",
    "warmups", "forward_repeats", "backward_repeats", "hvp_repeats",
    "memory_fraction", "max_tensorized_mib", "jax_batch_size", "jax_hvp_backend",
    "hvp_baseline_max_n", "hvp_keops_max_n", "hvp_baseline_max_d",
    "autotune", "block_m", "block_n",
)


def compare(local, author):
    local, author = Path(local), Path(author)
    environments = [json.loads((path / "environment.json").read_text(encoding="utf-8"))
                    for path in (local, author)]
    left, right = environments
    if left["args"].get("flash_implementation", "local") != "local":
        raise ValueError("The first run must use local Flash")
    if right["args"].get("flash_implementation") != "author":
        raise ValueError("The second run must use author Flash")
    if (right.get("author_source") or {}).get("revision") != AUTHOR_REVISION:
        raise ValueError("Author provenance is missing or uses a different revision")
    mismatches = [key for key in SHARED_CONTROLS
                  if left["args"].get(key) != right["args"].get(key)]
    for key in ("gpu", "capability", "torch", "torch_cuda", "packages", "cuda_visible_devices"):
        if left.get(key) != right.get(key):
            mismatches.append(key)
    # The timed author implementation differs intentionally. Shared coupling
    # setup and retained baselines must still use the same available hashes.
    for key in ("flashopw/solver.py", "flashopw/triton_kernels.py", "flashopw/kernel_tuning.py",
                "flashopw/transport.py", "flashopw/differentiation.py", "experiments/ott_hessian.py",
                "experiments/jax_hvp.py"):
        a = left.get("source_sha256", {}).get(key)
        b = right.get("source_sha256", {}).get(key)
        if a is not None and b is not None and a != b:
            mismatches.append(key)
    if mismatches:
        raise ValueError("Runs have different shared controls/environment/source: " + ", ".join(mismatches))

    def indexed(path):
        rows = json.loads((path / "paper_results.json").read_text(encoding="utf-8"))
        result = {}
        for row in rows:
            key = (row["experiment"], row["n"], row["m"], row["d"], row["method"])
            if key in result:
                raise ValueError(f"Duplicate result key: {key}")
            result[key] = row
        return result

    old, new = indexed(local), indexed(author)
    comparison = []
    for key in sorted(old.keys() | new.keys()):
        a, b = old.get(key, {}), new.get(key, {})
        ratio = None
        if a.get("status") == b.get("status") == "ok":
            if a["unit"] != b["unit"]:
                raise ValueError(f"Different units for {key}")
            if a["mean"] > 0 and b["mean"] > 0:
                ratio = a["mean"] / b["mean"]
        comparison.append(dict(
            experiment=key[0], n=key[1], m=key[2], d=key[3], method=key[4],
            unit=a.get("unit", b.get("unit")),
            local_status=a.get("status", "missing"), author_status=b.get("status", "missing"),
            local_mean=a.get("mean"), author_mean=b.get("mean"), local_over_author=ratio,
        ))
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local", type=Path)
    parser.add_argument("author", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = compare(args.local, args.author)
    if not rows:
        parser.error("both result files are empty")
    output = args.output or args.author / "comparison_with_local.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison: {output.resolve()}")
    print("local_over_author > 1: author faster (ms) or uses less memory (MB).")
    flash = [row for row in rows if row["method"].startswith("flash") and row["unit"] == "ms"
             and row["local_over_author"] is not None]
    print(f"Matched Flash timings: {len(flash)}; author faster: "
          f"{sum(row['local_over_author'] > 1 for row in flash)}")
    print("Only Flash uses the author source. KeOps/Tensorized/JAX are retained baselines.")


if __name__ == "__main__":
    main()
