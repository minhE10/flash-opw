"""Training selection artifacts and an independent batched FP64 cost oracle."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from flashopw import OPWParameters


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root / "flashopw").glob("*.py")),
             *sorted((root / "flashsinkhorn").glob("*.py"))]
    paths += [root / "experiments" / name for name in
              ("opw_parameters.py", "opw_tune.py", "opw_scaling.py", "opw_knn.py",
               "sequence_data.py", "sequence_metrics.py", "sequence_metrics_torch.py", "retrieval.py", "runtime.py")]
    return {p.relative_to(root).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths if p.exists()}


def load_selection(path, *, dataset=None, training_sha256=None, score=None, n_iters=None):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != "flashopw-training-selection-v1":
        raise ValueError("Not a FlashOPW training-selection artifact")
    if value.get("protocol") != "stratified holdout within official training split only":
        raise ValueError("Unknown hyperparameter selection protocol")
    for key, expected in (("dataset", dataset), ("training_sha256", training_sha256),
                          ("score", score), ("n_iters", n_iters)):
        if expected is not None and value.get(key) != expected:
            raise ValueError(f"Selection {key} mismatch: freeze the same training data and scoring settings")
    p = value["parameters"]
    OPWParameters(**{k:p[k] for k in ("lambda1", "lambda2", "sigma", "cost_scale")})
    if p["n_iters"] < 1 or p["n_iters"] != value["n_iters"]:
        raise ValueError("Invalid selection iteration count")
    return value


@torch.no_grad()
def dense_score_matrix(queries, gallery, parameters, *, score="pdf-loss", memory_mib=128):
    """Uniform f-then-g solve on explicit costs, grouping ragged gallery lengths.

    CPU FP64 verification/selection helper, never labelled Flash GPU performance.
    Working-array estimate bounds batches; inputs and allocator overhead excluded.
    """
    if score not in ("pdf-loss", "spatial") or memory_mib <= 0:
        raise ValueError("Invalid score or memory budget")
    p = OPWParameters(**{k:parameters[k] for k in ("lambda1", "lambda2", "sigma", "cost_scale")})
    iterations = parameters["n_iters"]
    if iterations < 1:
        raise ValueError("Iterations must be positive")
    distances = np.empty((len(queries), len(gallery)))
    residuals = np.empty_like(distances)
    groups = {}
    for j, y in enumerate(gallery):
        groups.setdefault(y.shape, []).append(j)
    for i, query in enumerate(queries):
        x = torch.as_tensor(query, dtype=torch.float64)
        n = len(x)
        for (m, _), indices in groups.items():
            budget = int(memory_mib * 1024**2)
            if 8 * 8 * n * m > budget:
                raise ValueError("Dense pair exceeds memory budget; use CUDA Flash selection")
            batch_size = max(1, budget // (8 * 8 * n * m))
            for start in range(0, len(indices), batch_size):
                ids = indices[start:start+batch_size]
                ys = torch.as_tensor(np.stack([gallery[j] for j in ids]), dtype=torch.float64)
                spatial = p.cost_scale * torch.cdist(x[None], ys, compute_mode="donot_use_mm_for_euclid_dist").square()
                t = torch.arange(1, n+1, dtype=x.dtype) / n
                s = torch.arange(1, m+1, dtype=x.dtype) / m
                cost = spatial + p.mu * (t[:, None] - s[None, :]).square()
                f, g = torch.zeros((len(ids), n)), torch.zeros((len(ids), m))
                f, g = f.double(), g.double()
                eps = p.lambda2
                for _ in range(iterations):
                    f = -eps * torch.logsumexp((g[:, None, :] - cost)/eps - math.log(m), dim=2)
                    g = -eps * torch.logsumexp((f[:, :, None] - cost)/eps - math.log(n), dim=1)
                plan = ((f[:, :, None] + g[:, None, :] - cost)/eps - math.log(n) - math.log(m)).exp()
                values = (f.mean(1) + g.mean(1) - eps + p.q0 if score == "pdf-loss"
                          else (plan * spatial).sum((1, 2)))
                distances[i, ids] = values.numpy()
                residuals[i, ids] = torch.maximum((plan.sum(2)-1/n).abs().sum(1),
                                                 (plan.sum(1)-1/m).abs().sum(1)).numpy()
    if not np.isfinite(distances).all() or not np.isfinite(residuals).all():
        raise FloatingPointError("Nonfinite score or marginal residual")
    return distances, residuals
