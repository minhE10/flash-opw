"""Run the pinned, unmodified author package through the existing data/timer.

The package is imported from its checkout, never installed over this project's
similarly named distribution. HVP setup uses the same coupling as the current
baselines; the timed HVP and CG are entirely the author's implementation.
"""

from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys

import torch

from flashopw import hessian_vector_product, sinkhorn_flash


AUTHOR_REVISION = "75d48cc42d2efe8d4f654d91152ccf6f857c993f"
AUTHOR_URL = "https://github.com/ot-triton-lab/flash-sinkhorn.git"


def source_provenance(source):
    source = Path(source).resolve()
    package = source / "torch-ext" / "flash_sinkhorn"
    if not (package / "__init__.py").is_file():
        raise FileNotFoundError(f"Author source missing at {source}; run scripts/setup_author_reference.sh")

    def git(*args):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={source.as_posix()}", "-C", str(source), *args],
            text=True, stderr=subprocess.PIPE).strip()

    revision = git("rev-parse", "HEAD")
    if revision != AUTHOR_REVISION:
        raise RuntimeError(f"Author revision {revision}; expected {AUTHOR_REVISION}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Author checkout has tracked modifications; use a clean separate checkout")
    hashes = {}
    for path in sorted(package.rglob("*.py")):
        content = path.read_bytes().replace(b"\r\n", b"\n")
        hashes[path.relative_to(source).as_posix()] = hashlib.sha256(content).hexdigest()
    return dict(path=str(source), revision=revision, url=AUTHOR_URL, source_sha256=hashes)


@lru_cache(maxsize=None)
def load_author(source):
    provenance = source_provenance(source)
    package = Path(provenance["path"]) / "torch-ext" / "flash_sinkhorn"
    sys.path.insert(0, str(package.parent))
    module = importlib.import_module("flash_sinkhorn")
    if Path(module.__file__).resolve() != package / "__init__.py":
        raise RuntimeError(f"Wrong author package imported: {module.__file__}")
    return module, provenance


def ott_potentials(result):
    """Convert log P = 2*c*x.y/eps + u + v to exp((f+g-C)/eps)."""
    return (result.epsilon * result.u + result.cost_scale * result.x.square().sum(1),
            result.epsilon * result.v + result.cost_scale * result.y.square().sum(1))


def relative_l2(actual, expected):
    return float((actual - expected).double().norm() / expected.double().norm().clamp_min(1e-30))


def author_operation(method, experiment, x, y, args):
    author, provenance = load_author(args.author_path)
    schedule = "symmetric" if method == "flash-sym" else "alternating"
    tag = f"author source {provenance['revision']}; CUDA events"
    if not experiment.startswith("hvp"):
        # Match the native bench_forward.py API, supplying the CURRENT tensors
        # instead of that script's Gaussian/random-weight default dataset.
        loss_fn = author.SamplesLoss(
            "sinkhorn", backend=schedule, use_epsilon_scaling=False,
            eps=args.epsilon, n_iters=args.iters, debias=False,
            potentials=False, normalize=False, half_cost=False,
            last_extrapolation=False, allow_tf32=args.precision == "tf32",
            use_exp2=True, autotune=args.autotune, threshold=None,
        )
        a = x.new_full((len(x),), 1.0 / len(x))
        b = y.new_full((len(y),), 1.0 / len(y))
        backward = experiment in ("backward_n", "backward_d", "memory_backward")
        tracked_x = x.detach().requires_grad_(backward)

        def operation():
            value = loss_fn(a, tracked_x, b, y)
            return torch.autograd.grad(value, tracked_x)[0] if backward else value

        def diagnose():
            from .paper_benchmarks import _flash_operation
            reference, _ = _flash_operation(method, experiment, x, y, args)
            actual, expected = operation(), reference()
            return dict(relative_l2_vs_local=relative_l2(actual, expected),
                        author_finite=bool(torch.isfinite(actual).all()),
                        author_norm=float(actual.norm()), local_norm=float(expected.norm()),
                        note="Original author initialization and row-normalized gradient retained; finite-iteration outputs may differ")

        operation.diagnostics = diagnose
        return operation, tag + f"; SamplesLoss {schedule}; n_iters={args.iters}; original initialization/gradient"

    # Shared state with CURRENT Flash/KeOps/JAX, not an author solve included
    # in the timed HVP. Potential conversion and direction are also untimed.
    base = sinkhorn_flash(
        x, y, epsilon=args.epsilon, n_iters=args.hvp_sinkhorn_iters,
        schedule="symmetric", precision="ieee",
        block_m=args.block_m, block_n=args.block_n,
    )
    f, g = ott_potentials(base)
    generator = torch.Generator(device=x.device).manual_seed(args.seed + 991)
    direction = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    direction /= direction.norm().clamp_min(torch.finfo(direction.dtype).tiny)

    def evaluate():
        return author.hvp_x_sqeuclid_from_potentials(
            x, y, f, g, direction, eps=args.epsilon,
            tau2=args.hvp_damping / args.epsilon,
            max_cg_iter=args.hvp_cg_iters, cg_rtol=0.0, cg_atol=0.0,
            preconditioner="none", use_preconditioner=False,
            allow_tf32=False, use_exp2=True, autotune=args.autotune,
        )

    def operation():
        value, _ = evaluate()
        return value

    def diagnose():
        from .paper_benchmarks import _diagnose_result
        actual, info = evaluate()
        expected = hessian_vector_product(
            base, direction, damping=args.hvp_damping,
            max_cg_iters=args.hvp_cg_iters, cg_rtol=0.0, cg_atol=0.0)
        return dict(**_diagnose_result(base, direction, args), author_cg=asdict(info),
                    author_finite=bool(torch.isfinite(actual).all()),
                    relative_l2_vs_local=relative_l2(actual, expected))

    operation.diagnostics = diagnose
    return operation, (tag + f"; shared current IEEE coupling; raw author HVP/CG; "
                       f"max_cg_iter={args.hvp_cg_iters}, rtol=atol=0; "
                       f"tau2={args.hvp_damping / args.epsilon:g}")
