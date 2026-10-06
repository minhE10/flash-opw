"""Affine/Taylor OPW from main (2).pdf, reusing the FlashSinkhorn engine.

This is the supplied approximation, not the original journal OPW formula.
The added temporal coordinate uses one-based positions i/N, j/M.
"""

from dataclasses import dataclass
import math

import torch

from flashsinkhorn import (SinkhornResult, apply_plan, materialize_plan,
                           sinkhorn_dense, sinkhorn_flash, sinkhorn_online)


@dataclass(frozen=True)
class OPWParameters:
    lambda1: float = 50.0
    lambda2: float = 0.1
    sigma: float = 1.0
    cost_scale: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.lambda1) or self.lambda1 < 0:
            raise ValueError("lambda1 must be finite and nonnegative")
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.lambda2, self.sigma, self.cost_scale)):
            raise ValueError("lambda2, sigma and cost_scale must be finite and positive")
        try:
            valid = math.isfinite(self.mu) and math.isfinite(self.q0)
        except (OverflowError, ZeroDivisionError):
            valid = False
        if not valid:
            raise ValueError("OPW parameters overflow their derived coefficients")

    @property
    def mu(self):
        return self.lambda1 + self.lambda2 / (2 * self.sigma**2)

    @property
    def q0(self):
        return self.lambda2 * (math.log(self.sigma) + 0.5 * math.log(2 * math.pi)) - self.lambda1


def temporal_features(x, parameters):
    """[sqrt(cost_scale)*x_i, sqrt(mu)*(i+1)/N], O(Nd) storage."""
    if x.ndim != 2 or min(x.shape) < 1 or x.dtype not in (torch.float32, torch.float64):
        raise ValueError("sequence must be a nonempty (frames, features) float32/float64 tensor")
    time = torch.arange(1, len(x) + 1, dtype=x.dtype, device=x.device) / len(x)
    return torch.cat((math.sqrt(parameters.cost_scale) * x,
                      (math.sqrt(parameters.mu) * time)[:, None]), dim=1).contiguous()


def effective_cost(x, y, parameters, *, max_entries=4_194_304):
    """Independent explicit C_new + q0 diagnostic; never used by Flash OPW."""
    if len(x) * len(y) > max_entries:
        raise ValueError("Dense cost exceeds max_entries")
    t = torch.arange(1, len(x) + 1, dtype=x.dtype, device=x.device) / len(x)
    s = torch.arange(1, len(y) + 1, dtype=y.dtype, device=y.device) / len(y)
    spatial = parameters.cost_scale * torch.cdist(
        x, y, compute_mode="donot_use_mm_for_euclid_dist").square()
    return spatial + parameters.mu * (t[:, None] - s[None, :]).square() + parameters.q0


@dataclass
class OPWResult:
    x: torch.Tensor
    y: torch.Tensor
    parameters: OPWParameters
    sinkhorn: SinkhornResult

    @property
    def f(self):
        """Literal main PDF Eq.17: shifted f + augmented squared norm."""
        return self.sinkhorn.f

    @property
    def g(self):
        """Literal main PDF Eq.18, with marginal logs outside potentials."""
        return self.sinkhorn.g

    @property
    def entropy_f(self):
        """Alternative potentials for entropy h(P)=-sum(P log P)."""
        return self.f + self.parameters.lambda2 * (self.sinkhorn.a.log() + 0.5)

    @property
    def entropy_g(self):
        return self.g + self.parameters.lambda2 * (self.sinkhorn.b.log() + 0.5)

    @property
    def loss(self):
        """Literal main PDF Eq.19; default FlashOPW ranking score.

        Uses Eq.17/18 potentials exactly. The engine's KL(P|a*b) convention
        differs from bare entropy; diagnostics report both conventions.
        """
        return ((self.sinkhorn.a * self.f).sum() + (self.sinkhorn.b * self.g).sum()
                - self.parameters.lambda2 + self.parameters.q0)


def _solve(backend, x, y, *, a=None, b=None, lambda1=50.0, lambda2=0.1,
           sigma=1.0, cost_scale=1.0, n_iters=200, schedule="alternating",
           tol=None, check_every=20, precision="ieee", block_m=32, block_n=64):
    parameters = OPWParameters(lambda1, lambda2, sigma, cost_scale)
    if x.device != y.device or x.dtype != y.dtype or x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x,y must have matching feature dimensions, dtype and device")
    if backend == "flash" and x.shape[1] + 1 > 1024:
        raise ValueError("Flash OPW supports at most 1023 spatial features plus one time coordinate")
    augmented_x, augmented_y = temporal_features(x, parameters), temporal_features(y, parameters)
    common = dict(a=a, b=b, epsilon=lambda2, cost_scale=1.0, n_iters=n_iters,
                  schedule=schedule, tol=tol, check_every=check_every)
    if backend == "flash":
        result = sinkhorn_flash(augmented_x, augmented_y, precision=precision,
                                block_m=block_m, block_n=block_n, **common)
    elif backend == "online":
        result = sinkhorn_online(augmented_x, augmented_y, block_m=block_m, block_n=block_n, **common)
    else:
        result = sinkhorn_dense(augmented_x, augmented_y, **common)
    return OPWResult(x.detach(), y.detach(), parameters, result)


def opw_flash(x, y, **kwargs):
    """Affine OPW with existing streaming Triton kernels; no N*M buffers."""
    return _solve("flash", x, y, **kwargs)


def opw_dense(x, y, **kwargs):
    """Dense affine OPW reference; explicitly distinct from journal exact OPW."""
    return _solve("dense", x, y, **kwargs)


def opw_online(x, y, **kwargs):
    """Tiled CPU/CUDA Torch reference, with the same affine transformation."""
    return _solve("online", x, y, **kwargs)


@torch.no_grad()
def _moments(result):
    base = result.sinkhorn
    rows_py = apply_plan(base, torch.cat((torch.ones_like(base.b[:, None]), result.y), 1))
    columns = apply_plan(base, torch.ones_like(base.a), transpose=True)
    return rows_py[:, 0], columns, rows_py[:, 1:]


@torch.no_grad()
def opw_distance(result):
    """Journal ranking convention <P,D_spatial>, WITHOUT regularization/time.

    D_spatial is squared Euclidean as explicitly specified in main (2).pdf.
    Uses actual finite-solve masses and streamed P@Y; never materializes P.
    """
    rows, columns, py = _moments(result)
    return result.parameters.cost_scale * (
        (rows * result.x.square().sum(1)).sum()
        + (columns * result.y.square().sum(1)).sum() - 2 * (result.x * py).sum())


@torch.no_grad()
def opw_diagnostics(result):
    rows, columns, py = _moments(result)
    eps, q0 = result.parameters.lambda2, result.parameters.q0
    mass = rows.sum()
    f, g = result.entropy_f, result.entropy_g
    primal = (rows * f).sum() + (columns * g).sum() - eps * mass + q0 * mass
    dual = ((result.sinkhorn.a * f).sum() + (result.sinkhorn.b * g).sum()
            - eps * mass + q0)
    spatial = result.parameters.cost_scale * (
        (rows * result.x.square().sum(1)).sum()
        + (columns * result.y.square().sum(1)).sum() - 2 * (result.x * py).sum())
    return {key: float(value) for key, value in dict(
        spatial_transport_cost=spatial, entropy_primal=primal, entropy_dual=dual,
        primal_minus_dual=primal-dual, pdf_eq19=result.loss, mass=mass,
        row_l1=(rows-result.sinkhorn.a).abs().sum(),
        col_l1=(columns-result.sinkhorn.b).abs().sum(),
    ).items()}


def materialize_opw_plan(result, *, max_entries=4_194_304):
    return materialize_plan(result.sinkhorn, max_entries=max_entries)
