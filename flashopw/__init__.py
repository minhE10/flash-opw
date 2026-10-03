"""Balanced squared-Euclidean entropic OT. Triton is imported lazily."""

from .solver import SinkhornResult, sinkhorn_dense, sinkhorn_flash, sinkhorn_online
from .transport import apply_plan, diagnostics, materialize_plan, point_gradients

__all__ = [
    "SinkhornResult", "sinkhorn_dense", "sinkhorn_flash", "sinkhorn_online",
    "apply_plan", "diagnostics", "materialize_plan", "point_gradients",
]
