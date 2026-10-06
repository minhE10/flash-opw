"""Balanced squared-Euclidean entropic OT. Triton is imported lazily."""

from .solver import SinkhornResult, sinkhorn_dense, sinkhorn_flash, sinkhorn_online
from .transport import (apply_plan, apply_plan_hadamard, diagnostics,
                        materialize_plan, point_gradients, source_gradient,
                        target_gradient)
from .differentiation import (HVPInfo, hessian_vector_product,
                              regularized_ot_cost, sinkhorn_cost)

__all__ = [
    "SinkhornResult", "sinkhorn_dense", "sinkhorn_flash", "sinkhorn_online",
    "apply_plan", "apply_plan_hadamard", "diagnostics", "materialize_plan",
    "point_gradients", "source_gradient", "target_gradient", "HVPInfo", "hessian_vector_product",
    "regularized_ot_cost", "sinkhorn_cost",
]
