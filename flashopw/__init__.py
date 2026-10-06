"""Affine/Taylor OPW; ordinary entropic OT lives in ``flashsinkhorn``."""

from .solver import (OPWParameters, OPWResult, effective_cost, materialize_opw_plan,
                     opw_dense, opw_diagnostics, opw_distance, opw_flash,
                     opw_online, temporal_features)

__all__ = ["OPWParameters", "OPWResult", "effective_cost", "materialize_opw_plan",
           "opw_dense", "opw_diagnostics", "opw_distance", "opw_flash",
           "opw_online", "temporal_features"]
