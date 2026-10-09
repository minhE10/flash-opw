"""Opt-in pytest convergence check for the author's 64-step double-backward case.

This changes only that case's iteration budget at runtime. Source, damping,
precision, stopping tolerances, and the original parity assertion stay intact.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import json
import math
from pathlib import Path

import pytest


CASE = "test_samplesloss_double_backward_matches_hvp_x_reference"


def convergence_record(path, info, parameters, *, finite_output):
    required = max(float(parameters["cg_atol"]),
                   float(parameters["cg_rtol"]) * float(info.cg_initial_residual))
    residual = float(info.cg_residual)
    initial = float(info.cg_initial_residual)
    confirmed = (bool(info.cg_converged) and finite_output and
                 all(math.isfinite(value) and value >= 0 for value in (required, residual, initial)) and
                 residual <= required)
    return {
        "path": path, "max_cg_iter": int(parameters["max_cg_iter"]),
        "cg_iters": int(info.cg_iters), "cg_converged": bool(info.cg_converged),
        "cg_residual": residual if math.isfinite(residual) else None,
        "cg_initial_residual": initial if math.isfinite(initial) else None,
        "required": required if math.isfinite(required) else None,
        "finite_output": finite_output, "confirmed": confirmed,
        "tau2": float(parameters["tau2"]), "cg_rtol": float(parameters["cg_rtol"]),
        "cg_atol": float(parameters["cg_atol"]),
    }


def pytest_addoption(parser):
    parser.addoption("--author-hvp-max-cg-iter", type=int)
    parser.addoption("--author-cg-report")


@pytest.fixture(autouse=True)
def check_author_hvp_convergence(request, monkeypatch):
    cap = request.config.getoption("--author-hvp-max-cg-iter")
    if cap is None or request.node.name != CASE or request.module.__name__.split(".")[-1] != "test_samples_loss_api":
        yield
        return
    report = Path(request.config.getoption("--author-cg-report"))
    module = request.module
    original_loss = module.SamplesLoss
    records, configurations = [], []

    @functools.wraps(original_loss)
    def with_iteration_budget(*args, **kwargs):
        # Fail closed if an upstream fixture change invalidates this targeted adaptation.
        assert kwargs["hvp_max_cg_iter"] == 64
        assert kwargs["hvp_tau2"] == 1e-5
        assert kwargs["hvp_cg_rtol"] == kwargs["hvp_cg_atol"] == 1e-6
        kwargs["hvp_max_cg_iter"] = cap
        configurations.append({key: value for key, value in kwargs.items() if key.startswith("hvp_")})
        return original_loss(*args, **kwargs)

    def record_calls(original, path):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            parameters = signature.bind(*args, **kwargs)
            parameters.apply_defaults()
            output, info = original(*args, **kwargs)
            finite = bool(module.torch.isfinite(output).all().item())
            records.append(convergence_record(path, info, parameters.arguments, finite_output=finite))
            return output, info
        return measured

    monkeypatch.setattr(module, "SamplesLoss", with_iteration_budget)
    monkeypatch.setattr(module, "hvp_x_sqeuclid_from_potentials",
                        record_calls(module.hvp_x_sqeuclid_from_potentials, "reference"))
    autograd = importlib.import_module("flash_sinkhorn._autograd")
    monkeypatch.setattr(autograd, "hvp_x_sqeuclid_from_potentials",
                        record_calls(autograd.hvp_x_sqeuclid_from_potentials, "autograd"))
    yield
    expected_paths = {record["path"] for record in records} == {"autograd", "reference"}
    settings_intact = (len(configurations) == 1 and all(
        record["max_cg_iter"] == cap and record["tau2"] == 1e-5 and
        record["cg_rtol"] == record["cg_atol"] == 1e-6 for record in records))
    confirmed = expected_paths and settings_intact and all(record["confirmed"] for record in records)
    report.write_text(json.dumps({
        "case": CASE, "max_cg_iter": cap, "status": "converged" if confirmed else "failed",
        "configurations": configurations, "records": records,
        "original_parity_assertion_unchanged": True,
        "note": "Convergence status is separate from pytest parity; both must pass.",
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert confirmed, f"HVP CG convergence was not confirmed on both paths; see {report}"
