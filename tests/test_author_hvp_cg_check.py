"""Convergence must be measured on both paths without relaxing author criteria."""
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from scripts import author_hvp_cg_check as cg_check
from scripts.author_hvp_cg_check import CASE, convergence_record


@pytest.mark.parametrize("flag,residual,finite,expected", [
    (True, 1e-7, True, True),
    (False, 1e-7, True, False),
    (True, 1.33e-4, True, False),
    (True, 1e-7, False, False),
    (True, float("nan"), True, False),
])
def test_convergence_requires_flag_true_residual_and_finite_output(flag, residual, finite, expected):
    info = SimpleNamespace(cg_converged=flag, cg_iters=128, cg_residual=residual, cg_initial_residual=0.1)
    record = convergence_record("autograd", info,
                                dict(max_cg_iter=128, cg_atol=1e-6, cg_rtol=1e-6, tau2=1e-5),
                                finite_output=finite)
    assert record["confirmed"] is expected
    assert record["required"] == 1e-6
    json.dumps(record, allow_nan=False)  # Failed numerical results must still leave valid diagnostics.


@pytest.mark.parametrize("reference_converged", [True, False])
def test_fixture_changes_only_budget_and_checks_reference_too(tmp_path, monkeypatch, reference_converged):
    report = tmp_path / "cg_convergence.json"
    module = ModuleType("test_samples_loss_api")
    module.torch = torch
    loss_calls = []

    def original_loss(**kwargs):
        loss_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    def hvp(converged):
        def original(*args, max_cg_iter=300, tau2=1e-5, cg_rtol=1e-6, cg_atol=1e-6):
            info = SimpleNamespace(cg_converged=converged, cg_iters=80 if converged else max_cg_iter,
                                   cg_residual=1e-7 if converged else 1.33e-4, cg_initial_residual=0.1)
            return torch.ones(2), info
        return original

    module.SamplesLoss = original_loss
    module.hvp_x_sqeuclid_from_potentials = hvp(reference_converged)
    autograd = ModuleType("flash_sinkhorn._autograd")
    autograd.hvp_x_sqeuclid_from_potentials = hvp(True)
    monkeypatch.setitem(sys.modules, autograd.__name__, autograd)
    options = {"--author-hvp-max-cg-iter": 128, "--author-cg-report": str(report)}
    request = SimpleNamespace(node=SimpleNamespace(name=CASE), module=module,
                              config=SimpleNamespace(getoption=options.__getitem__))
    fixture = cg_check.check_author_hvp_convergence.__wrapped__(request, monkeypatch)
    next(fixture)
    loss = module.SamplesLoss(hvp_max_cg_iter=64, hvp_tau2=1e-5,
                              hvp_cg_rtol=1e-6, hvp_cg_atol=1e-6, hvp_use_preconditioner=True,
                              allow_tf32=False, eps=0.2)
    assert loss_calls == [dict(hvp_max_cg_iter=128, hvp_tau2=1e-5, hvp_cg_rtol=1e-6,
                               hvp_cg_atol=1e-6, hvp_use_preconditioner=True, allow_tf32=False, eps=0.2)]
    for implementation in (autograd, module):
        implementation.hvp_x_sqeuclid_from_potentials(max_cg_iter=loss.hvp_max_cg_iter,
                                                      tau2=loss.hvp_tau2, cg_rtol=loss.hvp_cg_rtol,
                                                      cg_atol=loss.hvp_cg_atol)
    with pytest.raises(StopIteration if reference_converged else AssertionError):
        next(fixture)
    diagnostics = json.loads(report.read_text())
    assert diagnostics["status"] == ("converged" if reference_converged else "failed")
    assert {record["path"] for record in diagnostics["records"]} == {"autograd", "reference"}


def test_cg_runner_requires_explicit_larger_budget_before_starting_gpu(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/validate_author_flashsinkhorn.py"
    for args in (["--suite", "cg"], ["--suite", "cg", "--hvp-max-cg-iter", "64"],
                 ["--suite", "regressions", "--hvp-max-cg-iter", "128"]):
        result = subprocess.run([sys.executable, str(script), "--gpu", "--output", str(tmp_path), *args],
                                text=True, capture_output=True)
        assert result.returncode == 2
    assert not (tmp_path / "validation.json").exists()
