import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.distance import cdist
import torch

from experiments import opw_main_vs_journal as comparison
from experiments.opw_group3 import cost_tensors, dense_batch
from experiments.opw_parameters import atomic_json, source_hashes
from experiments.sequence_data import training_fingerprint
from experiments.sequence_metrics import reference_distance


@pytest.mark.parametrize("n,m", [(7,11),(17,29)])
def test_original_journal_cost_keeps_inverse_and_perpendicular_prior(n,m):
    rng = np.random.default_rng(6)
    x,y = rng.normal(size=(n,3)),rng.normal(size=(m,3))
    p = dict(lambda1=3.,lambda2=.3,sigma=5.,cost_scale=.7)
    cost,D,parameters = comparison.literal_cost("opw",x,y,p)
    delta = np.arange(1,n+1)[:,None]/n - np.arange(1,m+1)[None,:]/m
    E = 1/(1+delta**2)
    prior = np.exp(-delta**2/((1/n**2+1/m**2)*2*p["sigma"]**2))/(p["sigma"]*np.sqrt(2*np.pi))
    np.testing.assert_allclose(cost,D-p["lambda1"]*E-p["lambda2"]*np.log(prior),atol=1e-13)
    centered,_,_,_ = cost_tensors("opw",[x],[y],p)
    np.testing.assert_allclose(centered[0].numpy()+parameters.q0,cost,atol=1e-12)
    # Neither relative prior nor Taylor substitution can pass this oracle.
    main_cost,_,_ = comparison.literal_cost("flash-opw",x,y,p)
    assert np.max(np.abs(main_cost-cost)) > .1
    exact_relative = D-p["lambda1"]*E+p["lambda2"]*(delta**2/(2*p["sigma"]**2)+np.log(p["sigma"]*np.sqrt(2*np.pi)))
    assert np.max(np.abs(exact_relative-cost)) > .1


@pytest.mark.parametrize("metric", comparison.METHODS)
def test_native_score_matches_independent_literal_scipy(metric):
    rng = np.random.default_rng(9)
    x,y = rng.normal(size=(7,3)),rng.normal(size=(11,3))
    p = dict(lambda1=1.,lambda2=.3,sigma=3.,cost_scale=1.)
    values = dense_batch(metric,[x],[y],p,dict(tau=1e-30,max_iters=41,check_every=41))
    score = values["distances"][0]
    result = comparison.oracle_check(metric,x,y,p,41,score)
    expected = reference_distance("affine-opw-dense" if metric == "flash-opw" else "opw",x,y,**p,n_iters=41)
    assert score == pytest.approx(expected,abs=1e-11)
    assert result["score_check"] == "passed"
    with pytest.raises(AssertionError):
        comparison.oracle_check(metric,x,y,p,41,score+1.)


def test_frozen_two_method_runner_resume_and_cache_integrity(tmp_path,monkeypatch):
    root = Path(__file__).resolve().parents[1]
    selection = json.loads((root/"reports/opw_group3_gpu_20261007/selected_all_metrics.json").read_text())
    rng = np.random.default_rng(15)
    sequences = rng.normal(size=(6,5,1))
    labels = np.array(["a","b","a","b","a","b"])
    dataset = tmp_path/"train_only.npz"
    np.savez(dataset,train_x=sequences,train_labels=labels)
    selection.update(dataset="Toy",training_sha256=training_fingerprint(sequences,labels),
        splits=[dict(seed=s,gallery_indices=[0,1,2,3],query_indices=[4,5]) for s in selection["seeds"]],
        solver_policy=dict(tau=.01,max_iters=60,check_every=10,schedule="f_then_g"),sources=source_hashes())
    frozen = tmp_path/"selection.json"
    atomic_json(frozen,selection)
    output = tmp_path/"comparison"
    argv = ["comparison","--selection",str(frozen),"--dataset-file",str(dataset),
            "--device","cpu","--fixed-iters","23","--output",str(output)]
    monkeypatch.setattr(sys,"argv",argv)
    comparison.main()
    result = json.loads((output/"summary.json").read_text())
    assert result["test_used"] is False
    assert len(result["rows"]) == len(result["formula_and_score_checks"]) == 24
    assert {r["metric"] for r in result["rows"]} == {"flash-opw","opw"}
    for row in result["rows"]:
        if row["mode"] == "fixed_23":
            assert row["max_iterations"] == row["median_iterations"] == 23
    # Preserve each metric's independently frozen sigma and epsilon.
    for metric in comparison.METHODS:
        row = next(r for r in result["rows"] if r["profile"] == "selected" and r["metric"] == metric)
        assert json.loads(row["parameters"]) == selection["selected"][metric]["parameters"]
    monkeypatch.setattr(sys,"argv",argv+["--resume"])
    comparison.main()
    assert json.loads((output/"summary.json").read_text()) == result
    monkeypatch.setattr(sys,"argv",argv+["--resume","--fixed-iters","24"])
    with pytest.raises(ValueError,match="Resume refused"):
        comparison.main()
    monkeypatch.setattr(sys,"argv",argv+["--resume"])
    matrix = next((output/"matrices").glob("*.npz"))
    matrix.write_bytes(matrix.read_bytes()+b"corrupt")
    with pytest.raises(ValueError,match="hash/parameters"):
        comparison.main()
    # Selection/source drift cannot silently reuse an old frozen experiment.
    selection["sources"]["experiments/sequence_metrics.py"] = hashlib.sha256(b"changed").hexdigest()
    atomic_json(frozen,selection)
    with pytest.raises(ValueError,match="solver sources changed"):
        comparison.main()


@pytest.mark.gpu
@pytest.mark.parametrize("metric", comparison.METHODS)
def test_gpu_native_score_against_literal_paper_equations(metric):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rng = np.random.default_rng(17)
    x,y = rng.normal(size=(7,3)).astype(np.float32),rng.normal(size=(11,3)).astype(np.float32)
    p = dict(lambda1=1.,lambda2=.3,sigma=3.,cost_scale=1.)
    values = comparison.distance_matrix(metric,[x],[y],p,dict(tau=1e-30,max_iters=41,check_every=41),device=torch.device("cuda"))
    comparison.oracle_check(metric,x,y,p,41,values["distances"][0,0])
