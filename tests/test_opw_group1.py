import numpy as np
import pytest
import torch
import json
import sys

from experiments.opw_group1 import convergence_summary, ranking_comparison, reference_snapshots
from experiments.opw_scaling import synthetic_pair
from flashopw import materialize_opw_plan, opw_dense, opw_diagnostics
from experiments.opw_parameters import atomic_json
from experiments.sequence_data import load_training
import experiments.opw_group1 as group1


@pytest.mark.parametrize("sigma", [1., .031943828249997])
def test_incremental_oracle_checkpoints_match_dense_restart_cost_and_coupling(sigma):
    pairs = [synthetic_pair(7,11,3,seed) for seed in (9,10)]
    xs,ys = zip(*pairs)
    p = dict(lambda1=1.,lambda2=.1,sigma=sigma,cost_scale=.7)
    for iteration,values,plans in reference_snapshots(xs,ys,p,[1,7,20]):
        for i,(x,y) in enumerate(pairs):
            result = opw_dense(torch.from_numpy(x),torch.from_numpy(y),**p,n_iters=iteration)
            expected = opw_diagnostics(result)
            torch.testing.assert_close(plans[i],materialize_opw_plan(result),rtol=1e-10,atol=1e-12)
            for key,other in (("score","pdf_eq19"),("spatial","spatial_transport_cost"),
                              ("mass","mass"),("row_l1","row_l1"),("col_l1","col_l1"),
                              ("primal_minus_dual","primal_minus_dual")):
                assert float(values[key][i]) == pytest.approx(expected[other],abs=1e-11)


def test_ranking_reports_ambiguous_nearest_neighbor_changes_without_hiding_them():
    expected = np.array([[1.,1.00001,2.],[1.,2.,3.]])
    actual = np.array([[1.00002,1.00001,2.],[1.00002,2.,3.]])
    comparison = ranking_comparison(actual,expected)
    assert comparison["nn_agreement"] == .5
    assert comparison["full_ranking_agreement"] == .5
    assert comparison["ambiguous_queries_at_observed_error"] == 1
    assert comparison["max_abs_score_error"] == pytest.approx(.00002)
    with pytest.raises(FloatingPointError):
        ranking_comparison(actual*np.nan,expected)


def test_convergence_requires_both_marginals_and_never_assumes_last_is_converged():
    scores = np.array([[1.,2.]])
    snapshots = {}
    for it,row,col in ((1,.01,0.),(2,.0004,.004),(3,.0008,.0002),(4,.002,0.)):
        snapshots[it] = dict(score=scores,spatial=scores,mass=np.ones_like(scores),
                             row_l1=np.full_like(scores,row),col_l1=np.full_like(scores,col))
    summary = convergence_summary(snapshots,[.001,.0001])
    assert summary["checkpoints"][1]["reached_fraction"]["0.001"] == 0
    assert summary["first_reached"]["0.001"]["first_observed_checkpoint_counts"]["3"] == 2
    assert summary["first_reached"]["0.001"]["not_satisfied_at_cap"] == 2
    assert summary["first_reached"]["0.0001"]["never_reached_in_sweep"] == 2
    assert summary["checkpoints"][-1]["versus_last_checkpoint"]["max_abs_score_error"] == 0
    assert summary["checkpoints"][-1]["max_marginal_l1"] > .001


def test_incremental_reference_rejects_reordered_checkpoints():
    x,y = synthetic_pair(7,11,1,9)
    with pytest.raises(ValueError,match="increasing"):
        list(reference_snapshots([x],[y],dict(lambda1=1.,lambda2=.1,sigma=1.,cost_scale=1.),[20,7]))


def _run_tiny(tmp_path,monkeypatch,device):
    dataset = tmp_path/"train_only.npz"
    rng = np.random.default_rng(19)
    np.savez(dataset,train_x=rng.normal(size=(8,9,2)),train_labels=np.repeat([0,1],4))
    _,_,origin = load_training("tiny",tmp_path,dataset)
    parameters = dict(lambda1=1.,lambda2=.7,sigma=.2,cost_scale=1.,n_iters=3)
    artifact = dict(schema="flashopw-training-selection-v1",
                    protocol="stratified holdout within official training split only",dataset="tiny",
                    training_sha256=origin["training_sha256"],score="pdf-loss",n_iters=3,
                    parameters=parameters,default_validation=dict(parameters=dict(parameters,sigma=1.)))
    selection = tmp_path/"selection.json"
    atomic_json(selection,artifact)
    output = tmp_path/"group1"
    argv = ["opw_group1","--device",device,"--dataset","tiny","--dataset-file",str(dataset),
            "--flash-parameters",str(selection),"--gallery","4","--queries","2",
            "--checkpoints","1","3","--parity-iters","1","3","--parity-shapes","7:11:3",
            "--memory-fraction",".25","--output",str(output)]
    monkeypatch.setattr(sys,"argv",argv)
    group1.main()
    return output,argv,artifact


def test_cpu_group1_runs_without_test_split_freezes_parameters_and_resumes(tmp_path,monkeypatch):
    output,argv,artifact = _run_tiny(tmp_path,monkeypatch,"cpu")
    summary = json.loads((output/"summary.json").read_text())
    assert summary["status"] == "completed" and summary["parity_failures"] == 0
    assert summary["flash_verification"] == "not_run_CPU_only"
    assert summary["queries"] == 2 and summary["gallery"] == 4
    environment = json.loads((output/"environment.json").read_text())
    assert environment["signature"]["selection"] == artifact
    assert set(np.load(output/"tuned_reference_3.npz")["query_indices"]).isdisjoint(
        np.load(output/"tuned_reference_3.npz")["gallery_indices"])
    before = (output/"convergence.json").read_bytes()
    monkeypatch.setattr(sys,"argv",argv+["--resume"])
    def unexpected(*args,**kwargs):
        raise AssertionError("Completed reference/parity must not recompute on resume")
    monkeypatch.setattr(group1,"reference_snapshots",unexpected)
    group1.main()
    assert (output/"convergence.json").read_bytes() == before
    monkeypatch.setattr(sys,"argv",argv+["--resume","--taus",".002"])
    with pytest.raises(ValueError,match="Resume refused"):
        group1.main()


@pytest.mark.gpu
def test_gpu_group1_pipeline_scores_coupling_and_matrix_parity(tmp_path,monkeypatch):
    output,_,_ = _run_tiny(tmp_path,monkeypatch,"cuda")
    summary = json.loads((output/"summary.json").read_text())
    assert summary["status"] == "completed" and summary["parity_failures"] == 0
    assert summary["flash_verification"] == "executed"
    for profile in summary["profiles"].values():
        assert len(profile["matrix_parity"]) == 2
        assert all(r["status"] == "passed" for r in profile["matrix_parity"])
        assert profile["flash"]["cap"] == 3
