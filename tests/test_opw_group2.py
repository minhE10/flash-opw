import json
import math
import sys

import numpy as np
import pytest
import torch

import experiments.opw_group2 as group2
from experiments.opw_parameters import atomic_json
from experiments.opw_scaling import synthetic_pair
from experiments.sequence_data import load_training
from experiments.sequence_metrics import entropic_plan, reference_distance
from flashopw import materialize_opw_plan, opw_dense


P = dict(lambda1=1.,lambda2=.1,sigma=.031943828249996996,cost_scale=1.)


def test_exact_inverse_cost_gap_and_journal_geometry_are_separate_factors():
    x,y=synthetic_pair(7,11,3,19)
    a=group2.cost_components([x],[y],P,"affine")
    e=group2.cost_components([x],[y],P,"exact_relative")
    j=group2.cost_components([x],[y],P,"exact_journal")
    p,spatial,F,affine,gap,cost=a
    expected=spatial-p.lambda1/(1+F)+p.lambda2*(F/(2*p.sigma**2)+np.log(p.sigma*np.sqrt(2*np.pi)))
    torch.testing.assert_close(e[-1]+p.q0,expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(cost-e[-1],gap[None],rtol=1e-12,atol=1e-12)
    assert bool((gap>=0).all()) and float(gap.max())<=p.lambda1/2
    torch.testing.assert_close(j[-1]-e[-1],p.lambda2/(2*p.sigma**2)*F[None]*(1/(1/7**2+1/11**2)-1))
    matched=dict(P,sigma=p.sigma/math.sqrt(1/7**2+1/11**2))
    normalized_with_matched_units=group2.cost_components([x],[y],matched,"exact_journal")
    torch.testing.assert_close(normalized_with_matched_units[-1],e[-1],rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize("kind",group2.KINDS)
def test_all_scores_reuse_one_plan_and_match_independent_scipy(kind):
    x,y=synthetic_pair(7,11,3,19)
    sweep=group2.dense_sweep([x],[y],P,kind,[10,20],80,10,1e-3)
    values,plan=sweep["fixed_20"]
    p,spatial,_,affine,_,cost=group2.cost_components([x],[y],P,kind)
    expected,f,g,_,_=entropic_plan(cost[0].numpy(),p.lambda2,20,return_potentials=True)
    np.testing.assert_allclose(plan[0].numpy(),expected,rtol=1e-11,atol=1e-13)
    assert float(values["dual_score"][0])==pytest.approx(f.mean()+g.mean()-p.lambda2+p.q0,abs=1e-11)
    assert float(values["spatial_score"][0])==pytest.approx(np.sum(expected*spatial[0].numpy()),abs=1e-11)
    assert float(values["affine_score"][0])==pytest.approx(np.sum(expected*affine[0].numpy()),abs=1e-11)
    metric={"affine":"affine-opw-dense","exact_relative":"opw-exact-relative","exact_journal":"opw"}[kind]
    assert float(values["spatial_score"][0])==pytest.approx(
        reference_distance(metric,x,y,**P,n_iters=20,opw_score="spatial"),abs=1e-11)


def test_matched_tlp_is_the_affine_coupling_with_a_different_score():
    x,y=synthetic_pair(7,11,3,21)
    sweep=group2.dense_sweep([x],[y],P,"affine",[100,200],300,50,1e-3)
    for it in (100,200):
        values,plan=sweep[f"fixed_{it}"]
        tlp=reference_distance("tlp",x,y,epsilon=.1,tlp_weight=50,sinkhorn_iters=it)
        assert float(values["affine_score"][0])==pytest.approx(tlp,abs=1e-11)
        result=opw_dense(torch.from_numpy(x),torch.from_numpy(y),**P,n_iters=it)
        assert float(values["dual_score"][0])==pytest.approx(float(result.loss),abs=1e-11)
        torch.testing.assert_close(plan[0],materialize_opw_plan(result),rtol=1e-10,atol=1e-12)


@pytest.mark.parametrize("kind",group2.KINDS)
def test_dense_fp32_controls_match_fp64_at_fixed_iterations(kind):
    x,y=synthetic_pair(7,11,3,19)
    ref=group2.dense_sweep([x],[y],P,kind,[20],40,10,1e-3)
    actual=group2.dense_sweep([x],[y],P,kind,[20],40,10,1e-3,dtype=torch.float32)
    values,plan=actual["fixed_20"]
    expected,oracle=ref["fixed_20"]
    for score in group2.SCORES:
        torch.testing.assert_close(values[score].double(),expected[score],rtol=3e-3,atol=3e-4)
    assert float((plan.double()-oracle).norm()/oracle.norm())<5e-3


def test_stopping_snapshot_does_not_change_fixed_checkpoint_recurrence():
    x,y=synthetic_pair(7,11,1,3)
    p=dict(P,sigma=1.)
    sweep=group2.dense_sweep([x],[y],p,"affine",[3,7],8,2,10.)
    stopped,plan=sweep["residual_stop"]
    assert float(stopped["iterations"][0])==2
    torch.testing.assert_close(plan[0],materialize_opw_plan(opw_dense(torch.from_numpy(x),torch.from_numpy(y),**p,n_iters=2)))
    torch.testing.assert_close(sweep["fixed_7"][1][0],materialize_opw_plan(
        opw_dense(torch.from_numpy(x),torch.from_numpy(y),**p,n_iters=7)))


def test_capped_pairs_remain_finite_and_are_not_marked_converged():
    x,y=synthetic_pair(7,11,1,8)
    sweep=group2.dense_sweep([x],[y],P,"affine",[2,4],4,2,1e-16)
    values,_=sweep["residual_stop"]
    assert float(values["iterations"][0])==4 and float(values["converged"][0])==0
    assert all(bool(torch.isfinite(v).all()) for v in values.values())


def test_prior_ablation_refuses_to_change_epsilon_or_inverse_weight():
    selection=dict(parameters=dict(P),default_validation=dict(parameters=dict(P,sigma=1.)))
    assert group2.frozen_profiles(selection)["tuned"]==P
    selection["parameters"]["lambda2"]=.3
    with pytest.raises(ValueError,match="Prior-only"):
        group2.frozen_profiles(selection)


def _tiny(tmp_path,monkeypatch,device):
    dataset=tmp_path/"only_train.npz"
    rng=np.random.default_rng(7)
    np.savez(dataset,train_x=rng.normal(size=(8,9,2)),train_labels=np.repeat([0,1],4))
    _,_,origin=load_training("tiny",tmp_path,dataset)
    params=dict(P,n_iters=200)
    artifact=dict(schema="flashopw-training-selection-v1",
                  protocol="stratified holdout within official training split only",dataset="tiny",
                  training_sha256=origin["training_sha256"],score="pdf-loss",n_iters=200,
                  parameters=params,default_validation=dict(parameters=dict(params,sigma=1.)))
    selection=tmp_path/"selection.json"
    atomic_json(selection,artifact)
    before=selection.read_bytes()
    output=tmp_path/"group2"
    argv=["opw_group2","--device",device,"--dataset","tiny","--dataset-file",str(dataset),
          "--flash-parameters",str(selection),"--gallery","4","--queries","2","--ks","1","3",
          "--fixed-iters","10","20","--max-iters","80","--check-every","10","--batch","4",
          "--memory-fraction",".25","--output",str(output)]
    monkeypatch.setattr(sys,"argv",argv)
    group2.main()
    assert selection.read_bytes()==before
    return output,argv


def test_train_only_pipeline_preserves_selection_scores_and_resume(tmp_path,monkeypatch):
    output,argv=_tiny(tmp_path,monkeypatch,"cpu")
    summary=json.loads((output/"summary.json").read_text())
    assert summary["status"]=="completed" and summary["flash_verification"]=="not_run_CPU_only"
    assert len(summary["results"])==54
    with np.load(output/"CPU_FP64_affine_tuned_fixed_20.npz") as z:
        assert z["dual_score"].shape==(2,4)
        assert set(z["query_indices"]).isdisjoint(z["gallery_indices"])
    before=(output/"contrasts.json").read_bytes()
    def unexpected(*args,**kwargs):
        raise AssertionError("Resume should reuse complete chunks")
    monkeypatch.setattr(group2,"dense_sweep",unexpected)
    monkeypatch.setattr(sys,"argv",argv+["--resume"])
    group2.main()
    assert (output/"contrasts.json").read_bytes()==before
    monkeypatch.setattr(sys,"argv",argv+["--resume","--tau",".002"])
    with pytest.raises(ValueError,match="Resume refused"):
        group2.main()


@pytest.mark.gpu
def test_gpu_flash_affine_and_dense_exact_pipeline_match_fixed_iteration_fp64(tmp_path,monkeypatch):
    output,_=_tiny(tmp_path,monkeypatch,"cuda")
    summary=json.loads((output/"summary.json").read_text())
    assert summary["status"]=="completed" and summary["flash_verification"]=="executed"
    assert not summary["numerical_parity_failures"] and len(summary["results"])==108
