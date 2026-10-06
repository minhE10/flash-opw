import json
import sys

import numpy as np
import pytest
import torch

import experiments.opw_group3 as group3
from experiments.opw_group3_audit import audit
from experiments.opw_parameters import atomic_json
from experiments.opw_scaling import synthetic_pair
from experiments.sequence_metrics import reference_distance
from flashopw import OPWParameters, materialize_opw_plan, opw_flash


@pytest.mark.parametrize("dataset", ["FacesUCR", "FaceAll", "Other"])
def test_equal_budget_and_flash_tlp_search_identical_costs(dataset):
    grid = group3.candidate_sets(dataset,12)
    assert set(grid) == set(group3.METRICS)
    for metric, candidates in grid.items():
        assert len(candidates) == (1 if metric in group3.UNTUNABLE else 12)
        assert len({json.dumps(p,sort_keys=True) for p in candidates}) == len(candidates)
    flash = sorted((round(OPWParameters(**p).mu,9),p["lambda2"]) for p in grid["flash-opw"])
    tlp = sorted((round(p["tlp_weight"],9),p["epsilon"]) for p in grid["tlp"])
    assert flash == tlp
    for budget in (2,3,6):
        smaller = group3.candidate_sets(dataset,budget)
        assert all(smaller[m] == grid[m][:len(smaller[m])] for m in grid)


@pytest.mark.parametrize("metric", ["flash-opw","sinkhorn","tlp","tcot","opw","opw-kl"])
@pytest.mark.parametrize("candidate", [0,2])
def test_fixed_checkpoint_matches_independent_scipy_oracle(metric,candidate):
    x,y = synthetic_pair(7,11,3,19)
    p = group3.candidate_sets("FacesUCR",3)[metric][candidate]
    policy = dict(tau=1e-20,max_iters=23,check_every=10)
    result = group3.dense_batch(metric,[x],[y],p,policy)
    oracle = reference_distance("affine-opw-dense" if metric == "flash-opw" else metric,
                                x,y,**p,n_iters=23,sinkhorn_iters=23)
    assert result["distances"][0] == pytest.approx(oracle,abs=1e-10)
    assert result["iterations"][0] == 23
    assert result["residuals"][0] > policy["tau"]


def test_active_batch_removal_preserves_per_pair_first_passing_checkpoint():
    x,y = synthetic_pair(7,11,1,4)
    p = group3.candidate_sets("FacesUCR",3)["sinkhorn"][0]
    xs,ys = [np.zeros_like(x),x],[np.zeros_like(y),y]
    policy = dict(tau=1e-3,max_iters=200,check_every=10)
    batched = group3.dense_batch("sinkhorn",xs,ys,p,policy)
    assert batched["iterations"][0] == 10 and batched["iterations"][1] > 10
    for i in range(2):
        single = group3.dense_batch("sinkhorn",[xs[i]],[ys[i]],p,policy)
        for key in batched:
            np.testing.assert_allclose(batched[key][i],single[key][0],rtol=1e-12,atol=1e-12)


def test_ineligible_accuracy_winner_cannot_be_frozen_and_tie_is_declared():
    rows = [dict(candidate=0,eligible=True,mean_ACC1=.5,mean_MAP=.6),
            dict(candidate=1,eligible=False,mean_ACC1=1.,mean_MAP=1.),
            dict(candidate=2,eligible=True,mean_ACC1=.5,mean_MAP=.7),
            dict(candidate=3,eligible=True,mean_ACC1=.5,mean_MAP=.7)]
    assert group3.choose_candidate(rows)["candidate"] == 2
    assert group3.choose_candidate([rows[1]]) is None


def test_map_breaks_accuracy_ties_despite_final_float_bit_roundoff():
    rows = [dict(candidate=0,eligible=True,mean_ACC1=.5+2e-16,mean_MAP=.6),
            dict(candidate=1,eligible=True,mean_ACC1=.5,mean_MAP=.7)]
    assert group3.choose_candidate(rows)["candidate"] == 1


def test_runner_reads_train_only_freezes_all_metrics_and_refuses_changed_resume(tmp_path,monkeypatch):
    dataset = tmp_path/"data.npz"
    rng = np.random.default_rng(19)
    np.savez(dataset,train_x=rng.normal(size=(12,4,1)),train_labels=np.repeat([0,1],6))
    output = tmp_path/"run"
    argv = ["group3","--dataset","Toy","--dataset-file",str(dataset),"--device","cpu",
            "--gallery","4","--queries","2","--budget","3","--max-iters","200",
            "--check-every","5","--tau",".01","--output",str(output)]
    monkeypatch.setattr(sys,"argv",argv)
    group3.main()
    artifact = group3.load_frozen(output/"selected_all_metrics.json",dataset="Toy")
    audited = audit(output,dataset_file=dataset)
    assert audited["status"] == "passed" and audited["jobs_verified"] == 75
    assert artifact["test_used"] is False
    assert len(artifact["splits"]) == 3
    for split in artifact["splits"]:
        assert not set(split["gallery_indices"]) & set(split["query_indices"])
    assert artifact["selected"]["opw-kl"]["parameters"]["sigma"] >= 1.
    # Alter only metadata would be caught by artifact validation.
    frozen = json.loads((output/"selected_all_metrics.json").read_text())
    frozen["test_used"] = True
    bad = tmp_path/"bad.json"
    atomic_json(bad,frozen)
    with pytest.raises(ValueError,match="isolation"):
        group3.load_frozen(bad)
    with pytest.raises(ValueError,match="dataset mismatch"):
        group3.load_frozen(output/"selected_all_metrics.json",dataset="FacesUCR")
    tampered = json.loads((output/"selected_all_metrics.json").read_text())
    tampered["selected"]["soft-dtw"]["candidate"] = (tampered["selected"]["soft-dtw"]["candidate"]+1)%3
    atomic_json(output/"selected_all_metrics.json",tampered)
    with pytest.raises(ValueError,match="Frozen selected mismatch"):
        audit(output,dataset_file=dataset)
    atomic_json(output/"selected_all_metrics.json",artifact)
    monkeypatch.setattr(sys,"argv",argv+["--resume"])
    group3.main()
    monkeypatch.setattr(sys,"argv",argv+["--resume","--tau",".02"])
    with pytest.raises(ValueError,match="Resume refused"):
        group3.main()


@pytest.mark.gpu
def test_gpu_metric_scores_match_fp64_at_fixed_iterations():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x,y = synthetic_pair(7,11,3,19)
    policy = dict(tau=1e-20,max_iters=23,check_every=10)
    for metric in ("flash-opw","sinkhorn","tlp","tcot","opw","opw-kl"):
        for p in group3.candidate_sets("FacesUCR",6)[metric]:
            ref = group3.dense_batch(metric,[x],[y],p,policy)
            gpu = group3.distance_matrix(metric,[x],[y],p,policy,device=torch.device("cuda"))
            np.testing.assert_allclose(gpu["distances"][0,0],ref["distances"][0],rtol=3e-3,atol=3e-4)
            np.testing.assert_allclose(gpu["residuals"][0,0],ref["residuals"][0],rtol=3e-3,atol=3e-4)
            if metric == "flash-opw":
                result = opw_flash(torch.as_tensor(x,dtype=torch.float32,device="cuda"),
                                   torch.as_tensor(y,dtype=torch.float32,device="cuda"),**p,n_iters=23)
                plan = materialize_opw_plan(result).double().cpu().numpy()
                from experiments.sequence_metrics import entropic_plan
                cost,_,eps,_ = group3.cost_tensors(metric,[x],[y],p)
                oracle = entropic_plan(cost[0].numpy(),eps,23)
                assert np.linalg.norm(plan-oracle)/np.linalg.norm(oracle) < 5e-3
