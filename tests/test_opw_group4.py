import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from experiments import opw_group4 as group4
from experiments.opw_group4_audit import audit
from experiments.opw_group4_resume import LEGACY_RUNNER_SHA256, PATCHED_RUNNER_SHA256, RUNNER, MIGRATION, signature_migration
from experiments.opw_parameters import atomic_json, source_hashes
from experiments.paired_statistics import holm_adjust, paired_query_statistics
from experiments.sequence_data import training_fingerprint
from experiments.sequence_metrics import reference_distance


def toy_frozen(tmp_path):
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((root/"reports/opw_group3_gpu_20261007/selected_all_metrics.json").read_text())
    rng = np.random.default_rng(9)
    train,test = rng.normal(size=(6,4,1)),rng.normal(size=(3,4,1))
    train_labels,test_labels = np.array(["a","b"]*3),np.array(["a","b","a"])
    path = tmp_path/"toy.npz"
    np.savez(path,train_x=train,train_labels=train_labels,test_x=test,test_labels=test_labels)
    frozen.update(dataset="Toy",training_sha256=training_fingerprint(train,train_labels),sources=source_hashes(),
                  solver_policy=dict(tau=.01,max_iters=60,check_every=10,schedule="f_then_g"))
    selection = tmp_path/"frozen.json"
    atomic_json(selection,frozen)
    return path,selection,train,test,train_labels,test_labels


def test_exact_mcnemar_and_paired_map_uncertainty():
    labels = np.array(["a"]*10)
    left = dict(average_precision=[.8]*10,predictions={"1":["a"]*10})
    right = dict(average_precision=[.7]*10,predictions={"1":["b"]*10})
    result = paired_query_statistics(left,right,labels,resamples=100,seed=1)
    assert result["delta_MAP_pp"] == pytest.approx(10)
    assert result["map_difference_ci95_pp"] == pytest.approx([10,10])
    assert result["left_only_correct"] == 10 and result["right_only_correct"] == 0
    assert result["mcnemar_exact_two_sided_p"] == pytest.approx(2/2**10)
    same = paired_query_statistics(left,left,labels,resamples=100,seed=1)
    assert same["map_difference_ci95_pp"] == [0,0]
    assert same["mcnemar_exact_two_sided_p"] == 1
    flipped = paired_query_statistics(right,left,labels,resamples=100,seed=1)
    assert flipped["delta_ACC1_pp"] == -100


def test_bootstrap_is_paired_and_reproducible_and_holm_is_in_original_order():
    left = dict(average_precision=[.2,.3,.7,.9],predictions={"1":["a"]*4})
    right = dict(average_precision=[.1,.2,.6,.8],predictions={"1":["a"]*4})
    # Common query effects cancel; independent resampling would widen this CI.
    stats = paired_query_statistics(left,right,["a"]*4,resamples=500,seed=5)
    assert stats["map_difference_ci95_pp"] == pytest.approx([10,10])
    assert stats == paired_query_statistics(left,right,["a"]*4,resamples=500,seed=5)
    assert holm_adjust([.04,.01,.03]) == pytest.approx([.06,.03,.06])
    assert holm_adjust([]) == []
    with pytest.raises(ValueError):
        paired_query_statistics(left,right,["a"]*3,resamples=500,seed=5)
    with pytest.raises(ValueError,match="prediction lengths"):
        paired_query_statistics(left,dict(right,predictions={"1":["a"]}),["a"]*4,resamples=100)


def test_full_test_all_metrics_native_oracles_and_identical_profiles_are_reused(tmp_path,monkeypatch):
    data,selection,train,test,tl,ql = toy_frozen(tmp_path)
    output = tmp_path/"run"
    argv = ["group4","--selection",str(selection),"--dataset-file",str(data),"--device","cpu",
            "--ks","1","3","--query-chunk","2","--bootstrap-resamples","100","--output",str(output)]
    monkeypatch.setattr(sys,"argv",argv)
    group4.main()
    summary = json.loads((output/"summary.json").read_text())
    env = json.loads((output/"environment.json").read_text())
    assert summary["gallery"] == 6 and summary["queries"] == 3
    assert summary["test_used_for_selection"] is False
    assert len(summary["matrix_hashes"]) == len(summary["quality"]) == 22
    assert len(env["signature"]["configs"]) == 17
    assert summary["completed_chunks"] == summary["total_chunks"] == 34
    checked = audit(output,dataset_file=data)
    assert checked["status"] == "passed" and checked["chunks_verified"] == 34
    # Untunable and unchanged selected TLp configurations are solved once.
    for metric in ("dtw","ldtw","ndtw","ot","tlp"):
        assert summary["matrix_hashes"][f"selected__{metric}"] == summary["matrix_hashes"][f"preset__{metric}"]
    frozen = json.loads(selection.read_text())
    x,y = test[0].astype(np.float32).astype(np.float64),train[0].astype(np.float32).astype(np.float64)
    for metric in group4.DEFAULT_METRICS:
        path = output/summary["matrix_hashes"][f"selected__{metric}"]["file"]
        with np.load(path) as matrix:
            iterations = int(matrix["iterations"][0,0])
            p = frozen["selected"][metric]["parameters"]
            oracle = reference_distance("affine-opw-dense" if metric == "flash-opw" else metric,
                                       x,y,**p,n_iters=max(1,iterations),sinkhorn_iters=max(1,iterations))
            assert matrix["distances"][0,0] == pytest.approx(oracle,abs=1e-10)
            np.testing.assert_array_equal(matrix["query_indices"],np.arange(3))
    paired = json.loads((output/"paired_statistics.json").read_text())
    assert len(paired) == 20
    assert all(row["holm_family_size"] == 10 for row in paired)
    assert {r["right"] for r in paired if r["priority"] == "primary"} == {"opw","tlp"}
    # A completed resume must read all chunks without executing any solver.
    def no_solve(*args,**kwargs):
        raise AssertionError("cached chunks should be reused")
    monkeypatch.setattr(group4,"distance_matrix",no_solve)
    # Simulate the known v1 run metadata. Every saved numerical array stays
    # intact; the narrow migration must accept exactly this reader-only fix.
    old_environment = copy.deepcopy(env)
    old_environment["signature"]["sources"][RUNNER] = LEGACY_RUNNER_SHA256
    del old_environment["signature"]["sources"][MIGRATION]
    atomic_json(output/"environment.json",old_environment)
    monkeypatch.setattr(sys,"argv",argv+["--resume"])
    group4.main()
    assert json.loads((output/"paired_statistics.json").read_text()) == paired
    assert json.loads((output/"environment.before_checkpoint_validator_fix.json").read_text()) == old_environment
    migrated = json.loads((output/"environment.json").read_text())
    assert migrated["signature"] == env["signature"]
    assert migrated["checkpoint_validator_migration"]["old_runner_sha256"] == LEGACY_RUNNER_SHA256
    assert audit(output,dataset_file=data)["status"] == "passed"
    # Repeated resumes do not overwrite the original provenance backup.
    group4.main()
    assert json.loads((output/"environment.before_checkpoint_validator_fix.json").read_text()) == old_environment
    bad_stats = json.loads((output/"paired_statistics.json").read_text())
    bad_stats[0]["delta_MAP_pp"] += 1
    atomic_json(output/"paired_statistics.json",bad_stats)
    with pytest.raises(ValueError,match="paired_statistics.json mismatch"):
        audit(output,dataset_file=data)
    atomic_json(output/"paired_statistics.json",paired)
    monkeypatch.setattr(sys,"argv",argv+["--resume","--query-chunk","1"])
    with pytest.raises(ValueError,match="Resume refused"):
        group4.main()
    monkeypatch.setattr(sys,"argv",argv+["--resume"])
    chunk = next((output/"chunks").glob("*.npz"))
    chunk.write_bytes(chunk.read_bytes()+b"tamper")
    with pytest.raises(ValueError,match="identity/hash"):
        group4.main()


def test_interrupt_resume_keeps_completed_queries_and_test_bytes_are_locked(tmp_path,monkeypatch):
    data,selection,train,test,tl,ql = toy_frozen(tmp_path)
    output = tmp_path/"run"
    argv = ["group4","--selection",str(selection),"--dataset-file",str(data),"--device","cpu",
        "--profiles","selected","--metrics","flash-opw","opw","--ks","1","3",
        "--query-chunk","1","--bootstrap-resamples","100","--output",str(output)]
    original = group4.distance_matrix
    calls = []
    def interrupt(metric,queries,*args,**kwargs):
        calls.append(metric)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return original(metric,queries,*args,**kwargs)
    monkeypatch.setattr(sys,"argv",argv)
    monkeypatch.setattr(group4,"distance_matrix",interrupt)
    with pytest.raises(KeyboardInterrupt):
        group4.main()
    state = json.loads((output/"run_state.json").read_text())
    assert state["status"] == "interrupted" and state["completed_chunks"] == 1
    resumed = []
    def count(metric,queries,*args,**kwargs):
        resumed.append(metric)
        return original(metric,queries,*args,**kwargs)
    monkeypatch.setattr(group4,"distance_matrix",count)
    monkeypatch.setattr(sys,"argv",argv+["--resume"])
    group4.main()
    assert len(resumed) == 5  # First of six chunks survived interruption.
    test[0,0,0] += 1
    np.savez(data,train_x=train,train_labels=tl,test_x=test,test_labels=ql)
    with pytest.raises(ValueError,match="Resume refused"):
        group4.main()


def test_training_mismatch_is_refused_before_test_load(tmp_path,monkeypatch):
    data,selection,*_ = toy_frozen(tmp_path)
    frozen = json.loads(selection.read_text())
    frozen["training_sha256"] = "wrong"
    atomic_json(selection,frozen)
    def no_test_read(*args,**kwargs):
        raise AssertionError("TEST must not be opened with invalid TRAIN selection")
    monkeypatch.setattr(group4,"load_sequences",no_test_read)
    monkeypatch.setattr(sys,"argv",["group4","--selection",str(selection),"--dataset-file",str(data),
        "--device","cpu","--output",str(tmp_path/"run")])
    with pytest.raises(ValueError,match="TRAIN fingerprint"):
        group4.main()


def test_fp32_stop_diagnostic_boundary_is_quality_status_not_corrupt_cache(tmp_path):
    policy = dict(tau=.001,max_iters=4000,check_every=50)
    identity = dict(metric="flash-opw",parameters=dict(lambda1=1.,lambda2=.1,sigma=1.,cost_scale=1.),start=177,stop=178)
    # Exact reported failure: diagnostic reduction exceeds tau by 2.49e-9.
    residual = .001000002492219209
    values = dict(distances=np.array([[.2]]),residuals=np.array([[residual]]),iterations=np.array([[450.]]))
    path,jobpath = tmp_path/"chunk.npz",tmp_path/"chunk.json"
    group4.save_chunk(path,jobpath,values,identity,np.array(["a"]),np.array(["a"]),0.)
    loaded,_ = group4.load_chunk(path,jobpath,identity=identity,gallery_labels=np.array(["a"]),
                               query_labels=np.array(["a"]),policy=policy,entropic=True)
    assert loaded["residuals"][0,0] == residual  # No rounding or relaxed tau.
    config = dict(metric="flash-opw",parameters=identity["parameters"],aliases=[dict(profile="selected",candidate=1)])
    ev = dict(ACC={"1":1.},MAP=1.,queries_without_relevant_gallery=0)
    row = group4.result_rows(config,ev,loaded,policy,device_type="cuda",seconds=0,ks=[1])[0]
    assert row["unconverged_pairs"] == 1
    assert row["max_marginal_l1"] == residual


@pytest.mark.parametrize("iteration", [0,451,4050,450.5])
def test_invalid_iteration_metadata_still_refuses_checkpoint(tmp_path,iteration):
    identity = dict(metric="flash-opw",parameters={},start=0,stop=1)
    values = dict(distances=np.array([[.2]]),residuals=np.array([[.0005]]),iterations=np.array([[iteration]]))
    path,jobpath = tmp_path/"chunk.npz",tmp_path/"chunk.json"
    group4.save_chunk(path,jobpath,values,identity,np.array(["a"]),np.array(["a"]),0.)
    with pytest.raises(ValueError,match="Invalid cached"):
        group4.load_chunk(path,jobpath,identity=identity,gallery_labels=np.array(["a"]),query_labels=np.array(["a"]),
                         policy=dict(tau=.001,max_iters=4000,check_every=50),entropic=True)


def test_reader_signature_migration_cannot_bypass_numeric_or_settings_changes():
    actual = hashlib.sha256(Path(group4.__file__).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    assert actual == PATCHED_RUNNER_SHA256
    current = dict(sources={RUNNER:actual,MIGRATION:"helper", "flashsinkhorn/solver.py":"frozen"},
                   policy=dict(tau=.001),settings=dict(query_chunk=4),origin=dict(test_sha256="fixed"))
    prior = copy.deepcopy(current)
    prior["sources"][RUNNER] = LEGACY_RUNNER_SHA256
    del prior["sources"][MIGRATION]
    assert signature_migration(prior,current)["new_runner_sha256"] == actual
    assert signature_migration(current,current) is None
    for field,key,value in (("sources","flashsinkhorn/solver.py","changed"),("sources",RUNNER,"unknown-reader"),
                            ("settings","query_chunk",8),("policy","tau",.01),("origin","test_sha256","changed")):
        changed = copy.deepcopy(current)
        changed[field][key] = value
        with pytest.raises(ValueError,match="Resume refused"):
            signature_migration(prior,changed)


@pytest.mark.gpu
def test_gpu_full_test_pipeline_keeps_native_main_and_journal_scores(tmp_path,monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    data,selection,train,test,_,_ = toy_frozen(tmp_path)
    output = tmp_path/"gpu"
    monkeypatch.setattr(sys,"argv",["group4","--selection",str(selection),"--dataset-file",str(data),
        "--device","cuda","--profiles","selected","--metrics","flash-opw","opw","tlp", "--ks","1","3",
        "--query-chunk","2","--memory-fraction",".25","--bootstrap-resamples","100","--output",str(output)])
    group4.main()
    summary = json.loads((output/"summary.json").read_text())
    frozen = json.loads(selection.read_text())
    for metric in ("flash-opw","opw","tlp"):
        with np.load(output/summary["matrix_hashes"][f"selected__{metric}"]["file"]) as saved:
            iterations = int(saved["iterations"][0,0])
            expected = reference_distance("affine-opw-dense" if metric == "flash-opw" else metric,
                test[0].astype(np.float32).astype(np.float64),train[0].astype(np.float32).astype(np.float64),
                **frozen["selected"][metric]["parameters"],n_iters=iterations,sinkhorn_iters=iterations)
            assert saved["distances"][0,0] == pytest.approx(expected,rel=3e-3,abs=3e-4)
    assert audit(output,dataset_file=data)["status"] == "passed"
