"""Independent reference and provenance checks; no CUDA result is claimed."""
import hashlib

import pytest
import torch

from scripts import author_ott_hessian as dependency
from scripts.check_author_early_stopping import classify, marginal, reference, symmetric_fixed


def test_symmetric_schedule_and_checked_reference_reach_same_balanced_plan():
    rng = torch.Generator().manual_seed(7)
    x,y = [0.2*torch.randn(k,3,generator=rng,dtype=torch.float64) for k in (9,7)]
    a,b = [torch.rand(k,generator=rng,dtype=torch.float64)+0.1 for k in (9,7)]
    a,b = a/a.sum(), b/b.sum()
    ref,info = reference(x,y,a,b,0.7,cap=300,tolerance=1e-10)
    actual = symmetric_fixed(x,y,a,b,0.7,100)
    assert info["converged"] and marginal(ref,a,b) < 1e-10
    assert marginal(actual,a,b) < 1e-10
    torch.testing.assert_close(actual,ref,rtol=1e-9,atol=1e-11)


def test_checked_reference_does_not_certify_infeasible_rounded_masses():
    x,y = torch.zeros(3,2,dtype=torch.float64),torch.ones(4,2,dtype=torch.float64)
    a,b = torch.full((3,),1/3,dtype=x.dtype),torch.full((4,),0.3,dtype=x.dtype)
    _,info = reference(x,y,a,b,0.7,cap=100,tolerance=1e-6)
    assert not info["converged"] and info["marginal_l1"] >= 0.199


def test_budget_diagnosis_keeps_reference_failure_and_gpu_nonconvergence_distinct():
    good = [{"marginal_l1":1e-5,"plan_relative_l2":1e-4,"threshold":1e-3}]
    bad = [{"marginal_l1":0.1,"plan_relative_l2":0.2}]
    assert classify(good,{"converged":False}) == "reference_not_converged"
    assert classify(bad,{"converged":True}) == "not_confirmed_at_cap"
    assert classify(good,{"converged":True}) == "marginal_and_reference_confirmed"
    assert classify([dict(good[0],threshold=None)],{"converged":True}) == "fixed_budget_confirmed_stop_not_certified"
    assert classify(good+[{"error":"CUDA error"}],{"converged":True}) == "failed"


@pytest.mark.parametrize("mutation", [None,"tamper","extra","wrong_commit"])
def test_external_api_inspection_checks_content_and_does_not_alias_missing_class(tmp_path,monkeypatch,mutation):
    sources = {"torch_sinkhorn_hessian.py":"class TorchSinkhornHessian: pass\nclass TorchOTResult: pass\nclass _TorchGeometry: pass\n",
               "SinkhornHessian.py":"def HessianA(): pass\n"}
    pin = {"commit":"pinned", "repository":"official", "files":{
        name:hashlib.sha256(data.encode()).hexdigest() for name,data in sources.items()}}
    monkeypatch.setattr(dependency,"PIN",pin)
    monkeypatch.setattr(dependency,"git",lambda root,*args:
        "other" if mutation == "wrong_commit" and args[0] == "rev-parse" else
        "pinned" if args[0] == "rev-parse" else "official")
    for name,data in sources.items(): (tmp_path/name).write_bytes(data.encode())
    if mutation == "tamper": (tmp_path/"SinkhornHessian.py").write_text("def HessianALineax(): pass\n")
    if mutation == "extra": (tmp_path/"adapter.py").write_text("HessianALineax = None\n")
    record = dependency.inspect(tmp_path)
    assert record["status"] == ("verified" if mutation is None else "failed")
    assert record["keops_api_present"]
    if mutation is None:
        assert record["jax_api_present"] is False
        assert (tmp_path/"SinkhornHessian.py").read_text() == sources["SinkhornHessian.py"]
