"""Audit synthetic evidence only; these tests do not claim CUDA execution."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.audit_author_extended_validation import audit
from scripts.author_flashsinkhorn_profiles import expected_entries
from scripts.author_flashsinkhorn_sources import load_manifest
from scripts.run_author_extended_validation import summarize


@pytest.mark.parametrize("mutation,expected", [(None, "passed_extended_compatibility"),
                                             ("skip", "passed_with_coverage_gaps"),
                                             ("checksum", "failed_audit"), ("count", "failed_audit"),
                                             ("cg", "failed_audit"), ("numerics", "failed_audit"),
                                             ("subset", "passed_subset_compatibility"),
                                             ("subset_as_full", "failed_audit"),
                                             ("unknown_subset", "failed_audit")])
def test_saved_evidence_audit_recomputes_counts_convergence_and_checksums(tmp_path, mutation, expected):
    manifest = load_manifest()
    files = sorted(Path(name).name for name in manifest["files"]
                   if name.startswith("torch-ext/flash_sinkhorn/testing/test_") and name.endswith(".py"))
    inventory = files[:]
    is_subset = mutation in ("subset", "subset_as_full", "unknown_subset")
    if is_subset:
        files = ["test_unbalanced_sinkhorn.py"]
    payload = {}
    def put(name, data):
        payload[name] = json.dumps(data).encode()
    source = {"status":"verified", "commit":manifest["commit"], "files_checked":len(manifest["files"])}
    profile = {"files":expected_entries(), "upstream_commit":manifest["commit"]}
    runs = []
    for index, name in enumerate(files):
        skipped = mutation == "skip" and index == 0
        prefix = name.removesuffix(".py")
        xml = '<testsuites><testsuite><testcase name="synthetic_a"/>'
        xml += '<testcase name="synthetic_b"><skipped message="optional dependency"/></testcase>' if skipped else '<testcase name="synthetic_b"/>'
        payload[prefix+"/pytest.xml"] = (xml+'</testsuite></testsuites>').encode()
        counts = {"total":2,"passed":1 if skipped else 2,"skipped":int(skipped),"failed":0,"errors":0}
        status = "passed_with_skips_compatibility" if skipped else "passed_compatibility"
        runs.append({"file":name,"exit_code":0,"status":status,"tests":counts})
        put(prefix+"/validation.json", {"source":source,"source_after_tests":source,
                                      "profile_after_tests":{"status":"profile_verified"},
                                      "test_files":["torch-ext/flash_sinkhorn/testing/"+name],"status":status,"tests":counts})
        put(prefix+"/kernel-profile.json", profile)
    summary = {"source":source,"source_after_tests":source,"expected_files":files,"runs":runs,"cg_cap":256}
    if is_subset:
        summary.update(scope="subset", suite_inventory=inventory)
    summary.update(summarize(runs, "passed", files, scope="subset" if is_subset else "full"))
    if mutation == "subset_as_full":
        summary.update(scope="full", status="passed_extended_compatibility")
    if mutation == "unknown_subset":
        summary["expected_files"] = ["test_unknown.py"]
    if mutation == "count":
        summary["tests"]["passed"] += 1
    put("suite-summary.json", summary)
    cg = {"status":"converged","records":[
        {"path":path,"cg_iters":149,"cg_residual":5e-7,"cg_initial_residual":1.0,"finite_output":True,"cg_converged":True,
         "tau2":1e-5,"cg_rtol":1e-6,"cg_atol":1e-6,"max_cg_iter":256} for path in ("autograd","reference")]}
    if mutation == "cg":
        cg["records"][0]["cg_residual"] = 2e-6
    if not is_subset:
        put("test_samples_loss_api/cg_convergence.json", cg)
    cases = [{"shape":list(shape),"mode":mode,"cost_scale":scale,"status":"passed","allow_tf32":False,
              "apply_relative_l2":{"0":1e-7,"1":1e-7}, "reference_marginal_l1":1e-7,"gpu_marginal_l1":1e-7,
              "plan_relative_l2":1e-7,"hvp_relative_l2":1e-7,"direct_system_relative_residual":1e-12,
              "cg":{"cg_converged":True,"cg_residual":5e-7,"cg_initial_residual":1.0}}
             for shape in ((17,23,3),(32,24,16),(37,29,33)) for mode in ("symmetric","alternating") for scale in (0.5,1.0)]
    if mutation == "numerics":
        cases[0]["hvp_relative_l2"] = 0.1
    put("independent/numerics.json", {"status":"passed","cases":cases,
        "settings":{"eps":0.7,"forward_iterations":500,"reference_iterations":1000,"tau2":1e-5,
                    "cg_rtol":1e-6,"cg_atol":1e-6,"max_cg_iter":256,"apply_rtol":2e-4,"apply_atol":2e-6}})
    put("independent/kernel-profile.json", profile)
    put("artifact-sha256.json", {name:hashlib.sha256(data).hexdigest() for name,data in payload.items()})
    if mutation == "checksum":
        payload["independent/numerics.json"] += b" "
    archive = tmp_path / "synthetic-evidence.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for name, data in payload.items():
            zipped.writestr(name, data)
    result = audit(archive)
    assert result["status"] == expected, result
    assert bool(result["errors"]) == (expected == "failed_audit")
