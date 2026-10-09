"""Read a returned evidence ZIP without extracting/executing its contents.

Recompute checksums, JUnit counts and convergence/coverage decisions. Source
verification records are checked against the local pin; the ZIP is evidence
of a run, not a fresh GPU execution or a signature authenticating the server.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

try:
    from .author_flashsinkhorn_profiles import expected_entries
    from .author_flashsinkhorn_sources import load_manifest
    from .run_author_extended_validation import summarize
except ImportError:
    from author_flashsinkhorn_profiles import expected_entries
    from author_flashsinkhorn_sources import load_manifest
    from run_author_extended_validation import summarize


def audit(path):
    errors = []
    manifest = load_manifest()
    expected_files = sorted(Path(name).name for name in manifest["files"]
                            if name.startswith("torch-ext/flash_sinkhorn/testing/test_") and name.endswith(".py"))
    with zipfile.ZipFile(path) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)) or sum(p.file_size for p in zipped.infolist()) > 512*2**20:
            raise ValueError("Duplicate archive entries or expanded archive exceeds 512 MiB")
        def read_json(name):
            return json.loads(zipped.read(name))
        checksums = read_json("artifact-sha256.json")
        if set(checksums) != set(names)-{"artifact-sha256.json"}:
            errors.append("Archive membership differs from checksum manifest")
        for name, digest in checksums.items():
            if name not in names or hashlib.sha256(zipped.read(name)).hexdigest() != digest:
                errors.append(f"Checksum mismatch: {name}")
        summary = read_json("suite-summary.json")
        scope = summary.get("scope", "full")  # Older artifacts always ran the full inventory.
        selected = summary.get("expected_files", [])
        if scope == "subset":
            if (not selected or len(selected) != len(set(selected)) or
                not set(selected) <= set(expected_files) or summary.get("suite_inventory") != expected_files):
                errors.append("Subset selection differs from pinned author inventory")
            expected_files = selected
        elif scope != "full" or selected != expected_files:
            errors.append("Suite inventory differs from pinned author inventory")
        for key in ("source", "source_after_tests"):
            record = summary.get(key, {})
            if record.get("status") != "verified" or record.get("commit") != manifest["commit"] or record.get("files_checked") != len(manifest["files"]):
                errors.append(f"Missing or incompatible source verification record: {key}")
        def check_profile(prefix):
            profile = read_json(prefix + "/kernel-profile.json")
            if profile.get("files") != expected_entries() or profile.get("upstream_commit") != manifest["commit"]:
                errors.append(f"Profile hash metadata differs from the expected compatibility derivative: {prefix}")
        runs = []
        for run in summary["runs"]:
            prefix = run["file"].removesuffix(".py")
            record = read_json(prefix + "/validation.json")
            cases = list(ET.fromstring(zipped.read(prefix + "/pytest.xml")).iter("testcase"))
            counts = {"total": len(cases), "failed": sum(c.find("failure") is not None for c in cases),
                      "errors": sum(c.find("error") is not None for c in cases),
                      "skipped": sum(c.find("skipped") is not None for c in cases)}
            counts["passed"] = len(cases)-sum(counts[k] for k in ("failed", "errors", "skipped"))
            if counts != run.get("tests") or counts != record.get("tests"):
                errors.append(f"JUnit count mismatch: {run['file']}")
            if record.get("source_after_tests", {}).get("status") != "verified" or record.get("profile_after_tests", {}).get("status") != "profile_verified":
                errors.append(f"Source/profile verification not complete: {run['file']}")
            if record.get("source", {}).get("commit") != manifest["commit"]:
                errors.append(f"Unexpected upstream commit: {run['file']}")
            if len(record.get("test_files", [])) != 1 or Path(record["test_files"][0]).name != run["file"]:
                errors.append(f"Unexpected test selection: {run['file']}")
            check_profile(prefix)
            runs.append(dict(run, tests=counts, status=record["status"]))
        if len(runs) != len({r["file"] for r in runs}):
            errors.append("Duplicate test-file runs")
        if "test_samples_loss_api.py" in expected_files:
            cg_path = "test_samples_loss_api/cg_convergence.json"
            cg = read_json(cg_path) if cg_path in names else {}
            records = cg.get("records", [])
            if cg.get("status") != "converged" or len(records) != 2 or {r["path"] for r in records} != {"autograd", "reference"}:
                errors.append("CG evidence does not contain both converged paths")
            for record in records:
                residual, initial = record.get("cg_residual"), record.get("cg_initial_residual")
                steps = record.get("cg_iters")
                if (not record.get("finite_output") or not record.get("cg_converged") or
                    not isinstance(steps, int) or not 0 <= steps <= summary.get("cg_cap", 0) or
                    not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in (residual, initial)) or
                    residual > max(1e-6, 1e-6*initial) or record.get("tau2") != 1e-5 or
                    record.get("cg_rtol") != 1e-6 or record.get("cg_atol") != 1e-6 or
                    record.get("max_cg_iter") != summary.get("cg_cap")):
                    errors.append(f"CG convergence/parameters failed independent audit: {record.get('path')}")
        numerics = read_json("independent/numerics.json")
        check_profile("independent")
        settings = numerics.get("settings", {})
        for key, value in {"eps":0.7, "forward_iterations":500, "reference_iterations":1000,
                           "tau2":1e-5, "cg_rtol":1e-6, "cg_atol":1e-6, "max_cg_iter":256,
                           "apply_rtol":2e-4, "apply_atol":2e-6}.items():
            if settings.get(key) != value:
                errors.append(f"Unexpected independent numerical setting: {key}")
        cases = numerics.get("cases", [])
        case_ids = {(tuple(c["shape"]), c["mode"], c["cost_scale"]) for c in cases}
        expected_ids = {(shape, mode, scale) for shape in ((17,23,3), (32,24,16), (37,29,33))
                        for mode in ("symmetric", "alternating") for scale in (0.5, 1.0)}
        if len(cases) != 12 or case_ids != expected_ids or numerics.get("status") != "passed":
            errors.append("Independent numerical study is incomplete or failed")
        for case in cases:
            if case.get("allow_tf32") is not False or set(case.get("apply_relative_l2", {})) != {"0", "1"}:
                errors.append("Independent case lacks IEEE/both-axis apply coverage")
            for value in case.get("apply_relative_l2", {}).values():
                if not isinstance(value, (int,float)) or not math.isfinite(value) or value < 0:
                    errors.append("Independent apply metric is nonfinite or invalid")
            for metric, limit in (("reference_marginal_l1",1e-5), ("gpu_marginal_l1",1e-5),
                                  ("plan_relative_l2",2e-4), ("hvp_relative_l2",5e-4),
                                  ("direct_system_relative_residual",1e-9)):
                value = case.get(metric)
                if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= limit:
                    errors.append(f"Independent metric failed audit: {metric}, {case.get('shape')}")
            info = case.get("cg", {})
            residual, initial = info.get("cg_residual"), info.get("cg_initial_residual")
            if (case.get("status") != "passed" or not info.get("cg_converged") or
                not all(isinstance(x,(int,float)) and math.isfinite(x) and x >= 0 for x in (residual,initial)) or
                residual > max(1e-6,1e-6*initial)):
                errors.append("Independent numerical case lacks confirmed CG convergence")
        computed = summarize(runs, numerics.get("status"), expected_files,
                             scope=scope if scope in ("full", "subset") else "full")
        if computed["status"] != summary.get("status") or computed["tests"] != summary.get("tests"):
            errors.append("Aggregate status/counts mismatch")
    return {"status": "failed_audit" if errors else computed["status"], "scope": scope, "tests": computed["tests"],
            "coverage_gaps": computed["coverage_gaps"], "errors": errors,
            "limitation": "Checks saved evidence; does not rerun CUDA or authenticate the remote host."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.archive)
    except Exception as exc:
        result = {"status": "failed_audit", "errors": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return 0 if result["status"] in ("passed_extended_compatibility", "passed_subset_compatibility") else 3 if result["status"] == "passed_with_coverage_gaps" else 1


if __name__ == "__main__":
    raise SystemExit(main())
