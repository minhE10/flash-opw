"""Coverage accounting must not turn skipped or crashed files into passes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_author_extended_validation import summarize
from scripts import run_author_extended_validation as runner


def test_geomloss_requirement_rejects_other_version(monkeypatch):
    monkeypatch.setattr(runner.metadata, "version", lambda name: "0.2.6")
    with pytest.raises(ValueError, match="0.3.1 required.*0.2.6"):
        runner.check_geomloss_version("0.3.1")


def test_full_run_keeps_external_failure_and_continues_all_20_files(tmp_path, monkeypatch):
    # Simulate child boundaries: verifies orchestration, never claims a GPU pass.
    commands = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "verify", lambda: {"status":"verified"})
    monkeypatch.setattr(runner, "inspect_dependency", lambda root: {"status":"verified"})
    monkeypatch.setattr(runner.metadata, "version", lambda name: "0.3.1")
    monkeypatch.setattr(runner, "prepare_profile", lambda root: (root/"implementation-rtx5080", {}))
    monkeypatch.setattr(runner, "verify_profile", lambda root: {"status":"profile_verified"})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    dependency = tmp_path/"dependency"
    output = tmp_path/"outputs"/"full"
    monkeypatch.setattr(sys, "argv", ["runner", "--output", str(output),
        "--ott-hessian-root", str(dependency), "--require-geomloss-version", "0.3.1"])
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs:
        subprocess.CompletedProcess(command, 0, stdout="simulated", stderr=""))
    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.pid, self.stdout, self.code = 123, iter([]), 0
            target = Path(command[command.index("--output")+1])
            if any("check_author_flashsinkhorn_numerics.py" in arg for arg in command):
                target.write_text(json.dumps({"status":"passed"}))
            else:
                name = command[command.index("--test-file")+1]
                failed = name == "test_hvp_parity.py"
                self.code = int(failed)
                target.mkdir()
                (target/"validation.json").write_text(json.dumps({
                    "status":"failed" if failed else "passed_compatibility", "source":{"status":"verified"},
                    "tests":{"total":1,"passed":int(not failed),"failed":int(failed),"errors":0,"skipped":0}}))
        def wait(self): return self.code
        def poll(self): return self.code
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    assert runner.main() == 1
    summary = json.loads((output/"suite-summary.json").read_text())
    assert summary["scope"] == "full" and len(summary["runs"]) == 20
    assert summary["failed_files"] == ["test_hvp_parity.py"] and summary["missing_files"] == []
    assert summary["tests"]["passed"] == 19 and summary["tests"]["failed"] == 1
    for command in commands[1:]:
        name = command[command.index("--test-file")+1]
        assert ("--ott-hessian-root" in command) == (name == "test_hvp_parity.py")
        if name == "test_hvp_parity.py":
            assert command[command.index("--ott-hessian-root")+1] == str(dependency.resolve())
        if name == "test_samples_loss_api.py":
            assert command[command.index("--hvp-max-cg-iter")+1] == "256"
    assert summary["geomloss_before_tests"] == summary["geomloss_after_tests"]
    assert output.with_suffix(".zip").exists()


def test_selected_pass_is_labelled_as_subset():
    runs = [{"file":"a.py", "exit_code":0, "status":"passed_compatibility", "tests":{"passed":1}}]
    assert summarize(runs, "passed", ["a.py"], scope="subset")["status"] == "passed_subset_compatibility"
    with pytest.raises(ValueError, match="Unknown validation scope"):
        summarize(runs, "passed", ["a.py"], scope="unknown")


@pytest.mark.parametrize("selection", [["test_unknown.py"], ["test_half_cost.py", "test_half_cost.py"]])
def test_invalid_selection_rejected_before_gpu_or_output_creation(tmp_path, selection):
    script = Path(__file__).resolve().parents[1] / "scripts/run_author_extended_validation.py"
    command = [sys.executable, str(script), "--output", str(tmp_path / "unused")]
    for name in selection:
        command += ["--test-file", name]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 2
    assert "unique basenames" in result.stderr
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("kind,expected", [("passed", "passed_extended_compatibility"),
                                          ("skipped", "passed_with_coverage_gaps"),
                                          ("error", "failed"), ("missing", "failed")])
def test_extended_summary_preserves_coverage_gaps_and_failures(kind, expected):
    runs = [{"file": "a.py", "exit_code": 0, "status": "passed_compatibility",
             "tests": {"total": 3, "passed": 3, "failed": 0, "errors": 0, "skipped": 0}}]
    if kind != "missing":
        runs.append({"file": "b.py", "exit_code": {"passed": 0, "skipped": 3, "error": 1}[kind],
                     "status": {"passed": "passed_compatibility", "skipped": "not_validated_all_skipped", "error": "failed"}[kind],
                     "tests": {"total": 1, "passed": int(kind == "passed"), "skipped": int(kind == "skipped"),
                               "errors": int(kind == "error"), "failed": 0}})
    result = summarize(runs, "passed", ["a.py", "b.py"])
    assert result["status"] == expected
    assert result["tests"]["passed"] == 3 + int(kind == "passed")
    assert summarize(runs, "failed", ["a.py", "b.py"])["status"] == "failed"


@pytest.mark.parametrize("xml,child_exit,status,exit_code", [
    ('<testcase name="module"><skipped message="optional dependency absent"/></testcase>', 5, "not_validated_all_skipped", 3),
    ('<testcase name="module"><error message="dependency incompatible"/></testcase>', 2, "failed", 1),
    ('<testcase name="test"><failure message="wrong output"/></testcase>', 1, "failed", 1),
])
def test_one_file_runner_distinguishes_collection_skip_error_and_assertion_failure(tmp_path, xml, child_exit, status, exit_code):
    # Simulate subprocess boundaries only; no numerical/GPU pass is claimed.
    script = Path(__file__).resolve().parents[1] / "scripts/validate_author_flashsinkhorn.py"
    program = """
import json, pathlib, runpy, subprocess, sys
sys.path.insert(0, sys.argv[1])
import author_flashsinkhorn_sources as sources
sources.verify = lambda **kwargs: {'status':'verified'}
target, output, xml, code = sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4], int(sys.argv[5])
def probe(command, **kwargs):
    return subprocess.CompletedProcess(command, 0, stdout=json.dumps({'gpu':'simulated','free_vram_mib':8000,'total_vram_mib':16000}), stderr='')
class Process:
    def __init__(self, command, **kwargs):
        assert command[-1].endswith('test_ott_vs_triton.py')
        pathlib.Path(command[command.index('--junitxml')+1]).write_text('<testsuites><testsuite>'+xml+'</testsuite></testsuites>')
        self.stdout = iter([])
    def wait(self):
        return code
subprocess.run, subprocess.Popen = probe, Process
sys.argv = [target,'--gpu','--suite','full','--test-file','test_ott_vs_triton.py','--output',str(output)]
runpy.run_path(target, run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", program, str(script.parent), str(script), str(tmp_path), xml, str(child_exit)],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="1"), text=True, capture_output=True)
    assert result.returncode == exit_code, result.stdout + result.stderr
    summary = json.loads((tmp_path / "validation.json").read_text())
    assert summary["status"] == status
    assert summary["nonpassing_cases"][0]["outcomes"][0]["message"]
