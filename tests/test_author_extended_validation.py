"""Coverage accounting must not turn skipped or crashed files into passes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_author_extended_validation import summarize


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
