"""The author import must preserve Git blobs, detect drift, and protect its reference."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

from scripts.author_flashsinkhorn_sources import file_errors, verify


def _entry(data):
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(),
    }


def test_import_matches_pinned_manifest():
    result = verify(implementation_only=True)
    assert result["status"] == "verified", result["errors"]
    assert result["reference_checked"] is False


def test_detects_changed_missing_and_extra_source(tmp_path):
    data = b"# author\nvalue = 1\n"
    module = tmp_path / "kernel.py"
    module.write_bytes(data)
    entries = {"kernel.py": _entry(data)}
    assert file_errors(tmp_path, entries) == []
    # A line-ending conversion is a source change, even if Python still runs.
    module.write_bytes(data.replace(b"\n", b"\r\n"))
    assert any("Content differs" in error for error in file_errors(tmp_path, entries))
    module.unlink()
    assert any("Missing" in error for error in file_errors(tmp_path, entries))
    module.write_bytes(data)
    (tmp_path / "shadow.py").write_text("value = 2\n")
    assert any("Unexpected Python source" in error for error in file_errors(tmp_path, entries))


def test_requires_read_only_reference(tmp_path):
    path = tmp_path / "kernel.py"
    data = b"value = 1\n"
    path.write_bytes(data)
    entries = {"kernel.py": _entry(data)}
    assert any("writable" in error for error in file_errors(tmp_path, entries, read_only=True))
    old_mode = path.stat().st_mode
    try:
        path.chmod(old_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        assert file_errors(tmp_path, entries, read_only=True) == []
        if os.name == "nt":
            assert path.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY
    finally:
        path.chmod(old_mode)


def test_gpu_runner_refuses_to_report_pass_without_allocated_gpu(tmp_path):
    # Use a fake verified source to exercise the runner's admission rule offline.
    # No actual Torch/Triton import or CUDA execution occurs in this test.
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_author_flashsinkhorn.py"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    program = """
import runpy, sys
sys.path.insert(0, sys.argv[1])
import author_flashsinkhorn_sources as sources
sources.verify = lambda **kwargs: {'status': 'verified', 'reference_checked': True}
sys.argv = [sys.argv[2], '--gpu', '--output', sys.argv[3]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", program, str(script.parent), str(script), str(tmp_path)],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 2, result.stdout + result.stderr
    summary = json.loads((tmp_path / "validation.json").read_text())
    assert summary["gpu_status"] == "unavailable"
    assert summary["status"] == "failed"
    assert "tests" not in summary
