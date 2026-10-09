"""Guard the math-preserving derivative and actual matrix-kernel launch routing."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.author_flashsinkhorn_profiles import (
    FUNCTION, OVERRIDES, TARGET, patched_source, prepare_profile, verify_profile,
)
from scripts.author_flashsinkhorn_sources import IMPLEMENTATION, load_manifest


def test_profile_changes_only_launch_controls_and_preserves_all_author_tests(tmp_path):
    root, metadata = prepare_profile(tmp_path)
    assert metadata["verification"]["status"] == "profile_verified"
    for name in load_manifest()["files"]:
        if name != TARGET:
            assert (root / name).read_bytes() == (IMPLEMENTATION / name).read_bytes(), name
    original = ast.parse((IMPLEMENTATION / TARGET).read_bytes())
    modified = ast.parse((root / TARGET).read_bytes())
    function = next(node for node in modified.body if isinstance(node, ast.FunctionDef) and node.name == FUNCTION)
    # Remove only our six leading launch assignments (after the untouched docstring).
    del function.body[1:1 + len(OVERRIDES)]
    assert ast.dump(original, include_attributes=False) == ast.dump(modified, include_attributes=False)
    assert metadata["numerical_tolerances_unchanged"]
    assert "num_stages = 1" in (tmp_path / "kernel-profile.patch").read_text()
    # The verifier must use pinned expectations, not trust edited output metadata.
    test_file = root / "torch-ext/flash_sinkhorn/testing/test_samples_loss_tf32.py"
    test_file.write_bytes(test_file.read_bytes() + b"\n# drift\n")
    assert verify_profile(root)["status"] == "failed"
    with pytest.raises(ValueError, match="overwrite"):
        prepare_profile(tmp_path)


@pytest.mark.parametrize("axis", [0, 1])
def test_matrix_launch_overrides_reach_both_axis_kernels_without_changing_math_flags(axis):
    # Execute the actual patched Python launcher with recording GPU call sites.
    # No fake numerical results are compared and no CUDA pass is claimed.
    source = patched_source((IMPLEMENTATION / TARGET).read_bytes())
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == FUNCTION)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             function], type_ignores=[])
    ast.fix_missing_locations(module)

    class Tensor:
        is_cuda = True
        device = "cuda"
        def __init__(self, shape):
            self.shape, self.ndim = shape, len(shape)
        def stride(self, axis):
            return self.shape[1] if self.ndim == 2 and axis == 0 else 1
        def float(self):
            return self
        def contiguous(self):
            return self

    calls = []
    class Kernel:
        def __init__(self, name):
            self.name = name
        def __getitem__(self, grid):
            def record(*args, **kwargs):
                calls.append((self.name, grid, kwargs))
            return record

    namespace = {
        "torch": SimpleNamespace(float32="fp32", empty=lambda shape, **kwargs: Tensor(shape)),
        "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
        "_validate_device": lambda *args: None,
        "_cache_key_bucket": lambda size: size,
    }
    for direction in (0, 1):
        for suffix in ("", "_autotune"):
            name = f"_apply_plan_axis{direction}_mat_flashstyle_kernel{suffix}"
            namespace[name] = Kernel(name)
    exec(compile(module, str(IMPLEMENTATION / TARGET), "exec"), namespace)
    n, m, d = 37, 23, 40  # Non-multiples exercise the tiling grid, including output D.
    namespace[FUNCTION](Tensor((n, d)), Tensor((m, d)), Tensor((n,)), Tensor((m,)),
                        Tensor((n,)), Tensor((m,)), Tensor((m if axis else n, d)),
                        eps=0.2, axis=axis, block_m=128, block_n=128, block_k=64, block_d=64,
                        num_stages=3, autotune=True, allow_tf32=True, use_exp2=False,
                        cost_scale=0.5)
    name, grid, options = calls[0]
    assert name == f"_apply_plan_axis{axis}_mat_flashstyle_kernel"
    assert grid == (((n if axis else m) + 31) // 32, (d + 15) // 16)
    assert options["num_stages"] == 1
    assert {key: options[key] for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "BLOCK_D")} == {
        "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_D": 16,
    }
    assert options["ALLOW_TF32"] is True and options["USE_EXP2"] is False


def test_offline_runner_labels_compatibility_and_keeps_original_source_verified(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/validate_author_flashsinkhorn.py"
    result = subprocess.run([sys.executable, str(script), "--kernel-profile", "rtx5080",
                             "--implementation-only", "--output", str(tmp_path)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "validation.json").read_text())
    assert summary["status"] == "static_profile_verified_gpu_pending"
    assert summary["gpu_status"] == "not_run"
    assert summary["source"]["status"] == "verified"
    assert summary["profile"]["changed_files"] == [TARGET]
    assert "tests" not in summary


@pytest.mark.parametrize("tamper", [False, True])
def test_gpu_runner_routes_profile_and_checks_post_run_hashes(tmp_path, tamper):
    # Simulate child processes to test orchestration, never execute CUDA tests.
    script = Path(__file__).resolve().parents[1] / "scripts/validate_author_flashsinkhorn.py"
    program = """
import json, pathlib, runpy, subprocess, sys
sys.path.insert(0, sys.argv[1])
import author_flashsinkhorn_sources as sources
sources.verify = lambda **kwargs: {'status': 'verified', 'reference_checked': True}
target, output, tamper = sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4] == 'True'
expected = output / 'implementation-rtx5080'
def probe(command, **kwargs):
    assert command[-1] == str(expected)
    assert kwargs['cwd'] == expected
    assert kwargs['env']['PYTHONPATH'] == str(expected / 'torch-ext')
    return subprocess.CompletedProcess(command, 0, stdout=json.dumps({
        'gpu': 'simulated', 'free_vram_mib': 8000, 'total_vram_mib': 16000}), stderr='')
class Process:
    def __init__(self, command, **kwargs):
        assert command[5] == str(expected)
        assert str(expected / 'pyproject.toml') in command
        assert kwargs['env']['PYTHONPATH'] == str(expected / 'torch-ext')
        assert command[-3].endswith('test_samples_loss_tf32.py::test_default_tf32_hvp_reaches_input')
        report = pathlib.Path(command[command.index('--junitxml') + 1])
        report.write_text('<testsuites><testsuite><testcase name="simulated"/></testsuite></testsuites>')
        if tamper:
            path = expected / 'torch-ext/flash_sinkhorn/kernels/apply_flash.py'
            path.write_bytes(path.read_bytes() + b'\\n# unapproved change\\n')
        self.stdout = iter(['simulated child log\\n'])
    def wait(self):
        return 0
subprocess.run, subprocess.Popen = probe, Process
sys.argv = [target, '--gpu', '--suite', 'regressions', '--kernel-profile', 'rtx5080', '--output', str(output)]
runpy.run_path(target, run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", program, str(script.parent), str(script), str(tmp_path), str(tamper)],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="1"), text=True, capture_output=True)
    assert result.returncode == (1 if tamper else 0), result.stdout + result.stderr
    summary = json.loads((tmp_path / "validation.json").read_text())
    assert summary["status"] == ("failed" if tamper else "passed_compatibility")
    assert summary["profile_after_tests"]["status"] == ("failed" if tamper else "profile_verified")
