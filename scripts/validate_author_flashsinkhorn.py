"""Validate the author's source without importing the legacy flashsinkhorn package.

Static verification works on CPU. GPU validation executes the unmodified
author tests, in a separate process, and refuses to count a CPU skip as a pass.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from author_flashsinkhorn_sources import IMPLEMENTATION, REFERENCE, ROOT, verify


CORE_TESTS = (
    "test_flashstyle_parity.py", "test_apply_plan_flashstyle.py",
    "test_sinkhorn_triton.py", "test_half_cost.py", "test_samples_loss_api.py",
    "test_samples_loss_tf32.py", "test_hvp_sqeuclid.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--suite", choices=("core", "full"), default="core")
    parser.add_argument("--implementation-only", action="store_true")
    parser.add_argument("--preflight-timeout", type=float, default=180,
                        help="Maximum seconds for imports and CUDA environment checks (default: 180).")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "author_flashsinkhorn_validation")
    args = parser.parse_args()
    if args.preflight_timeout <= 0:
        parser.error("--preflight-timeout must be positive")
    if args.gpu and args.implementation_only:
        parser.error("GPU validation requires both the immutable reference and the working copy")
    output = args.output.resolve()
    if any(output == protected or protected in output.parents for protected in (IMPLEMENTATION, REFERENCE)):
        parser.error("Validation outputs must be outside both author source trees")
    if args.gpu and any((output / name).exists() for name in ("validation.json", "pytest.xml", "pytest.log")):
        parser.error("GPU output already contains validation artifacts; choose a fresh --output")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "gpu_requested": args.gpu, "gpu_status": "not_run", "suite": args.suite,
               "status": "running", "phase": "verifying_source"}

    def save():
        (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    save()
    print(f"Validation output: {output}", flush=True)
    print("[source] Verifying pinned author files and read-only reference...", flush=True)
    summary["source"] = verify(implementation_only=args.implementation_only)
    save()
    if summary["source"]["status"] != "verified":
        summary["status"] = "failed"
        save()
        print(json.dumps(summary["source"], indent=2))
        return 1
    # Parse tracked Python files only, without importing kernels or fabricating Triton.
    print("[source] Source verified; parsing Python files...", flush=True)
    summary["phase"] = "parsing_source"
    save()
    manifest = json.loads((IMPLEMENTATION / "UPSTREAM.json").read_text(encoding="utf-8"))
    python_names = [name for name in manifest["files"] if name.endswith(".py")]
    for name in python_names:
        ast.parse((IMPLEMENTATION / name).read_bytes(), filename=name)
    summary["python_files_parsed"] = len(python_names)
    print(f"[source] Parsed {len(python_names)} Python files.", flush=True)
    if not args.gpu:
        summary["status"] = "static_verified_gpu_pending"
        summary["phase"] = "complete"
        save()
        print(f"Source verified; {len(python_names)} Python files parsed. CUDA tests NOT run.")
        return 0

    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selected.strip() or "," in selected or selected.strip() == "-1":
        summary["gpu_status"] = "unavailable"
        summary["status"] = "failed"
        summary["error"] = "Set CUDA_VISIBLE_DEVICES to exactly one allocated GPU."
        save()
        print(summary["error"], file=sys.stderr)
        return 2
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # Both package and test resolution must prefer this exact copy over site-packages.
    env["PYTHONPATH"] = str(IMPLEMENTATION / "torch-ext")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    probe = """
import json, pathlib, sys, torch, triton
assert torch.cuda.is_available(), 'CUDA PyTorch required'
assert torch.cuda.device_count() == 1, 'Exactly one visible GPU required'
import flash_sinkhorn
package = pathlib.Path(flash_sinkhorn.__file__).resolve()
assert pathlib.Path(sys.argv[1]).resolve() in package.parents, str(package)
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
print(json.dumps({'python': sys.version, 'torch': torch.__version__,
    'triton': triton.__version__, 'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0), 'package': str(package),
    'version': flash_sinkhorn.__version__,
    'free_vram_mib': free_bytes / 2**20, 'total_vram_mib': total_bytes / 2**20}))
"""
    summary["phase"] = "checking_gpu"
    save()
    print(f"[gpu] Importing Torch/Triton and checking CUDA (timeout {args.preflight_timeout:g}s)...", flush=True)
    try:
        preflight = subprocess.run([sys.executable, "-u", "-B", "-c", probe, str(IMPLEMENTATION)],
                                   cwd=IMPLEMENTATION, env=env, text=True, capture_output=True,
                                   timeout=args.preflight_timeout)
    except subprocess.TimeoutExpired as exc:
        summary["gpu_status"] = "unavailable"
        summary["status"] = "failed"
        summary["error"] = f"GPU preflight exceeded {args.preflight_timeout:g}s; CUDA tests were not started."
        save()
        partial = []
        for stream in (exc.stdout, exc.stderr):
            if stream:
                partial.append(stream.decode("utf-8", errors="replace") if isinstance(stream, bytes) else stream)
        (output / "preflight.log").write_text("".join(partial) + "\n" + summary["error"] + "\n", encoding="utf-8")
        print(summary["error"], file=sys.stderr, flush=True)
        return 2
    (output / "preflight.log").write_text(preflight.stdout + preflight.stderr, encoding="utf-8")
    if preflight.returncode:
        summary["gpu_status"] = "unavailable"
        summary["status"] = "failed"
        summary["error"] = preflight.stderr.strip()
        save()
        print(preflight.stderr, file=sys.stderr)
        return 2
    summary["environment"] = json.loads(preflight.stdout.strip().splitlines()[-1])
    gpu_env = summary["environment"]
    print(f"[gpu] {gpu_env['gpu']}; free VRAM {gpu_env['free_vram_mib']:.0f}/{gpu_env['total_vram_mib']:.0f} MiB.", flush=True)
    test_root = IMPLEMENTATION / "torch-ext" / "flash_sinkhorn" / "testing"
    tests = [test_root] if args.suite == "full" else [test_root / name for name in CORE_TESTS]
    report = output / "pytest.xml"
    # Each run has a fresh XML file so an interrupted run cannot reuse old results.
    if report.exists():
        parser.error(f"Output already contains a test report: {report}; choose a new --output")
    test_program = """
import sys, torch, pytest
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.45)
raise SystemExit(pytest.main(sys.argv[1:]))
"""
    command = [sys.executable, "-u", "-B", "-c", test_program, "-v", "-ra", "-c",
               str(IMPLEMENTATION / "pyproject.toml"), "--rootdir", str(IMPLEMENTATION),
               "-o", f"cache_dir={output / 'pytest-cache'}", "--junitxml", str(report),
               *map(str, tests)]
    summary["gpu_status"] = "running"
    summary["phase"] = "running_tests"
    summary["test_files"] = [str(path.relative_to(IMPLEMENTATION)) for path in tests]
    save()
    print(f"[tests] Starting {args.suite} author suite; Triton compilation/autotuning may take time.", flush=True)
    with (output / "pytest.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=IMPLEMENTATION, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    summary["pytest_exit_code"] = returncode
    summary["phase"] = "verifying_source_after_tests"
    save()
    print("[source] Checking source hashes after tests...", flush=True)
    summary["source_after_tests"] = verify()
    if report.exists():
        cases = list(ET.parse(report).getroot().iter("testcase"))
        summary["tests"] = {
            "total": len(cases), "failed": sum(c.find("failure") is not None for c in cases),
            "errors": sum(c.find("error") is not None for c in cases),
            "skipped": sum(c.find("skipped") is not None for c in cases),
        }
        summary["tests"]["passed"] = len(cases) - sum(summary["tests"][k] for k in ("failed", "errors", "skipped"))
    passed = summary.get("tests", {}).get("passed", 0)
    ok = returncode == 0 and passed > 0 and summary["source_after_tests"]["status"] == "verified"
    summary["gpu_status"] = ("passed_with_skips" if summary["tests"]["skipped"] else "passed") if ok else "failed"
    summary["status"] = summary["gpu_status"]
    summary["phase"] = "complete"
    save()
    print(f"[result] {summary['status']}: {summary.get('tests', {})}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
