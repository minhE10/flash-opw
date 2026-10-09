"""Run independent numerics and every pinned author test file, then bundle evidence.

Requires the existing author CUDA environment. Does not install dependencies,
edit upstream tests, suppress warnings, or terminate other GPU processes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile

try:
    from .author_flashsinkhorn_profiles import prepare_profile, verify_profile
    from .author_flashsinkhorn_sources import ROOT, load_manifest, verify
except ImportError:  # Direct script execution.
    from author_flashsinkhorn_profiles import prepare_profile, verify_profile
    from author_flashsinkhorn_sources import ROOT, load_manifest, verify


def summarize(runs, independent, expected_files, scope="full"):
    if scope not in ("full", "subset"):
        raise ValueError(f"Unknown validation scope: {scope}")
    covered = {r["file"] for r in runs}
    failures = [r["file"] for r in runs if r["exit_code"] not in (0, 3)
                or r.get("tests", {}).get("failed", 0) or r.get("tests", {}).get("errors", 0)
                or (r["exit_code"] == 0 and not r.get("status", "").startswith("passed"))
                or (r["exit_code"] == 3 and r.get("status") != "not_validated_all_skipped")]
    gaps = [r["file"] for r in runs if r["exit_code"] == 3 or r.get("tests", {}).get("skipped", 0)]
    counts = {key: sum(r.get("tests", {}).get(key, 0) for r in runs)
              for key in ("total", "passed", "failed", "errors", "skipped")}
    # These counts also include module-level collection skips/errors from JUnit;
    # they are not estimates of how many parametrized tests would have existed.
    if failures or independent != "passed" or covered != set(expected_files):
        status = "failed"
    else:
        status = "passed_with_coverage_gaps" if gaps else (
            "passed_subset_compatibility" if scope == "subset" else "passed_extended_compatibility")
    return {"status": status, "tests": counts, "failed_files": failures, "coverage_gaps": gaps,
            "missing_files": sorted(set(expected_files)-covered), "independent_status": independent}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--test-file", action="append", help="Repeat to rerun selected pinned test basenames; default is all 20 files")
    parser.add_argument("--cg-cap", type=int, default=256, help="Previously validated scoped fixture budget (default 256)")
    args = parser.parse_args()
    if args.cg_cap <= 64:
        parser.error("--cg-cap must exceed the original fixture budget of 64")
    manifest = load_manifest()
    inventory = sorted(Path(name).name for name in manifest["files"]
                       if name.startswith("torch-ext/flash_sinkhorn/testing/test_") and name.endswith(".py"))
    if args.test_file and (len(set(args.test_file)) != len(args.test_file) or not set(args.test_file) <= set(inventory)):
        parser.error("--test-file must contain unique basenames from the pinned author inventory")
    files = sorted(args.test_file) if args.test_file else inventory
    scope = "subset" if args.test_file else "full"
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not device or "," in device or device == "-1":
        parser.error("Set CUDA_VISIBLE_DEVICES to exactly one allocated GPU")
    output = (args.output or ROOT / "outputs" / (
        "author_flashsinkhorn_extended_" + datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    )).resolve()
    # Fresh outputs only, inside the ignored outputs tree.
    if ROOT / "outputs" not in output.parents:
        parser.error("--output must be a fresh directory inside outputs/")
    if output.exists():
        parser.error("Refusing to reuse an existing output directory")
    output.mkdir(parents=True)
    print(f"[extended] Run: {output}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(ROOT / "scripts")
    # Avoid JAX reserving most of VRAM; each test file still gets a fresh process.
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = "2"
    source = verify()
    summary = {"status": "running", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "source": source, "kernel_profile": "rtx5080", "cg_cap": args.cg_cap,
               "scope": scope, "suite_inventory": inventory,
               "expected_files": files, "runs": [], "independent_status": "not_run",
               "limitations": ["Small independent balanced cases are not a full Hessian proof or speed benchmark.",
                               "Matrix-apply autotuning is disabled by the compatibility profile.",
                               "Author skips remain coverage gaps; no dependency is auto-installed.",
                               "JUnit totals may include module-level skip/collection-error entries."]}

    def save():
        (output / "suite-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def run(command, log_path):
        print(f"[extended] {' '.join(map(str, command))}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            done, started = threading.Event(), time.monotonic()
            def heartbeat():
                while not done.wait(30):
                    if process.poll() is None:
                        print(f"[extended] Child PID {process.pid} still running ({time.monotonic()-started:.0f}s); log: {log_path}", flush=True)
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return process.wait()
            finally:
                done.set()
                thread.join(timeout=1)

    def bundle():
        paths = [p for p in output.rglob("*") if p.is_file() and p.suffix in (".json", ".log", ".xml", ".patch", ".txt")
                 and not any(part.startswith("implementation-") or part == "pytest-cache" for part in p.relative_to(output).parts)]
        hashes = {str(p.relative_to(output)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        checksum_path = output / "artifact-sha256.json"
        checksum_path.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
        archive = output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
            for path in paths + [checksum_path]:
                zipped.write(path, str(path.relative_to(output)))
        print(f"[extended] Artifact bundle: {archive}", flush=True)

    save()
    if source["status"] != "verified":
        summary.update(status="failed", error="Source integrity verification failed")
        save()
        bundle()
        return 1
    for command, name in (([sys.executable, "-m", "pip", "freeze"], "pip-freeze.txt"),
                          (["git", "rev-parse", "HEAD"], "project-commit.txt"),
                          (["git", "status", "--short"], "project-status.txt"),
                          (["nvidia-smi", "-i", device], "nvidia-smi.txt")):
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
            (output / name).write_text(result.stdout + result.stderr, encoding="utf-8")
        except (OSError, subprocess.TimeoutExpired) as exc:
            (output / name).write_text(str(exc), encoding="utf-8")
    try:
        independent_dir = output / "independent"
        independent_dir.mkdir()
        implementation, summary["independent_profile"] = prepare_profile(independent_dir)
        code = run([sys.executable, "-u", "-B", ROOT / "scripts/check_author_flashsinkhorn_numerics.py",
                    "--implementation", implementation, "--output", independent_dir / "numerics.json"],
                   independent_dir / "numerics.log")
        result_path = independent_dir / "numerics.json"
        summary["independent_status"] = (json.loads(result_path.read_text())["status"] if result_path.exists() else "failed")
        summary["independent_profile_after"] = verify_profile(implementation)
        if code or summary["independent_profile_after"]["status"] != "profile_verified":
            summary["independent_status"] = "failed"
        save()
        for index, name in enumerate(files, 1):
            print(f"[extended] File {index}/{len(files)}: {name}", flush=True)
            run_dir = output / name.removesuffix(".py")
            command = [sys.executable, "-u", "-B", ROOT / "scripts/validate_author_flashsinkhorn.py",
                       "--gpu", "--suite", "full", "--test-file", name, "--kernel-profile", "rtx5080", "--output", run_dir]
            if name == "test_samples_loss_api.py":
                command += ["--hvp-max-cg-iter", args.cg_cap]
            code = run(command, output / (name + ".log"))
            path = run_dir / "validation.json"
            result = json.loads(path.read_text()) if path.exists() else {}
            summary["runs"].append({"file": name, "exit_code": code, "status": result.get("status", "missing_report"),
                                    "tests": result.get("tests", {}), "report": str(path.relative_to(output))})
            save()
            if result.get("gpu_status") == "unavailable" or result.get("source", {}).get("status") != "verified":
                print("[extended] Stopping after failed environment/source preflight; remaining files are not validated.", flush=True)
                break
        summary.update(summarize(summary["runs"], summary["independent_status"], files, scope=scope))
        summary["source_after_tests"] = verify()
        if summary["source_after_tests"]["status"] != "verified":
            summary["status"] = "failed"
    except KeyboardInterrupt:
        summary.update(status="interrupted", error="User interrupted; unfinished files are not validated")
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    save()
    bundle()
    print(f"[extended] {summary['status']}: {summary.get('tests', {})}", flush=True)
    return 0 if summary["status"] in ("passed_extended_compatibility", "passed_subset_compatibility") else 3 if summary["status"] == "passed_with_coverage_gaps" else 1


if __name__ == "__main__":
    raise SystemExit(main())
