"""Investigate external HVP tests and the exact early-stop fixture on one GPU."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import zipfile

from author_flashsinkhorn_profiles import prepare_profile, verify_profile
from author_flashsinkhorn_sources import ROOT, verify
from author_ott_hessian import prepare, inspect, history


def main():
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not device or "," in device or device == "-1":
        raise SystemExit("Set CUDA_VISIBLE_DEVICES to one allocated GPU")
    output = ROOT/"outputs"/("author_flashsinkhorn_steps12_"+datetime.now().strftime("%Y%m%d_%H%M%S")+f"_{os.getpid()}")
    output.mkdir(parents=True, exist_ok=False)
    print(f"[steps12] Run: {output}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(ROOT/"scripts")
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = "2"
    report = {"status":"running", "timestamp_utc":datetime.now(timezone.utc).isoformat(),
              "scope":"Steps 1 and 2 only; public baseline API investigation and exact early-stop fixture",
              "source":verify(), "limitations":["No API aliases or baseline fixes are applied.",
                 "Missing HessianALineax remains a coverage gap.", "Saved evidence requires audit; not a speed benchmark."]}
    def save():
        (output/"steps12-summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    def run(command, name):
        print(f"[steps12] Starting {name}",flush=True)
        with (output/name).open("w",encoding="utf-8") as log:
            process = subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,text=True,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            done = threading.Event()
            def beat():
                while not done.wait(30):
                    if process.poll() is None:
                        print(f"[steps12] Child PID {process.pid} still running; log: {output/name}",flush=True)
            thread = threading.Thread(target=beat,daemon=True)
            thread.start()
            try:
                for line in process.stdout:
                    print(line,end="",flush=True)
                    log.write(line)
                    log.flush()
                return process.wait()
            finally:
                done.set()
                thread.join(timeout=1)
    save()
    try:
        if report["source"]["status"] != "verified":
            raise RuntimeError("Author source verification failed")
        dependency = ROOT/"outputs/author-dependencies/OTT-Hessian"
        report["dependency"] = prepare(dependency)
        report["dependency"]["history"] = history(dependency)
        save()
        for command,name in (([sys.executable,"-m","pip","freeze"],"pip-freeze.txt"),
                             (["git","rev-parse","HEAD"],"project-commit.txt"),
                             (["nvidia-smi","-i",device],"nvidia-smi.txt")):
            result = subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True,timeout=60)
            (output/name).write_text(result.stdout+result.stderr,encoding="utf-8")
        code = run([sys.executable,"-u","-B",ROOT/"scripts/validate_author_flashsinkhorn.py",
            "--gpu","--suite","full","--test-file","test_hvp_parity.py","--kernel-profile","rtx5080",
            "--ott-hessian-root",dependency,"--output",output/"hvp-parity"],"hvp-parity.log")
        path = output/"hvp-parity/validation.json"
        report["hvp_exit_code"] = code
        report["hvp"] = json.loads(path.read_text()) if path.exists() else {"status":"missing_report"}
        save()
        # Numerical diagnosis is independent of an optional baseline test failing.
        profile_dir = output/"early-profile"
        profile_dir.mkdir()
        implementation,report["early_profile"] = prepare_profile(profile_dir)
        code = run([sys.executable,"-u","-B",ROOT/"scripts/check_author_early_stopping.py",
                    "--implementation",implementation,"--output",output/"early-stopping"],"early-stopping.log")
        path = output/"early-stopping/early-stopping.json"
        report["early_exit_code"] = code
        report["early_status"] = json.loads(path.read_text())["status"] if path.exists() else "missing_report"
        report["early_profile_after"] = verify_profile(implementation)
        report["source_after"] = verify()
        report["dependency_after"] = inspect(dependency)
        counts = report["hvp"].get("tests",{})
        integrity = (report["source_after"]["status"] == "verified" and
                     report["dependency_after"]["status"] == "verified" and
                     report["early_profile_after"]["status"] == "profile_verified")
        report["status"] = "failed_or_incomplete"
        if (integrity and report["hvp_exit_code"] == 0 and counts.get("passed",0) > 0 and
            not counts.get("failed",0) and not counts.get("errors",0) and
            report["early_status"] in ("marginal_and_reference_confirmed", "fixed_budget_confirmed_stop_not_certified")):
            gaps = counts.get("skipped",0) or report["early_status"] == "fixed_budget_confirmed_stop_not_certified"
            report["status"] = "completed_with_coverage_gaps" if gaps else "steps12_confirmed"
    except Exception as exc:
        report.update(status="failed_or_incomplete",error=f"{type(exc).__name__}: {exc}")
    finally:
        report["source_final"] = verify()
        if report["source_final"]["status"] != "verified":
            report["status"] = "failed_or_incomplete"
        save()
        paths = [p for p in output.rglob("*") if p.is_file() and p.suffix in (".json",".log",".xml",".txt",".patch")
                 and not any(part.startswith("implementation-") or part == "pytest-cache" for part in p.relative_to(output).parts)]
        hashes = {p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        checksum = output/"artifact-sha256.json"
        checksum.write_text(json.dumps(hashes,indent=2)+"\n",encoding="utf-8")
        with zipfile.ZipFile(output.with_suffix(".zip"),"x",zipfile.ZIP_DEFLATED) as z:
            for p in paths+[checksum]: z.write(p,p.relative_to(output).as_posix())
        print(f"[steps12] Artifact bundle: {output.with_suffix('.zip')}",flush=True)
        print(f"[steps12] {report['status']}; HVP: {report.get('hvp',{}).get('tests',{})}; early: {report.get('early_status')}",flush=True)
    return 0 if report["status"] == "steps12_confirmed" else 3 if report["status"] == "completed_with_coverage_gaps" else 1


if __name__ == "__main__":
    raise SystemExit(main())
