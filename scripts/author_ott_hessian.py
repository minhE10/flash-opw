"""Pin and inspect the external public baseline; never patch its API or formulas."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

PIN = json.loads(Path(__file__).with_name("author_ott_hessian_pin.json").read_text())


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def inspect(root):
    root = Path(root).resolve()
    errors = []
    if git(root, "rev-parse", "HEAD") != PIN["commit"]:
        errors.append("Dependency HEAD differs from pin")
    if git(root, "remote", "get-url", "origin") != PIN["repository"]:
        errors.append("Dependency origin differs from official repository")
    for name, digest in PIN["files"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f"Dependency hash mismatch: {name}")
    unexpected = [p.relative_to(root).as_posix() for p in root.rglob("*.py")
                  if ".git" not in p.relative_to(root).parts and p.relative_to(root).as_posix() not in PIN["files"]]
    errors.extend(f"Unexpected dependency Python file: {name}" for name in unexpected)
    names = {}
    for name in ("torch_sinkhorn_hessian.py", "SinkhornHessian.py"):
        tree = ast.parse((root / name).read_bytes())
        names[name] = sorted(n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)))
    return {"status": "verified" if not errors else "failed", "root": str(root),
            "repository": PIN["repository"], "commit": PIN["commit"], "files_checked": len(PIN["files"]),
            "files_sha256": PIN["files"], "definitions": names,
            "keops_api_present": {"TorchSinkhornHessian", "TorchOTResult", "_TorchGeometry"} <= set(names["torch_sinkhorn_hessian.py"]),
            "jax_api_present": "HessianALineax" in names["SinkhornHessian.py"], "errors": errors}


def prepare(root):
    root = Path(root).resolve()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-c", "core.autocrlf=false", PIN["repository"], str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", PIN["commit"]], check=True)
    # Existing trees are only inspected; never reset or overwrite them.
    record = inspect(root)
    if record["status"] != "verified":
        raise ValueError("Dependency verification failed: " + "; ".join(record["errors"]))
    return record


def history(root):
    """Search all fetched reachable Python histories, without asserting private code is absent."""
    commits = git(root, "rev-list", "--all").splitlines()
    matches = []
    for commit in commits:
        result = subprocess.run(["git", "-C", str(root), "grep", "-l", "HessianALineax", commit, "--", "*.py"],
                                text=True, capture_output=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        matches.extend(result.stdout.splitlines())
    return {"reachable_commits_searched": len(commits), "shallow": git(root, "rev-parse", "--is-shallow-repository"),
            "refs": git(root, "for-each-ref", "--format=%(refname)", "refs/remotes", "refs/tags").splitlines(),
            "HessianALineax_matches": matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = prepare(args.root) if args.prepare else inspect(args.root)
    if args.history:
        record["history"] = history(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in record.items() if k not in ("files_sha256", "definitions")}, indent=2))
    return 0 if record["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
