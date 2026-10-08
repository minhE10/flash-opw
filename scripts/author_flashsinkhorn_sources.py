"""Prepare and verify the immutable author checkout and its verbatim working copy.

This script has no Torch/Triton dependency. The upstream commit and file hashes
are pinned in flash_sinkhorn_author/UPSTREAM.json; it never follows remote HEAD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "flash_sinkhorn_author"
REFERENCE = ROOT / "external" / "flash-sinkhorn-upstream"
MANIFEST = IMPLEMENTATION / "UPSTREAM.json"


def git(*args: str, cwd: Path) -> bytes:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.check_output(["git", *args], cwd=cwd, env=env)


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def file_errors(root: Path, entries: dict, *, read_only: bool = False) -> list[str]:
    errors = []
    for name, expected in entries.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest path: {name}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            errors.append(f"Missing or non-regular file: {name}")
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected["sha256"]:
            errors.append(f"Content differs from author commit: {name}")
        # Git blob identity also verifies the recorded SHA256 against Git.
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != expected["git_blob"]:
            errors.append(f"Git blob differs from author commit: {name}")
        if read_only:
            mode = path.stat()
            if os.name == "nt":
                locked = bool(mode.st_file_attributes & stat.FILE_ATTRIBUTE_READONLY)
            else:
                locked = not bool(mode.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            if not locked:
                errors.append(f"Reference file is writable: {name}")
    # Prevent an additional Python module from silently changing the import tree.
    if root.is_dir():
        for path in root.rglob("*.py"):
            name = path.relative_to(root).as_posix()
            if ".git" not in path.relative_to(root).parts and name not in entries:
                errors.append(f"Unexpected Python source: {name}")
    return errors


def verify(*, implementation_only: bool = False) -> dict:
    manifest = load_manifest()
    entries = manifest["files"]
    errors = file_errors(IMPLEMENTATION, entries)
    if not implementation_only:
        if not (REFERENCE / ".git").is_dir():
            errors.append("Author checkout missing; run the prepare command first.")
        else:
            head = git("rev-parse", "HEAD", cwd=REFERENCE).decode().strip()
            if head != manifest["commit"]:
                errors.append(f"Reference HEAD {head} differs from pinned commit.")
            # Compare the manifest to the actual pinned Git tree, not just two copies.
            tree = {}
            for record in git("ls-tree", "-rz", "--full-tree", manifest["commit"], cwd=REFERENCE).split(b"\0"):
                if not record:
                    continue
                metadata, name = record.split(b"\t", 1)
                mode, kind, blob = metadata.decode().split()
                tree[name.decode("utf-8")] = {"mode": mode, "git_blob": blob, "kind": kind}
            if set(tree) != set(entries):
                errors.append("Manifest paths differ from the pinned author Git tree.")
            for name in set(tree) & set(entries):
                if tree[name]["kind"] != "blob" or any(
                    tree[name][key] != entries[name][key] for key in ("mode", "git_blob")
                ):
                    errors.append(f"Manifest metadata differs from pinned Git tree: {name}")
            errors.extend(file_errors(REFERENCE, entries, read_only=True))
            if os.name != "nt":
                for path in [REFERENCE, *[p for p in REFERENCE.rglob("*") if p.is_dir()]]:
                    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                        errors.append(f"Reference directory is writable: {path.relative_to(REFERENCE)}")
    result = {
        "commit": manifest["commit"], "version": manifest["version"],
        "files_checked": len(entries), "implementation": str(IMPLEMENTATION),
        "reference_checked": not implementation_only,
        "status": "verified" if not errors else "failed", "errors": errors,
    }
    return result


def seal_reference() -> None:
    # On Windows the native read-only file attribute is used; it is not an ACL.
    # On POSIX remove all write bits from both files and directories, including .git.
    for path in REFERENCE.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Unexpected reference symlink: {path}")
        if path.is_file() or (os.name != "nt" and path.is_dir()):
            path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    if os.name != "nt":
        REFERENCE.chmod(REFERENCE.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def prepare() -> dict:
    manifest = load_manifest()
    if not REFERENCE.exists():
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "git", "clone", "--no-checkout", manifest["repository"], str(REFERENCE)
        ], check=True)
        subprocess.run([
            "git", "-c", "core.autocrlf=false", "checkout", "--detach", manifest["commit"]
        ], cwd=REFERENCE, check=True)
    # Existing checkouts are never reset, cleaned, pulled or overwritten.
    if not (REFERENCE / ".git").is_dir():
        raise RuntimeError(f"Refusing to replace existing non-Git directory: {REFERENCE}")
    head = git("rev-parse", "HEAD", cwd=REFERENCE).decode().strip()
    if head != manifest["commit"]:
        raise RuntimeError(f"Existing checkout is at {head}, expected {manifest['commit']}; no changes made.")
    errors = file_errors(REFERENCE, manifest["files"])
    if errors:
        raise RuntimeError("Refusing to seal a modified author checkout:\n" + "\n".join(errors))
    seal_reference()
    return verify()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--implementation-only", action="store_true",
                        help="Offline copy check only; does not verify the author checkout.")
    args = parser.parse_args()
    if args.command == "prepare" and args.implementation_only:
        parser.error("--implementation-only applies only to verify")
    try:
        result = prepare() if args.command == "prepare" else verify(implementation_only=args.implementation_only)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
