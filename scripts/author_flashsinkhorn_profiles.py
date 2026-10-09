"""Build an auditable launch-configuration derivative of the pinned author copy.

Only apply-plan matrix launch controls change. Both original source trees and
all author tests remain intact. This module needs neither Torch nor Triton.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path

try:
    from .author_flashsinkhorn_sources import IMPLEMENTATION, file_errors, load_manifest
except ImportError:  # Direct script execution.
    from author_flashsinkhorn_sources import IMPLEMENTATION, file_errors, load_manifest


PROFILE = "rtx5080"
TARGET = "torch-ext/flash_sinkhorn/kernels/apply_flash.py"
FUNCTION = "apply_plan_mat_flashstyle"
OVERRIDES = {
    "block_m": 32, "block_n": 32, "block_k": 16, "block_d": 16,
    "num_stages": 1, "autotune": False,
}


def patched_source(source: bytes) -> bytes:
    """Insert assignments before the original function body; preserve its bytes."""
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == FUNCTION]
    if len(functions) != 1:
        raise ValueError(f"Expected exactly one {FUNCTION} in pinned source")
    function = functions[0]
    first_statement = function.body[1] if ast.get_docstring(function) is not None else function.body[0]
    lines = source.splitlines(keepends=True)
    newline = b"\r\n" if lines[first_statement.lineno - 1].endswith(b"\r\n") else b"\n"
    insertion = [b"    # RTX 5080 compatibility: launch controls only; author math/tests unchanged." + newline]
    insertion.extend(f"    {key} = {value!r}".encode() + newline for key, value in OVERRIDES.items())
    insertion.append(newline)
    result = b"".join(lines[:first_statement.lineno - 1] + insertion + lines[first_statement.lineno - 1:])
    ast.parse(result)
    return result


def expected_entries() -> dict:
    """Derive hashes from pinned input, never from the potentially modified output."""
    entries = {name: dict(entry) for name, entry in load_manifest()["files"].items()}
    data = patched_source((IMPLEMENTATION / TARGET).read_bytes())
    entries[TARGET]["sha256"] = hashlib.sha256(data).hexdigest()
    entries[TARGET]["git_blob"] = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    return entries


def verify_profile(root: Path) -> dict:
    entries = expected_entries()
    errors = file_errors(root, entries)
    return {"status": "profile_verified" if not errors else "failed",
            "files_checked": len(entries), "errors": errors}


def prepare_profile(output: Path) -> tuple[Path, dict]:
    """Create a fresh per-run copy, plus a diff and hashes for every source file."""
    destination = output / "implementation-rtx5080"
    if destination.exists():
        raise ValueError(f"Refusing to overwrite an existing profile tree: {destination}")
    manifest = load_manifest()
    errors = file_errors(IMPLEMENTATION, manifest["files"])
    if errors:
        raise ValueError("Cannot derive a profile from modified author sources: " + "; ".join(errors))
    original = (IMPLEMENTATION / TARGET).read_bytes()
    modified = patched_source(original)
    for name in manifest["files"]:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(modified if name == TARGET else (IMPLEMENTATION / name).read_bytes())
    patch = "".join(difflib.unified_diff(
        original.decode().splitlines(keepends=True), modified.decode().splitlines(keepends=True),
        fromfile=f"upstream/{TARGET}", tofile=f"{PROFILE}/{TARGET}",
    ))
    (output / "kernel-profile.patch").write_text(patch, encoding="utf-8")
    metadata = {
        "name": PROFILE, "upstream_commit": manifest["commit"],
        "implementation": str(destination), "changed_files": [TARGET],
        "function": FUNCTION, "overrides": OVERRIDES,
        "scope": "All calls to apply_plan_mat_flashstyle, including HVP and direct/re-exported calls",
        "author_tests_unchanged": True, "numerical_tolerances_unchanged": True,
        "limitations": [
            "Matrix-apply autotuning is disabled; autotune-parity tests in this profile compare fixed launches and do not validate that autotuner.",
            "This fixed configuration is for correctness validation, not an optimized performance benchmark.",
        ],
        "files": expected_entries(),
    }
    (output / "kernel-profile.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata = {key: value for key, value in metadata.items() if key != "files"}
    metadata["verification"] = verify_profile(destination)
    return destination, metadata
