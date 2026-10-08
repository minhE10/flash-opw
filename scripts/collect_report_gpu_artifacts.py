"""Collect existing GPU artifacts for the experiment report; no GPU operations."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--dry-run", action="store_true", help="inspect without writing an archive")
    parser.add_argument("--export-dir", type=Path,
                        help="copy artifacts and manifest into a new repository directory instead of an archive")
    args = parser.parse_args()
    root = args.root.resolve()
    checkpoints = (20, 100, 200, 500, 1000, 2000)
    groups = {
        "outputs/opw_group1_gpu_v1": [
            "summary.json", "run_state.json", "parity.json", "environment.json",
            "convergence.json", "convergence.csv", "reference_state.json",
            *[f"{profile}_{kind}_{step}.npz" for profile in ("default", "tuned")
              for kind in ("flash", "reference") for step in checkpoints],
        ],
        "outputs/paper_20261005T090633.880402Z": [
            "environment.json", "paper_results.json", "paper_results.csv",
            "autotuning.json", "diagnostics.json",
        ],
    }
    files = []
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "groups": {}}
    for directory, names in groups.items():
        folder = root / directory
        found, missing = [], []
        print(f"\n{directory}")
        for name in names:
            source = folder / name
            if source.is_file() and not source.is_symlink():
                source.resolve().relative_to(root)
                files.append(source)
                found.append(name)
            else:
                missing.append(name)
        print(f"Found {len(found)}/{len(names)} files")
        if missing:
            print("Missing: " + ", ".join(missing))
        info = {"found": found, "missing": missing}
        for name in ("environment.json", "summary.json"):
            if name in found:
                try:
                    data = json.loads((folder / name).read_text(encoding="utf-8-sig"))
                    if not isinstance(data, dict):
                        raise ValueError("Expected a JSON object")
                    fields = ("device", "gpu", "status", "flash_verification", "parity_failures")
                    info[name] = {key: data[key] for key in fields if key in data}
                    print(f"{name}: {json.dumps(info[name], ensure_ascii=True)}")
                    if data.get("device") == "cpu" or data.get("flash_verification") == "not_run_CPU_only":
                        print("CPU-only artifact: this file does not verify a GPU run.")
                except (ValueError, TypeError) as exc:
                    info[name] = {"read_error": str(exc)}
                    print(f"Cannot parse {name}: {exc}")
        manifest["groups"][directory] = info
        if not folder.is_dir():
            outputs = root / "outputs"
            candidates = sorted(p.name for p in outputs.glob(folder.name.split("_2026")[0] + "*")
                                if p.is_dir())
            if candidates:
                print("Other matching output directories: " + ", ".join(candidates))
    if args.dry_run:
        print("\nDry run: no files written.")
        return
    if not files:
        print("\nNo artifact files found. Check the server output directory names; no archive written.")
        if args.export_dir is not None:
            raise SystemExit(1)
        return
    if args.export_dir is not None:
        destination = (root / args.export_dir).resolve()
        destination.relative_to(root)
        if destination == root or destination.exists():
            raise SystemExit("Export directory must be new; existing files will not be overwritten.")
        oversized = [str(source.relative_to(root)) for source in files
                     if source.stat().st_size >= 95 * 1024 * 1024]
        if oversized:
            raise SystemExit("Files too large for this Git handoff: " + ", ".join(oversized))
        destination.mkdir(parents=True)
        for source in files:
            target = destination / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (destination / "collection_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nExport directory: {destination}")
        print(f"Files collected: {len(files)}")
        print("Missing files are listed in collection_manifest.json. No experiments were rerun.")
        return
    output = root / "outputs" / "artifact_handoff"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive = output / f"gpu_artifacts_{stamp}.tar.gz"
    with tarfile.open(archive, "x:gz") as bundle:
        for source in files:
            bundle.add(source, arcname=source.relative_to(root).as_posix(), recursive=False)
        payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        import io
        item = tarfile.TarInfo("collection_manifest.json")
        item.size = len(payload)
        bundle.addfile(item, io.BytesIO(payload))
    print(f"\nArchive: {archive}")
    print(f"Files collected: {len(files)}; size: {archive.stat().st_size / 1e6:.2f} MB")
    print("Missing files are listed in collection_manifest.json. No experiments were rerun.")


if __name__ == "__main__":
    main()
