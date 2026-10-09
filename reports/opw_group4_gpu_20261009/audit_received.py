"""Recheck uploaded group4 reports; matrices/chunks remain on the server.

Run from repo root with the original report directory as the first argument.
The original reports are backed up under ignored outputs before cleanup.
"""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.stats import binomtest, bootstrap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.sequence_data import read_ts, training_fingerprint


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def main():
    target = Path(__file__).parent
    backup = Path("outputs/opw_group4_gpu_import_20261009/original_reports")
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else backup
    backup.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "upload_manifest.json").read_text())
    verified = {}
    for name, expected in manifest["files"].items():
        check(Path(name).name == name, "Unexpected manifest path")
        raw = (source / name).read_bytes()
        lf = raw.replace(b"\r\n", b"\n")
        variants = {"working_tree": raw, "LF": lf, "CRLF": lf.replace(b"\n", b"\r\n")}
        matched = [(mode, value) for mode, value in variants.items()
                   if len(value) == expected["bytes"]
                   and hashlib.sha256(value).hexdigest() == expected["sha256"]]
        check(bool(matched), f"Upload checksum mismatch: {name}")
        mode, value = matched[0]
        (backup / name).write_bytes(value)
        verified[name] = dict(expected, recovered_line_endings=mode)
    if source.resolve() != backup.resolve():
        shutil.copy2(source / "upload_manifest.json", backup / "upload_manifest.json")

    def read(name):
        return json.loads((backup / name).read_text(encoding="utf-8"))

    summary, environment = read("summary.json"), read("environment.json")
    state, selection, evaluations = read("run_state.json"), read("frozen_selection.json"), read("evaluations.json")
    saved_paired, provenance = read("paired_statistics.json"), read("data_manifest.json")
    signature = environment["signature"]
    settings = signature["settings"]
    check(summary["status"] == state["status"] == "completed_with_nonconvergence", "Run status mismatch")
    check(summary["test_used_for_selection"] is False and environment["test_used_for_selection"] is False,
          "Frozen TRAIN selection declaration missing")
    for path, digest in signature["sources"].items():
        check(hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest,
              f"Numerical source mismatch: {path}")
    check(signature["sources"] == summary["sources"], "Summary sources mismatch")
    check(all(signature["sources"].get(k) == v for k, v in selection["sources"].items()), "Selection sources mismatch")
    frozen_hash = hashlib.sha256((backup / "frozen_selection.json").read_bytes()).hexdigest()
    check(frozen_hash == signature["selection_sha256"] == summary["selection_sha256"], "Frozen selection hash mismatch")
    check(selection == json.loads(Path("reports/opw_group3_gpu_20261007/selected_all_metrics.json").read_text()),
          "Group3 frozen selection changed")
    check(selection["solver_policy"] == signature["policy"] == summary["policy"], "Policy mismatch")
    if "checkpoint_validator_migration" in environment:
        from experiments.opw_group4_resume import signature_migration
        migration = signature_migration(read("environment.before_checkpoint_validator_fix.json")["signature"], signature)
        check(migration is not None and all(environment["checkpoint_validator_migration"].get(k) == v
              for k, v in migration.items()), "Reader migration provenance mismatch")

    train_path = Path("data/opw/FacesUCR/FacesUCR_TRAIN.ts")
    test_path = Path("data/opw/FacesUCR/FacesUCR_TEST.ts")
    train, train_labels = read_ts(train_path)
    test, labels = read_ts(test_path)
    for path in (train_path, test_path):
        check(hashlib.sha256(path.read_bytes()).hexdigest() == provenance["files"][path.name], "Dataset file mismatch")
    check(training_fingerprint(train, train_labels) == selection["training_sha256"] == provenance["training_sha256"],
          "TRAIN values mismatch")
    check(training_fingerprint(test, labels) == provenance["test_sha256"] == signature["origin"]["test_sha256"],
          "TEST values mismatch")
    check(provenance == signature["origin"], "Data manifest identity mismatch")
    check(len(train) == summary["gallery"] == 200 and len(test) == summary["queries"] == 2050,
          "Full split dimensions mismatch")
    check(provenance["gallery_indices"] == list(range(200)) and provenance["query_indices"] == list(range(2050)),
          "Full split indices mismatch")

    configs = {}
    expected_evalkeys = set()
    for profile in settings["profiles"]:
        for metric in settings["metrics"]:
            key = f"{profile}__{metric}"
            expected_evalkeys.add(key)
            chosen = selection["selected" if profile == "selected" else "presets"][metric]
            p = chosen["parameters"]
            check(p == selection["grids"][metric][chosen["candidate"]], "Candidate parameters mismatch")
            if profile == "selected":
                check(chosen["eligible"] and chosen["capped_pairs"] == 0, "Selected TRAIN candidate ineligible")
            config_key = metric + "__" + hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
            configs.setdefault(config_key, dict(key=config_key, metric=metric, parameters=p, aliases=[]))["aliases"].append(
                dict(profile=profile, candidate=chosen["candidate"]))
    check(list(configs.values()) == signature["configs"] and len(configs) == 17, "Configuration mismatch")
    check(set(evaluations) == expected_evalkeys == set(summary["quality"]), "Evaluation coverage mismatch")
    check(all(value[k] == 8721 for value in (summary, state) for k in ("completed_chunks", "total_chunks")),
          "Incomplete chunk count")

    rows = list(csv.DictReader((backup / "results.csv").open(encoding="utf-8", newline="")))
    seen = set()
    for row in rows:
        key, k = f"{row['profile']}__{row['metric']}", row["k"]
        check((key, k) not in seen, "Duplicate result row")
        seen.add((key, k))
        ev = evaluations[key]
        ap = np.asarray(ev["average_precision"], dtype=float)
        prediction = np.asarray(ev["predictions"][k])
        check(ap.shape == prediction.shape == labels.shape and np.isfinite(ap).all()
              and ((ap >= 0) & (ap <= 1)).all(), "Invalid per-query arrays")
        close(float(row["MAP"]), ap.mean())
        close(ev["MAP"], ap.mean())
        close(float(row["ACC"]), np.mean(prediction == labels))
        close(ev["ACC"][k], np.mean(prediction == labels))
        check(row["queries_without_relevant_gallery"] == "0" and ev["queries_without_relevant_gallery"] == 0,
              "Unexpected zero-relevant query")
        check(int(row["unconverged_pairs"]) == summary["quality"][key]["unconverged_pairs"], "Quality count mismatch")
        chosen = selection["selected" if row["profile"] == "selected" else "presets"][row["metric"]]
        check(json.loads(row["parameters"]) == chosen["parameters"] and int(row["candidate"]) == chosen["candidate"],
              "CSV parameters mismatch")
    check(seen == {(key, str(k)) for key in expected_evalkeys for k in settings["ks"]} and len(rows) == 132,
          "ACC@k coverage mismatch")

    recomputed = []
    for saved in saved_paired:
        profile, metric = saved["profile"], saved["right"]
        left, right = evaluations[f"{profile}__flash-opw"], evaluations[f"{profile}__{metric}"]
        delta = np.asarray(left["average_precision"]) - np.asarray(right["average_precision"])
        ci = bootstrap((delta,), np.mean, method="percentile", confidence_level=.95,
                       n_resamples=settings["bootstrap_resamples"], batch=256,
                       rng=np.random.default_rng(settings["bootstrap_seed"])).confidence_interval
        a, b = np.asarray(left["predictions"]["1"]) == labels, np.asarray(right["predictions"]["1"]) == labels
        counts = {"both_correct": int(np.sum(a & b)), "left_only_correct": int(np.sum(a & ~b)),
                  "right_only_correct": int(np.sum(~a & b)), "both_wrong": int(np.sum(~a & ~b))}
        discordant = counts["left_only_correct"] + counts["right_only_correct"]
        p = float(binomtest(counts["left_only_correct"], discordant, p=.5).pvalue) if discordant else 1.
        calculated = dict(delta_MAP_pp=100*float(delta.mean()), map_difference_ci95_pp=[100*ci.low, 100*ci.high],
                          delta_ACC1_pp=100*float(a.mean()-b.mean()), mcnemar_exact_two_sided_p=p, **counts)
        for field, value in calculated.items():
            close(saved[field], value)
        convergence = all(summary["quality"][f"{profile}__{m}"]["unconverged_pairs"] == 0 for m in ("flash-opw", metric))
        check(saved["convergence_valid"] == convergence, "Comparison quality mismatch")
        recomputed.append(dict(profile=profile, right=metric, **calculated))
    for profile in settings["profiles"]:
        family = [r for r in saved_paired if r["profile"] == profile]
        check(len(family) == 10, "Holm family mismatch")
        ordered = sorted(family, key=lambda r: r["mcnemar_exact_two_sided_p"])
        maximum = 0.
        for i, row in enumerate(ordered):
            maximum = min(1., max(maximum, (len(ordered)-i)*row["mcnemar_exact_two_sided_p"]))
            close(row["mcnemar_holm_p"], maximum)
    paired_csv = list(csv.DictReader((backup / "paired_statistics.csv").open(encoding="utf-8", newline="")))
    check(paired_csv == [{k: str(v) for k, v in row.items()} for row in saved_paired], "Paired CSV mismatch")

    compact = [r for r in rows if r["k"] == "1"]
    for row in compact:
        print(row["profile"], row["metric"], "ACC", round(100*float(row["ACC"]), 6),
              "MAP", round(100*float(row["MAP"]), 6), "unconverged", row["unconverged_pairs"],
              "max_residual", row["max_marginal_l1"], "max_iters", row["max_iterations"],
              "hours", round(float(row["solve_wall_seconds"])/3600, 4))
    unique_times = {(r["metric"], r["parameters"]): float(r["solve_wall_seconds"]) for r in compact}
    audit = dict(status="passed", source_commit="8cfb172", uploaded_files_verified=verified,
        numerical_sources_verified=len(signature["sources"]), dataset="FacesUCR", gallery=200, queries=2050,
        unique_configurations=17, method_profile_evaluations=22, metric_k_rows_verified=132,
        paired_comparisons_recomputed=len(recomputed), total_recorded_solve_hours=sum(unique_times.values())/3600,
        server_full_audit=read("audit.json"), recorded_quality=summary["quality"],
        checks=["upload SHA256 with explicit Git line-ending recovery", "frozen sources, parameters, policy and migration",
                "full local TRAIN/TEST bytes, values and indices", "MAP aggregates from stored per-query AP",
                "ACC at every k from stored predictions and official TEST labels", "paired bootstrap, McNemar and Holm",
                "CSV consistency and configuration coverage"],
        limits=["No uploaded distance matrices or chunks: local audit cannot recompute AP/ranking from distances or residuals from couplings",
                "Full chunk/matrix readback passed on server according to uploaded audit; not rerun locally",
                "No GPU solver rerun; wall times include conversion/diagnostics and are not group5 benchmark",
                "TRAIN-frozen current run; prior TEST pilots were viewed during development, so not a wholly blind development process"])
    (target / "audit_received.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    for name in ("results.csv", "paired_statistics.csv", "paired_statistics.json", "summary.json", "audit.json"):
        shutil.copy2(backup / name, target / name)
    print("AUDIT PASSED; backed up original reports:", backup)
    print("Total recorded solve hours:", audit["total_recorded_solve_hours"])


if __name__ == "__main__":
    main()
