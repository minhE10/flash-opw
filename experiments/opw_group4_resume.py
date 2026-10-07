"""Narrow provenance migration for the group4 checkpoint-validator fix.

Only the exact original runner and this exact replacement are compatible.
All numerical engines, data, parameters, policy and environment stay pinned.
"""

import copy


RUNNER = "experiments/opw_group4.py"
MIGRATION = "experiments/opw_group4_resume.py"
LEGACY_RUNNER_SHA256 = "ead0d10a6088c5cd749686da72bfc5e2236c6a63357adb0086bc292564fa59e0"
PATCHED_RUNNER_SHA256 = "7f6fced8c1d715165fa7c0e8b534fec4cf7861529002cf285664396e41cc44b4"


def signature_migration(prior, current):
    if prior == current:
        return None
    old_sources,new_sources = prior.get("sources",{}),current.get("sources",{})
    if (old_sources.get(RUNNER) != LEGACY_RUNNER_SHA256
            or new_sources.get(RUNNER) != PATCHED_RUNNER_SHA256
            or MIGRATION in old_sources or MIGRATION not in new_sources):
        raise ValueError("Resume refused: data/code/selection/settings/environment changed")
    translated = copy.deepcopy(prior)
    translated["sources"][RUNNER] = new_sources[RUNNER]
    translated["sources"][MIGRATION] = new_sources[MIGRATION]
    if translated != current:
        raise ValueError("Resume refused: data/code/selection/settings/environment changed")
    return dict(reason="checkpoint validation/progress reporting only; numerical sources and cached values unchanged",
                old_runner_sha256=LEGACY_RUNNER_SHA256,new_runner_sha256=PATCHED_RUNNER_SHA256,
                migration_source_sha256=new_sources[MIGRATION])
