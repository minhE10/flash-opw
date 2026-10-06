#!/usr/bin/env bash
# Fetch a pinned source checkout without installing over the local distribution.
set -euo pipefail
target="${1:-outputs/third_party/flash-sinkhorn-author}"
revision=75d48cc42d2efe8d4f654d91152ccf6f857c993f
if [[ -e "$target" ]]; then
  current="$(git -C "$target" rev-parse HEAD)"
  if [[ "$current" != "$revision" ]]; then
    printf 'Existing author checkout is %s; expected %s. Supply a new directory.\n' "$current" "$revision" >&2
    exit 2
  fi
else
  git clone https://github.com/ot-triton-lab/flash-sinkhorn.git "$target"
  git -C "$target" checkout --detach "$revision"
fi
if [[ -n "$(git -C "$target" status --porcelain --untracked-files=no)" ]]; then
  printf '%s\n' 'Author checkout has tracked modifications. Supply a clean separate checkout.' >&2
  exit 2
fi
printf 'Author source ready: %s at %s (no pip installation)\n' "$target" "$revision"
