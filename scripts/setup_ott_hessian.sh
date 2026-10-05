#!/usr/bin/env bash
# Install the read-only external baseline into ignored outputs; no GPU needed.
set -euo pipefail
target="${1:-outputs/third_party/OTT-Hessian}"
revision=7eb189fe39982f587da935044480655b65939637
if [[ -e "$target" ]]; then
  current="$(git -C "$target" rev-parse HEAD)"
  if [[ "$current" != "$revision" ]]; then
    printf 'Existing checkout is %s; expected %s. Supply a new target directory.\n' "$current" "$revision" >&2
    exit 2
  fi
else
  git clone https://github.com/yexf308/OTT-Hessian.git "$target"
  git -C "$target" checkout --detach "$revision"
fi
printf 'OTT-Hessian ready: %s at %s\n' "$target" "$revision"
printf '%s\n' 'Uses ott-jax, jax, lineax, optax and jaxtyping from your baseline environment.'
