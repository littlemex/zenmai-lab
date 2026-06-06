#!/usr/bin/env bash
# DCV-mode pre-flight check.
# Verifies that the local environment can run scripts/run_bench.sh.
#
# Source bench.env first so that overrides take effect:
#   source ./bench.env
#   bash scripts/check_local.sh
#
# Exit code 0 means OK to run; non-zero means at least one check failed.
set -o pipefail

# Same defaults as run_bench.sh, kept in sync intentionally.
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/ubuntu/IsaacLab}"
CONDA_SH="${CONDA_SH:-/home/ubuntu/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
DEST_BASE="${DEST_BASE:-/home/ubuntu/bench}"

ok()   { printf '\033[32m[ OK ]\033[0m %s\n' "$*"; }
fail() { printf '\033[31m[FAIL]\033[0m %s\n' "$*"; }
note() { printf '\033[33m[note]\033[0m %s\n' "$*"; }

errors=0

# 1. python3
if command -v python3 >/dev/null 2>&1; then
  ok "python3: $(python3 --version 2>&1)"
else
  fail "python3 not found in PATH. Install with: sudo apt install python3"
  errors=$((errors + 1))
fi

# 2. nvidia-smi + GPU visible
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi -L >/dev/null 2>&1; then
    gpus=$(nvidia-smi -L | wc -l | tr -d ' ')
    ok "nvidia-smi: ${gpus} GPU(s) visible"
  else
    fail "nvidia-smi exists but listing GPUs failed. Driver issue?"
    errors=$((errors + 1))
  fi
else
  fail "nvidia-smi not found. NVIDIA driver missing?"
  errors=$((errors + 1))
fi

# 3. conda init script
if [ -f "$CONDA_SH" ]; then
  ok "conda init: $CONDA_SH"
else
  fail "conda init script not found at $CONDA_SH"
  note "Override with: export CONDA_SH=/path/to/conda.sh"
  errors=$((errors + 1))
fi

# 4. IsaacLab dir
if [ -d "$ISAACLAB_DIR" ]; then
  ok "IsaacLab dir: $ISAACLAB_DIR"
else
  fail "IsaacLab dir not found at $ISAACLAB_DIR"
  note "Override with: export ISAACLAB_DIR=/path/to/IsaacLab"
  errors=$((errors + 1))
fi

# 5. conda env (only if conda.sh is sourceable)
if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  if source "$CONDA_SH" 2>/dev/null && conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    ok "conda env: $CONDA_ENV"
  else
    fail "conda env '$CONDA_ENV' not found"
    note "List envs with: source \"$CONDA_SH\" && conda env list"
    errors=$((errors + 1))
  fi
fi

# 6. DEST_BASE writable
DEST_PARENT="$(dirname "$DEST_BASE")"
if [ -w "$DEST_PARENT" ] || [ -w "$DEST_BASE" ]; then
  ok "results dest: $DEST_BASE"
else
  fail "DEST_BASE parent ($DEST_PARENT) is not writable"
  note "Override with: export DEST_BASE=/path/you/can/write"
  errors=$((errors + 1))
fi

echo
if [ "$errors" -eq 0 ]; then
  echo "All checks passed. Ready: bash scripts/run_bench.sh <task> <tag>"
  exit 0
else
  echo "$errors check(s) failed. Fix the above before running scripts/run_bench.sh."
  exit 1
fi
