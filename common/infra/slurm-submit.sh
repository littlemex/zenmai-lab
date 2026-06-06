#!/usr/bin/env bash
# Submit a Slurm job to a HyperPod cluster.
#
# Usage:
#   slurm-submit.sh <sbatch_script> [extra sbatch args]
#
# Reads from push.env:
#   HYPERPOD_HEAD_NODE    e.g. ubuntu@<head-ip>
#   HYPERPOD_SSH_KEY      Optional
#
# This is a thin wrapper. For complex orchestration use the physai CLI in
# aws-samples/sample-physical-ai-scaffolding-kit/physai/.

set -euo pipefail

[ -f push.env ] || { echo "push.env not found in $(pwd)"; exit 1; }
source push.env

: "${HYPERPOD_HEAD_NODE:?HYPERPOD_HEAD_NODE not set in push.env}"

SBATCH_SCRIPT="${1:?usage: slurm-submit.sh <sbatch_script> [extra args]}"
shift || true

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[ -n "${HYPERPOD_SSH_KEY:-}" ] && SSH_OPTS+=(-i "${HYPERPOD_SSH_KEY}")

# copy script + submit
SCRIPT_NAME="$(basename "${SBATCH_SCRIPT}")"
scp "${SSH_OPTS[@]}" "${SBATCH_SCRIPT}" "${HYPERPOD_HEAD_NODE}:/tmp/${SCRIPT_NAME}"
ssh "${SSH_OPTS[@]}" "${HYPERPOD_HEAD_NODE}" "sbatch $* /tmp/${SCRIPT_NAME}"
