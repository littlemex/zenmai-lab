#!/usr/bin/env bash
# Run a command on a single EC2 host over SSH.
#
# Usage:
#   ssh-run.sh "<command>"
#
# Reads from push.env in the caller's CWD:
#   EC2_HOST       e.g. ubuntu@<eip-or-hostname>  (the SSH target)
#   EC2_SSH_KEY    Path to private key (optional, if not in ssh-config)

set -euo pipefail

[ -f push.env ] || { echo "push.env not found in $(pwd)"; exit 1; }
source push.env

: "${EC2_HOST:?EC2_HOST not set in push.env}"

CMD="${1:?usage: ssh-run.sh '<command>'}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
[ -n "${EC2_SSH_KEY:-}" ] && SSH_OPTS+=(-i "${EC2_SSH_KEY}")

ssh "${SSH_OPTS[@]}" "${EC2_HOST}" "${CMD}"
