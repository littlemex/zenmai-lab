#!/usr/bin/env bash
# Collect summary.csv files from experiments/<name>/results/ into
# benchmarks/cross-experiment/summary.csv.
#
# Run after an experiment completes (manual; intended to be added to the
# experiment's run.sh as a final step).

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
DEST="${ROOT}/benchmarks/cross-experiment/summary.csv"
mkdir -p "$(dirname "${DEST}")"

# header
[ -f "${DEST}" ] || echo "experiment,task,seed,metric,value,timestamp" > "${DEST}"

for f in "${ROOT}"/experiments/*/results/summary.csv; do
  [ -e "$f" ] || continue
  EXP="$(basename "$(dirname "$(dirname "$f")")")"
  TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # take all rows except header, prepend experiment name + timestamp
  tail -n +2 "$f" | while IFS= read -r line; do
    echo "${EXP},${line},${TS}" >> "${DEST}"
  done
  echo "Appended ${EXP}: $(wc -l < "$f") rows from $f"
done
