#!/usr/bin/env bash
# Drive an Isaac Lab benchmark remotely via SSM Run Command.
#
# Steps:
#   1. Read ssm.env (INSTANCE_ID, AWS_REGION, optional overrides).
#   2. Upload scripts/run_bench.sh to the instance under /home/ubuntu/.
#   3. Launch the benchmark in the background (nohup) and poll until the
#      worker process disappears.
#   4. Pull summary.csv and per-seed dmon logs back to ./results/<tag>/.
#   5. Print dmon stats locally via scripts/dmon_stats.py.
#
# Usage:
#   bash orchestrate.sh <profile>
#
# Profiles (see PROFILES below) bundle the (task, seeds, iters, num_envs)
# combos used in our recurring evaluations. Tweak / add to taste — they are
# the only thing here that needs to change for new sweep designs.
set -o pipefail

usage() {
  cat <<'EOF'
Usage: bash orchestrate.sh <profile>

Profiles:
  cartpole        Isaac-Cartpole-v0,         5 seeds, 100 iters, num_envs=4096
  g1              Isaac-Velocity-Rough-G1-v0, 5 seeds, 100 iters, num_envs=4096
  g1-quick        same task, 1 seed,         50 iters, num_envs=4096
  g1-num-envs     num_envs sweep {4096,8192,16384}, 1 seed, 100 iters

Override profile presets via env vars (e.g. NUM_ENVS=2048 SEEDS="42 7").
Required in ssm.env: INSTANCE_ID, AWS_REGION.
Optional in ssm.env: REMOTE_USER, ISAACLAB_DIR, CONDA_SH, CONDA_ENV,
                      DEST_BASE, RUN_TAG_PREFIX.
EOF
}

PROFILE="${1:-}"
[ -n "$PROFILE" ] || { usage; exit 1; }

# Pre-flight: dependency check (no silent failures).
for _cmd in aws jq python3; do
  command -v "$_cmd" >/dev/null 2>&1 || {
    echo "[orchestrate] ERROR: '$_cmd' not found in PATH." >&2
    case "$_cmd" in
      aws)     echo "  Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html" >&2 ;;
      jq)      echo "  Install: brew install jq  /  sudo apt-get install jq" >&2 ;;
      python3) echo "  Install: brew install python3  /  sudo apt-get install python3" >&2 ;;
    esac
    echo "  (DCV mode — running on the EC2 itself — does not need this. Use scripts/run_bench.sh directly.)" >&2
    exit 1
  }
done
unset _cmd

[ -f ssm.env ]   || { echo "ssm.env not found in $(pwd). Copy ssm.env.sample first." >&2; exit 1; }

# shellcheck disable=SC1091
source ssm.env
: "${INSTANCE_ID:?INSTANCE_ID not set in ssm.env}"
: "${AWS_REGION:?AWS_REGION not set in ssm.env}"

REMOTE_USER="${REMOTE_USER:-ubuntu}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/${REMOTE_USER}/IsaacLab}"
CONDA_SH="${CONDA_SH:-/home/${REMOTE_USER}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
DEST_BASE="${DEST_BASE:-/mnt/s3files/bench}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-bench}"

# --- Resolve profile -------------------------------------------------------

case "$PROFILE" in
  cartpole)
    TASK="Isaac-Cartpole-v0"
    SEEDS="${SEEDS:-42 123 456 789 1337}"
    ITERS="${ITERS:-100}"
    NUM_ENVS="${NUM_ENVS:-4096}"
    SWEEP=("$NUM_ENVS")
    ;;
  g1)
    TASK="Isaac-Velocity-Rough-G1-v0"
    SEEDS="${SEEDS:-42 123 456 789 1337}"
    ITERS="${ITERS:-100}"
    NUM_ENVS="${NUM_ENVS:-4096}"
    SWEEP=("$NUM_ENVS")
    ;;
  g1-quick)
    TASK="Isaac-Velocity-Rough-G1-v0"
    SEEDS="${SEEDS:-42}"
    ITERS="${ITERS:-50}"
    NUM_ENVS="${NUM_ENVS:-4096}"
    SWEEP=("$NUM_ENVS")
    ;;
  g1-num-envs)
    TASK="Isaac-Velocity-Rough-G1-v0"
    SEEDS="${SEEDS:-42}"
    ITERS="${ITERS:-100}"
    SWEEP=(${NUM_ENVS_SWEEP:-4096 8192 16384})
    ;;
  *)
    echo "unknown profile: $PROFILE" >&2
    usage
    exit 1
    ;;
esac

STAMP=$(date +%Y%m%d-%H%M%S)
LOCAL_OUT="$(pwd)/results/${PROFILE}-${STAMP}"
mkdir -p "$LOCAL_OUT"

ssm_send() {
  # Send a multi-line shell snippet via SSM. We pipeline JSON so quoting is
  # less painful than the plain-string parameters form.
  local script="$1"
  local cmd_id
  cmd_id=$(aws ssm send-command \
    --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID" \
    --document-name AWS-RunShellScript \
    --cli-input-json "$(jq -n --arg s "$script" '{Parameters: {commands: [$s]}}')" \
    --output text --query 'Command.CommandId') || return 1
  echo "$cmd_id"
}

ssm_wait() {
  local cmd_id="$1"
  local timeout="${2:-300}"
  local elapsed=0
  while [ "$elapsed" -lt "$timeout" ]; do
    local status
    status=$(aws ssm get-command-invocation \
      --region "$AWS_REGION" \
      --command-id "$cmd_id" \
      --instance-id "$INSTANCE_ID" \
      --query 'Status' --output text 2>/dev/null) || status=Pending
    case "$status" in
      Success|Failed|TimedOut|Cancelled) echo "$status"; return 0 ;;
    esac
    sleep 4
    elapsed=$((elapsed + 4))
  done
  echo "PollTimeout"
}

ssm_get() {
  aws ssm get-command-invocation \
    --region "$AWS_REGION" \
    --command-id "$1" \
    --instance-id "$INSTANCE_ID" \
    --query 'StandardOutputContent' --output text
}

# --- 1. Upload run_bench.sh -----------------------------------------------

echo "[orchestrate] uploading scripts/run_bench.sh to $INSTANCE_ID"
SCRIPT_B64=$(base64 < scripts/run_bench.sh | tr -d '\n')
UPLOAD_CMD=$(printf 'umask 022; echo %s | base64 -d > /home/%s/run_bench.sh && chmod +x /home/%s/run_bench.sh && chown %s:%s /home/%s/run_bench.sh && wc -l /home/%s/run_bench.sh' \
  "$SCRIPT_B64" "$REMOTE_USER" "$REMOTE_USER" "$REMOTE_USER" "$REMOTE_USER" "$REMOTE_USER" "$REMOTE_USER")
CID=$(ssm_send "$UPLOAD_CMD")
ssm_wait "$CID" 60 >/dev/null
ssm_get "$CID" | tail -3

# --- 2. Launch each (task, num_envs) combo --------------------------------

run_one() {
  local task="$1" tag="$2" seeds="$3" iters="$4" num_envs="$5"
  echo "[orchestrate] launching tag=$tag task=$task num_envs=$num_envs"
  # Pass env vars to run_bench.sh so it can target user-relative paths.
  local launch
  launch=$(printf 'sudo -u %s env DEST_BASE=%s ISAACLAB_DIR=%s CONDA_SH=%s CONDA_ENV=%s nohup bash /home/%s/run_bench.sh %s %s %q %s %s > /home/%s/%s.log 2>&1 &\nsleep 4\npgrep -af benchmark_rsl_rl | head -3' \
    "$REMOTE_USER" "$DEST_BASE" "$ISAACLAB_DIR" "$CONDA_SH" "$CONDA_ENV" \
    "$REMOTE_USER" "$task" "$tag" "$seeds" "$iters" "$num_envs" \
    "$REMOTE_USER" "$tag")
  CID=$(ssm_send "$launch")
  ssm_wait "$CID" 60 >/dev/null
  ssm_get "$CID"

  # Poll until benchmark process is gone.
  echo "[orchestrate] polling tag=$tag …"
  while true; do
    sleep 60
    local poll_cmd
    poll_cmd=$(printf 'pgrep -c -f benchmark_rsl_rl; tail -2 /home/%s/%s.log 2>/dev/null; echo --- summary ---; cat %s/%s/summary.csv 2>/dev/null' \
      "$REMOTE_USER" "$tag" "$DEST_BASE" "$tag")
    CID=$(ssm_send "$poll_cmd")
    ssm_wait "$CID" 30 >/dev/null
    local out
    out=$(ssm_get "$CID")
    local n
    n=$(echo "$out" | head -1)
    echo "[orchestrate] [$(date +%H:%M:%S)] tag=$tag pgrep=$n"
    if [ "$n" = "0" ]; then
      echo "$out" | sed -n '/--- summary ---/,$p'
      break
    fi
  done
}

# --- 3. Loop over the sweep and record results ----------------------------

for env_count in "${SWEEP[@]}"; do
  TAG="${RUN_TAG_PREFIX}-${PROFILE}-n${env_count}-${STAMP}"
  run_one "$TASK" "$TAG" "$SEEDS" "$ITERS" "$env_count"

  # --- 4. Pull summary.csv + dmon logs back -------------------------------
  echo "[orchestrate] pulling results for tag=$TAG"
  REMOTE_DIR="${DEST_BASE}/${TAG}"
  TARBALL_CMD=$(printf 'cd %s && tar -czf /tmp/%s.tar.gz . && wc -c /tmp/%s.tar.gz && base64 -w0 /tmp/%s.tar.gz' \
    "$REMOTE_DIR" "$TAG" "$TAG" "$TAG")
  CID=$(ssm_send "$TARBALL_CMD")
  ssm_wait "$CID" 120 >/dev/null
  ssm_get "$CID" | tail -1 | tr -d '\n' | base64 -d > "$LOCAL_OUT/${TAG}.tar.gz"
  ( cd "$LOCAL_OUT" && tar -xzf "${TAG}.tar.gz" -C . && rm -f "${TAG}.tar.gz" )
  echo "[orchestrate] saved to $LOCAL_OUT"
done

# --- 5. Local dmon analysis ------------------------------------------------

if command -v python3 >/dev/null; then
  echo
  echo "=== dmon summary (markdown) ==="
  python3 scripts/dmon_stats.py --format md "$LOCAL_OUT"/run-seed*/nvidia-smi-dmon.log || true
fi

echo
echo "[orchestrate] done. results in $LOCAL_OUT"
