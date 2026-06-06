#!/usr/bin/env bash
# Isaac Lab benchmark harness — runs on the EC2 instance.
# Driven by orchestrate.sh from the operator's laptop via SSM.
#
# Args:
#   $1  TASK         e.g. Isaac-Cartpole-v0 / Isaac-Velocity-Rough-G1-v0
#   $2  TAG          run tag, becomes a subdir under $DEST_BASE
#   $3  SEEDS        space-separated seeds, default "42 123 456 789 1337"
#   $4  ITERS        max_iterations, default 100
#   $5  NUM_ENVS     parallel envs, default 4096
#
# Env (override at the call site):
#   ISAACLAB_DIR     default /home/ubuntu/IsaacLab
#   CONDA_SH         default /home/ubuntu/miniconda3/etc/profile.d/conda.sh
#   CONDA_ENV        default env_isaaclab
#   DEST_BASE        default /mnt/s3files/bench
#                    (override when /mnt/s3files is unavailable: e.g. /home/ubuntu/bench)
#
# Output: $DEST_BASE/$TAG/
#   summary.csv                                     ← parsed metrics, one row per seed
#   run-seed<seed>-<stamp>/bench.log                ← raw stdout from benchmark_rsl_rl.py
#   run-seed<seed>-<stamp>/nvidia-smi-dmon.log      ← GPU sample (1Hz)
#
# DO NOT use 'set -u' — Isaac Lab's setup_conda_env.sh reads $ZSH_VERSION
# which is unset under bash and trips set -u.
set -o pipefail

TASK="${1:?usage: run_bench.sh <task> <tag> [seeds] [iters] [num_envs]}"
TAG="${2:?usage: run_bench.sh <task> <tag> [seeds] [iters] [num_envs]}"
SEEDS="${3:-42 123 456 789 1337}"
ITERS="${4:-100}"
NUM_ENVS="${5:-4096}"

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/ubuntu/IsaacLab}"
CONDA_SH="${CONDA_SH:-/home/ubuntu/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
DEST_BASE="${DEST_BASE:-/mnt/s3files/bench}"

DEST="$DEST_BASE/$TAG"
mkdir -p "$DEST"

SUMMARY="$DEST/summary.csv"
echo "seed,iters,task,num_envs,app_launch_ms,total_start_ms,mean_total_fps,mean_collection_fps,max_rewards,wall_clock_s" > "$SUMMARY"

# Isaac Lab's tabs(1) crashes when TERM is empty/unknown (cloud-init / SSM).
# Force a real terminfo entry so tabs(1) succeeds and `set -e` inside isaaclab.sh
# does not abort the run.
export TERM="${TERM:-xterm}"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$ISAACLAB_DIR"

for SEED in $SEEDS; do
  STAMP=$(date +%Y%m%d-%H%M%S)
  RUN_DIR="$DEST/run-seed${SEED}-${STAMP}"
  mkdir -p "$RUN_DIR"
  echo "[$(date +%H:%M:%S)] === START seed=$SEED iters=$ITERS task=$TASK num_envs=$NUM_ENVS ==="

  # 1 Hz GPU sample (sm/mem/pwr/clocks). Keep the same column set used in our
  # earlier dmon_stats.py so existing parsing keeps working.
  nvidia-smi dmon -s upcm -d 1 -o DT > "$RUN_DIR/nvidia-smi-dmon.log" 2>&1 &
  DMON_PID=$!

  T0=$(date +%s)
  python scripts/benchmarks/benchmark_rsl_rl.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --max_iterations "$ITERS" \
    > "$RUN_DIR/bench.log" 2>&1
  RC=$?
  T1=$(date +%s)
  WALL=$((T1 - T0))

  kill "$DMON_PID" 2>/dev/null || true
  wait "$DMON_PID" 2>/dev/null || true

  # benchmark_rsl_rl.py prints "<key>: <value> ms" / "<value> float" lines.
  # The "ms" suffix appears even on FPS rows — that's an upstream quirk.
  APP_LAUNCH=$(grep -E "App Launch Time:" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.]+ ms" | head -1 | awk '{print $1}')
  TOTAL_START=$(grep -E "Total Start Time" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.]+ ms" | head -1 | awk '{print $1}')
  MEAN_TOTAL_FPS=$(grep -E "Mean Total FPS:" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.]+ ms" | head -1 | awk '{print $1}')
  MEAN_COLL_FPS=$(grep -E "Mean Collection FPS:" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.]+ ms" | head -1 | awk '{print $1}')
  MAX_R=$(grep -E "Max Rewards:" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.-]+ float" | head -1 | awk '{print $1}')

  echo "$SEED,$ITERS,$TASK,$NUM_ENVS,${APP_LAUNCH:-NA},${TOTAL_START:-NA},${MEAN_TOTAL_FPS:-NA},${MEAN_COLL_FPS:-NA},${MAX_R:-NA},$WALL" >> "$SUMMARY"

  echo "[$(date +%H:%M:%S)] === END seed=$SEED rc=$RC wall=${WALL}s mean_total_fps=${MEAN_TOTAL_FPS:-NA} ==="
done

echo "ALL_SEEDS_DONE for task=$TASK tag=$TAG"
echo "=== summary.csv ==="
cat "$SUMMARY"
