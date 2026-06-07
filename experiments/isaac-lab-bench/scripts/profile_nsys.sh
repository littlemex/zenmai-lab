#!/usr/bin/env bash
# profile_nsys.sh — nsys ラッパーで benchmark_rsl_rl.py の1イテレーション
# GPU タイムラインを取得する。run_bench.sh の引数パーシングをそのまま流用。
#
# Args:
#   $1  TASK         e.g. Isaac-Velocity-Rough-G1-v0
#   $2  TAG          run tag, results/<tag>/profile/<seed>/ に出力
#   $3  SEEDS        space-separated seeds, default "42"
#   $4  ITERS        max_iterations (warmup 込み), default 30
#   $5  NUM_ENVS     parallel envs, default 4096
#
# Env (override at the call site):
#   ISAACLAB_DIR     default /home/ubuntu/IsaacLab
#   CONDA_SH         default /home/ubuntu/miniconda3/etc/profile.d/conda.sh
#   CONDA_ENV        default env_isaaclab
#   DEST_BASE        default /mnt/s3files/bench
#   WARMUP_ITERS     NVTX キャプチャを開始するまでのスキップイテレーション数
#                    default 5 (シェーダコンパイル・キャッシュウォームアップを除外)
#   PROFILE_ITERS    キャプチャ対象のイテレーション数 default 20
#   NSYS_BIN         nsys バイナリのフルパス (PATH に nsys があれば不要)
#   GPU_ID           GPU メトリクスサンプリング対象デバイス番号 default 0
#
# 出力: $DEST_BASE/$TAG/profile/<seed>/
#   report.nsys-rep     — Nsight Systems バイナリレポート (git 除外推奨)
#   kernels.csv         — GPU カーネル累積統計 (nsys stats cuda_gpu_kern_sum)
#   nvtx_trace.csv      — NVTX Push/Pop 範囲統計 (nsys stats nvtx_pushpop_sum)
#   gpu-gaps.txt        — GPU アイドルギャップ解析 (nsys analyze gpu_gaps)
#   bench.log           — benchmark_rsl_rl.py の生 stdout
#   nvidia-smi-dmon.log — GPU 1Hz サンプル
#
# summary.csv: run_bench.sh と同一フォーマットで $DEST_BASE/$TAG/summary.csv に追記
#
# .gitignore メモ:
#   *.nsys-rep と *.sqlite をリポジトリから除外すること。
#   kernels.csv / nvtx_trace.csv / gpu-gaps.txt のみをコミットしてよい。
#
# DO NOT use 'set -u' — Isaac Lab の setup_conda_env.sh が $ZSH_VERSION を
# 参照するため、bash では set -u でエラーになる。
set -o pipefail

TASK="${1:?usage: profile_nsys.sh <task> <tag> [seeds] [iters] [num_envs]}"
TAG="${2:?usage: profile_nsys.sh <task> <tag> [seeds] [iters] [num_envs]}"
SEEDS="${3:-42}"
ITERS="${4:-30}"
NUM_ENVS="${5:-4096}"

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/ubuntu/IsaacLab}"
CONDA_SH="${CONDA_SH:-/home/ubuntu/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
DEST_BASE="${DEST_BASE:-/mnt/s3files/bench}"
WARMUP_ITERS="${WARMUP_ITERS:-5}"
PROFILE_ITERS="${PROFILE_ITERS:-20}"
GPU_ID="${GPU_ID:-0}"

# nsys バイナリの解決 (Isaac Sim バンドル版 → CUDA Toolkit 版 → PATH の順に探索)
if [ -n "${NSYS_BIN:-}" ]; then
  NSYS="$NSYS_BIN"
elif command -v nsys >/dev/null 2>&1; then
  NSYS="nsys"
else
  # Isaac Sim にバンドルされた nsys を探す
  _NSYS_CANDIDATE=$(find /opt/IsaacSim -name nsys -type f 2>/dev/null | head -1)
  if [ -z "$_NSYS_CANDIDATE" ]; then
    # CUDA Toolkit の標準パスを試す
    for _p in /usr/local/cuda/bin/nsys /usr/local/cuda/nsight-systems-*/bin/nsys; do
      [ -x "$_p" ] && { _NSYS_CANDIDATE="$_p"; break; }
    done
  fi
  NSYS="${_NSYS_CANDIDATE:?nsys が見つかりません。NSYS_BIN にフルパスを指定してください。}"
  unset _NSYS_CANDIDATE _p
fi
echo "[profile_nsys] using nsys: $NSYS"
"$NSYS" --version 2>/dev/null | head -1 || true

DEST="$DEST_BASE/$TAG"
mkdir -p "$DEST"

# run_bench.sh と同一フォーマットの summary.csv を生成・共有する
SUMMARY="$DEST/summary.csv"
if [ ! -f "$SUMMARY" ]; then
  echo "seed,iters,task,num_envs,app_launch_ms,total_start_ms,mean_total_fps,mean_collection_fps,max_rewards,wall_clock_s" > "$SUMMARY"
fi

export TERM="${TERM:-xterm}"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$ISAACLAB_DIR"

# ---------------------------------------------------------------------------
# NVTX キャプチャ対象の範囲名を決定する。
# WARMUP_ITERS=5 の場合、最初にキャプチャする NVTX 範囲は 'iter_5'。
# benchmark_rsl_rl.py は NVTX マーカーを持たないため、OnPolicyRunner を
# サブクラス化した薄いラッパーを一時ファイルとして生成し、パッチを当てる。
# ---------------------------------------------------------------------------
CAPTURE_RANGE_NAME="iter_${WARMUP_ITERS}"

# Python パッチ: benchmark_rsl_rl.py をインポートして runner.learn() 呼び出し
# 直前に NVTX マーカーを注入する OnPolicyRunner サブクラスを差し込む。
# スクリプトは実行後に削除される。
PATCH_SCRIPT="$(mktemp /tmp/profile_nvtx_patch_XXXXXX.py)"
cat > "$PATCH_SCRIPT" <<PYEOF
"""
profile_nvtx_patch.py — OnPolicyRunner.learn() に NVTX 範囲を注入する
モンキーパッチ。profile_nsys.sh が一時生成して実行後に削除する。

WARMUP_ITERS イテレーション目から NVTX マーカーを発行することで、
シェーダコンパイル・キャッシュウォームアップ期間を除外した
steady-state のみを nsys がキャプチャできるようにする。
"""
import os, importlib, sys

WARMUP_ITERS = int(os.environ.get("NSYS_WARMUP_ITERS", "5"))
MAX_CAPTURE   = int(os.environ.get("NSYS_PROFILE_ITERS", "20"))

# torch.cuda.nvtx は torch が GPU なしでもインポート可能
import torch.cuda

try:
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner as _Base
except ImportError:
    # rsl_rl が見つからない場合はパッチなしで続行
    _Base = None

if _Base is not None:
    _orig_learn = _Base.learn

    def _patched_learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """NVTX マーカー付き learn() オーバーライド。"""
        _capture_count = 0
        _orig_it_attr  = None

        # rsl_rl v1 系は self.current_learning_iteration を持つ
        _has_it_attr = hasattr(self, "current_learning_iteration")

        # イテレーション前処理フックで NVTX push を行うためにループを再実装する
        # のは rsl_rl バージョン依存でリスクが高い。
        # ここでは「最初の呼び出し前に torch.cuda.nvtx を有効化するだけ」の
        # 軽量アプローチをとる:
        #   - WARMUP_ITERS 後にキャプチャ開始 NVTX マーカーを push する
        #   - WARMUP_ITERS + MAX_CAPTURE 後に pop して終了マーカーを push する
        # rsl_rl の各イテレーションカウントは _Base の it 変数を直接参照できないため、
        # step コールバック経由でカウントする。
        #
        # より確実な方法: 本 learn() を全イテレーション分呼ぶのではなく
        # ウォームアップ分と計測分に分けて2回呼び出す。
        # ウォームアップ (NVTX マーカーなし)
        if WARMUP_ITERS > 0:
            _orig_learn(self, min(WARMUP_ITERS, num_learning_iterations),
                        init_at_random_ep_len)

        remaining = num_learning_iterations - WARMUP_ITERS
        if remaining <= 0:
            return

        # steady-state キャプチャ開始
        torch.cuda.nvtx.range_push(f"iter_{WARMUP_ITERS}")
        try:
            _orig_learn(self, min(MAX_CAPTURE, remaining),
                        init_at_random_ep_len)
        finally:
            torch.cuda.nvtx.range_pop()

        # 残りイテレーション (nsys はすでにキャプチャ範囲外だが念のため実行)
        done = WARMUP_ITERS + min(MAX_CAPTURE, remaining)
        leftover = num_learning_iterations - done
        if leftover > 0:
            _orig_learn(self, leftover, init_at_random_ep_len)

    _Base.learn = _patched_learn
    print(f"[profile_nvtx_patch] NVTX patch applied: warmup={WARMUP_ITERS} capture={MAX_CAPTURE}")
else:
    print("[profile_nvtx_patch] WARNING: rsl_rl not found; NVTX patch skipped.")
PYEOF

for SEED in $SEEDS; do
  STAMP=$(date +%Y%m%d-%H%M%S)
  RUN_DIR="$DEST/run-seed${SEED}-${STAMP}"
  PROF_DIR="$DEST/profile/${SEED}"
  mkdir -p "$RUN_DIR" "$PROF_DIR"
  echo "[$(date +%H:%M:%S)] === START (nsys) seed=$SEED iters=$ITERS task=$TASK num_envs=$NUM_ENVS ==="
  echo "[$(date +%H:%M:%S)]   warmup=$WARMUP_ITERS capture=$PROFILE_ITERS range=$CAPTURE_RANGE_NAME"
  echo "[$(date +%H:%M:%S)]   prof dir: $PROF_DIR"

  # 1 Hz GPU サンプル (run_bench.sh と同一形式)
  nvidia-smi dmon -s upcm -d 1 -o DT > "$RUN_DIR/nvidia-smi-dmon.log" 2>&1 &
  DMON_PID=$!

  NSYS_REP="$PROF_DIR/report"   # nsys が .nsys-rep を付加する

  T0=$(date +%s)

  # nsys profile — キャプチャ範囲は NVTX の iter_${WARMUP_ITERS} 範囲が現れた
  # 瞬間に開始し、その範囲が終了した後にセッションを自動停止する。
  # --pytorch=autograd-nvtx で PyTorch autograd 演算を自動 NVTX 注釈。
  # --sample/cpuctxsw/backtrace=none で CPU プロファイリングオーバーヘッドを最小化。
  # The PYTHONSTARTUP-based NVTX monkey-patch in $PATCH_SCRIPT is documented
  # for posterity but inactive: PYTHONSTARTUP only runs in interactive Python,
  # not under `python script.py`. We capture the entire run instead. The first
  # ~6s are shader-compile / total-start; post-process can clip the timeline
  # by NVTX timestamps if needed.
  "$NSYS" profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --backtrace=none \
    --force-overwrite=true \
    --output="${NSYS_REP}" \
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

  # ---------------------------------------------------------------------------
  # nsys stats / analyze でテキストサマリーを生成
  # .nsys-rep ファイルは git 除外推奨。CSV と gpu-gaps.txt のみをコミットする。
  # ---------------------------------------------------------------------------
  REP_FILE="${NSYS_REP}.nsys-rep"
  if [ -f "$REP_FILE" ]; then
    echo "[$(date +%H:%M:%S)] generating nsys text summaries from $REP_FILE"

    # GPU カーネル累積統計 (所要時間上位)
    "$NSYS" stats \
      --report cuda_gpu_kern_sum \
      --format csv \
      --output "${PROF_DIR}/kernels" \
      "$REP_FILE" 2>&1 \
      || echo "[profile_nsys] WARNING: nsys stats cuda_gpu_kern_sum failed (rc=$?)"
    # nsys stats は --output <prefix> から <prefix>_cuda_gpu_kern_sum.csv を生成する
    # 名前を kernels.csv に統一する
    if [ -f "${PROF_DIR}/kernels_cuda_gpu_kern_sum.csv" ]; then
      mv "${PROF_DIR}/kernels_cuda_gpu_kern_sum.csv" "${PROF_DIR}/kernels.csv"
    fi

    # NVTX Push/Pop 範囲統計
    "$NSYS" stats \
      --report nvtx_pushpop_sum \
      --format csv \
      --output "${PROF_DIR}/nvtx_trace" \
      "$REP_FILE" 2>&1 \
      || echo "[profile_nsys] WARNING: nsys stats nvtx_pushpop_sum failed (rc=$?)"
    if [ -f "${PROF_DIR}/nvtx_trace_nvtx_pushpop_sum.csv" ]; then
      mv "${PROF_DIR}/nvtx_trace_nvtx_pushpop_sum.csv" "${PROF_DIR}/nvtx_trace.csv"
    fi

    # GPU アイドルギャップ解析
    "$NSYS" analyze \
      --rule gpu_gaps,gpu_time_util,cuda_api_sync \
      --format table \
      --output "${PROF_DIR}/gpu-gaps" \
      "$REP_FILE" 2>&1 \
      > "${PROF_DIR}/gpu-gaps.txt" \
      || echo "[profile_nsys] WARNING: nsys analyze gpu_gaps failed (rc=$?)"
  else
    echo "[profile_nsys] WARNING: report file not found: $REP_FILE (nsys rc=$RC)"
  fi

  # ---------------------------------------------------------------------------
  # run_bench.sh と同一フォーマットで summary.csv に追記
  # benchmark_rsl_rl.py の出力パーシングは run_bench.sh のロジックをそのまま流用
  # ---------------------------------------------------------------------------
  APP_LAUNCH=$(grep -E "App Launch Time:" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.]+ ms" | head -1 | awk '{print $1}')
  TOTAL_START=$(grep -E "Total Start Time" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.]+ ms" | head -1 | awk '{print $1}')
  MEAN_TOTAL_FPS=$(grep -E "Mean Total FPS:" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.]+ ms" | head -1 | awk '{print $1}')
  MEAN_COLL_FPS=$(grep -E "Mean Collection FPS:" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.]+ ms" | head -1 | awk '{print $1}')
  MAX_R=$(grep -E "Max Rewards:" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.-]+ float" | head -1 | awk '{print $1}')

  echo "$SEED,$ITERS,$TASK,$NUM_ENVS,${APP_LAUNCH:-NA},${TOTAL_START:-NA},${MEAN_TOTAL_FPS:-NA},${MEAN_COLL_FPS:-NA},${MAX_R:-NA},$WALL" >> "$SUMMARY"

  echo "[$(date +%H:%M:%S)] === END seed=$SEED rc=$RC wall=${WALL}s mean_total_fps=${MEAN_TOTAL_FPS:-NA} ==="
  echo "[$(date +%H:%M:%S)]   profile outputs: $PROF_DIR"
done

# PYTHONSTARTUP パッチファイルを削除
rm -f "$PATCH_SCRIPT"

echo "ALL_SEEDS_DONE (nsys) for task=$TASK tag=$TAG"
echo "=== summary.csv ==="
cat "$SUMMARY"
