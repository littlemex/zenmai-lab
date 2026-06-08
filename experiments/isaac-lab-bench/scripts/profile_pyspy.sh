#!/usr/bin/env bash
# profile_pyspy.sh — py-spy フレームグラフプロファイラで benchmark_rsl_rl.py の
# CPU スタックトレースを SVG フレームグラフとして記録する。
# run_bench.sh の引数パーシング・ディレクトリ構造をそのまま流用。
#
# 動作原理:
#   1. conda env に py-spy がなければ pip install (冪等)
#   2. benchmark_rsl_rl.py をバックグラウンドで起動し PID を取得
#   3. Isaac Sim のシェーダコンパイル・キャッシュウォームアップが完了するまで
#      PYSPY_WARMUP_SEC 秒待機 (この間は py-spy をアタッチしない)
#   4. sudo py-spy record で PYSPY_DURATION_SEC 秒分のフレームグラフを記録
#      (sudo が必要な理由: Linux の ptrace_scope=1 制限を回避するため)
#   5. py-spy dump でその時点のスタックスナップショットを1回取得
#   6. ベンチプロセスが自然終了するのを待ち、タイムアウトなら SIGTERM で終了
#   7. 出力先に README.txt を書き出して実行パラメータを記録
#
# Args:
#   $1  TASK         e.g. Isaac-Velocity-Rough-G1-v0
#   $2  TAG          run tag, results/pyspy/<tag>/ に出力
#   $3  SEEDS        space-separated seeds, default "42"
#   $4  ITERS        max_iterations, default 100
#   $5  NUM_ENVS     parallel envs, default 16384
#
# Env (override at the call site):
#   ISAACLAB_DIR        default /home/ubuntu/IsaacLab
#   CONDA_SH            default /home/ubuntu/miniconda3/etc/profile.d/conda.sh
#   CONDA_ENV           default env_isaaclab
#   DEST_BASE           default /mnt/s3files/bench
#   PYSPY_WARMUP_SEC    シェーダウォームアップ待機秒数 default 60
#   PYSPY_DURATION_SEC  py-spy record の記録秒数     default 60
#   PYSPY_RATE          サンプリングレート (Hz)        default 100
#   PYSPY_IDLE          1 なら --idle (GPU 待ちスレッドも捕捉、推奨) default 1
#   PYSPY_SUBPROCESSES  1 なら --subprocesses           default 1
#   PYSPY_NATIVE        1 なら --native (C 拡張も捕捉、低速) default 0
#
# 出力: $DEST_BASE/pyspy/$TAG/run-seed<seed>-<stamp>/
#   pyspy.svg       — フレームグラフ (ブラウザで開く)
#   pyspy-dump.txt  — py-spy dump によるスタックスナップショット
#   bench.log       — benchmark_rsl_rl.py の生 stdout/stderr
#   README.txt      — 実行パラメータのメモ
#
# DO NOT use 'set -u' — Isaac Lab の setup_conda_env.sh が $ZSH_VERSION を
# 参照するため、bash では set -u でエラーになる。
set -o pipefail

TASK="${1:?usage: profile_pyspy.sh <task> <tag> [seeds] [iters] [num_envs]}"
TAG="${2:?usage: profile_pyspy.sh <task> <tag> [seeds] [iters] [num_envs]}"
SEEDS="${3:-42}"
ITERS="${4:-100}"
NUM_ENVS="${5:-16384}"

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/ubuntu/IsaacLab}"
CONDA_SH="${CONDA_SH:-/home/ubuntu/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
DEST_BASE="${DEST_BASE:-/mnt/s3files/bench}"

PYSPY_WARMUP_SEC="${PYSPY_WARMUP_SEC:-60}"
PYSPY_DURATION_SEC="${PYSPY_DURATION_SEC:-60}"
PYSPY_RATE="${PYSPY_RATE:-100}"
PYSPY_IDLE="${PYSPY_IDLE:-1}"
PYSPY_SUBPROCESSES="${PYSPY_SUBPROCESSES:-1}"
PYSPY_NATIVE="${PYSPY_NATIVE:-0}"

# Isaac Lab の tabs(1) は TERM が空/不明だとクラッシュする (cloud-init / SSM 環境)。
# 実在する terminfo エントリを強制設定して isaaclab.sh 内の `set -e` が
# abort しないようにする。
export TERM="${TERM:-xterm}"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

# py-spy バイナリを conda env 内で解決する。
# conda activate 後に command -v を実行することで PATH が確実に更新されている。
# py-spy が見つからなければ pip install (冪等チェック付き)。
pip show py-spy >/dev/null 2>&1 || pip install py-spy
PYSPY_BIN=$(command -v py-spy)
if [ -z "$PYSPY_BIN" ]; then
  echo "[profile_pyspy] ERROR: py-spy のインストールに失敗しました。" >&2
  exit 1
fi
echo "[$(date +%H:%M:%S)] [profile_pyspy] using py-spy: $PYSPY_BIN"
"$PYSPY_BIN" --version 2>/dev/null || true

# 出力ルートは $DEST_BASE/pyspy/$TAG に分離することで run_bench.sh の
# $DEST_BASE/$TAG と衝突しないようにする。
DEST="$DEST_BASE/pyspy/$TAG"
mkdir -p "$DEST"

# run_bench.sh と同一フォーマットの summary.csv を生成・共有する
SUMMARY="$DEST/summary.csv"
if [ ! -f "$SUMMARY" ]; then
  echo "seed,iters,task,num_envs,app_launch_ms,total_start_ms,mean_total_fps,mean_collection_fps,max_rewards,wall_clock_s" > "$SUMMARY"
fi

cd "$ISAACLAB_DIR"

# py-spy record に渡すオプションフラグを事前に組み立てる。
# --idle:        GPU 待ちや sleep 中のスレッドも捕捉 (推奨)
# --subprocesses: Isaac Sim が fork するサブプロセスも同時プロファイル
# --native:      C 拡張・Cython コードのシンボルも解決 (低速)
PYSPY_FLAGS=""
[ "$PYSPY_IDLE" = "1" ]        && PYSPY_FLAGS="$PYSPY_FLAGS --idle"
[ "$PYSPY_SUBPROCESSES" = "1" ] && PYSPY_FLAGS="$PYSPY_FLAGS --subprocesses"
[ "$PYSPY_NATIVE" = "1" ]      && PYSPY_FLAGS="$PYSPY_FLAGS --native"

# ---------------------------------------------------------------------------
# シード毎ループ — run_bench.sh / profile_nsys.sh と同一の構造
# ---------------------------------------------------------------------------
for SEED in $SEEDS; do
  STAMP=$(date +%Y%m%d-%H%M%S)
  RUN_DIR="$DEST/run-seed${SEED}-${STAMP}"
  mkdir -p "$RUN_DIR"

  echo "[$(date +%H:%M:%S)] === START (py-spy) seed=$SEED iters=$ITERS task=$TASK num_envs=$NUM_ENVS ==="
  echo "[$(date +%H:%M:%S)]   warmup=${PYSPY_WARMUP_SEC}s  duration=${PYSPY_DURATION_SEC}s  rate=${PYSPY_RATE}Hz"
  echo "[$(date +%H:%M:%S)]   flags: $PYSPY_FLAGS"
  echo "[$(date +%H:%M:%S)]   run dir: $RUN_DIR"

  # ------------------------------------------------------------------
  # benchmark_rsl_rl.py をバックグラウンドで起動する。
  # フォアグラウンドで実行すると py-spy のアタッチタイミングを制御できない。
  # PID を取得してウォームアップ完了後にアタッチする。
  # ------------------------------------------------------------------
  python scripts/benchmarks/benchmark_rsl_rl.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --max_iterations "$ITERS" \
    > "$RUN_DIR/bench.log" 2>&1 &
  PY_PID=$!
  T0=$(date +%s)
  echo "[$(date +%H:%M:%S)]   bench PID=$PY_PID"

  # ------------------------------------------------------------------
  # Isaac Sim のシェーダコンパイル・テクスチャキャッシュウォームアップを
  # 待機する。この期間は CPU バウンドなコンパイル処理が支配的であり、
  # steady-state の Python プロファイルと混在させると結果が汚れる。
  # ------------------------------------------------------------------
  echo "[$(date +%H:%M:%S)]   シェーダウォームアップ待機 ${PYSPY_WARMUP_SEC}s ..."
  sleep "$PYSPY_WARMUP_SEC"

  # ウォームアップ完了時点でベンチプロセスがまだ生きているか確認する。
  # 起動直後にクラッシュしている場合はスキップして次のシードへ進む。
  if ! kill -0 "$PY_PID" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)]   WARNING: bench process (PID=$PY_PID) はウォームアップ前に終了しました。bench.log を確認してください。"
    T1=$(date +%s); WALL=$((T1 - T0))
    echo "  (skipped),,,, rc=crashed" >&2
    echo "$SEED,$ITERS,$TASK,$NUM_ENVS,NA,NA,NA,NA,NA,$WALL" >> "$SUMMARY"
    continue
  fi

  # ------------------------------------------------------------------
  # py-spy record — SVG フレームグラフを生成する。
  # sudo が必要な理由: Linux の /proc/sys/kernel/yama/ptrace_scope=1 では
  # 子プロセス以外の ptrace を一般ユーザが行えないため。
  # --duration: 指定秒数だけサンプリングして自動終了 (プロセスは生かしたまま)
  # --output:   出力ファイルパス (拡張子 .svg)
  # --pid:      アタッチ先の Python PID
  # ------------------------------------------------------------------
  SVG_PATH="$RUN_DIR/pyspy.svg"
  echo "[$(date +%H:%M:%S)]   py-spy record 開始 (duration=${PYSPY_DURATION_SEC}s, rate=${PYSPY_RATE}Hz) ..."
  # shellcheck disable=SC2086
  sudo "$PYSPY_BIN" record \
    --pid "$PY_PID" \
    --output "$SVG_PATH" \
    --duration "$PYSPY_DURATION_SEC" \
    --rate "$PYSPY_RATE" \
    $PYSPY_FLAGS \
    || echo "[profile_pyspy] WARNING: py-spy record が失敗しました (rc=$?)"
  echo "[$(date +%H:%M:%S)]   py-spy record 完了 -> $SVG_PATH"

  # ------------------------------------------------------------------
  # py-spy dump — 現時点のスタックスナップショットを1回取得する。
  # record とは異なり瞬時にスタックを出力するため、フレームグラフを
  # 補完するデバッグ情報として有用。
  # ------------------------------------------------------------------
  DUMP_PATH="$RUN_DIR/pyspy-dump.txt"
  echo "[$(date +%H:%M:%S)]   py-spy dump (スタックスナップショット) ..."
  if kill -0 "$PY_PID" 2>/dev/null; then
    sudo "$PYSPY_BIN" dump --pid "$PY_PID" > "$DUMP_PATH" 2>&1 \
      || echo "[profile_pyspy] WARNING: py-spy dump が失敗しました (rc=$?)"
    echo "[$(date +%H:%M:%S)]   py-spy dump 完了 -> $DUMP_PATH"
  else
    echo "[profile_pyspy] WARNING: dump 時点でベンチプロセスは終了済みでした。" | tee "$DUMP_PATH"
  fi

  # ------------------------------------------------------------------
  # ベンチプロセスの後処理:
  # py-spy は非侵入的 (プロセスを kill しない) なので、benchmark_rsl_rl.py は
  # 引き続き残りのイテレーションを実行中の場合がある。
  # 自然終了を最大 (PYSPY_WARMUP_SEC + PYSPY_DURATION_SEC + 300) 秒待ち、
  # タイムアウトしたら SIGTERM で強制終了する。
  # ------------------------------------------------------------------
  MAX_WAIT=$(( PYSPY_WARMUP_SEC + PYSPY_DURATION_SEC + 300 ))
  ELAPSED=$(( $(date +%s) - T0 ))
  REMAINING=$(( MAX_WAIT - ELAPSED ))
  # REMAINING が 0 以下になると sleep に負数が渡され失敗し、&& により
  # kill が実行されずウォッチドッグが無効化されて wait が永久ハングする。
  # 最低 1 秒を保証してウォッチドッグの動作を保護する。
  [ "$REMAINING" -le 0 ] && REMAINING=1

  if kill -0 "$PY_PID" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)]   ベンチプロセスの自然終了を最大 ${REMAINING}s 待機 ..."
    # タイムアウト付き wait: サブシェルで wait し、タイムアウトなら親で SIGTERM
    ( sleep "$REMAINING" && kill "$PY_PID" 2>/dev/null ) &
    WATCHDOG_PID=$!
    wait "$PY_PID" 2>/dev/null
    BENCH_RC=$?
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  else
    BENCH_RC=0
  fi

  T1=$(date +%s)
  WALL=$((T1 - T0))

  # ------------------------------------------------------------------
  # run_bench.sh と同一のパーシングロジックで summary.csv に追記
  # ------------------------------------------------------------------
  APP_LAUNCH=$(grep -E "App Launch Time:" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.]+ ms" | head -1 | awk '{print $1}')
  TOTAL_START=$(grep -E "Total Start Time" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.]+ ms" | head -1 | awk '{print $1}')
  MEAN_TOTAL_FPS=$(grep -E "Mean Total FPS:" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.]+ ms" | head -1 | awk '{print $1}')
  MEAN_COLL_FPS=$(grep -E "Mean Collection FPS:" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.]+ ms" | head -1 | awk '{print $1}')
  MAX_R=$(grep -E "Max Rewards:" "$RUN_DIR/bench.log" | head -1 | grep -oE "[0-9.-]+ float" | head -1 | awk '{print $1}')

  echo "$SEED,$ITERS,$TASK,$NUM_ENVS,${APP_LAUNCH:-NA},${TOTAL_START:-NA},${MEAN_TOTAL_FPS:-NA},${MEAN_COLL_FPS:-NA},${MAX_R:-NA},$WALL" >> "$SUMMARY"

  # ------------------------------------------------------------------
  # README.txt — 実行パラメータと再現コマンドをそのディレクトリに記録
  # ------------------------------------------------------------------
  cat > "$RUN_DIR/README.txt" <<README
py-spy profile run
==================
task:              $TASK
tag:               $TAG
seed:              $SEED
iters:             $ITERS
num_envs:          $NUM_ENVS
stamp:             $STAMP
wall_clock_s:      $WALL

py-spy settings
---------------
warmup_sec:        $PYSPY_WARMUP_SEC
duration_sec:      $PYSPY_DURATION_SEC
rate_hz:           $PYSPY_RATE
idle:              $PYSPY_IDLE
subprocesses:      $PYSPY_SUBPROCESSES
native:            $PYSPY_NATIVE
extra_flags:       $PYSPY_FLAGS

outputs
-------
flamegraph SVG:    $SVG_PATH
stack dump:        $DUMP_PATH
bench log:         $RUN_DIR/bench.log
summary csv:       $SUMMARY

view flamegraph
---------------
  # ブラウザで直接開く
  xdg-open "$SVG_PATH"   # Linux
  open "$SVG_PATH"        # macOS

download (S3 マウント経由ではなく直接 S3 から取得する場合)
---------
  aws s3 cp "s3://$(echo "$RUN_DIR" | sed 's|/mnt/s3files/||')/pyspy.svg" ./pyspy.svg

offline parse (SVG から関数名・時間を抽出)
------
  # 関数名トップ20 を呼び出し時間降順でリスト
  grep -oP '(?<=title>)[^<]+' "$SVG_PATH" | sort | uniq -c | sort -rn | head -20
README

  echo "[$(date +%H:%M:%S)] === END seed=$SEED rc=$BENCH_RC wall=${WALL}s mean_total_fps=${MEAN_TOTAL_FPS:-NA} ==="
  echo "[$(date +%H:%M:%S)]   SVG:  $SVG_PATH"
  echo "[$(date +%H:%M:%S)]   DUMP: $DUMP_PATH"
done

echo "ALL_SEEDS_DONE (py-spy) for task=$TASK tag=$TAG"
echo "=== summary.csv ==="
cat "$SUMMARY"

echo ""
echo "=== フレームグラフの確認方法 ==="
echo "  ブラウザで開く:  open/xdg-open \$SVG_PATH"
echo "  S3 からダウンロード (例):"
echo "    aws s3 cp s3://YOUR_BUCKET/pyspy/$TAG/ . --recursive --exclude '*.log'"
echo "  オフライン解析 (top 関数):"
echo "    grep -oP '(?<=title>)[^<]+' pyspy.svg | sort | uniq -c | sort -rn | head -20"
