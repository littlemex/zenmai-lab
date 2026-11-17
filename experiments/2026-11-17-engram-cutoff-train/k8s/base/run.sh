# Inside the training pod: build the training-text 3-gram counts once, train every <config>:<seed> pair in
# RUNS side by side (one GPU each), then evaluate each checkpoint, and the earlier Engram run of each seed
# (PRIOR_RUN), under the evaluation-time cutoff settings. Results go under /work/runs/$RUN_NAME/; an
# existing run directory is refused. One pod runs everything because the volume can be mounted by one node.
set -euo pipefail
OUT=/work/runs/${RUN_NAME}
mkdir -p /work/runs
mkdir "$OUT" || { echo "$OUT exists; choose another RUN_NAME" >&2; exit 1; }
apt-get update -qq
apt-get install -y -qq --no-install-recommends gcc libc6-dev > /dev/null
pip install --quiet --no-cache-dir -r /code/requirements.txt
cd /code
python -m engram_lm.data --out-dir /work/data
python -m engram_lm.counts --data-dir /work/data
steps=()
if [ -n "${MAX_STEPS:-}" ]; then steps=(--max-steps "$MAX_STEPS"); fi
SETTINGS="0:hard,5:hard,5:soft"

run_all() {  # run_all <command template using $v $s $gpu>... : one process per entry of $LIST, fail if any fails
  local pids=() gpu=0 v s
  for item in $LIST; do
    v=${item%%:*}; s=${item##*:}
    CUDA_VISIBLE_DEVICES=$gpu eval "$1" &
    pids+=("$!"); gpu=$((gpu + 1))
  done
  local fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  return "$fail"
}
LIST=$RUNS
run_all 'python -m engram_lm.train --config "configs/$v.json" --data-dir /work/data --out-dir "$OUT/$v-seed$s" --seed "$s" "${steps[@]}" > "$OUT/$v-seed$s.log" 2>&1' \
  || { tail -n 20 "$OUT"/*.log; echo "training failed" >&2; exit 1; }
run_all 'python -m engram_lm.eval_cut --checkpoint "$OUT/$v-seed$s/ckpt.pt" --data-dir /work/data --settings "$SETTINGS" --out "$OUT/$v-seed$s/eval_cut.json" > "$OUT/$v-seed$s.eval.log" 2>&1' \
  || { tail -n 20 "$OUT"/*.eval.log; echo "evaluation failed" >&2; exit 1; }
LIST=""
for s in $(for item in $RUNS; do echo "${item##*:}"; done | sort -u); do
  if [ -f "/work/runs/$PRIOR_RUN/engram-seed$s/ckpt.pt" ]; then LIST="$LIST engram:$s"; mkdir -p "$OUT/prior-engram-seed$s"; fi
done
run_all 'python -m engram_lm.eval_cut --checkpoint "/work/runs/$PRIOR_RUN/engram-seed$s/ckpt.pt" --data-dir /work/data --settings "$SETTINGS" --out "$OUT/prior-engram-seed$s/eval_cut.json" > "$OUT/prior-engram-seed$s.eval.log" 2>&1' \
  || { echo "evaluation of the prior runs failed" >&2; exit 1; }
touch "$OUT/DONE"
echo "done: $OUT"
