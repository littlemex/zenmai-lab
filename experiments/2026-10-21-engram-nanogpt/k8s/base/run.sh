# Inside the training pod: prepare the data once, then for each seed train the Engram and the
# baseline model side by side on GPU 0 and GPU 1 and evaluate both. Everything is written under
# /work/runs/$RUN_NAME/<variant>-seed<seed>; an existing run directory is refused.
set -euo pipefail
# "{pod}" in RUN_NAME becomes this pod's name, for runs that should never collide, such as smoke runs
OUT=/work/runs/${RUN_NAME//\{pod\}/${POD_NAME:-}}
mkdir -p /work/runs
mkdir "$OUT" || { echo "$OUT exists; choose another RUN_NAME" >&2; exit 1; }  # create-or-fail in one step
# torch.compile runs Triton, which builds a small C launcher at start-up; the runtime image has no compiler
apt-get update -qq
apt-get install -y -qq --no-install-recommends gcc libc6-dev > /dev/null
pip install --quiet --no-cache-dir -r /code/requirements.txt
cd /code
python -m engram_lm.data --out-dir /work/data
steps=()
if [ -n "${MAX_STEPS:-}" ]; then steps=(--max-steps "$MAX_STEPS"); fi

run_pair() {  # run_pair <module> <args for seed>: one process per GPU, fail if either fails
  local seed=$1; shift
  local pids=()
  for variant in engram baseline; do
    gpu=$([ "$variant" = engram ] && echo 0 || echo 1)
    CUDA_VISIBLE_DEVICES=$gpu "$@" "$variant" "$seed" &
    pids+=("$!")
  done
  local fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  return "$fail"
}
train_one() {
  python -m engram_lm.train --config "configs/$1.json" --data-dir /work/data --out-dir "$OUT/$1-seed$2" \
    --seed "$2" "${steps[@]}" > "$OUT/$1-seed$2.log" 2>&1
}
evaluate_one() {
  python -m engram_lm.evaluate --checkpoint "$OUT/$1-seed$2/ckpt.pt" --data-dir /work/data \
    --out "$OUT/$1-seed$2/eval.json" > "$OUT/$1-seed$2.eval.log" 2>&1
}
for seed in ${SEEDS:-1337}; do
  run_pair "$seed" train_one || { tail -n 20 "$OUT"/*-seed"$seed".log; echo "training failed (seed $seed)" >&2; exit 1; }
  run_pair "$seed" evaluate_one || { tail -n 20 "$OUT"/*-seed"$seed".eval.log; echo "evaluation failed (seed $seed)" >&2; exit 1; }
  tail -n 1 "$OUT"/*-seed"$seed".log
done
touch "$OUT/DONE"
echo "done: $OUT"
