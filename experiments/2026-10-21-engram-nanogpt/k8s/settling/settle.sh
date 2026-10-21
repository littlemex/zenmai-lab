# Inside the pod: for each seed, download the published Engram and baseline models and count where
# their predictions settle over the validation split. Writes /work/settling/settling-seed<seed>.json.
set -euo pipefail
pip install --quiet --no-cache-dir -r /code/requirements.txt
cd /code
python -m engram_lm.data --out-dir /work/data
mkdir -p /work/settling
for seed in ${SEEDS:-1337}; do
  hf download "$HF_REPO" --revision "$HF_REVISION" --include "*-seed$seed/model.safetensors" --local-dir /work/hub
  python -m engram_lm.settling --engram "/work/hub/engram-seed$seed/model.safetensors" \
    --baseline "/work/hub/baseline-seed$seed/model.safetensors" --data /work/data/val.bin --out "/work/settling/settling-seed$seed.json"
done
