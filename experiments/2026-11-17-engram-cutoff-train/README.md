# Engram trained so that its table never reads the text it learned from

The Engram model of [`2026-10-21-engram-nanogpt`](../2026-10-21-engram-nanogpt) memorises its training text. At the end of training, its loss is 0.151 below the baseline's on the training text, but only 0.010 below on the validation text. On held-out text it loses at positions whose 3-gram never occurs in training. This experiment retrains it in three ways, all aimed at that loss. The setup is the same as in that experiment: model, 3,600 steps, the same training windows from the same random draws, and one NVIDIA RTX PRO 4500 Blackwell per run.

| Config | What changes in training |
|---|---|
| `engram-cut5.json` | Count cutoff: at positions whose 3-gram occurs fewer than 5 times in the training text, the eight 3-gram rows are read as zero. While training, the count leaves out the position's own occurrence (count − 1); at evaluation it does not |
| `engram-soft5.json` | The same count, as a weight instead of a cut: the eight 3-gram rows are multiplied by c / (c + 5) |
| `engram-cross.json` | Cross tables: the training articles alternate between halves A and B, and the layer has two tables. A position in an A article reads table B's values, and its gradient goes to table A; a position in a B article does the reverse. At evaluation the layer reads the mean of the two tables. Each table has half the rows (32,768 per head), so the parameter count is unchanged |
| `engram-cross-full.json` | Cross tables with 65,536 rows per head each, twice the table parameters |
| `engram-cross3.json`, `engram-cross4.json`, `engram-cross8.json` | N cross tables: the articles go to N parts in turn, and a position in part f reads the mean of the other N − 1 tables and sends its gradient to table f. Each table has 1/N of the rows, so the parameter count is unchanged |
| `engram-cross-chunk.json` | Two cross tables, with the parts taken as 512-token pieces of the token stream in turn instead of articles |

With cross tables, no table is read at a position it learns from, so every position in training sees what a held-out position sees: rows written only by other text.

## Results

Perplexity on the full validation and test splits (`engram_lm/eval_cut.py`, bfloat16 on GPU). The earlier Engram and baseline are the runs of `2026-10-21-engram-nanogpt`, evaluated the same way. "Train gain" is the Engram run's training loss minus the baseline's over the last 200 steps. "Val gain" is the same difference in validation loss at the end. `results/retrain.json` holds every value, and `results/eval_cut-*.json` the raw evaluations.

| Model | Seeds | Evaluation-time cutoff | Validation | Test | Train gain | Val gain |
|---|---|---|---|---|---|---|
| Baseline (no Engram) | 3 | — | 29.26 | 29.35 | — | — |
| Engram | 3 | none | 28.97 | 29.40 | −0.151 | −0.010 |
| Engram | 3 | c / (c + 5) | 28.22 | 28.60 | | |
| `engram-cut5` | 1337 | as trained | 28.47 | 28.92 | −0.150 | −0.028 |
| `engram-soft5` | 1337 | as trained | 28.42 | 28.80 | −0.150 | −0.029 |
| `engram-cross` | 3 | none | 28.54 | 28.66 | −0.007 | −0.025 |
| `engram-cross` | 3 | c / (c + 5) | 28.53 | 28.65 | | |
| `engram-cross-full` | 1337 | none | 28.29 | 28.46 | −0.010 | −0.034 |
| `engram-cross-chunk` | 1337 | none | 28.41 | 28.63 | −0.010 | −0.029 |
| `engram-cross3` | 3 | none | 28.38 | 28.50 | −0.011 | −0.029 |
| `engram-cross4` | 3 | none | 28.41 | 28.52 | −0.011 | −0.029 |
| `engram-cross8` | 1337 | none | 28.57 | 28.64 | −0.011 | −0.024 |

Cross tables close the gap between training and held-out text. For all three seeds, the gain on the training text falls from about −0.15 to between −0.005 and −0.009, and the model is lower than the earlier Engram on both validation and test. The cross-table model needs no count table at inference, and its test perplexity is about that of the earlier Engram with an evaluation-time cutoff, which does read the training counts. Adding the cutoff to the cross-table model changes almost nothing. On validation the earlier Engram with the cutoff is still 0.32 lower.

Three tables are better than two: for each of the three seeds, `engram-cross3` is lower than `engram-cross` on both validation and test. Its test perplexity, 28.50, is 3.1% below the earlier Engram and 2.9% below the baseline, and below the earlier Engram with the evaluation-time cutoff (28.60); on validation that combination is still 0.16 lower. Four tables are about the same as three, and eight are worse. With the parameter count fixed, more parts give each position tables that learned from more of the text, but fewer rows per table. Splitting by 512-token pieces instead of articles changes almost nothing (seed 1337: 28.41 and 28.63, against 28.42 and 28.62 by articles).

Training with the count cutoff does not remove the memorisation: the gain on the training text stays at −0.150. Both cutoff runs are above the earlier Engram with the cutoff applied only at evaluation.

## Running

The Job reuses the PersistentVolumeClaim `engram-lm` of `2026-10-21-engram-nanogpt` and needs no other preparation. It finds the tokenised WikiText-103 there and builds `counts3.npz`, the 3-gram counts of the training text (about 460 MB), once. It also evaluates that experiment's Engram checkpoint of each seed in `RUNS` when it is on the volume. `RUNS` lists `<config>:<seed>` pairs, one GPU each, so the GPU limit must be at least the number of pairs. The runs above used `engram-cut5:1337 engram-soft5:1337 engram-cross:1337` (4 GPUs), `engram-cross:1338 engram-cross:1339 engram-cross-full:1337` (3), `engram-cross3:1337 engram-cross4:1337 engram-cross8:1337 engram-cross-chunk:1337` (4) and `engram-cross3:1338 engram-cross3:1339 engram-cross4:1338 engram-cross4:1339` (4); the GPU limit also has to cover the earlier-experiment evaluations, one per seed.

```bash
export NAMESPACE=default
kubectl kustomize --load-restrictor LoadRestrictionsNone k8s/base | kubectl apply -n "$NAMESPACE" -f -
kubectl -n "$NAMESPACE" wait --for=condition=complete job/engram-cut-train --timeout=2h
```

Each run writes `/work/runs/$RUN_NAME/<config>-seed<seed>/` with the training files of the earlier experiment and `eval_cut.json`. The Job refuses an existing `RUN_NAME`, so change it before running again.

`tests/test_cutoff.py` checks the parts this experiment adds on a tiny model on CPU, with nothing to download:

- the 3-gram keys match the counted ones;
- training uses count − 1;
- a zero weight equals reading zero 3-gram rows;
- a cross-table position reads the mean of the other tables and sends its gradient to its own, for two and three tables.

```bash
uv run --with torch --with numpy --with pytest python -m pytest -q tests
```
