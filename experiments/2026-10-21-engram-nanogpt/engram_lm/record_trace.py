"""Record an engram-trace JSON of the trained models on one text, for the Engram trace viewer.

The text is a prompt continued greedily by the Engram model. Every variant then reads that same
token sequence, so the variants differ only in the model, not in what they read:

* engram: the Engram model;
* engram_off: the same weights with the Engram layer skipped;
* baseline: the model trained without Engram from the same seed.

For each position and block the trace holds the LogitLens top-k tokens, the probability the block
gives to the variant's own final prediction and to the actual next token, and KL(final || block).
It also holds the Engram model's row ids and gates, so the same file opens in the viewer's table view.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from .evaluate import load_pair
from .tokenizer import EOT_ID, build_compressed_token_map, load_tokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    """The commit of the recorder, with "-dirty" when this experiment's files have uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=Path(__file__).parents[1]).stdout.strip()
    try:
        return git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain", "--", ".") else "")
    except (OSError, subprocess.CalledProcessError):
        return None


@torch.no_grad()
def generate(model, ids: list[int], new_tokens: int) -> list[int]:
    out = list(ids)
    for _ in range(new_tokens):
        logits, _ = model(torch.tensor([out]))
        out.append(int(logits[0, -1].argmax()))
    return out


@torch.no_grad()
def lens_of(model, ids: torch.Tensor, top_k: int, engram_off: bool) -> tuple[list, dict]:
    """[position][block] -> {"top": [[id, p], ...], "p_final": p, "p_next": p | None, "kl": x}"""
    rec = {}
    logits, _ = model(ids, record=rec, engram_off=engram_off)
    final = F.log_softmax(logits.float()[0], dim=-1)
    final_top = final.argmax(-1)
    next_ids = ids[0, 1:]
    per_block = []
    for hidden in rec["hidden"]:
        lens = F.log_softmax(model.lm_head(model.ln_f(hidden)).float()[0], dim=-1)
        kl = (final.exp() * (final - lens)).sum(-1)  # KL(final || block), per position
        p = lens.exp()
        probs, top = p.topk(top_k, dim=-1)
        p_final = p.gather(-1, final_top.unsqueeze(-1)).squeeze(-1)
        p_next = p[:-1].gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
        per_block.append((top, probs, p_final, p_next, kl))
    positions = []
    for t in range(ids.shape[1]):
        positions.append([
            {"top": [[int(i), round(float(p), 5)] for i, p in zip(top[t], probs[t])],
             "p_final": round(float(p_final[t]), 5),
             "p_next": round(float(p_next[t]), 5) if t < ids.shape[1] - 1 else None,
             "kl": round(float(kl[t]), 5)}
            for top, probs, p_final, p_next, kl in per_block
        ])
    return positions, rec


def label_byte_runs(tokens: list[dict], tokenizer) -> None:
    """Label tokens that hold part of one character, such as "庁 1/2" (see the viewer's recorder)."""
    start = 0
    while start < len(tokens):
        if "\ufffd" not in tokens[start]["text"]:
            start += 1
            continue
        for end in range(start + 1, len(tokens) + 1):
            joined = tokenizer.decode([t["id"] for t in tokens[start:end]])
            if "\ufffd" not in joined:
                for i, token in enumerate(tokens[start:end]):
                    token["display"] = f"{joined.strip()} {i + 1}/{end - start}"
                start = end
                break
        else:
            start += 1


def record(engram_ckpt: Path, baseline_ckpt: Path, prompt: str, new_tokens: int, top_k: int, title: str) -> dict:
    tokenizer = load_tokenizer()
    token_map, _ = build_compressed_token_map(tokenizer)
    engram, engram_state, baseline, baseline_state = load_pair(engram_ckpt, baseline_ckpt, "cpu")
    if top_k < 1 or new_tokens < 0:
        raise ValueError("--top-k must be at least 1 and --new-tokens at least 0")

    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids or len(prompt_ids) + new_tokens > engram.config.block_size:
        raise ValueError(f"the prompt ({len(prompt_ids)} tokens) and --new-tokens must fit in the "
                         f"{engram.config.block_size}-token context and the prompt must not be empty")
    ids = generate(engram, prompt_ids, new_tokens)
    x = torch.tensor([ids])

    lens, engram_record = {}, None
    for name, model, off in (("engram", engram, False), ("engram_off", engram, True), ("baseline", baseline, False)):
        lens[name], rec = lens_of(model, x, top_k, off)
        if name == "engram":
            engram_record = rec

    layout = engram.engram_hash.layout
    layers = []
    for index, layer_id in enumerate(layout.layer_ids):
        columns, offset = [], 0
        for c, size in enumerate(layout.primes[index]):
            columns.append({"n": layout.column_ngram(c), "head": c % layout.n_heads, "offset": offset, "size": size})
            offset += size
        layers.append({"layer_id": layer_id, "num_rows": offset, "columns": columns})
    rows = engram_record["rows"][0].tolist()  # [L, n_layers, n_columns]
    gates = [[[round(float(engram_record["gates"][layer_id][0, t]), 5)] for layer_id in layout.layer_ids]
             for t in range(len(ids))]

    tokens = [{"id": i, "text": tokenizer.decode([i]), "compressed_id": token_map[i]} for i in ids]
    label_byte_runs(tokens, tokenizer)
    vocab = sorted({i for variant in lens.values() for per_pos in variant for per_block in per_pos
                    for i, _ in per_block["top"]})
    commit = git_commit()
    return {
        "format": "engram-trace",
        "version": 1,
        "title": title,
        "producer": {"name": "engram_lm.record_trace", "commit": commit},
        "source": {
            "model": "zenmai-lab experiments/2026-10-21-engram-nanogpt",
            "seed": engram_state.get("seed"),
            "files": {engram_ckpt.name + " (engram)": {"sha256": sha256(engram_ckpt)},
                      baseline_ckpt.name + " (baseline)": {"sha256": sha256(baseline_ckpt)}},
        },
        "provenance": {
            "hash_rows": "Computed by the trained Engram model's own NgramHash.",
            "gates": "Recorded from the trained Engram model's forward pass over this text.",
            "lens": "LogitLens of each variant over this text: each block's residual stream through the final LayerNorm "
                    "and head, in float32 on the CPU.",
        },
        "model": {"max_ngram_size": layout.max_ngram_size, "n_heads": layout.n_heads,
                  "head_dim": engram.config.engram_head_dim, "hc_mult": 1, "pad_compressed_id": token_map[EOT_ID]},
        "layers": layers,
        "tokens": tokens,
        "hash_rows": rows,
        "gates": gates,
        "generation": {"prompt_tokens": len(prompt_ids), "new_tokens": len(ids) - len(prompt_ids),
                       "by": "engram", "method": "greedy"},
        "lens": {
            "n_layer": engram.config.n_layer,
            "top_k": top_k,
            "engram_before_block": list(layout.layer_ids),
            # "engram" says what happens at engram_before_block; gates are recorded for the "added" variant
            "variants": [
                {"name": "engram", "label": "with Engram", "engram": "added", "checkpoint_step": engram_state["step"]},
                {"name": "engram_off", "label": "same weights, Engram skipped", "engram": "skipped",
                 "checkpoint_step": engram_state["step"]},
                {"name": "baseline", "label": "trained without Engram", "engram": "absent",
                 "checkpoint_step": baseline_state["step"]},
            ],
            "vocab": {str(i): tokenizer.decode([i]) for i in vocab},
            "values": lens,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engram", type=Path, required=True, help="the Engram model: ckpt.pt or model.safetensors")
    parser.add_argument("--baseline", type=Path, required=True, help="the baseline of the same seed")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--new-tokens", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--title", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    trace = record(args.engram, args.baseline, args.prompt, args.new_tokens, args.top_k, args.title or args.prompt[:60])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(trace['tokens'])} tokens ({trace['generation']['new_tokens']} generated)")


if __name__ == "__main__":
    main()
