"""Record the Engram n-gram lookups of a text as an engram-trace JSON file.

The row ids are not re-implemented here. The recorder downloads the model's own
`inference/engram.py`, tokenizer and config at a pinned revision and calls its
`NgramHashState`, so every row id in the trace is the one the released model reads.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

DEFAULT_REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_REVISION = "2cba9e42aa026125f3ed06c6d98c1db82f7ca027"
SOURCE_FILES = ("inference/config.json", "inference/engram.py", "tokenizer.json", "tokenizer_config.json")
FORMAT_VERSION = 1


def resolve_revision(repo: str, revision: str) -> str:
    """The commit a branch, tag or commit names, so the trace pins what it read."""
    from huggingface_hub import HfApi

    return HfApi().model_info(repo, revision=revision).sha


def fetch_sources(repo: str, commit: str) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    return {name: Path(hf_hub_download(repo, name, revision=commit)) for name in SOURCE_FILES}


def load_reference_module(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("engram_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_layers(layout) -> list[dict]:
    layers = []
    for layer_index, layer_id in enumerate(layout.layer_ids):
        columns, offset = [], 0
        for n_index, primes in enumerate(layout.primes[layer_index]):
            for head, size in enumerate(primes):
                columns.append({"n": n_index + 2, "head": head, "offset": offset, "size": size})
                offset += size
        if offset != layout.num_embeddings[layer_index]:
            raise ValueError(
                f"layer {layer_id}: columns cover {offset} rows, config says {layout.num_embeddings[layer_index]}"
            )
        layers.append({"layer_id": layer_id, "num_rows": offset, "columns": columns})
    return layers


def label_byte_runs(tokens: list[dict], backend) -> None:
    """Give tokens that hold part of one character a readable label, such as "庁 1/2".

    Byte-level BPE splits some characters over several tokens, and each alone decodes to U+FFFD.
    A run of such tokens is grown one token at a time until it decodes to text, so each run is one
    character; its tokens are then labelled with that character and their place in the run. The
    text field keeps the per-token decode unchanged.
    """
    start = 0
    while start < len(tokens):
        if "\ufffd" not in tokens[start]["text"]:
            start += 1
            continue
        for end in range(start + 1, len(tokens) + 1):
            joined = backend.decode([t["id"] for t in tokens[start:end]], skip_special_tokens=False)
            if "\ufffd" not in joined:
                for i, token in enumerate(tokens[start:end]):
                    token["display"] = f"{joined.strip()} {i + 1}/{end - start}"
                start = end
                break
        else:
            start += 1


def record(text: str, title: str, repo: str, revision: str, max_tokens: int, add_bos: bool) -> dict:
    import torch
    from transformers import PreTrainedTokenizerFast

    commit = resolve_revision(repo, revision)
    paths = fetch_sources(repo, commit)
    reference = load_reference_module(paths["inference/engram.py"])
    config = json.loads(paths["inference/config.json"].read_text(encoding="utf-8"))
    # the same files the model's generate.py loads with AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(paths["tokenizer.json"].parent)

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if add_bos:
        token_ids = [tokenizer.bos_token_id] + token_ids
    if not token_ids:
        raise ValueError("the text produced no tokens")
    if len(token_ids) > max_tokens:
        raise ValueError(f"the text has {len(token_ids)} tokens; raise --max-tokens (now {max_tokens}) to record it")

    args = types.SimpleNamespace(**{**config, "max_batch_size": 1, "max_seq_len": len(token_ids)})
    layout = reference.EngramLayout.from_args(args)
    state = reference.NgramHashState(args, layout, tokenizer)
    hash_rows = state(torch.tensor([token_ids]), 0)[0]  # [L, n_engram_layers, n_hash_cols]

    backend = tokenizer.backend_tokenizer
    tokens = [
        {
            "id": token_id,
            "text": backend.decode([token_id], skip_special_tokens=False),
            "compressed_id": int(state.token_map[token_id]),
        }
        for token_id in token_ids
    ]
    label_byte_runs(tokens, backend)

    return {
        "format": "engram-trace",
        "version": FORMAT_VERSION,
        "title": title,
        "producer": {"name": "record_trace.py", "command": " ".join(sys.argv)},
        "source": {
            "model": repo,
            "revision": commit,
            "files": {name: {"sha256": sha256(path)} for name, path in paths.items()},
        },
        "provenance": {
            "hash_rows": f"Computed by NgramHashState in {repo}@{commit[:12]} inference/engram.py, for these token ids from position 0.",
            "gates": None,
        },
        "model": {
            "max_ngram_size": layout.max_ngram_size,
            "n_heads": layout.n_heads,
            "head_dim": layout.head_dim,
            "hc_mult": config["hc_mult"],
            "pad_compressed_id": int(state.pad_id),
        },
        "layers": build_layers(layout),
        "tokens": tokens,
        "hash_rows": hash_rows.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="The recorder executes inference/engram.py from the model repository. Pin --revision to a commit you trust.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="text to record")
    source.add_argument("--text-file", type=Path, help="file holding the text to record")
    parser.add_argument("--out", type=Path, required=True, help="where to write the trace JSON")
    parser.add_argument("--title", default="", help="title shown by the viewer")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="commit, branch or tag; the trace records the commit")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--add-bos", action="store_true", help="prepend the BOS token, as a chat prompt starts")
    args = parser.parse_args()

    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    trace = record(text, args.title or text[:60], args.repo, args.revision, args.max_tokens, args.add_bos)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(trace['tokens'])} tokens, {len(trace['layers'])} Engram layers")


if __name__ == "__main__":
    main()
