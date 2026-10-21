"""WikiText-103 tokenised with GPT-2 into flat uint16 token files, and random training windows."""

import json
from pathlib import Path

import numpy as np
import torch

from .tokenizer import EOT_ID, load_tokenizer

DATASET_REPO = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SPLITS = {"train": "train", "val": "validation", "test": "test"}


def cached(out_dir: Path) -> dict | None:
    """The metadata of a complete earlier preparation of the same dataset, or None."""
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("dataset") != f"{DATASET_REPO}/{DATASET_CONFIG}@{DATASET_REVISION}":
        raise ValueError(f"{out_dir} holds {meta.get('dataset')}, not this dataset")
    for name, count in meta["tokens"].items():
        if (out_dir / f"{name}.bin").stat().st_size != 2 * count:
            raise ValueError(f"{out_dir / name}.bin does not hold the {count} tokens meta.json records")
    return meta


def prepare(out_dir: Path) -> dict:
    """Writes {train,val,test}.bin and meta.json; articles are joined with <|endoftext|>.

    Under an exclusive lock, each file is written to a temporary name and renamed when complete,
    and meta.json is written last, so a reader never maps a partial file.
    """
    import fcntl

    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        meta = cached(out_dir)
        if meta is not None:
            return meta
        return _prepare(out_dir, load_dataset)


def _prepare(out_dir: Path, load_dataset) -> dict:
    meta_path = out_dir / "meta.json"
    tokenizer = load_tokenizer()
    dataset = load_dataset(DATASET_REPO, DATASET_CONFIG, revision=DATASET_REVISION)
    meta = {"dataset": f"{DATASET_REPO}/{DATASET_CONFIG}@{DATASET_REVISION}", "tokens": {}}
    for name, split in SPLITS.items():
        ids: list[int] = []
        article: list[str] = []

        def flush():
            if article:
                ids.extend(tokenizer.encode("".join(article)).ids)
                ids.append(EOT_ID)
                article.clear()

        for line in dataset[split]["text"]:
            # an article starts at a level-1 heading " = Title = "
            if line.startswith(" = ") and not line.startswith(" = = ") and article:
                flush()
            article.append(line)
        flush()
        array = np.array(ids, dtype=np.int64)
        if array.min() < 0 or array.max() >= 2**16:
            raise ValueError("token ids do not fit uint16")
        tmp = out_dir / f"{name}.bin.tmp"
        array.astype(np.uint16).tofile(tmp)
        tmp.rename(out_dir / f"{name}.bin")
        meta["tokens"][name] = len(ids)
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.rename(meta_path)
    return meta


class TokenFile:
    def __init__(self, path: Path):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")

    def batch(self, batch_size: int, block_size: int, generator: np.random.Generator, device: str):
        # a window needs block_size + 1 tokens, so the last valid start is len - block_size - 1
        starts = generator.integers(0, len(self.tokens) - block_size, size=batch_size)
        x = np.stack([self.tokens[s : s + block_size] for s in starts]).astype(np.int64)
        y = np.stack([self.tokens[s + 1 : s + 1 + block_size] for s in starts]).astype(np.int64)
        return torch.from_numpy(x).to(device, non_blocking=True), torch.from_numpy(y).to(device, non_blocking=True)

    def windows(self, block_size: int, limit: int | None = None):
        """Consecutive non-overlapping full windows from the start, for deterministic evaluation.

        Each window starts with no history: attention, positions and n-gram look-back reset at its
        first token, exactly as in training. The tokens after the last full window are not scored."""
        n = (len(self.tokens) - 1) // block_size
        for i in range(n if limit is None else min(n, limit)):
            s = i * block_size
            x = torch.from_numpy(self.tokens[s : s + block_size].astype(np.int64))
            y = torch.from_numpy(self.tokens[s + 1 : s + 1 + block_size].astype(np.int64))
            yield x, y


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Tokenise WikiText-103 into flat token files.")
    parser.add_argument("--out-dir", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().out_dir)), flush=True)


if __name__ == "__main__":
    main()
