"""Count every 3-gram of compressed ids in the training text, for the count cutoff.

Writes counts3.npz (keys: sorted int64 3-gram keys, counts: int32) next to train.bin. The key is the one
NgramHash.trigram_keys computes. Built once from train.bin; the validation and test texts are not read.
"""

import argparse
from pathlib import Path

import numpy as np

from .tokenizer import build_compressed_token_map, load_tokenizer


def build(data_dir: Path) -> Path:
    out = data_dir / "counts3.npz"
    if out.exists():
        return out
    token_map, k = build_compressed_token_map(load_tokenizer())
    c = np.asarray(token_map, dtype=np.int64)[np.fromfile(data_dir / "train.bin", dtype=np.uint16).astype(np.int64)]
    keys = (c[:-2] * k + c[1:-1]) * k + c[2:]
    del c
    u, n = np.unique(keys, return_counts=True)
    tmp = data_dir / "counts3.tmp.npz"
    np.savez(tmp, keys=u, counts=n.astype(np.int32))
    tmp.rename(out)
    return out


def load(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(build(data_dir))
    return z["keys"], z["counts"]


def folds(tokens: np.ndarray, eot_id: int, n: int, split: str) -> np.ndarray:
    """uint8 per token: the part of the training text it is in, 0..n-1. "article": articles in turn
    (articles are separated by <|endoftext|>); "chunk512": 512-token pieces of the token stream in turn."""
    if split == "article":
        return (np.cumsum(tokens == eot_id) % n).astype(np.uint8)
    if split == "chunk512":
        return ((np.arange(len(tokens)) // 512) % n).astype(np.uint8)
    raise ValueError(f"unknown engram_cross_split {split!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, required=True)
    print(build(ap.parse_args().data_dir))
