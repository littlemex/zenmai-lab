"""Perplexity of one checkpoint on val and test under several count-cutoff settings applied at evaluation.

The settings are (k, mode) pairs; k = 0 reads the table as is. The model's own training-time setting is
always included. Writes one JSON.
"""

import argparse
import json
from pathlib import Path

import torch

from . import counts as count_table
from .data import TokenFile
from .evaluate import load, perplexity


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settings", default="0:hard,5:hard,5:soft")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, state = load(a.checkpoint, device)
    model.attach_counts(*count_table.load(a.data_dir))
    trained = (model.config.engram_cut_k, model.config.engram_cut_mode)
    settings = [(int(k), m) for k, m in (x.split(":") for x in a.settings.split(","))]
    if trained not in settings:
        settings.insert(0, trained)
    block = model.config.block_size
    out = {"checkpoint": str(a.checkpoint), "step": state["step"], "trained_with": list(trained), "results": []}
    for k, mode in settings:
        model.config.engram_cut_k, model.config.engram_cut_mode = k, mode
        row = {"k": k, "mode": mode}
        for split in ("val", "test"):
            row[split] = perplexity(model, TokenFile(a.data_dir / f"{split}.bin"), block, device)["ppl"]
        out["results"].append(row)
        print(json.dumps(row), flush=True)
        a.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
