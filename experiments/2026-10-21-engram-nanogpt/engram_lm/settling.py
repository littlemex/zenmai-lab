"""Where predictions settle over a whole split, for an Engram model, the same weights with Engram
skipped, and the baseline of the same seed.

A prediction settles at the first block from which the LogitLens top-1 equals the final top-1 and
stays equal up to the last block, as in the trace viewer's ladder view. A position can settle on a
guess its model gives little probability, so the top-1 probability at the settled block is reported
too. The variants are compared over every position, and over the positions where both end on the
token that actually comes next.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import TokenFile
from .evaluate import load_pair
from .record_trace import git_commit, sha256

VARIANTS = (("engram", False), ("engram_off", True), ("baseline", False))
PAIRS = (("engram", "baseline"), ("engram", "engram_off"))


@torch.no_grad()
def settle(model, x: torch.Tensor, engram_off: bool, device: str) -> dict:
    """Per position: the settled block, the final top-1, and the top-1 probability at the settled block
    and at the last block. The forward pass runs under the evaluation's autocast (bfloat16 on a GPU,
    unlike the recorder's float32 on the CPU); each block's projection runs in float32, one block at
    a time."""
    rec = {}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        model(x.unsqueeze(0).to(device), record=rec, engram_off=engram_off)
    tops, probs = [], []
    with torch.autocast(device_type="cuda", enabled=False):
        for hidden in rec["hidden"]:
            p, top = model.lm_head(model.ln_f(hidden.float()))[0].float().softmax(-1).max(-1)
            tops.append(top)
            probs.append(p)
    tops, probs = torch.stack(tops), torch.stack(probs)  # [block, position]
    # the blocks from the top down that agree with the last one
    run = (tops == tops[-1]).flip(0).int().cumprod(0).sum(0)
    settled = len(tops) - run
    return {"settled": settled.cpu(), "final": tops[-1].cpu(), "p": probs.gather(0, settled.unsqueeze(0))[0].cpu(),
            "p_last": probs[-1].cpu()}


def median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def compare(a: dict, b: dict, where: np.ndarray, n_layer: int) -> dict:
    """a against b over the positions in where. For the positions where a settles sooner, the top-1
    probability each gives at its own settled block, so an earlier block reached with less
    confidence shows."""
    sa, sb = a["settled"][where], b["settled"][where]
    sooner = sa < sb
    above = lambda s: float((n_layer - 1 - s).mean()) if len(s) else None  # noqa: E731
    return {"positions": int(where.sum()), "sooner": int(sooner.sum()), "later": int((sa > sb).sum()),
            "same": int((sa == sb).sum()), "blocks_above": [above(sa), above(sb)],
            "median_top1_probability_at_settled_block_where_sooner": [median(a["p"][where][sooner]),
                                                                       median(b["p"][where][sooner])]}


def run(engram_path: Path, baseline_path: Path, data: TokenFile, windows: int | None, device: str) -> dict:
    engram, engram_state, baseline, _ = load_pair(engram_path, baseline_path, device)
    models = {"engram": engram, "engram_off": engram, "baseline": baseline}
    n_layer = engram.config.n_layer
    out = {name: [] for name, _ in VARIANTS}
    targets = []
    for x, y in data.windows(engram.config.block_size, windows):
        targets.append(y.numpy())
        for name, off in VARIANTS:
            out[name].append(settle(models[name], x, off, device))
    if not targets:
        raise ValueError("no full window in the data")
    y = np.concatenate(targets)
    arrays = {name: {k: np.concatenate([w[k].numpy() for w in windows_]) for k in windows_[0]} for name, windows_ in out.items()}
    everywhere = np.ones(len(y), dtype=bool)
    result = {"seed": engram_state.get("seed"), "files": {"engram": sha256(engram_path), "baseline": sha256(baseline_path)},
              "commit": git_commit(), "positions": int(len(y)), "n_layer": n_layer, "variants": {}, "pairs": {}}
    for name, a in arrays.items():
        result["variants"][name] = {
            "mean_settled_block": float(a["settled"].mean()),
            "settled_block_histogram": np.bincount(a["settled"], minlength=n_layer).tolist(),
            "median_top1_probability_at_settled_block": median(a["p"]),
            "median_top1_probability_at_last_block": median(a["p_last"]),
            "final_top1_is_next_token": float((a["final"] == y).mean()),
        }
    for first, second in PAIRS:
        a, b = arrays[first], arrays[second]
        result["pairs"][f"{first}/{second}"] = {
            "all": compare(a, b, everywhere, n_layer),
            "both_end_on_next_token": compare(a, b, (a["final"] == y) & (b["final"] == y), n_layer),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engram", type=Path, required=True, help="the Engram model: ckpt.pt or model.safetensors")
    parser.add_argument("--baseline", type=Path, required=True, help="the baseline of the same seed")
    parser.add_argument("--data", type=Path, required=True, help="a token file written by engram_lm.data, such as val.bin")
    parser.add_argument("--windows", type=int, default=None, help="the first N windows; all full windows by default")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {"data": args.data.name, **run(args.engram, args.baseline, TokenFile(args.data), args.windows, device)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
