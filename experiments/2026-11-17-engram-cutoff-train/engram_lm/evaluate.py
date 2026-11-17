"""Evaluate a trained model: perplexity over every full window of a split, per-layer LogitLens KL, Engram gate statistics.

LogitLens projects the residual stream after each block through the final LayerNorm and the
output head and reports KL(final || layer), the divergence of the layer's distribution from the
model's final one, as LogitLens studies usually do; a smaller value at a layer means the model is
closer to its final prediction there. The projections run under the same autocast as the final
logits, so the last layer's value is zero up to rounding.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_model

from .data import TokenFile
from .model import ModelConfig
from .train import build_model


def load(checkpoint: Path, device: str):
    """A training checkpoint (ckpt.pt) or its published form (model.safetensors, see export.py)."""
    if checkpoint.suffix == ".safetensors":
        with safe_open(checkpoint, "pt") as f:
            meta = f.metadata()
        state = {"config": json.loads(meta["config"]), "step": int(meta["step"]), "seed": int(meta["seed"]),
                 "val_loss": float(meta["val_loss"])}
        model = build_model(ModelConfig(**state["config"]))
        load_model(model, checkpoint)
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = build_model(ModelConfig(**state["config"]))
        model.load_state_dict(state["model"])
    return model.to(device).eval(), state


def load_pair(engram_path: Path, baseline_path: Path, device: str):
    """An Engram model and the baseline of the same seed, refused unless they differ only in Engram."""
    engram, engram_state = load(engram_path, device)
    baseline, baseline_state = load(baseline_path, device)
    if not engram.engrams or baseline.engrams:
        raise ValueError("the first model must have Engram and the second must not")
    backbone = lambda config: {k: v for k, v in config.items() if not k.startswith("engram_")}  # noqa: E731
    if backbone(engram_state["config"]) != backbone(baseline_state["config"]):
        raise ValueError("the two models differ outside Engram, so they are not a pair")
    if engram_state.get("seed") != baseline_state.get("seed"):
        raise ValueError(f"seeds differ: {engram_state.get('seed')} and {baseline_state.get('seed')}")
    return engram, engram_state, baseline, baseline_state


@torch.no_grad()
def perplexity(model, data: TokenFile, block_size: int, device: str, limit=None, engram_off: bool = False) -> dict:
    nll, count = 0.0, 0
    for x, y in data.windows(block_size, limit):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, _ = model(x.unsqueeze(0).to(device), engram_off=engram_off)
        nll += F.cross_entropy(logits.float()[0], y.to(device), reduction="sum").item()
        count += y.numel()
    if count == 0:
        raise ValueError("no evaluation windows")
    return {"loss": nll / count, "ppl": float(np.exp(nll / count)), "tokens_scored": count,
            "predictable_tokens_in_split": len(data.tokens) - 1}


@torch.no_grad()
def logit_lens(model, data: TokenFile, block_size: int, windows: int, device: str) -> dict:
    per_layer = np.zeros(model.config.n_layer)
    gate_sum, gate_open, gate_count = {}, {}, {}
    processed = 0
    for x, _ in data.windows(block_size, windows):
        record = {}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, _ = model(x.unsqueeze(0).to(device), record=record)
            lenses = [model.lm_head(model.ln_f(hidden)) for hidden in record["hidden"]]
        final = F.log_softmax(logits.float()[0], dim=-1)
        for layer, lens_logits in enumerate(lenses):
            lens = F.log_softmax(lens_logits.float()[0], dim=-1)
            # kl_div(input, target) is KL(target || input)
            per_layer[layer] += F.kl_div(lens, final, log_target=True, reduction="batchmean").item()
        processed += 1
        for layer_id, gate in record["gates"].items():
            g = gate.float()[0]
            gate_sum[layer_id] = gate_sum.get(layer_id, 0.0) + g.sum().item()
            gate_open[layer_id] = gate_open.get(layer_id, 0.0) + (g > 0.5).sum().item()
            gate_count[layer_id] = gate_count.get(layer_id, 0) + g.numel()
    if processed == 0:
        raise ValueError("no evaluation windows")
    result = {"kl_final_to_layer": (per_layer / processed).tolist(), "windows": processed}
    if gate_count:
        result["gates"] = {str(k): {"mean": gate_sum[k] / gate_count[k], "share_above_half": gate_open[k] / gate_count[k]}
                           for k in gate_count}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lens-windows", type=int, default=200)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, state = load(args.checkpoint, device)
    block = model.config.block_size
    result = {
        "checkpoint": str(args.checkpoint),
        "step": state["step"],
        "parameters": model.num_parameters(),
        "val": perplexity(model, TokenFile(args.data_dir / "val.bin"), block, device),
        "test": perplexity(model, TokenFile(args.data_dir / "test.bin"), block, device),
        "logit_lens": logit_lens(model, TokenFile(args.data_dir / "val.bin"), block, args.lens_windows, device),
    }
    if model.engrams:
        # the same weights with Engram skipped: how much the trained model relies on it
        result["val_engram_off"] = perplexity(model, TokenFile(args.data_dir / "val.bin"), block, device, engram_off=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
