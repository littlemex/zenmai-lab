"""Train one model from a JSON config; writes config.json, metrics.jsonl, best.pt and ckpt.pt (the last step) to --out-dir.

There is no resume: a run is short, and a failed one is rerun into a new directory."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import TokenFile, prepare
from .model import GPT, ModelConfig
from .tokenizer import EOT_ID, build_compressed_token_map, load_tokenizer

TRAIN_DEFAULTS = {
    "batch_size": 32,
    "grad_accum": 4,
    "max_steps": 3600,
    "lr": 6e-4,
    "min_lr": 6e-5,
    "warmup_steps": 300,
    "weight_decay": 0.1,
    "engram_lr_scale": 5.0,
    "grad_clip": 1.0,
    "eval_interval": 200,
    "eval_windows": None,
    "seed": 1337,
    "compile": True,
}


def load_config(path: Path) -> tuple[ModelConfig, dict]:
    raw = json.loads(path.read_text())
    unknown = set(raw.get("train", {})) - set(TRAIN_DEFAULTS)
    if unknown:
        raise ValueError(f"{path}: unknown train settings {sorted(unknown)}")
    return ModelConfig(**raw.get("model", {})), {**TRAIN_DEFAULTS, **raw.get("train", {})}


def build_model(model_config: ModelConfig) -> GPT:
    if not model_config.engram_layer_ids:
        return GPT(model_config)
    token_map, compressed_vocab_size = build_compressed_token_map(load_tokenizer())
    return GPT(model_config, token_map, compressed_vocab_size, EOT_ID)


def learning_rate(step: int, train: dict) -> float:
    if step < train["warmup_steps"]:
        return train["lr"] * (step + 1) / train["warmup_steps"]
    progress = (step - train["warmup_steps"]) / max(1, train["max_steps"] - train["warmup_steps"])
    return train["min_lr"] + 0.5 * (train["lr"] - train["min_lr"]) * (1 + math.cos(math.pi * min(progress, 1.0)))


def optimizers(model: GPT, train: dict):
    """AdamW for the dense weights; Adam with a larger rate and no decay for the Engram tables, as the paper trains them."""
    tables = {id(p) for p in model.engram_parameters()}
    decay, no_decay = [], []
    for p in model.parameters():
        if id(p) in tables:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    fused = torch.cuda.is_available()
    dense = torch.optim.AdamW(
        [{"params": decay, "weight_decay": train["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=train["lr"], betas=(0.9, 0.95), fused=fused)
    sparse = torch.optim.Adam(model.engram_parameters(), lr=train["lr"] * train["engram_lr_scale"], betas=(0.9, 0.95),
                              fused=fused) if tables else None
    return dense, sparse


def save(payload: dict, path: Path) -> None:
    """Write next to the target and rename, so an interrupted save never replaces a good checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)


def clip(model: GPT, max_norm: float) -> dict:
    """Clip the backbone, the Engram projections and the Engram tables as three separate groups, so
    nothing Engram adds can shrink the backbone's update: the backbone is clipped exactly as in the
    baseline. Returns the norm of each group before clipping."""
    engram = {id(p) for p in model.engrams.parameters()}
    tables = {id(p) for p in model.engram_parameters()}
    groups = {
        "backbone": [p for p in model.parameters() if id(p) not in engram],
        "engram": [p for p in model.engrams.parameters() if id(p) not in tables],
        "tables": model.engram_parameters(),
    }
    return {name: float(torch.nn.utils.clip_grad_norm_(params, max_norm)) if params else 0.0
            for name, params in groups.items()}


@torch.no_grad()
def evaluate(model, data: TokenFile, block_size: int, windows: int | None, device: str) -> float:
    model.eval()
    losses = []
    for x, y in data.windows(block_size, windows):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x.unsqueeze(0).to(device), y.unsqueeze(0).to(device))
        losses.append(loss.item())
    model.train()
    if not losses:
        raise ValueError("no validation windows")
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, help="override the config, for a smoke run")
    parser.add_argument("--seed", type=int, help="override the config's seed")
    args = parser.parse_args()

    model_config, train = load_config(args.config)
    if args.max_steps:
        train["max_steps"] = args.max_steps
    if args.seed is not None:
        train["seed"] = args.seed
    if train["eval_windows"] is not None and train["eval_windows"] < 1:
        raise ValueError("eval_windows must be at least 1, or null for every window")
    try:
        args.out_dir.mkdir(parents=True, exist_ok=False)  # one atomic step: a second run cannot share it
    except FileExistsError:
        raise SystemExit(f"{args.out_dir} exists; give each run its own directory") from None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(train["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    meta = prepare(args.data_dir)
    train_data, val_data = TokenFile(args.data_dir / "train.bin"), TokenFile(args.data_dir / "val.bin")
    model = build_model(model_config).to(device)
    counts = model.num_parameters()
    (args.out_dir / "config.json").write_text(json.dumps(
        {"model": model_config.to_dict(), "train": train, "data": meta, "parameters": counts}, indent=2))
    print(json.dumps({"parameters": counts, "tokens": meta["tokens"]}), flush=True)

    step_model = torch.compile(model) if train["compile"] else model
    dense, sparse = optimizers(model, train)
    generator = np.random.default_rng(train["seed"])
    tokens_per_step = train["batch_size"] * train["grad_accum"] * model_config.block_size
    log = (args.out_dir / "metrics.jsonl").open("a")
    best = float("inf")
    started = time.time()
    for step in range(train["max_steps"]):
        lr = learning_rate(step, train)
        for group in dense.param_groups:
            group["lr"] = lr
        if sparse:
            for group in sparse.param_groups:
                group["lr"] = lr * train["engram_lr_scale"]
        t0 = time.time()
        total = 0.0
        for _ in range(train["grad_accum"]):
            x, y = train_data.batch(train["batch_size"], model_config.block_size, generator, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = step_model(x, y)
            (loss / train["grad_accum"]).backward()
            total += loss.item() / train["grad_accum"]
        norms = clip(model, train["grad_clip"])
        dense.step()
        if sparse:
            sparse.step()
        dense.zero_grad(set_to_none=True)
        if sparse:
            sparse.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        record = {"step": step, "loss": total, "lr": lr, "grad_norm": norms,
                  "tokens": (step + 1) * tokens_per_step, "tokens_per_s": tokens_per_step / (time.time() - t0),
                  "elapsed_s": time.time() - started}
        last = step == train["max_steps"] - 1
        if step % train["eval_interval"] == 0 or last:
            record["val_loss"] = evaluate(model, val_data, model_config.block_size, train["eval_windows"], device)
            payload = {"model": model.state_dict(), "config": model_config.to_dict(), "step": step,
                       "val_loss": record["val_loss"], "seed": train["seed"]}
            if record["val_loss"] < best:
                best = record["val_loss"]
                save(payload, args.out_dir / "best.pt")
            if last:
                save(payload, args.out_dir / "ckpt.pt")
            print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()
    log.close()


if __name__ == "__main__":
    main()
