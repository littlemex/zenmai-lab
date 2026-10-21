"""Write a training checkpoint as model.safetensors, with its config, step, seed and validation loss
in the file's metadata, so evaluate.load reads it without unpickling anything."""

import argparse
import json
from pathlib import Path

from safetensors.torch import save_model

from .evaluate import load


def export(checkpoint: Path, out: Path) -> None:
    model, state = load(checkpoint, "cpu")
    meta = {"config": json.dumps(state["config"]), "step": str(state["step"]), "seed": str(state["seed"]),
            "val_loss": repr(state["val_loss"])}
    save_model(model, str(out), metadata=meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export(args.checkpoint, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
