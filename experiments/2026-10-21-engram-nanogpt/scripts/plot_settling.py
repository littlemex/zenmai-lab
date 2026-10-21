"""Combine engram_lm.settling outputs of several seeds into results/settling.json and settling.png."""

import argparse
import json
from pathlib import Path

import numpy as np

VARIANTS = {"engram": ("with Engram", "#3bd16f"), "engram_off": ("same weights, Engram skipped", "#9aa5b1"),
            "baseline": ("trained without Engram", "#2f9bff")}


def stat(values) -> dict:
    """Mean and sd over seeds; a seed whose value is undefined (an empty subset) is left out."""
    defined = [float(v) for v in values if v is not None]
    return {"mean": float(np.mean(defined)) if defined else None,
            "std": float(np.std(defined, ddof=1)) if len(defined) > 1 else None, "values": list(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", type=Path, nargs="+", help="settling JSON files, one per seed")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = sorted((json.loads(p.read_text()) for p in args.inputs), key=lambda r: r["seed"])
    for key in ("n_layer", "positions", "data"):
        if len({r.get(key) for r in runs}) != 1:
            raise SystemExit(f"the inputs differ in {key}")
    if len({r["seed"] for r in runs}) != len(runs):
        raise SystemExit("two inputs have the same seed")
    missing = [v for v in VARIANTS for r in runs if v not in r["variants"]]
    if missing:
        raise SystemExit(f"an input has no variant {missing[0]}")
    n_layer = runs[0]["n_layer"]
    share = {v: np.array([np.array(r["variants"][v]["settled_block_histogram"]) / r["positions"] for r in runs]) for v in VARIANTS}
    summary = {"seeds": [r["seed"] for r in runs], "data": runs[0].get("data"), "positions_per_seed": runs[0]["positions"],
               "files": {r["seed"]: r.get("files") for r in runs}, "commits": sorted({r["commit"] for r in runs if r.get("commit")}),
               "variants": {}, "pairs": {}}
    for v in VARIANTS:
        keys = ("mean_settled_block", "median_top1_probability_at_settled_block", "median_top1_probability_at_last_block",
                "final_top1_is_next_token")
        summary["variants"][v] = {k: stat([r["variants"][v][k] for r in runs]) for k in keys}
        summary["variants"][v]["settled_block_share"] = [stat(share[v][:, b]) for b in range(n_layer)]
    for pair in runs[0]["pairs"]:
        summary["pairs"][pair] = {}
        for subset in runs[0]["pairs"][pair]:
            counts = [r["pairs"][pair][subset] for r in runs]
            summary["pairs"][pair][subset] = {
                "positions": [c["positions"] for c in counts],
                **{k: stat([c[k] / c["positions"] if c["positions"] else None for c in counts]) for k in ("sooner", "later", "same")},
                **{k: [stat([c[k][i] for c in counts]) for i in range(2)]
                   for k in ("blocks_above", "median_top1_probability_at_settled_block_where_sooner")},
            }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "settling.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": "#000000", "axes.facecolor": "#0d1117", "axes.edgecolor": "#5a6270",
                         "axes.labelcolor": "#e8edf2", "xtick.color": "#c9d3dc", "ytick.color": "#c9d3dc",
                         "text.color": "#e8edf2", "grid.color": "#2a323c", "legend.facecolor": "#0d1117",
                         "legend.edgecolor": "#5a6270", "font.size": 13})
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(VARIANTS)
    for i, (v, (label, color)) in enumerate(VARIANTS.items()):
        x = np.arange(n_layer) + (i - (len(VARIANTS) - 1) / 2) * width
        yerr = 100 * share[v].std(0, ddof=1) if len(runs) > 1 else None
        ax.bar(x, 100 * share[v].mean(0), width, yerr=yerr, color=color, label=label, error_kw={"ecolor": "#e8edf2", "lw": 1})
    ax.set_xticks(range(n_layer))
    ax.set_xlabel("block from which the top-1 prediction no longer changes")
    ax.set_ylabel(f"% of positions in {runs[0].get('data') or 'the data'}")
    ax.grid(axis="y", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "settling.png", dpi=150)


if __name__ == "__main__":
    main()
