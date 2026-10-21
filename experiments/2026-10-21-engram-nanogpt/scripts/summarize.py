"""Summarise a run directory (one <variant>-seed<seed>/ per training) into results/summary.json and figures.

Reports, per variant, the mean and the sample standard deviation over seeds of validation and test
perplexity, the Engram-off validation perplexity, the LogitLens KL per layer and the gate statistics,
and the validation loss at every evaluation of every seed; and draws the validation loss curves and
the per-layer KL.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def load_runs(run_dir: Path) -> dict:
    runs: dict[str, dict[int, dict]] = {}
    for path in sorted(run_dir.glob("*-seed*/eval.json")):
        match = re.fullmatch(r"(\w+)-seed(\d+)", path.parent.name)
        if not match:
            continue
        variant, seed = match.group(1), int(match.group(2))
        metrics = [json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines()]
        runs.setdefault(variant, {})[seed] = {"eval": json.loads(path.read_text()), "metrics": metrics,
                                              "config": json.loads((path.parent / "config.json").read_text())}
    if not runs:
        raise SystemExit(f"no <variant>-seed<seed>/eval.json under {run_dir}")
    return runs


def stat(values: list[float]) -> dict:
    a = np.array(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": len(a),
            "values": a.tolist()}


def summarise(runs: dict) -> dict:
    summary = {}
    for variant, seeds in runs.items():
        evals = [r["eval"] for r in seeds.values()]
        entry = {
            "seeds": sorted(seeds),
            "parameters": evals[0]["parameters"],
            "val_ppl": stat([e["val"]["ppl"] for e in evals]),
            "test_ppl": stat([e["test"]["ppl"] for e in evals]),
            "kl_final_to_layer": [stat(list(col)) for col in zip(*[e["logit_lens"]["kl_final_to_layer"] for e in evals])],
            "tokens_trained": seeds[min(seeds)]["metrics"][-1]["tokens"],
            "tokens_per_s": stat([float(np.median([m["tokens_per_s"] for m in r["metrics"][10:]])) for r in seeds.values()]),
            # the validation loss at every evaluation, [step, loss], per seed
            "val_loss_curve": {str(seed): [[m["step"], m["val_loss"]] for m in r["metrics"] if "val_loss" in m]
                               for seed, r in sorted(seeds.items())},
        }
        if "val_engram_off" in evals[0]:
            entry["val_ppl_engram_off"] = stat([e["val_engram_off"]["ppl"] for e in evals])
            entry["gates"] = {k: {"mean": stat([e["logit_lens"]["gates"][k]["mean"] for e in evals]),
                                  "share_above_half": stat([e["logit_lens"]["gates"][k]["share_above_half"] for e in evals])}
                              for k in evals[0]["logit_lens"]["gates"]}
        summary[variant] = entry
    return summary


def plot(runs: dict, summary: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": "#000000", "axes.facecolor": "#0d1117", "axes.edgecolor": "#5a6270",
                         "axes.labelcolor": "#e8edf2", "xtick.color": "#c9d3dc", "ytick.color": "#c9d3dc",
                         "text.color": "#e8edf2", "grid.color": "#2a323c", "legend.facecolor": "#0d1117",
                         "legend.edgecolor": "#5a6270", "font.size": 13})
    colors = {"engram": "#3bd16f", "baseline": "#2f9bff"}
    styles = ["-", "--", ":"]  # one line style per seed, so seeds are told apart without colour
    fig, (ax, gap) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    curves = {}
    for variant, seeds in runs.items():
        for i, (seed, run) in enumerate(sorted(seeds.items())):
            points = [(m["tokens"] / 1e6, m["val_loss"]) for m in run["metrics"] if "val_loss" in m and m["step"] > 0]
            curves[(variant, seed)] = dict(points)
            ax.plot(*zip(*points), color=colors.get(variant), ls=styles[i % 3], alpha=0.9, lw=2, label=f"{variant} {seed}")
    ax.set_ylabel("validation loss")
    ax.set_ylim(3.2, 6.0)
    ax.legend(ncol=2, fontsize=11)
    ax.grid(alpha=0.3)
    if "engram" in runs and "baseline" in runs:
        for i, seed in enumerate(sorted(set(runs["engram"]) & set(runs["baseline"]))):
            e, b = curves[("engram", seed)], curves[("baseline", seed)]
            xs = sorted(set(e) & set(b))
            gap.plot(xs, [e[x] - b[x] for x in xs], color="#fbd332", ls=styles[i % 3], alpha=0.9, lw=2, label=str(seed))
        gap.axhline(0, color="#5a6270", lw=1)
        gap.set_ylabel("engram − baseline")
        gap.legend(fontsize=11, title="seed", ncol=3)
        gap.grid(alpha=0.3)
    gap.set_xlabel("training tokens (millions)")
    fig.tight_layout()
    fig.savefig(out_dir / "val-loss.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    for variant, entry in summary.items():
        # the last block's KL is zero by definition: its lens is the final distribution
        kl = entry["kl_final_to_layer"][:-1]
        ax.errorbar(np.arange(len(kl)), [s["mean"] for s in kl], yerr=[s["std"] for s in kl], marker="o", capsize=3,
                    lw=2, color=colors.get(variant), label=variant)
    ax.set_xlabel("block")
    ax.set_ylabel("KL(final || LogitLens at this block)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "logit-lens-kl.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.run_dir)
    summary = summarise(runs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(runs, summary, args.out_dir)
    for variant, entry in summary.items():
        line = f"{variant}: val ppl {entry['val_ppl']['mean']:.2f} ± {entry['val_ppl']['std']:.2f}, test {entry['test_ppl']['mean']:.2f} ± {entry['test_ppl']['std']:.2f}"
        if "val_ppl_engram_off" in entry:
            line += f", Engram off {entry['val_ppl_engram_off']['mean']:.2f}"
        print(line)


if __name__ == "__main__":
    main()
