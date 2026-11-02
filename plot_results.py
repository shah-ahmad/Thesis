"""Create thesis-ready PNG charts from validation/test benchmark results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    "Accuracy", "Balanced Accuracy", "Precision (attack)",
    "Recall / Detection Rate (attack)", "F1 (attack)", "F1 (macro)", "ROC AUC",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("thesis_benchmark_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("visuals"))
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Results file not found: {args.input}")

    results = pd.read_csv(args.input)
    required = {"Model", "Split", *METRICS, "False Positives", "False Negatives", "Training Time (s)"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results CSV is missing columns: {', '.join(sorted(missing))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Validation": "#2563EB", "Test": "#DC2626"}

    # Validation and test scores use separate panels; the full 0-to-1 axis avoids hiding changes.
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, split in zip(axes, ("Validation", "Test")):
        subset = results.loc[results["Split"] == split].set_index("Model")
        subset[METRICS].plot(kind="bar", ax=ax, width=0.82, colormap="tab10")
        ax.set_title(f"{split} results")
        ax.set_xlabel("")
        ax.set_ylabel("Score (0 to 1)")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Model comparison by held-out partition")
    fig.tight_layout()
    fig.savefig(args.output_dir / "performance_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, split in zip(axes, ("Validation", "Test")):
        subset = results.loc[results["Split"] == split]
        ax.scatter(subset["False Positives"], subset["False Negatives"],
                   s=100, color=colors[split])
        for _, row in subset.iterrows():
            ax.annotate(row["Model"], (row["False Positives"], row["False Negatives"]),
                        xytext=(6, 6), textcoords="offset points", fontsize=8)
        ax.set_title(f"{split}: errors (lower is better)")
        ax.set_xlabel("False positives")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("False negatives")
    fig.tight_layout()
    fig.savefig(args.output_dir / "test_set_errors.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    test_results = results.loc[results["Split"] == "Test"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(test_results["Model"], test_results["Training Time (s)"], color="#43AA8B")
    ax.bar_label(bars, fmt="%.2f s", padding=3)
    ax.set_title("Training time (models trained on Monday–Wednesday)")
    ax.set_ylabel("Seconds (lower is faster)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Created validation/test charts in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
