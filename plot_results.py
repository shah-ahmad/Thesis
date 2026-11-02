"""Create readable charts from thesis_benchmark_results.csv without retraining models."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("thesis_benchmark_results.csv"),
        help="Benchmark results CSV (default: thesis_benchmark_results.csv).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("visuals"),
        help="Folder for PNG chart files (default: visuals).",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Results file not found: {args.input}")

    results = pd.read_csv(args.input)
    required = {
        "Model", "Accuracy", "Precision (attack)",
        "Recall / Detection Rate (attack)", "F1 (attack)", "ROC AUC",
        "False Alarm Rate", "Training Time (s)", "False Positives", "False Negatives",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results CSV is missing required columns: {', '.join(sorted(missing))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = results["Model"].astype(str).tolist()
    colors = ["#577590", "#43aa8b", "#f9c74f", "#f94144"][:len(models)]

    # Core classification measures share the 0-to-1 scale.
    metric_columns = [
        ("Accuracy", "Accuracy"),
        ("Precision (attack)", "Attack precision"),
        ("Recall / Detection Rate (attack)", "Attack recall"),
        ("F1 (attack)", "Attack F1"),
        ("ROC AUC", "ROC AUC"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.15
    x = range(len(models))
    for index, (column, label) in enumerate(metric_columns):
        offsets = [position + (index - (len(metric_columns) - 1) / 2) * width for position in x]
        bars = ax.bar(offsets, results[column], width, label=label)
        ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2, rotation=90)
    ax.set_title("Model performance metrics")
    ax.set_ylabel("Score (higher is better)")
    ax.set_ylim(0.97, 1.005)
    ax.set_xticks(list(x), models, rotation=12, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "performance_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Show operational errors directly: fewer false alarms and missed attacks are better.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(results["False Positives"], results["False Negatives"], s=110, c=colors)
    for _, row in results.iterrows():
        ax.annotate(str(row["Model"]), (row["False Positives"], row["False Negatives"]),
                    xytext=(7, 7), textcoords="offset points", fontsize=9)
    ax.set_title("Errors on the test set (lower is better)")
    ax.set_xlabel("False positives (benign flows flagged as attacks)")
    ax.set_ylabel("False negatives (attacks missed)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "test_set_errors.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Training time is a speed comparison, separate from prediction quality.
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, results["Training Time (s)"], color=colors)
    ax.bar_label(bars, fmt="%.2f s", padding=3)
    ax.set_title("Model training time")
    ax.set_ylabel("Seconds (lower is faster)")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Created charts in: {args.output_dir.resolve()}")
    print(" - performance_metrics.png")
    print(" - test_set_errors.png")
    print(" - training_time.png")


if __name__ == "__main__":
    main()
