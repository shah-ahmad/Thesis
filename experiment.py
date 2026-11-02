"""Binary CIC-IDS flow benchmark with explicit data selection and reproducible metrics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET_COLUMN = "Label"
# Change this to the CSV you want to benchmark. The file must be inside the data folder.
# Example: DATA_FILE = "Monday-WorkingHours.pcap_ISCX.csv"
DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
# Interactive dashboard appearance (dimensions are in pixels).
CHART_WIDTH = 1200
CHART_HEIGHT = 720
CHART_COLORS = ["#2563EB", "#16A34A", "#F59E0B", "#DC2626", "#7C3AED"]
IDENTIFIER_COLUMNS = {
    "flow id", "source ip", "destination ip", "source port",
    "destination port", "timestamp", "src ip", "dst ip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark binary benign-versus-attack classifiers on a CIC-IDS CSV."
    )
    parser.add_argument(
        "--sample-size", type=int, default=100_000,
        help="Maximum rows to sample before splitting; use 0 for all rows (default: 100000).",
    )
    parser.add_argument("--test-size", type=float, default=0.20, help="Test fraction (default: 0.20).")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed (default: 42).")
    parser.add_argument(
        "--output", type=Path, default=Path("thesis_benchmark_results.csv"),
        help="CSV path for the results table.",
    )
    return parser.parse_args()


def select_dataset() -> Path:
    dataset = Path("data") / DATA_FILE
    if not dataset.is_file():
        available = sorted(path.name for path in Path("data").glob("*.csv"))
        names = "\n".join(f"  - {name}" for name in available) or "  (no CSV files found)"
        raise FileNotFoundError(
            f"Selected dataset was not found: {dataset}\n"
            f"Edit DATA_FILE near the top of experiment.py. CSV files in data/:\n{names}"
        )
    return dataset


def load_data(csv_path: Path, sample_size: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    print(f"[1/5] Reading {csv_path}")
    frame = pd.read_csv(csv_path, low_memory=False)
    frame.columns = frame.columns.astype(str).str.strip()
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' is missing from {csv_path}.")

    original_rows = len(frame)
    frame = frame.drop_duplicates()
    duplicate_rows = original_rows - len(frame)
    labels = frame[TARGET_COLUMN].astype(str).str.strip()
    valid_labels = labels.notna() & labels.ne("")
    frame, labels = frame.loc[valid_labels].copy(), labels.loc[valid_labels]
    y = (labels.str.casefold() != "benign").astype("int8")

    # Remove identifiers and constant columns. Numeric coercion handles CIC-IDS whitespace
    # and occasional malformed values; missing values are imputed inside each model pipeline.
    drop_columns = [
        col for col in frame.columns
        if col != TARGET_COLUMN and col.strip().casefold() in IDENTIFIER_COLUMNS
    ]
    X = frame.drop(columns=[TARGET_COLUMN, *drop_columns]).copy()
    X.columns = [str(column).strip() for column in X.columns]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        X = X.drop(columns=all_missing)
    constant = X.columns[X.nunique(dropna=True) <= 1].tolist()
    if constant:
        X = X.drop(columns=constant)
    if X.empty:
        raise ValueError("No usable numeric feature columns remain after cleaning.")

    if sample_size > 0 and len(X) > sample_size:
        class_counts = y.value_counts()
        if sample_size < len(class_counts):
            raise ValueError("Sample size must be at least the number of classes (2).")
        allocation = (class_counts / len(y) * sample_size).round().astype(int).clip(lower=1)
        while allocation.sum() > sample_size:
            largest = allocation.idxmax()
            if allocation[largest] <= 1:
                break
            allocation[largest] -= 1
        while allocation.sum() < sample_size:
            allocation[class_counts.idxmax()] += 1
        sampled_indices = pd.concat([
            X.loc[y.index[y == label]].sample(n=count, random_state=seed).index.to_series()
            for label, count in allocation.items()
        ]).to_numpy(copy=True)
        rng = np.random.default_rng(seed)
        rng.shuffle(sampled_indices)
        X, y = X.loc[sampled_indices], y.loc[sampled_indices].to_numpy()
    else:
        y = y.to_numpy()

    if len(np.unique(y)) < 2:
        raise ValueError("The selected data contains only one class after cleaning/sampling.")
    print(f"       Usable rows: {len(X):,} | features: {X.shape[1]} | duplicate rows removed: {duplicate_rows:,}")
    print(f"       Benign: {(y == 0).sum():,} | attack: {(y == 1).sum():,}")
    return X, y


def build_models(seed: int) -> dict[str, object]:
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=20, min_samples_leaf=2,
                                   class_weight="balanced", random_state=seed),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=200, max_features="sqrt", min_samples_leaf=2,
                                   class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        ),
        "XGBoost": make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          n_jobs=-1, random_state=seed, eval_metric="logloss"),
        ),
    }


def evaluate_models(
    models: dict[str, object], X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: np.ndarray, y_test: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        print(f"[3/5] Training {name}...")
        started = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - started

        started = time.perf_counter()
        predictions = model.predict(X_test)
        predict_seconds = time.perf_counter() - started
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        try:
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        except (AttributeError, ValueError):
            auc = float("nan")

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision (attack)": precision_score(y_test, predictions, zero_division=0),
            "Recall / Detection Rate (attack)": recall_score(y_test, predictions, zero_division=0),
            "F1 (attack)": f1_score(y_test, predictions, zero_division=0),
            "ROC AUC": auc,
            "False Alarm Rate": fp / (fp + tn) if fp + tn else 0.0,
            "True Negatives": int(tn), "False Positives": int(fp),
            "False Negatives": int(fn), "True Positives": int(tp),
            "Training Time (s)": train_seconds,
            "Prediction Time (ms/sample)": predict_seconds * 1000 / len(y_test),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.sample_size < 0:
        raise ValueError("--sample-size must be 0 (all rows) or a positive integer.")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    dataset = select_dataset()
    X, y = load_data(dataset, args.sample_size, args.seed)
    print("[2/5] Creating stratified train/test split (preprocessing is fit on training data only)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"       Train rows: {len(y_train):,} | test rows: {len(y_test):,}")

    results = evaluate_models(build_models(args.seed), X_train, X_test, y_train, y_test)
    print("[4/5] Results (higher is better except false alarm rate and time):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False, float_format="%.6f")
    print(f"[5/5] Saved {len(results)} model results to {args.output}")
    print("       Note: random row splits can overstate performance on traffic from the same capture/day.")

    # Open an interactive dashboard in the browser after every benchmark run.
    chart = px.bar(
        results.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision (attack)",
                        "Recall / Detection Rate (attack)", "F1 (attack)", "ROC AUC"],
            var_name="Metric",
            value_name="Score",
        ),
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title=f"Interactive model performance — {dataset.name}",
        hover_data={"Score": ":.5f"},
        color_discrete_sequence=CHART_COLORS,
    )
    chart.update_yaxes(range=[0.97, 1.005], title="Score (zoomed to show differences)")
    chart.update_layout(
        template="plotly_white",
        legend_title="Metric",
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
        font=dict(family="Arial, sans-serif", size=13, color="#1F2937"),
        title_font=dict(size=20, color="#111827"),
        margin=dict(l=80, r=40, t=90, b=100),
    )
    chart_path = args.output.with_name("benchmark_dashboard.html").resolve()
    chart.write_html(chart_path, auto_open=True)
    print(f"       Interactive chart opened; saved to {chart_path}")


if __name__ == "__main__":
    main()
