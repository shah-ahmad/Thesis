"""Day-held-out CIC-IDS benchmark with train, validation, and final test results."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET_COLUMN = "Label"
DATA_FOLDER = Path("data")

# Keep whole capture days together so related records do not get randomly split
# across training, validation, and test partitions.
TRAIN_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]
VALIDATION_FILES = [
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]
TEST_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

# Maximum sampled rows from each input CSV. Set to 0 to use every row.
SAMPLE_SIZE_PER_FILE = 25_000

# Interactive dashboard appearance (dimensions are in pixels).
CHART_WIDTH = 1500
CHART_HEIGHT = 780
METRIC_COLORS = {
    "Accuracy": "#2563EB",
    "Balanced Accuracy": "#16A34A",
    "Precision (attack)": "#F59E0B",
    "Recall / Detection Rate (attack)": "#DC2626",
    "F1 (attack)": "#7C3AED",
    "F1 (macro)": "#0891B2",
    "ROC AUC": "#64748B",
}
MODEL_COLORS = {
    "Logistic Regression": "#2563EB",
    "Decision Tree": "#16A34A",
    "Random Forest": "#F59E0B",
    "XGBoost": "#7C3AED",
}

IDENTIFIER_COLUMNS = {
    "flow id", "source ip", "destination ip", "source port",
    "destination port", "timestamp", "src ip", "dst ip",
}
METRIC_COLUMNS = [
    "Accuracy", "Balanced Accuracy", "Precision (attack)",
    "Recall / Detection Rate (attack)", "F1 (attack)", "F1 (macro)",
    "ROC AUC", "False Alarm Rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare binary CIC-IDS classifiers using separate capture days."
    )
    parser.add_argument(
        "--sample-size", type=int, default=SAMPLE_SIZE_PER_FILE,
        help="Maximum rows sampled from each CSV; use 0 for all rows.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Sampling seed (default: 42).")
    parser.add_argument(
        "--output", type=Path, default=Path("thesis_benchmark_results.csv"),
        help="Path for the validation/test results CSV.",
    )
    return parser.parse_args()


def sample_frame(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size == 0 or len(frame) <= sample_size:
        return frame
    labels = frame[TARGET_COLUMN].astype(str).str.strip().str.casefold()
    classes = labels.unique()
    if sample_size < len(classes):
        raise ValueError("Sample size must be at least the number of labels in each file.")

    counts = labels.value_counts()
    allocation = (counts / len(frame) * sample_size).round().astype(int).clip(lower=1)
    while allocation.sum() > sample_size:
        largest = allocation.idxmax()
        if allocation[largest] <= 1:
            break
        allocation[largest] -= 1
    while allocation.sum() < sample_size:
        allocation[counts.idxmax()] += 1

    pieces = [
        frame.loc[labels == label].sample(n=count, random_state=seed)
        for label, count in allocation.items()
    ]
    return pd.concat(pieces).sample(frac=1, random_state=seed)


def load_partition(file_names: list[str], name: str, sample_size: int, seed: int) -> pd.DataFrame:
    frames = []
    print(f"Loading {name} files:")
    for file_name in file_names:
        path = DATA_FOLDER / file_name
        if not path.is_file():
            raise FileNotFoundError(f"Required {name} CSV not found: {path}")

        frame = pd.read_csv(path, low_memory=False)
        frame.columns = frame.columns.astype(str).str.strip()
        if TARGET_COLUMN not in frame.columns:
            raise ValueError(f"Required '{TARGET_COLUMN}' column is missing from {path}.")

        labels = frame[TARGET_COLUMN]
        valid = labels.notna() & labels.astype(str).str.strip().ne("")
        frame = frame.loc[valid].drop_duplicates()
        frame = sample_frame(frame, sample_size, seed)
        frames.append(frame)
        label_text = frame[TARGET_COLUMN].astype(str).str.strip().str.casefold()
        attack_rows = int(label_text.ne("benign").sum())
        print(f"  {file_name}: {len(frame):,} rows ({attack_rows:,} attack)")
        del frame
        gc.collect()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    del frames
    gc.collect()
    return combined


def prepare_features(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Clean features and select columns using training data only."""
    partitions = []
    labels = []
    for frame in (train_frame, validation_frame, test_frame):
        label_text = frame[TARGET_COLUMN].astype(str).str.strip().str.casefold()
        y = (label_text != "benign").astype("int8").to_numpy()
        drop_columns = [
            column for column in frame.columns
            if column != TARGET_COLUMN and column.strip().casefold() in IDENTIFIER_COLUMNS
        ]
        X = frame.drop(columns=[TARGET_COLUMN, *drop_columns]).copy()
        X.columns = [str(column).strip() for column in X.columns]
        X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        partitions.append(X)
        labels.append(y)

    X_train, X_validation, X_test = partitions
    usable_columns = [
        column for column in X_train.columns
        if X_train[column].notna().any() and X_train[column].nunique(dropna=True) > 1
    ]
    if not usable_columns:
        raise ValueError("No usable numeric features remain in the training partition.")
    X_train = X_train[usable_columns]
    X_validation = X_validation.reindex(columns=usable_columns)
    X_test = X_test.reindex(columns=usable_columns)

    for split_name, y in zip(("train", "validation", "test"), labels):
        counts = np.bincount(y, minlength=2)
        print(f"{split_name.title():12} rows: {len(y):,} | benign: {counts[0]:,} | attack: {counts[1]:,}")
        if len(np.unique(y)) < 2:
            raise ValueError(f"The {split_name} partition must contain benign and attack rows.")
    return X_train, labels[0], X_validation, labels[1], X_test, labels[2]


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


def score_partition(
    y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_true, predictions),
        "Precision (attack)": precision_score(y_true, predictions, zero_division=0),
        "Recall / Detection Rate (attack)": recall_score(y_true, predictions, zero_division=0),
        "F1 (attack)": f1_score(y_true, predictions, zero_division=0),
        "F1 (macro)": f1_score(y_true, predictions, average="macro", zero_division=0),
        "ROC AUC": roc_auc_score(y_true, probabilities),
        "False Alarm Rate": fp / (fp + tn) if fp + tn else 0.0,
        "True Negatives": int(tn), "False Positives": int(fp),
        "False Negatives": int(fn), "True Positives": int(tp),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for name, model in build_models(seed).items():
        print(f"Training {name} on train days...")
        started = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - started

        for split_name, X_split, y_split in (
            ("Validation", X_validation, y_validation),
            ("Test", X_test, y_test),
        ):
            started = time.perf_counter()
            prediction = model.predict(X_split)
            probabilities = model.predict_proba(X_split)[:, 1]
            predict_seconds = time.perf_counter() - started
            row = {
                "Model": name,
                "Split": split_name,
                **score_partition(y_split, prediction, probabilities),
                "Training Time (s)": train_seconds,
                "Prediction Time (ms/sample)": predict_seconds * 1000 / len(y_split),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def make_dashboard(results: pd.DataFrame, output_path: Path) -> None:
    try:
        import plotly.express as px
    except ImportError:
        print("Plotly is not installed; metrics were saved, but no interactive chart was created.")
        print("Install it with: python -m pip install plotly")
        return

    performance_metrics = [metric for metric in METRIC_COLUMNS if metric != "False Alarm Rate"]
    chart_data = results.melt(
        id_vars=["Model", "Split"], value_vars=performance_metrics,
        var_name="Metric", value_name="Score",
    )
    chart = px.bar(
        chart_data, x="Model", y="Score", color="Metric", facet_col="Split",
        barmode="group", title="Day-held-out CIC-IDS model comparison",
        hover_data={"Score": ":.5f"}, color_discrete_map=METRIC_COLORS,
    )
    chart.update_yaxes(range=[0.0, 1.02], title="Score")
    chart.update_layout(
        template="plotly_white", legend_title="Metric", width=CHART_WIDTH,
        height=CHART_HEIGHT, font=dict(family="Arial, sans-serif", size=13, color="#1F2937"),
        title_font=dict(size=20, color="#111827"), margin=dict(l=70, r=40, t=90, b=120),
    )
    dashboard_path = output_path.with_name("benchmark_dashboard.html").resolve()
    chart.write_html(dashboard_path, auto_open=True)
    print(f"Interactive validation/test chart saved to {dashboard_path}")

    # False alarm rate is a cost metric, so show it separately with an explicit lower-is-better axis.
    far_chart = px.bar(
        results, x="Model", y="False Alarm Rate", color="Model", facet_col="Split",
        title="False alarm rate by held-out partition (lower is better)",
        hover_data={"False Alarm Rate": ":.5%"}, color_discrete_map=MODEL_COLORS,
    )
    far_max = max(float(results["False Alarm Rate"].max()) * 1.2, 0.01)
    far_chart.update_yaxes(range=[0, far_max], tickformat=".1%", title="False alarm rate (lower is better)")
    far_chart.update_layout(
        template="plotly_white", width=CHART_WIDTH, height=560, showlegend=False,
        font=dict(family="Arial, sans-serif", size=13, color="#1F2937"),
        title_font=dict(size=20, color="#111827"), margin=dict(l=75, r=40, t=90, b=100),
    )
    far_path = output_path.with_name("false_alarm_dashboard.html").resolve()
    far_chart.write_html(far_path, auto_open=False)
    print(f"False alarm chart saved to {far_path}")


def main() -> None:
    args = parse_args()
    if args.sample_size < 0:
        raise ValueError("--sample-size must be 0 (all rows) or a positive integer.")

    print("Split protocol: train = Monday-Wednesday, validation = Thursday, final test = Friday")
    train_frame = load_partition(TRAIN_FILES, "training", args.sample_size, args.seed)
    validation_frame = load_partition(VALIDATION_FILES, "validation", args.sample_size, args.seed)
    test_frame = load_partition(TEST_FILES, "test", args.sample_size, args.seed)
    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_features(
        train_frame, validation_frame, test_frame
    )
    del train_frame, validation_frame, test_frame
    gc.collect()

    results = evaluate_models(
        X_train, y_train, X_validation, y_validation, X_test, y_test, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False, float_format="%.6f")
    print("\nValidation and final test results (higher is better except FAR and time):")
    display_columns = ["Model", "Split", *METRIC_COLUMNS, "Training Time (s)"]
    print(results[display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved results to {args.output.resolve()}")
    print("The Friday test partition was held out by capture day and was not used to fit preprocessing or models.")
    make_dashboard(results, args.output)


if __name__ == "__main__":
    main()
