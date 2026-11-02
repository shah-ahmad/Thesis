import os
import glob
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ==========================================
# 1. PURE NUMPY METRIC CALCULATIONS
# (Bypasses Windows DLL Application Control blocks)
# ==========================================
def calculate_metrics(y_true, y_pred):
    """
    Computes standard cybersecurity intrusion detection metrics
    directly using NumPy without relying on blocked C-extensions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Alarm Rate

    return acc, prec, rec, f1, far


# ==========================================
# 2. DATA PREPROCESSING PIPELINE
# ==========================================
def load_and_preprocess_data(csv_filepath: str, sample_size: int = 100000):
    print(f"[*] Loading data from: {csv_filepath}")
    df = pd.read_csv(csv_filepath, low_memory=False)

    # 1. Fix CIC-IDS-2017 leading/trailing whitespace in column names
    df.columns = df.columns.str.strip()

    # 2. Subsample for prototype speed (remove sample_size or increase for final paper)
    if sample_size and sample_size < len(df):
        print(f"[*] Subsampling to {sample_size:,} records for fast prototyping...")
        df = df.sample(n=sample_size, random_state=42)

    # 3. Clean infinite and missing values
    print("[*] Cleaning infinite and missing values...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 4. Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # 5. Drop socket/metadata features to prevent data leakage
    leakage_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Timestamp']
    existing_leakage_cols = [col for col in leakage_cols if col in df.columns]
    if existing_leakage_cols:
        df.drop(columns=existing_leakage_cols, inplace=True)

    # 6. Separate features (X) and label (y)
    target_col = 'Label'
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # Convert binary label: BENIGN -> 0, ANY ATTACK -> 1
    y = y_raw.apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1).values

    print(f"[+] Cleaned dataset: {X.shape[0]:,} records, {X.shape[1]} features")
    benign_count = np.sum(y == 0)
    attack_count = np.sum(y == 1)
    print(f"[+] Distribution: {benign_count:,} Benign (0), {attack_count:,} Malicious (1)")

    return X, y


# ==========================================
# 3. MODEL BENCHMARKING HARNESS
# ==========================================
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, eval_metric='logloss')
    }

    results = []

    for name, model in models.items():
        print(f"\n--- Training {name} ---")

        # 1. Train and record duration
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        # 2. Predict and measure inference latency (in milliseconds per sample)
        start_pred = time.time()
        y_pred = model.predict(X_test)
        total_pred_time = time.time() - start_pred
        latency_ms = (total_pred_time / len(X_test)) * 1000

        # 3. Compute metrics via NumPy
        acc, prec, rec, f1, far = calculate_metrics(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall (DR)": round(rec, 4),
            "F1-Score": round(f1, 4),
            "False Alarm Rate (FAR)": round(far, 4),
            "Train Time (s)": round(train_time, 2),
            "Inference Latency (ms/sample)": round(latency_ms, 4)
        })

    return pd.DataFrame(results)


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Automatically find the CSV file in 'data' or current directory
    csv_files = glob.glob("data/*.csv") or glob.glob("*.csv")
    
    if not csv_files:
        raise FileNotFoundError("[-] No CSV found! Put your CIC-IDS CSV inside the 'data' folder.")

    DATASET_PATH = csv_files[0]
    print(f"[+] Found dataset file: {DATASET_PATH}")

    # 2. Preprocess
    X, y = load_and_preprocess_data(DATASET_PATH, sample_size=100000)

    # 3. Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Feature Scaling
    print("[*] Normalizing features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Run Benchmark
    results_df = evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # 6. Display & Save Results
    print("\n" + "=" * 65)
    print("           FINAL THESIS BENCHMARK TABLE")
    print("=" * 65)
    print(results_df.to_string(index=False))

    results_df.to_csv("thesis_benchmark_results.csv", index=False)
    print("\n[+] Benchmark saved successfully to 'thesis_benchmark_results.csv'!")