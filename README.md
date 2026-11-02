# CIC-IDS Intrusion Detection Benchmark

This project compares four classifiers for the binary task of distinguishing benign network flows from attack flows in CIC-IDS-style CSV data:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The script reports attack precision, attack recall (detection rate), attack F1, ROC AUC, false alarm rate, confusion-matrix counts, training time, and prediction latency. It writes one row per model to a results CSV.

## Requirements

- Python 3.9 or newer
- CIC-IDS CSV files with a `Label` column and numeric flow features

## Install

Run these commands from the repository folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn xgboost
```

If PowerShell blocks activation, use `\.venv\Scripts\python.exe` in place of `python` in the commands below.

## Choose the data

Put the CIC-IDS CSV file or files in `data/`. If there is exactly one CSV in `data/`, the script selects it automatically. If there are multiple files, pass the exact one you want with `--data`:

```powershell
python .\experiment.py --data .\data\Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv
```

The program refuses to guess when it finds multiple CSVs. That avoids accidentally benchmarking a different capture than intended. The `Label` column is required. `BENIGN` becomes class 0; every other non-empty label is treated as attack (class 1). Identifiers such as IP addresses, ports, flow IDs, and timestamps are excluded from the feature set. Numeric features are cleaned, and missing values are imputed within each model using training data only.

## Run

Quick run using a maximum of 100,000 rows:

```powershell
python .\experiment.py --data .\data\your-dataset.csv
```

Use the entire selected CSV (can take substantially longer and use more memory):

```powershell
python .\experiment.py --data .\data\your-dataset.csv --sample-size 0
```

Change the sample size, test fraction, random seed, or output path:

```powershell
python .\experiment.py --data .\data\your-dataset.csv --sample-size 200000 --test-size 0.25 --seed 42 --output .\results\benchmark.csv
```

The sample is drawn approximately in proportion to the benign/attack classes. The train/test split is stratified to preserve class proportions. Scaling for Logistic Regression and missing-value imputation are fitted on training data inside each model pipeline, which prevents information from the test set leaking into training.

## Understanding the metrics

- **Precision (attack):** Of flows predicted as attacks, the fraction that are attacks.
- **Recall / Detection Rate (attack):** Of all attacks, the fraction detected.
- **F1 (attack):** Harmonic mean of attack precision and recall.
- **ROC AUC:** How well the model ranks attacks above benign flows across thresholds.
- **False Alarm Rate:** False positives divided by all truly benign flows.
- **Confusion counts:** True negatives, false positives, false negatives, and true positives.
- **Prediction time:** Average prediction time per test sample on the current machine; it is hardware-dependent.

There is no single best score for every use case. High attack recall can matter when missed attacks are costly, while a low false alarm rate can matter when analysts have limited time. Read these together and choose based on the thesis question.

## Results and reproducibility

By default, results are saved to `thesis_benchmark_results.csv` in the current directory; an existing file at that path is overwritten. The fixed default seed is 42. The output includes the selected metrics and timing for all four models.

The default stratified random row split is a useful baseline, but CIC-IDS contains correlated flows from the same capture periods. A random split can put very similar traffic in both train and test sets and overstate performance on genuinely new days, networks, or attack campaigns. For a thesis claim about generalization, add a time-based or capture/day-level evaluation with an untouched test period. Also document the exact CSV, dataset version, sample size, software versions, and machine for repeatable comparisons.

## Limitations

- All non-`BENIGN` labels are combined into one attack class, so this is binary detection rather than attack-family classification.
- Metrics depend on the chosen CSV, sample, split, and hardware.
- Sampling changes the dataset composition; use `--sample-size 0` when feasible for full-data results.
- Benchmark results are experimental and should be validated with a split that matches the intended deployment setting.
