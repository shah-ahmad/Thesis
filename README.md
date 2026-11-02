# Thesis: Network Intrusion Detection Benchmark

This project benchmarks four machine learning classifiers for binary network intrusion detection using the CIC-IDS-2017 flow data:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The script cleans the selected CSV, maps `BENIGN` to 0 and all other labels to 1, samples up to 100,000 rows, creates a stratified 80/20 train/test split, scales the features, and reports accuracy, precision, recall, F1 score, false alarm rate, training time, and inference latency.

## Requirements

- Python 3.9 or newer
- pip
- CIC-IDS-2017 CSV dataset files

## Setup

Open a terminal in the repository folder and create/activate a virtual environment (recommended):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn xgboost
```

## Add the dataset

Place one or more CIC-IDS-2017 CSV files in the `data` folder. For example:

```text
data\Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv
```

The script reads the first CSV returned from `data/*.csv` (or, if none are in `data`, from the project root). If you need a specific dataset file, keep only that CSV in `data` before running; file selection is not currently configurable from the command line.

## Run

From the repository root:

```powershell
python .\experiment.py
```

The benchmark table is printed in the terminal and saved to `thesis_benchmark_results.csv` in the current directory. Each run overwrites that results file.

## Notes

- The project folder should contain `experiment.py`, `data/`, and this README.
- The script uses a fixed random seed (`42`) for sampling and splitting.
- A full run can take a while, especially with the larger CSV files and ensemble models.
- Results depend on which CSV is selected and the machine running the benchmark.
