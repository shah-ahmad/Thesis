# CIC-IDS Intrusion Detection Benchmark

This project compares Logistic Regression, Decision Tree, Random Forest, and XGBoost for binary intrusion detection. It combines the CIC-IDS2017 daily flow CSVs and reports validation and final test results for each classifier.

## Evaluation design

The benchmark keeps capture days separate:

| Partition | CIC-IDS2017 files | Purpose |
| --- | --- | --- |
| Train | Monday, Tuesday, Wednesday | Fit preprocessing and model parameters |
| Validation | Both Thursday files | Compare models and make development decisions |
| Final test | Friday morning, DDoS, and PortScan files | Final evaluation on capture days excluded from training |

Monday contains benign traffic only. The training partition also includes Tuesday and Wednesday attack traffic, so it contains both benign and attack classes. Thursday validation and Friday test are checked to contain both classes before the benchmark continues.

Keeping whole capture days together is more demanding than a random row split: closely related flows from one day cannot be divided between train and test. Validation results can be used during development; reserve the Friday test results for the final report and do not tune models against them. The Friday test measures performance on those held-out CIC-IDS2017 days, not a guarantee of performance on modern networks or future attacks.

All non-empty labels other than `BENIGN` are mapped to the binary attack class. Identifier fields such as IP addresses, ports, flow IDs, and timestamps are excluded. Numeric features are selected using training data only. Missing values are imputed inside each model pipeline, fit on training data only.

## Requirements and install

- Python 3.9 or newer
- CIC-IDS2017 daily CSV files in `data/`

From the project folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn xgboost
```

Plotly is optional and is only needed for the interactive browser chart:

```powershell
python -m pip install plotly
```

Matplotlib is optional and is only needed for the static PNG chart script:

```powershell
python -m pip install matplotlib
```

## Data files and split configuration

Place the CIC-IDS2017 CSV files in `data/` using the standard filenames. The train, validation, and test filename lists are configured near the top of `experiment.py` as `TRAIN_FILES`, `VALIDATION_FILES`, and `TEST_FILES`. Keep a complete capture day in only one list. The script stops with an error if any configured file is missing.

The default caps each CSV at 25,000 sampled rows to keep the run practical. Sampling happens separately inside each CSV and preserves its benign/attack proportions. To use every row, set `SAMPLE_SIZE_PER_FILE = 0` near the top of `experiment.py`. Full-data runs can require substantial memory and time.

## Run

From the repository root:

```powershell
python .\experiment.py
```

Override the per-file sample cap or seed if needed:

```powershell
python .\experiment.py --sample-size 50000 --seed 42 --output .\results\day_split_results.csv
```

The script trains each model on Monday–Wednesday, scores it on Thursday validation and Friday final test, prints a table, and saves one row per model per split to `thesis_benchmark_results.csv` by default. A majority-class baseline is included for comparison. It also saves `attack_family_test_results.csv`, with each Friday attack family's training support, test support, detections, misses, and recall.

## Reading the results

- **Validation** scores are for model comparison during development.
- **Test** scores are the final held-out evaluation; avoid using them to choose settings.
- **Attack recall / detection rate** measures the fraction of attacks found.
- **Attack precision** measures how many attack alerts are correct.
- **F1 (attack)** balances attack precision and recall.
- **F1 (macro)** gives benign and attack classes equal weight.
- **False alarm rate** is the share of benign flows incorrectly flagged.
- **Confusion counts** show true negatives, false positives, false negatives, and true positives.
- **Majority-class baseline** predicts the most common training label every time; compare model performance with it, especially when classes are imbalanced.
- **Training and prediction time** depend on the computer and sample size.

The day-based result may be lower than the earlier random row-split result. That is useful evidence about how well the model transfers across capture days. Scores are also affected by the specific CIC-IDS2017 files and label quality.

For H3, compare `Train Support` with per-family test recall, but interpret this carefully: attack families with zero training examples are unseen families. Their detection performance measures generalization to novel attacks, not the isolated effect of class imbalance. Limit an imbalance claim to families represented in training and evaluated under a design that controls for attack-family novelty.

## Charts

If Plotly is installed, the benchmark automatically opens an interactive chart comparing validation and test performance metrics and saves `benchmark_dashboard.html` beside the results CSV. Every metric has its own distinct color. False alarm rate is in a separate chart, clearly marked lower-is-better, saved as `false_alarm_dashboard.html`. Per-family test recall is saved as `attack_family_dashboard.html`. Metrics and CSV results still save when Plotly is not installed; only the interactive charts are skipped.

Customize the interactive chart size and palettes by editing `CHART_WIDTH`, `CHART_HEIGHT`, `METRIC_COLORS`, and `MODEL_COLORS` near the top of `experiment.py`.

To create static PNG charts from a completed results CSV (requires Matplotlib):

```powershell
python .\plot_results.py
```

The charts are saved in `visuals/` and separate validation scores from final test scores. Performance metrics use distinct colors; false alarm rate has its own figure and is labeled lower-is-better. If `attack_family_test_results.csv` is present, the script also creates `attack_family_recall.png`.

## Reproducibility and limitations

The default random seed is 42. Record the dataset version, configured filenames, per-file sample cap, Python and package versions, and machine used. CIC-IDS2017 is a dated lab dataset, so even a day-held-out test does not establish performance on live or current network traffic. Binary grouping also hides which attack families are detected or missed; a later multiclass experiment could report per-attack results.
