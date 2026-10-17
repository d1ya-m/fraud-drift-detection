# Implementation Pipeline — Concept-Drift-Aware Credit-Card Fraud Detection

A single, follow-along guide: create the repo, get the data, set up the environment, build the pipeline in the right order, run it, test it, and push. Pure classical ML (scikit-learn, River, imbalanced-learn) — no deep learning.

The guiding principle: **the pipeline runs on a built-in synthetic stream from day one**, so you can build and test every module before downloading anything. Real datasets (ULB, then IEEE-CIS) slot in behind the same loader interface later, changing nothing else.

---

## 0. Prerequisites

Install these on your machine first:
- **Git** and a **GitHub account**
- **GitHub CLI** (`gh`) — optional but makes repo creation one command ([cli.github.com](https://cli.github.com))
- **Python 3.11+**
- A **Kaggle account** (the datasets live on Kaggle)

---

## 1. Create the GitHub repository

Pick one person as the "setup owner" to run this once.

**With the GitHub CLI:**
```bash
gh auth login                       # one-time, follow the prompts
gh repo create fraud-drift-detection --private --clone \
  --description "Concept-drift-aware credit-card fraud detection using online classical ML"
cd fraud-drift-detection
```

**Or without the CLI:** create the repo on github.com (New → name it `fraud-drift-detection` → Private → Create), then:
```bash
git clone https://github.com/<your-username>/fraud-drift-detection.git
cd fraud-drift-detection
```

---

## 2. Scaffold the folder structure

```bash
mkdir -p src/data src/models src/drift src/evaluation src/imbalance src/viz \
         experiments notebooks results/figures tests docs data/raw .github/workflows
# make the src folders importable Python packages
touch src/__init__.py src/data/__init__.py src/models/__init__.py \
      src/drift/__init__.py src/evaluation/__init__.py \
      src/imbalance/__init__.py src/viz/__init__.py
# keep otherwise-empty folders in git
touch data/raw/.gitkeep results/figures/.gitkeep
```

Final layout:
```
fraud-drift-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                 # CSVs live here (git-ignored)
│   └── README.md            # how to download the data
├── src/
│   ├── data/      loader.py, stream.py, synthetic.py
│   ├── models/    base.py, baselines.py, retrain.py, online.py
│   ├── drift/     detectors.py
│   ├── evaluation/metrics.py, prequential.py
│   ├── imbalance/ samplers.py
│   └── viz/       plots.py
├── experiments/   run_experiment.py
├── notebooks/     01_eda.ipynb
├── results/       figures/ + metric CSVs (CSVs git-ignored)
├── tests/         test_data.py, test_metrics.py, test_prequential.py
└── docs/          synopsis, report, this file
```

---

## 3. Create the base files

**`.gitignore`**
```gitignore
__pycache__/
*.py[cod]
.venv/
.ipynb_checkpoints/
data/raw/*
!data/raw/.gitkeep
results/*.csv
results/figures/*
!results/figures/.gitkeep
.DS_Store
.vscode/
.idea/
```

**`requirements.txt`**
```
numpy
pandas
scikit-learn
imbalanced-learn
river
xgboost
lightgbm
shap
matplotlib
seaborn
jupyter
pytest
```

**`README.md`**
```markdown
# Concept-Drift-Aware Credit-Card Fraud Detection
Classical (non-deep) ML for detecting fraudulent card transactions that keeps
up as fraud patterns drift over time.

## Setup
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Run (works immediately on synthetic data, no download needed)
python experiments/run_experiment.py

## Data
See data/README.md to download the real datasets into data/raw/.
```

---

## 4. Get the data

### Why synthetic first
You do **not** need any download to start. `src/data/synthetic.py` generates a
time-ordered, imbalanced stream whose fraud pattern *changes halfway through*, so
the whole pipeline runs end-to-end on day one. Build and debug everything against
it; only then bring in the real CSVs.

### Dataset A — ULB Credit Card Fraud (develop + clean sanity check)
- **Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Size:** 284,807 transactions, 492 fraud (0.172%), 30 columns (V1–V28 PCA + Time + Amount + Class), ~150 MB.
- **Role:** small and clean — perfect for getting the pipeline correct and for the *controlled* (synthetic-drift) experiment.
- **Download:** on the Kaggle page click **Download**, unzip, and place `creditcard.csv` in `data/raw/`.
- **Or via Kaggle CLI:**
  ```bash
  pip install kaggle
  # put your kaggle.json API token in ~/.kaggle/ (Kaggle → Account → Create New Token)
  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
  ```

### Dataset B — IEEE-CIS Fraud Detection (the "real" temporal experiment)
- **Link:** https://www.kaggle.com/competitions/ieee-fraud-detection/data
- **Size:** 590,540 transactions, ~3.5% fraud, ~390 real (non-PCA) features including categoricals, with a real `TransactionDT` timestamp over a long window.
- **Role:** where the concept-drift story becomes *natural* rather than injected, and where SHAP gives interpretable explanations. Use `train_transaction.csv` (optionally merged with `train_identity.csv` on `TransactionID`).
- **Download (CLI, accept the competition rules on the page first):**
  ```bash
  kaggle competitions download -c ieee-fraud-detection -p data/raw --unzip
  ```
- **Heads-up for the loader:** it has missing values and categorical columns, so its loader needs median-impute for numeric, a fill-and-encode step for categoricals, and it should sort by `TransactionDT` and rename `isFraud`→`Class`, `TransactionDT`→`Time`.

**Important:** the two datasets have different feature spaces, so you never train on one and test on the other. You *develop* on ULB/synthetic and *re-run the same pipeline* on IEEE-CIS. The real experiment is always **train on earlier time, test on later time within one dataset.**

---

## 5. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 6. The architecture — four contracts

Everything connects through four small interfaces. Agree on these first; once they exist, each module can be built and tested in isolation against the others' stubs.

**Contract 1 — the data stream** (`src/data/stream.py`)
```python
FEATURES = [all columns except "Time" and "Class"]   # inferred, so ULB & synthetic both work
TARGET   = "Class"                                    # 1 = fraud, 0 = legitimate

def load_dataset(path) -> DataFrame        # load, validate, sort by Time
def make_stream(df, batch_size=2000)       # yield (X_batch, y_batch) in time order; never shuffle
```

**Contract 2 — every model looks the same** (`src/models/base.py`)
```python
class BaseDetector:
    name: str
    def fit(X, y)            # batch training / warm-up
    def partial_fit(X, y)    # per-batch update (no-op for a frozen model)
    def predict_proba(X)     # P(fraud) in [0,1], shape (n,)
    def refit(X, y)          # adapt after a detected drift (default: full re-fit)
```

**Contract 3 — metrics and the evaluation loop** (`src/evaluation/`)
```python
def compute_metrics(y_true, y_score, precision_target=0.90) -> dict
    # {pr_auc, roc_auc, recall_at_p, fpr, threshold}
def prequential_eval(model, stream, drift_detector=None) -> DataFrame
    # test-then-train down the stream; one row per batch:
    # [batch_idx, n, pr_auc, recall_at_p, fpr, drift_flag]
```

**Contract 4 — the results format.** Every experiment writes a tidy CSV to `results/` with a `model_name` column plus the per-batch metric columns above. The plots only read these CSVs, so visualization can be built against a fake results file before any model exists.

---

## 7. Build order (the detailed pipeline)

Build in this sequence. Each stage is runnable and testable before the next, and the whole thing works on synthetic data throughout.

**Stage 1 — Data layer.**
`synthetic.py` (drifting stream generator) → `loader.py` (`load_dataset`, `get_feature_columns`) → `stream.py` (`make_stream`). Milestone: you can generate a synthetic DataFrame, confirm it is time-sorted and imbalanced, and iterate `make_stream` to get `(X, y)` batches.

**Stage 2 — Metrics + the prequential harness.**
`evaluation/metrics.py` (PR-AUC, recall-at-fixed-precision, FPR, chosen threshold; returns NaNs cleanly for a batch with no fraud) → `evaluation/prequential.py` (warm up the model on the first batch, then for each later batch: predict → score → update). Milestone: feed it a trivial constant-score "model" and get a results DataFrame with one row per batch.

**Stage 3 — The three model families (all behind `BaseDetector`).**
`models/baselines.py` — **static** logistic regression, random forest, gradient boosting (fit once at warm-up; `partial_fit` is a no-op; use `class_weight='balanced'` where supported). → `models/retrain.py` — **periodic-retrain** wrapper that buffers recent batches and refits every *N* batches. → `models/online.py` — **online** model via River (an incremental logistic pipeline or `forest.ARFClassifier`), updating every batch. Milestone: each model runs through `prequential_eval` and produces a sane PR-AUC-over-time curve.

**Stage 4 — Drift detection.**
`drift/detectors.py` — wrap River's **ADWIN** and **DDM** (`.update(error) → .drift_detected`) behind a tiny uniform class, plus a no-op fallback. Wire into `prequential_eval`: feed the per-batch error signal to the detector; when it fires, set `drift_flag=1` and call `model.refit(current_batch)`. Milestone: on the synthetic data, drift flags cluster around the midpoint where the pattern changes.

**Stage 5 — Imbalance handling.**
`imbalance/samplers.py` — a class-weight helper and a SMOTE wrapper (`imbalanced-learn`). Milestone: you can run each model with and without resampling and compare.

**Stage 6 — Visualization + the experiment runner.**
`viz/plots.py` — plot a chosen metric over time, one line per model, saved to `results/figures/`. → `experiments/run_experiment.py` — the entry point: load data (synthetic by default, `--data data/raw/creditcard.csv` for ULB), build the model set, run each through the harness, write `results/*.csv`, and render the comparison plot. Milestone: one command produces the headline figure — the frozen model's recall sliding down over time versus the adaptive models holding steadier and recovering after drift.

**Stage 7 — Explainability (optional, strong for the report).**
Use **SHAP** on a tree model (most meaningful on IEEE-CIS's real features) to show which features drive fraud predictions and whether their importance shifts after drift.

---

## 8. How to run

```bash
# Runs immediately on synthetic data — no download required
python experiments/run_experiment.py

# On the real ULB data once creditcard.csv is in data/raw/
python experiments/run_experiment.py --data data/raw/creditcard.csv --batch-size 2000

# Later, on IEEE-CIS
python experiments/run_experiment.py --data data/raw/train_transaction.csv --batch-size 5000
```
Output: per-model CSVs in `results/` and the comparison figure in `results/figures/`.

---

## 9. Testing

Keep a few `pytest` tests so nothing silently breaks:
- `tests/test_data.py` — synthetic df has the required columns, is time-sorted, fraud rate ≈ target.
- `tests/test_metrics.py` — perfectly separable scores give PR-AUC ≈ 1; an all-legitimate batch returns the metric keys without crashing.
- `tests/test_prequential.py` — a static model on small synthetic data returns a non-empty DataFrame with the expected columns.

```bash
pytest -q
```

---

## 10. Git workflow and pushing

Keep `main` always working; do each piece on its own short branch and merge via pull request.

```bash
# first push of the scaffold
git add -A
git commit -m "chore: initial project scaffold, pipeline doc, and contracts"
git push -u origin main

# each new piece of work
git checkout -b feat/data-stream
# ...build Stage 1, commit in small steps...
git add -A && git commit -m "feat: synthetic stream, loader, and batch stream"
git push -u origin feat/data-stream
gh pr create --fill --base main          # open a PR; a teammate reviews and merges
```

Conventional commit prefixes keep history readable and make the report's contribution section easy: `feat:`, `fix:`, `test:`, `docs:`, `chore:`.

For each teammate, after the scaffold is on `main`:
```bash
git clone https://github.com/<owner>/fraud-drift-detection.git
cd fraud-drift-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                 # should pass
git checkout -b feat/<your-piece>
```

---

## 11. How this maps to the synopsis

- **Baselines + imbalance-aware metrics** (Stages 2–3) → Objective 1.
- **Prequential evaluation across static / periodic / online** (Stages 2–3, 6) → Objective 2.
- **ADWIN/DDM drift detection + recovery** (Stage 4) → Objective 3.
- **ULB develop + IEEE-CIS real run** (Section 4) → Data Source, and the rigor point that your method is validated on two datasets (controlled synthetic drift + natural temporal drift).

Start at Stage 1 on synthetic data, get a green `pytest`, push the scaffold, and build up the stages one branch at a time.
