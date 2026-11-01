# Implementation Plan: Project Context and Remaining Work

This is the hand-off guide for the project. Read sections 1–4 to understand **what the project
is, what has been built, and why it was built that way**. Then pick a task from section 7.

[PIPELINE.md](../PIPELINE.md) is the original design document. This file records what
actually happened and what's left.

---

## Contents

1. [The project in five minutes](#1-the-project-in-five-minutes)
2. [What has been done, stage by stage](#2-what-has-been-done-stage-by-stage)
3. [Design decisions: don't change these without discussion](#3-design-decisions-dont-change-these-without-discussion)
4. [Glossary](#4-glossary)
5. [Getting set up](#5-getting-set-up)
6. [How the code fits together](#6-how-the-code-fits-together)
7. [Remaining tasks](#7-remaining-tasks)
   - [Task A: SHAP explainability (Stage 7)](#task-a-shap-explainability-stage-7)
   - [Task B: IEEE-CIS natural-drift experiment](#task-b-ieee-cis-natural-drift-experiment)
   - [Task C: EDA notebook](#task-c-eda-notebook)
   - [Task D: Report write-up](#task-d-report-write-up)
   - [Task E (optional): Automatic tests on GitHub](#task-e-optional-automatic-tests-on-github)
   - [Task F (optional): Robustness checks](#task-f-optional-robustness-checks)
8. [Team rules](#8-team-rules)
9. [Suggested split and order](#9-suggested-split-and-order)
10. [Results so far (reference numbers)](#10-results-so-far-reference-numbers)
11. [Known limitations](#11-known-limitations)
12. [Gotchas we already hit](#12-gotchas-we-already-hit)

---

## 1. The project in five minutes

### The problem
We detect **fraudulent credit-card transactions** with classical machine learning
(scikit-learn, River, imbalanced-learn; no deep learning). Two things make this hard:

1. **Fraud is extremely rare.** In the ULB dataset, 492 of 284,807 transactions (0.17%) are
   fraud. A model that says "never fraud" is 99.8% accurate and completely useless, so we
   never use accuracy as a measure.
2. **Fraud patterns change over time** (*concept drift*). Fraudsters change tactics, so a
   model trained on last month's fraud slowly or suddenly stops working.

### The research question
**How do we keep a fraud model accurate as fraud patterns change?** We compare:
- **static** models: trained once, then frozen
- **periodic retraining**: refit on recent data every few batches
- **online** models: learn from every transaction
- **drift-triggered retraining**: a detector (ADWIN or DDM) notices when performance drops
  and triggers a refit
- **our improvement, "accumulate after drift"**: drift-triggered, but the model keeps
  learning from new data for a few batches after the alarm

### The three objectives (from the synopsis, mapped in PIPELINE.md section 11)
1. Baselines with imbalance-aware metrics (PR-AUC, recall at a fixed precision) and
   imbalance handling.
2. A prequential (test-then-train) comparison of static, periodic and online learning over time.
3. Drift detection (ADWIN, DDM) and recovery after drift.

> The synopsis is in the repo root: `MLL_project_synopsis_format.pdf`. Objective 3 explicitly asks for the
> **recovery time** after drift, and the problem statement argues that shuffled test sets hide decay; both are
> now measured (see "Completing the evaluation" in section 2).

### Datasets
| Dataset | Role | Status |
|---|---|---|
| **Synthetic stream** (`src/data/synthetic.py`) | 50,000 transactions, 1% fraud, fraud pattern flips exactly halfway. We know the truth, so it's ideal for building and testing | ✅ used |
| **ULB credit card** ([Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) | Real data, 48 hours, 0.17% fraud, features V1–V28 (anonymised) + Time + Amount. Has no real drift in 48 hours, so we **inject** a known drift | ✅ used |
| **IEEE-CIS** ([Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)) | Real data, ~6 months, 3.5% fraud, ~390 features. Expected to have **natural** drift | ⬜ Task B |

### How a model is evaluated
Transactions are processed **in time order, in batches** (2,000 for synthetic, 10,000 for ULB).
For each batch: the model **predicts first**, we **score** those predictions, and **only then**
does the model learn from that batch, the way a bank would see transactions arrive and labels
come in later. This gives a score **per batch**, so we can watch performance change over time
and see it crash when drift hits. Data is never shuffled.

### The main findings so far
1. Static models **collapse** after drift: PR-AUC drops from ~0.7 to ~0.01–0.23.
2. **ADWIN** fires exactly once at the real drift, and never when there's no drift. **DDM**
   also catches drift but raises false alarms that hurt performance.
3. Our **accumulate-after-drift** retraining is the best strategy in 5 of 6 tests, while
   retraining far less often than periodic retraining.
4. On real data, **imbalance handling matters**: class weights work best for logistic
   regression and random forest; SMOTE works best for gradient boosting after drift.

---

## 2. What has been done, stage by stage

Each stage lists what was built, what it showed, and the decisions made along the way.
The decisions matter: section 3 collects the important ones.

### Stage 1: Data layer ✅
- **Built:** a synthetic drifting stream; a loader that converts any dataset to one format
  (`Time`, `Class`, numeric features; handles ULB and the IEEE-CIS column names, missing
  values and text columns); a batch generator that yields batches in time order.
- **Why synthetic first:** the whole pipeline could be built and tested before downloading
  anything, against data where we *know* exactly when and how the drift happens.
- **Synthetic design:** before the drift, fraud shows up as high V1–V3 and large amounts;
  after it, as low V4–V6 and small amounts ("card testing"). V7+ are pure noise.

### Stage 2: Metrics and the evaluation loop ✅
- **Built:** `compute_metrics` (PR-AUC, ROC-AUC, recall at 90% precision, false-positive rate,
  threshold; returns NaN for a batch with no fraud instead of crashing) and `prequential_eval`
  (the test-then-train loop).
- **Decision:** PR-AUC is the headline metric. It focuses on how well fraud is ranked above
  legitimate transactions and isn't fooled by the 99.8% majority.

### Stage 3: Models ✅
- **Built:** static logistic regression, random forest and gradient boosting (class-weighted);
  `PeriodicRetrain` (refits every 3 batches on the last 6); online logistic regression and an
  adaptive random forest via River.
- **Found:** on synthetic data, static models drop from ~0.7 PR-AUC to ~0.03 after the drift;
  retrained and online models recover to ~0.65–0.85.
- **Decisions:**
  - River models have no class weights, so they use River's `RandomUnderSampler`
    (target: fraud ≈ 20% of what they learn from).
  - `warmup_batches` (default 2 in the experiments) was added because ULB batches hold so
    few frauds that one batch is too little to train on.
  - A static model's refit is **skipped** if a batch contains no fraud (it can't be trained
    on a single class).

### Stage 4: Drift detection ✅
- **Built:** wrappers around River's ADWIN and DDM with one interface, `update(errors) -> bool`;
  the evaluation loop refits the model when it returns True.
- **Found:** on synthetic data, ADWIN flags batch 13 (the drift starts partway through batch 12)
  for every model. DDM flags it too, but also raises false alarms.
- **Decisions:**
  - **Drift signal = "was each fraud missed?"** (1 if a fraud scored below 0.5), fed one value
    per fraud. The first version fed one PR-AUC value per batch, but ADWIN normally checks
    only every 32 values and DDM only accepts 0/1, so they barely worked. ADWIN uses
    `clock=1` (checks after every value).
  - **The detector is reset after every alarm**, so the model's new error rate after a refit
    isn't compared with the old model's.

### Stage 6: Experiment runner and plots ✅
(Built before Stage 5, to see results sooner.)
- **Built:** `experiments/run_experiment.py`, one command that runs every model × strategy,
  writes a CSV per run plus a summary CSV, and draws three figures (PR-AUC over time,
  recall over time, summary ranking).
- **Decisions:** each strategy has a fixed colour in every figure; lines are smoothed with a
  3-batch rolling mean; DDM is drawn thicker *underneath* so identical lines stay visible; a
  panel notes when a detector never fired (its line is then hidden under "base").

### Real data: ULB, then injected drift ✅
- **ULB as-is (no injection):** ADWIN **never fired** for any model, and static random forest
  was best (0.76) with no adaptation at all. DDM's false alarms *lowered* scores.
  **Interpretation:** 48 hours is too short for fraud tactics to change. This is a real
  finding, not a bug, and it's why PIPELINE.md gives ULB the role of the controlled
  experiment.
- **Built:** `src/data/drift_injection.py`, a feature swap. From the halfway point on, the
  4 features most correlated with fraud (V17, V14, V12, V10) swap values with the 4 least
  correlated (V22, V23, V25, V15). The data stays real, but a model that learned "low V14 =
  fraud" now looks in the wrong place. Run with `--inject-drift 0.5`.
- **Found (ULB + injected drift):** frozen models collapse (logistic regression → 0.01),
  ADWIN fires exactly once at the drift for each static model, and drift-triggered
  retraining recovers.
- **Bug caught:** the first version of the swap copied one column onto the other instead of
  exchanging them (a pandas 3 view issue; see section 12). A sanity check caught it before
  any results were used, and a test now guards it.
- **Decision:** ULB runs use **batch size 10,000** (~17 frauds per batch). At 2,000 there
  are ~3 frauds per batch, which makes per-batch scores mostly noise.

### Our improvement: accumulate after drift ✅
- **Problem noticed:** after an alarm, a static model was refit on **one** batch (~17 frauds on
  ULB) and then frozen, so the new model was underfed.
- **Built:** `AccumulateAfterDrift` (in `src/models/drift_retrain.py`). After an alarm, it
  refits on *all batches since the alarm* (1, then 2, then 3… up to 6), then freezes again.
  It costs nothing between drifts.
- **Needed fix:** as the model recovers, its error rate *falls*, and ADWIN flags any change,
  including improvement. That would reset the growing window. The detector now **ignores
  alarms where the error rate went down**.
- **Found:** best strategy in 5 of 6 static-model tests (e.g. ULB random forest 0.75 vs 0.70
  for single-batch refits and 0.65 for periodic retraining), at roughly half the runtime of
  periodic retraining.

### Stage 5: Imbalance handling ✅
- **Built:** `src/imbalance/samplers.py`, offering none / class weights / SMOTE /
  undersampling for any static model, plus `experiments/run_imbalance.py`.
- **Decisions:**
  - Resampling happens **only at training time**; scoring always uses the real transactions.
  - SMOTE runs **after** feature scaling (it measures distances) and adapts to batches with
    very few frauds (plain SMOTE crashes with fewer than 6).
  - Resampling targets 1 fraud per 2 legitimate transactions (`--ratio 0.5`).
  - **Class weights stay the default** (simplest, and best or tied-best for 2 of 3 models).
- **Found:** see section 10. On synthetic data (1% fraud, easy to separate), "no handling" is
  often fine; on real data (0.17%), handling matters a lot.

### Completing the evaluation (step 14) ✅
Reading the synopsis showed two gaps, now closed:
- **Recovery time and degradation** (`src/evaluation/drift_metrics.py`): for each run, PR-AUC before/after the
  drift, the drop (absolute and %), the worst point, batches spent below 90% of the pre-drift level, and
  **recovery time**: batches until the 3-batch rolling PR-AUC is back at 90% of its pre-drift level *after having
  dropped* (a model that looks fine for one batch and then collapses counts as never recovered). Also a trend
  (change per 10 batches) for data without a known drift. These columns are added to every new
  `*_summary.csv`, and the analysis notebook computes them from the per-batch CSVs.
- **Reported vs deployed** (`experiments/run_reported_vs_deployed.py`): tests the synopsis's claim that shuffled
  test sets hide decay. A model trained once on the first 40% of the stream reports a score (random 70/30 split or
  time-ordered split), then is scored on each later 10% slice. With drift: reported 0.67–0.89, deployed after the
  drift 0.004–0.069. Without drift: reported ≈ deployed.
- A first design (random vs time-ordered split of the first 70%) was dropped: that training window already
  contained post-drift data, so it didn't test the claim.
- **Found while checking metrics:** recall at 90% precision is often unreachable per batch (few frauds per batch,
  and logistic regression gives many transactions an identical top score of 1.0), so it reads 0. This is why
  PR-AUC is the headline metric; see the analysis notebook, section 1.2.

### EDA and results analysis (steps 15–16) ✅
- `notebooks/01_eda.ipynb`: data quality, imbalance, amounts (37% of frauds are ≤ 1, "card testing"), time
  (quiet hours hold 10% of transactions but 27% of frauds), feature signal, the injected drift, the batch-size
  choice (at 2,000, 25 batches have no fraud at all), and the synthetic stream.
- `notebooks/02_results_analysis.ipynb`: every result organised by synopsis objective, with what worked, what
  didn't and why. All tables are computed from `results/` CSVs. **Start here for the report.**
- Running notebooks: open them in VS Code and pick the `.venv` interpreter as the kernel. (On Diya's machine,
  command-line Jupyter couldn't reach a kernel from the assistant's sandbox, so the saved outputs were produced
  by executing the cells in-process; the notebooks themselves are ordinary `.ipynb` files.)

### Infrastructure ✅
- GitHub repo: https://github.com/d1ya-m/fraud-drift-detection
- 71 tests (`pytest -q`, about 1 minute; no dataset needed)
- Results CSVs and figures **are committed** (`.gitignore` lines commented out by Sanjay, 2026-10-25), so the
  report numbers are versioned. Rerunning an experiment changes them, so commit reruns deliberately.

---

## 3. Design decisions: don't change these without discussion

These are deliberate. Each has a reason, and changing one silently would invalidate the
results so far. If you think one is wrong, raise it with the team first.

| Decision | Why | Where |
|---|---|---|
| Never shuffle; always time order | Shuffling lets the model see the future, so results would be fake | `stream.py` |
| Predict first, learn after (prequential) | Matches how a real system works | `prequential.py` |
| PR-AUC as the headline metric, not accuracy | Accuracy is ~99.8% for a useless model | `metrics.py` |
| ULB batch size 10,000 | At 2,000, ~3 frauds per batch means noisy scores | commands in section 5 |
| 2 warm-up batches | One batch has too few frauds to train on | `--warmup-batches` |
| Drift signal = per-fraud "missed?" (score < 0.5) | ADWIN/DDM need a stream of 0/1 errors, not one value per batch | `prequential.py` |
| ADWIN `clock=1` | The default (32) barely checks with our batch counts | `detectors.py` |
| Detector reset after each alarm | Otherwise the refit model gets compared with the old one | `detectors.py` |
| ADWIN ignores "errors went down" alarms | Recovery is not drift; it would break accumulate-after-drift | `detectors.py` |
| Skip a refit when a batch has no fraud | Can't train a classifier on one class | `baselines.py` |
| Class weights as the default imbalance method | Simple, no data change, best or tied-best for 2 of 3 models | `baselines.py` |
| Resampling only at training time, after scaling | Scoring must use real data; SMOTE needs scaled distances | `samplers.py` |
| Seed 42 everywhere by default | Reproducible results | all factories |
| Drift injection = swap the top-4 with the bottom-4 fraud-correlated features | Keeps real data while creating a known, abrupt concept drift | `drift_injection.py` |

---

## 4. Glossary

| Term | Meaning |
|---|---|
| **Concept drift** | The relationship between features and fraud changes over time, so a trained model goes stale |
| **Abrupt / gradual drift** | The change happens at one moment / phases in over a period. All our drifts so far are abrupt |
| **Batch** | A chunk of consecutive transactions (e.g. 10,000) processed together |
| **Warm-up** | The first batches, used only for training before any scoring |
| **Prequential evaluation** | "Predictive sequential", or test-then-train: for each batch, predict, score, then learn |
| **PR-AUC** | Area under the precision-recall curve: how well the model ranks fraud above legitimate transactions. 1 = perfect; a random guess scores about the fraud rate (0.0017 on ULB) |
| **Precision / recall** | Of flagged transactions, the share that are really fraud / of real frauds, the share we flagged |
| **Recall at 90% precision** | The share of fraud we catch if we insist that 9 of 10 alerts are real fraud |
| **FPR (false-positive rate)** | The share of legitimate customers wrongly flagged |
| **Class imbalance** | One class (fraud) is far rarer than the other |
| **Class weights** | Each fraud counts more in the model's training loss (~600× on ULB) |
| **SMOTE** | Creates synthetic frauds by blending a real fraud with its nearest fraud neighbours |
| **Undersampling** | Throws away legitimate transactions at random to balance the classes |
| **Static model** | Trained once at the start and never updated |
| **Periodic retraining** | Refit on the most recent batches every N batches, whether needed or not |
| **Online learning** | The model updates after every single transaction (River library) |
| **ARF** | Adaptive Random Forest, River's online random forest with built-in drift handling |
| **ADWIN** | ADaptive WINdowing: keeps a window of recent errors and alarms when recent errors differ significantly from older ones |
| **DDM** | Drift Detection Method: alarms when the error rate rises well above its lowest point so far |
| **Drift injection / feature swap** | Our way of creating a known drift in real data by swapping feature columns from a point in time onwards |
| **Accumulate after drift** | Our method: after an alarm, refit on every new batch until 6 have been gathered, then freeze |
| **SHAP** | A method that splits a prediction into per-feature contributions, used for explaining models |

---

## 5. Getting set up

Requires Python 3.11+ and Git.

```powershell
git clone https://github.com/d1ya-m/fraud-drift-detection.git
cd fraud-drift-detection
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt     # takes a few minutes
pytest -q                           # expect: 71 passed (about 1 minute)
```

**Get the ULB data** (not in Git; the file is 150 MB): download `creditcard.csv` from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and put it in `data/raw/`.
See [data/README.md](../data/README.md).

**Reproduce the existing results** (run from the project root with the venv active):

```powershell
# synthetic data (~3 min)
python experiments/run_experiment.py
# real ULB data, no drift (~10 min)
python experiments/run_experiment.py --data data/raw/creditcard.csv --batch-size 10000
# real ULB data with drift injected halfway, the main experiment (~10-15 min)
python experiments/run_experiment.py --data data/raw/creditcard.csv --batch-size 10000 --inject-drift 0.5
# imbalance comparison (~10-15 min)
python experiments/run_imbalance.py --data data/raw/creditcard.csv --batch-size 10000 --inject-drift 0.5
# reported vs deployed: synthetic (~20 s), ULB as recorded and ULB with drift (~5 min each)
python experiments/run_reported_vs_deployed.py
python experiments/run_reported_vs_deployed.py --data data/raw/creditcard.csv --batch-size 10000
python experiments/run_reported_vs_deployed.py --data data/raw/creditcard.csv --batch-size 10000 --inject-drift 0.5
```

Outputs go to `results/` (CSVs) and `results/figures/` (PNGs). They are committed to Git (see
"Infrastructure" in section 2), so commit reruns deliberately. The two notebooks in `notebooks/` read them.

Tips: `--models logreg,online_logreg` runs a subset, which is handy for quick checks.
`--detectors adwin` skips DDM. `python experiments/run_experiment.py --help` lists every option.
The online adaptive random forest (`online_arf`) is by far the slowest model.

---

## 6. How the code fits together

```
src/data/loader.py            load_dataset(path=None, **synthetic_kwargs) -> DataFrame
                              get_feature_columns(df) -> every column except Time and Class
src/data/synthetic.py         generate_synthetic_stream(...): 1% fraud; pattern flips halfway
                              (fraud on V1-V3 before, on V4-V6 after)
src/data/drift_injection.py   inject_feature_swap(df, at=0.5, k=4) -> (drifted_df, pairs)
src/data/stream.py            make_stream(df, batch_size) -> yields (X, y) in time order

src/models/base.py            BaseDetector: fit / partial_fit / predict_proba / refit
src/models/baselines.py       make_logreg / make_random_forest / make_gradient_boosting
                              (seed=42, imbalance="weights") -> StaticModel (fitted model in .model)
src/models/retrain.py         PeriodicRetrain(inner, every=3, window=6)
src/models/drift_retrain.py   AccumulateAfterDrift(inner, accumulate=6)   <- our improved method
src/models/online.py          make_online_logreg / make_arf (River, learn every transaction)

src/drift/detectors.py        make_adwin() / make_ddm() / NoDriftDetector
                              .update(errors) -> bool   (errors = 1 per missed fraud)
src/imbalance/samplers.py     with_imbalance_handling(estimator, method), METHODS
src/evaluation/metrics.py     compute_metrics(y, scores) -> pr_auc, roc_auc, recall_at_p, fpr, threshold
src/evaluation/prequential.py prequential_eval(model, stream, drift_detector=None,
                              warmup_batches=1) -> one row per batch
src/viz/plots.py              plot_metric_over_time / plot_summary / plot_grouped_bars + colour constants

experiments/run_experiment.py main experiment; also exposes load_data(args), evaluate(...)
experiments/run_imbalance.py  Stage 5 experiment
tests/                        one test file per module; all use synthetic or tiny fake data
```

**Every model has the same four methods** (`BaseDetector`), so the evaluation loop treats
them all alike. A new model only needs `fit`, `predict_proba`, and optionally `partial_fit`
(learn from a new batch) and `refit` (respond to a drift alarm).

**Results format** (every experiment writes this): one CSV row per batch with columns
`model_name, batch_idx, n, n_fraud, pr_auc, roc_auc, recall_at_p, fpr, threshold, drift_flag`,
plus `family`, `strategy`, `runtime_s`. Plots only ever read these columns, so a plot can
be built or changed without running any model.

**Model names in results:** `logreg` (static), `logreg_retrain3` (periodic),
`logreg+adwin` (drift-triggered), `logreg_acc6+adwin` (accumulate after drift),
`logreg_smote` (SMOTE instead of class weights), `online_logreg`, `online_arf`.

---

## 7. Remaining tasks

Every task lists: **Goal**, **Steps**, **Done when** and **Pitfalls**. Rough effort is given
for someone new to the codebase.

---

### Task A: SHAP explainability (Stage 7)

**Effort:** 1–2 days · **Depends on:** nothing · **Branch:** `feat/shap`

**Goal:** show *which features* the model uses to flag fraud, and how that changes after
drift. The injected drift on ULB swaps the 4 most fraud-revealing columns (V17, V14, V12, V10)
with 4 weak ones (V22, V23, V25, V15). SHAP should therefore show:
1. **Before the drift:** a model relies on V14, V17, V12, V10.
2. **After the drift:** a retrained model relies on V23, V22, V25, V15 instead.
3. **The frozen model after the drift:** it still leans on V14 etc., which no longer carry
   the signal. That *explains* why it collapses, which is the strongest visual for the report.

**What SHAP is:** for each prediction, SHAP splits the model's output into a contribution
from each feature. Averaging |contribution| over many transactions gives a "global
importance" per feature.

#### Steps

1. **Create** `src/explain/__init__.py` (empty) and `src/explain/shap_analysis.py`:

   ```python
   def shap_importance(model, X, max_rows=2000, seed=42) -> pd.Series:
       """Mean |SHAP value| per feature for the fraud class, sorted descending.

       model: a fitted StaticModel (use model.model, the fitted scikit-learn estimator).
       X: DataFrame of transactions to explain; sample down to max_rows for speed,
          keeping ALL frauds plus a random sample of legitimate rows.
       """
   ```
   - Use `shap.TreeExplainer(model.model)`. It works for the random forest and gradient
     boosting models when they use class weights or no handling. With SMOTE or undersampling,
     `model.model` is a Pipeline, and you'd explain its last step, `model.model.steps[-1][1]`,
     because the sampler isn't used at prediction time.
   - **Output shapes differ (checked with shap 0.53):**
     - random forest: `shap_values(X)` → array `(rows, features, 2)` → take `[:, :, 1]` (fraud)
     - gradient boosting: → array `(rows, features)` already for the fraud class
   - Return `pd.Series(np.abs(values).mean(axis=0), index=X.columns).sort_values(ascending=False)`.
   - Skip logistic regression for SHAP. If you want it, its standardised coefficients are the
     equivalent and need no SHAP.

2. **Create** `experiments/run_shap.py` (copy the import/`sys.path` header from
   `run_imbalance.py`; reuse `load_data(args)` from `run_experiment.py`). It should:
   1. Load ULB with `--inject-drift 0.5` (and support synthetic data when `--data` is omitted).
   2. Split rows at the drift point: `pre = df.iloc[:drift_row]`, `post = df.iloc[drift_row:]`.
   3. Train **model_pre** on the *last* `--train-rows` (default 40,000) rows before the drift,
      and **model_post** on the *first* `--train-rows` rows after it. Use `make_random_forest()`
      (and optionally `make_gradient_boosting()`).
   4. Compute three importance tables:
      - `pre`: model_pre explained on held-out pre-drift rows (rows not used for training)
      - `post`: model_post explained on held-out post-drift rows
      - `frozen_after`: **model_pre** explained on post-drift rows (the frozen model)
   5. Save `results/<tag>_shap_importance.csv` (columns: `feature, pre, post, frozen_after`).
   6. Plot the top 10 features as horizontal bars, one panel per table, side by side,
      to `results/figures/<tag>_shap.png`. Use one colour (`SINGLE_SERIES`) and values at
      the bar ends, like `plot_summary`. Add the function to `src/viz/plots.py` as
      `plot_feature_importance(tables: dict[str, pd.Series], out_path, top=10, title=None)`.
   7. Optional extra figure: `shap.summary_plot(values, X_sample, show=False)` (the
      "beeswarm"), saved with `plt.savefig`. It shows direction too (low V14 → fraud).

3. **Test** in `tests/test_shap.py`, using the synthetic stream (fast, and we know the truth):
   - Train a random forest on the first 40% of a synthetic stream (`load_dataset(n_samples=20000,
     fraud_rate=0.02)`). Among the **V columns**, its top 3 SHAP features must be {V1, V2, V3}.
   - Train one on the last 40%. Among the V columns, its top 3 must be {V4, V5, V6}.
   - Leave `Amount` out of these two checks: the synthetic fraud amounts also change at the
     drift, so `Amount` legitimately ranks high (checked: after the drift it scores 0.108,
     just behind V5 at 0.117), and including it would make the test fragile.
   - `shap_importance` returns one non-negative value per feature column.

#### Done when
- [ ] `python experiments/run_shap.py --data data/raw/creditcard.csv --inject-drift 0.5` writes the CSV and figure
- [ ] On ULB, the `pre` top features include V14/V17/V12/V10 and the `post` top features include V23/V22/V25/V15
- [ ] `frozen_after` still ranks V14/V17 highly (that's the explanation of the collapse)
- [ ] `tests/test_shap.py` passes, and so does the full `pytest -q`

#### Pitfalls
- SHAP on a 200-tree forest is slow: always sample (≤2,000 rows; keep every fraud).
- Explain *held-out* rows, not the rows the model trained on.
- Don't explain the SMOTE pipeline as a whole; use its final step (see step 1).

---

### Task B: IEEE-CIS natural-drift experiment

**Effort:** 2–4 days (mostly waiting on runs) · **Depends on:** nothing · **Branch:** `feat/ieee-cis`

**Goal:** rerun the main experiment on a dataset with **natural** drift. IEEE-CIS covers
about 6 months of real e-commerce transactions (590,540 rows, ~3.5% fraud, ~390 columns).
ULB only covers 48 hours, which is why its drift had to be injected. This answers "does any
of this matter on real, uninjected data?"

#### Steps

1. **Download** (1.3 GB). Log in to Kaggle, open
   https://www.kaggle.com/competitions/ieee-fraud-detection/data, click **"I understand and
   accept"** on the rules, then:
   ```powershell
   pip install kaggle      # put kaggle.json (Kaggle → Settings → Create New Token) in %USERPROFILE%\.kaggle\
   kaggle competitions download -c ieee-fraud-detection -p data/raw
   ```
   Unzip it and keep **`train_transaction.csv`** in `data/raw/`. Ignore `test_*` (no labels)
   and `train_identity.csv` for now; merging it on `TransactionID` is optional.

2. **Make loading feasible.** The loader already renames `TransactionDT`→`Time` and
   `isFraud`→`Class`, drops `TransactionID`, median-fills numbers and integer-encodes text
   columns. But the full file as float64 is ~1.8 GB in memory, and many columns are mostly
   empty. In `src/data/loader.py`, extend `load_dataset` with two options and keep the
   defaults so nothing else changes:
   ```python
   def load_dataset(path=None, max_missing=None, float32=False, **synthetic_kwargs):
       # max_missing: drop feature columns whose fraction of missing values > max_missing
       #              (do this BEFORE imputing; 0.9 is a sensible value for IEEE-CIS)
       # float32:     downcast float64 feature columns to float32 after cleaning
   ```
   Add `--max-missing` and `--float32` flags to `run_experiment.py` and pass them through
   in `load_data`. Add a test to `tests/test_data.py` (extend the existing IEEE-layout test):
   a column that is 95% missing is dropped with `max_missing=0.9`, and one that is 50%
   missing is kept.

3. **Try a small slice first.** Make a 100k-row sample to check that everything runs
   end to end in minutes:
   ```powershell
   python -c "import pandas as pd; pd.read_csv('data/raw/train_transaction.csv', nrows=100000).to_csv('data/raw/ieee_sample.csv', index=False)"
   python experiments/run_experiment.py --data data/raw/ieee_sample.csv --batch-size 5000 --max-missing 0.9 --float32 --models logreg,random_forest
   ```

4. **Full run.** Batch size **20000** gives ~30 batches with ~700 frauds each, so per-batch
   scores are stable. Leave out the online adaptive forest: per-row Python over 590k rows
   × ~390 columns would take hours.
   ```powershell
   python experiments/run_experiment.py --data data/raw/train_transaction.csv --batch-size 20000 --max-missing 0.9 --float32 --models logreg,random_forest,gradient_boosting,online_logreg
   ```
   There is no `--inject-drift` here: the drift (if any) is natural, so no "true drift"
   line is drawn. Watch which batches ADWIN/DDM flag, and whether frozen models decay over
   the 6 months.

5. **Optional:** run `run_imbalance.py` on IEEE-CIS as well. With 3.5% fraud, imbalance
   handling may matter less than on ULB (0.17%), which is an interesting comparison.

#### Done when
- [ ] The loader options exist, with tests, and the default behaviour is unchanged (all old tests pass)
- [ ] The full run completes and produces `results/train_transaction_*.csv` + figures
- [ ] A short write-up (bullet points are fine) for the report: Do frozen models decay over time? Where did detectors fire? Does accumulate-after-drift still beat periodic retraining?

#### Pitfalls
- **Memory:** close other programs; use `--float32` and `--max-missing`. If it still fails,
  read with `usecols=` to keep `TransactionDT, isFraud`, `TransactionAmt`, `card1-6`,
  `C1-C14`, `D1-D15` and `V1-V100`.
- **Encoding leakage:** text columns are integer-encoded using the whole file, and medians
  are computed over the whole file. That's a mild look-ahead; mention it in the report's
  limitations (it's already noted in a docstring in `loader.py`).
- **Runtime:** expect gradient boosting with periodic retraining to be the slowest static
  run. Run it in the background and keep the terminal output.
- Results may show **little natural drift.** That's a legitimate finding. Report it
  honestly rather than tuning until something appears.

---

### Task C: EDA notebook

> ✅ **Done for ULB** (`notebooks/01_eda.ipynb`). If Task B happens, add IEEE-CIS cells following the same sections.

**Effort:** 1 day · **Depends on:** nothing (Task B's file is optional) · **Branch:** `feat/eda`

**Goal:** `notebooks/01_eda.ipynb`, a short exploration that feeds the report's "Data"
section. Start with ULB; add IEEE-CIS cells once Task B has the file.

#### Steps

Start the notebook with this so it can import the project code:
```python
import sys; sys.path.insert(0, "..")
from src.data.loader import load_dataset
from src.data.drift_injection import inject_feature_swap
from src.viz.plots import apply_style, SINGLE_SERIES   # for consistent styling
df = load_dataset("../data/raw/creditcard.csv")
```

Sections, each a chart plus 2–3 sentences of interpretation:
1. **Shape and class balance:** rows, columns, fraud count and rate (492 / 0.173%), missing
   values (none), and duplicates (1,081 exact duplicate rows; keep them and say why).
2. **Amounts:** the distribution of `Amount` for fraud vs legitimate (log scale).
3. **Time:** transactions per hour and frauds per hour over the 48 hours. This shows the
   day/night cycle and why ULB has little natural drift.
4. **Which features separate fraud:** absolute correlation of each feature with `Class`
   (top 10), plus fraud-vs-legit distributions for the top 4 (expect V17, V14, V12, V10).
5. **The injected drift, visually:** using `inject_feature_swap(df, at=0.5)`, plot V14 and
   V23 for frauds before vs after the drift point, to show the swap.
6. **Batching check:** frauds per batch at batch size 2,000 vs 10,000. This justifies the
   choice of 10,000: at 2,000 there are only ~3 frauds per batch.

#### Done when
- [ ] The notebook runs top to bottom with **Restart & Run All**
- [ ] Each section has a chart and short text
- [ ] Outputs are kept but the file stays under ~5 MB (lower the figure dpi if needed)

---

### Task D: Report write-up

**Effort:** 3–5 days · **Depends on:** A, B and C for their sections; start the rest now ·
**Branch:** `docs/report`

**Goal:** the final report in `docs/`. Adapt the structure to your course template; the
mapping below shows where each result comes from. Sections 1–3 of this document are a good
starting draft for the Introduction and Method.

| Report section | Content | Source |
|---|---|---|
| Introduction | Fraud is rare and its patterns change (concept drift); a model trained once goes stale | Section 1 here, synopsis |
| Data | ULB + synthetic (+ IEEE-CIS); class imbalance; how drift was injected | Task C notebook, `drift_injection.py` docstring |
| Method: evaluation | Prequential test-then-train; why not a random train/test split; PR-AUC and recall at 90% precision instead of accuracy | Section 2 here, `prequential.py`, `metrics.py` |
| Method: models | Static (LR, RF, GB), periodic retrain, online (River LR, ARF) | `src/models/` |
| Method: drift detection | ADWIN and DDM on the "missed fraud" signal; ignoring improvements | Section 3 here, `detectors.py` docstring |
| Method: our improvement | Accumulate-after-drift retraining | Section 2 here, `drift_retrain.py` docstring |
| Method: imbalance | Class weights / SMOTE / undersampling, applied only at training time | `samplers.py` |
| Results 1: objective 1 | Baselines + imbalance comparison | `*_imbalance.png`, section 10 tables |
| Results 2: objective 2 | Static vs periodic vs online over time | `*_pr_auc_over_time.png` |
| Results 3: objective 3 | Detectors + recovery; accumulate beats single-batch refits and periodic retraining | `*_summary.png`, section 10 |
| Explainability | Features before/after drift | Task A figure |
| Natural drift | IEEE-CIS findings | Task B |
| Limitations | Section 11 here | |
| Reproducibility | The exact commands from section 5 | |
| Contributions | Who did what | `git log --format="%an: %s"` |

**Figures to include** (regenerate them with the commands in section 5):
- `synthetic_pr_auc_over_time.png`: the clearest picture of the core idea
- `creditcard_drift50_pr_auc_over_time.png`: the same story on real data
- `creditcard_drift50_summary.png`: the ranking of every strategy after drift
- `creditcard_drift50_imbalance.png`: Stage 5
- `creditcard_pr_auc_over_time.png` (raw ULB, no drift): shows ADWIN correctly stays quiet
  when there is no drift, and DDM's false alarms
- the Task A SHAP figure; the Task B IEEE-CIS figure

#### Done when
- [ ] Every number in the report can be traced to a CSV in `results/` produced by a listed command
- [ ] Limitations from section 11 are discussed
- [ ] A teammate other than the author has proofread it

---

### Task E (optional): Automatic tests on GitHub

**Effort:** 1–2 hours · **Branch:** `chore/ci`

Create `.github/workflows/tests.yml` so GitHub runs `pytest` on every push and pull request.
No data is needed: all tests use the synthetic stream or tiny fake CSVs.

```yaml
name: tests
on: [push, pull_request]
jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt
      - run: pytest -q
```

**Done when:** the "Actions" tab on GitHub shows a green tick on a pull request.

---

### Task F (optional): Robustness checks

**Effort:** 1–3 days · **Depends on:** nothing · **Branch:** `feat/robustness`

These make the conclusions more convincing. Pick whichever fits the time you have.

1. **Several seeds.** Run the main ULB-drift experiment with `--seed 1`, `2` and `3`, and
   report mean ± standard deviation per strategy. Write a small script that reads the three
   `*_summary.csv` files and aggregates them. (The injected drift is deterministic; the
   seed changes model randomness and SMOTE.) Use `--out results/seed1` etc. so runs don't
   overwrite each other.
2. **Sensitivity of our method.** Rerun with `--accumulate 3`, `6` and `10`. How many
   post-drift batches are needed?
3. **Gradual drift.** Today both drifts are abrupt (switched at one row). Add a
   `drift_width` option (fraction of the stream over which the change phases in), first to
   `generate_synthetic_stream` and then to `inject_feature_swap`. During the window, apply
   the new pattern to each row with a probability that rises linearly from 0 to 1. Test it,
   then compare how ADWIN vs DDM cope. Gradual drift is usually harder to detect.
4. **ADWIN sensitivity.** `make_adwin(delta=...)` with 0.0002, 0.002 (default) and 0.02:
   fewer vs more alarms.

**Done when:** a results table per check, plus 2–3 sentences each for the report.

---

## 8. Team rules

1. **One branch per task**, merged through a pull request:
   ```powershell
   git checkout main; git pull
   git checkout -b feat/shap          # name it after the task
   # ...work, commit in small steps...
   git push -u origin feat/shap
   ```
   Then open a pull request on GitHub and have one teammate review it before merging.
2. **Commit messages** start with `feat:`, `fix:`, `test:`, `docs:` or `chore:`.
3. **`pytest -q` must pass before you push.** Add tests for any new module (see each task).
4. **Never commit** anything in `data/raw/` or `.venv/`; `.gitignore` excludes them, so don't force-add
   them. Results in `results/` *are* committed: only commit reruns you mean to keep.
5. **Never shuffle the data or use future batches** to train or tune a model.
6. **Plots:** reuse the colour constants and `apply_style()` in `src/viz/plots.py` so every figure
   matches. A new strategy gets the next unused colour in `STRATEGY_COLORS` (8 at most);
   never reuse a colour for a different meaning inside one figure.
7. **Seeds:** pass `seed=` through so results are reproducible (the default is 42).
8. **Check section 3 before changing behaviour** that existing results depend on. If you
   must change it, rerun the affected experiments and update section 10.
9. **Update this file** when you finish a task: tick it off in section 2 and add any new
   results to section 10.

---

## 9. Suggested split and order

```
Week 1   A: SHAP        B: IEEE-CIS download + loader + sample run     C: EDA notebook     D: report outline + method sections
Week 2   A: done → helps D     B: full runs + findings                 E/F (whoever is free)       D: results sections
Week 3   D: finish, proofread, final reruns of all commands from section 5
```

- **A, B and C are independent**: three people can start in parallel on day one.
- **D can start immediately** with the Introduction and Method sections; results sections
  fill in as A–C land.
- Before the final submission, someone reruns **every command in section 5** from a fresh
  clone, so the report's numbers are reproducible.

---

## 10. Results so far (reference numbers)

All runs use seed 42 and are reproducible with the commands in section 5.
PR-AUC: 0 = useless, 1 = perfect; a random guess scores about the fraud rate.

**Mean PR-AUC after the drift, static models**

| Data | Model | No adaptation | Periodic retrain | + ADWIN | + DDM | **+ ADWIN + accumulate** |
|---|---|---|---|---|---|---|
| Synthetic | Logistic regression | 0.04 | 0.65 | 0.79 | 0.79 | **0.82** |
| Synthetic | Random forest | 0.03 | **0.72** | 0.69 | 0.69 | 0.71 |
| Synthetic | Gradient boosting | 0.03 | 0.64 | 0.73 | 0.73 | **0.76** |
| ULB + injected drift | Logistic regression | 0.01 | 0.48 | 0.48 | 0.39 | **0.58** |
| ULB + injected drift | Random forest | 0.23 | 0.65 | 0.70 | 0.70 | **0.75** |
| ULB + injected drift | Gradient boosting | 0.03 | 0.54 | 0.49 | 0.54 | **0.60** |

**Online models, ULB + injected drift (PR-AUC after the drift):** online logistic regression
0.52 (ADWIN never fires: the model adapts by itself); adaptive random forest 0.65, or 0.70
with DDM.

**Raw ULB (no injected drift):** static random forest is best at 0.76 with no adaptation.
ADWIN never fires, which is correct, since there's no drift in 48 hours. DDM raises false
alarms that *lower* scores (random forest 0.76 → 0.70).

**Imbalance handling, ULB + injected drift, "+ ADWIN + accumulate" (mean PR-AUC):**

| Model | No handling | Class weights | SMOTE | Undersampling |
|---|---|---|---|---|
| Logistic regression | 0.46 | **0.52** | 0.38 | 0.37 |
| Random forest | 0.71 | **0.75** | 0.74 | 0.57 |
| Gradient boosting | 0.46 | 0.61 | **0.70** | 0.60 |

Undersampling is the fastest (random forest 6s vs 25–37s) but the weakest after drift.
On synthetic data (1% fraud, easy to separate), "no handling" is often as good as or better
than balancing.

---

## 11. Known limitations

Discuss these in the report.

1. **The drift signal only watches missed frauds.** A drift that mainly causes *false alarms*
   on legitimate customers would go unnoticed.
2. **"Recall at 90% precision" is optimistic.** The threshold is chosen per batch *using
   that batch's labels*, so it's an upper bound on what a live system would get.
3. **Labels are assumed to arrive immediately.** In reality, fraud is confirmed days or
   weeks later (label delay).
4. **ULB's drift is injected**, not natural (which is why Task B matters). ULB itself covers
   only 48 hours.
5. **Imputation and encoding use the whole file** (a mild look-ahead; mainly IEEE-CIS).
6. **Results come from single runs with one seed** (Task F.1 fixes this).

---

## 12. Gotchas we already hit

- **pandas 3 views:** `df[cols].to_numpy()` can return a *view*. When swapping or
  overwriting columns, copy first (`.to_numpy(copy=True)`). This silently broke the drift
  injection once; `tests/test_drift_injection.py` now guards it.
- **Shared y-axes in matplotlib:** calling `ax.invert_yaxis()` on every subplot of a
  `sharey=True` figure flips it back and forth. Invert once.
- **Overlapping lines:** when two strategies produce identical results, one line hides the
  other. `plot_metric_over_time` draws DDM wider underneath and notes detectors that never
  fired; keep that behaviour if you add strategies.
- **Batch size on ULB:** 2,000 rows means only ~3 frauds per batch, and very noisy scores.
  Use 10,000.
- **SMOTE needs at least 2 frauds**, and `k_neighbors` below the fraud count. `samplers.smote`
  already handles this; reuse it, don't call imblearn's SMOTE directly.
- **PowerShell:** Git prints normal progress to stderr, which PowerShell shows in red as an
  "error". Check the last line (`main -> main`) instead.
- **Scripts must run from the project root** with the venv active; they add the root to
  `sys.path` themselves.
