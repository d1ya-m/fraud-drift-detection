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

### Project title (final report)
**Keeping Credit-Card Fraud Detectors Reliable Under Concept Drift: A Comparison of Adaptive
Model-Maintenance Strategies**

### The research question
**How do we best keep a fraud model reliable as fraud patterns change?** We compare the adaptive strategies a
bank could use, against a frozen reference:
- **periodic retraining**: refit on the latest 6 batches every 3 batches, on a fixed schedule
- **online** models: update after every batch
- **drift-triggered retraining**: a detector (ADWIN or DDM) notices a rise in missed frauds and triggers a refit
- **DDM warning window** (Gama et al., 2004): retrain on the examples since DDM's warning
- **accumulate-after-drift** (our variant): after an ADWIN alarm, keep refitting on new batches until 6 are
  gathered
- reference: a **static** model, trained once and then frozen

### Three research questions (the structure of notebook 02 and of the report)
1. How well do classical models detect rare fraud, measured properly (PR-AUC, recall at 90% precision,
   false-positive rate; imbalance handling)?
2. How badly does a model that is never updated degrade, and would a standard (shuffled) evaluation notice?
3. Which adaptive strategy keeps a fraud model working best, and why?

### Storyline for the report and PPT (lead with the strong results; report everything)
1. **The hidden problem.** Random forest *reports* 0.81–0.89 PR-AUC on a test split, then *deploys* at 0.07
   after the drift. Frozen models lose 79–99%.
2. **Strong baselines.** Random forest 0.76 PR-AUC on ULB, about 440× random, holding ~0.9 precision.
3. **Adaptation works.** Held-out post-drift PR-AUC goes from 0.16 (frozen) to 0.65 (periodic retraining), and
   operational fraud recall rises by +0.43. The best combination is random forest + periodic retraining (0.75).
4. **Ranking of strategies:**
   1. periodic retraining
   2. the DDM methods
   3. accumulate
   4. ADWIN refit-on-alarm
   5. online logistic regression

   The ranking is confirmed with paired 95% CIs and on 4 fresh scenarios.
5. **Why detectors trail.** They learn from ~13 frauds per batch and react 1–9 batches late. Accumulate still
   improves on standard refit-on-alarm (14 of 18), and ADWIN never raised a false alarm, so it makes a good
   monitor.
6. **SHAP** shows what changed, and how retraining moves to the new features.
7. **Rigour:**
   - 10 scenarios × 3 seeds, block-bootstrap CIs
   - held-out and fresh scenarios, label delay
   - 89 tests, reproducible scripts
8. **Recommendation:** retrain on a schedule and use detectors as alarms. Future work: combining label-based
   and label-free drift signals (offline check in notebook 02 section 3.1), and a natural-drift dataset.

Wording to use for our variant: *"Accumulate-after-drift fixes the main weakness of detector-triggered
retraining (refitting on a single batch) and improves it significantly, but it inherits the detector's delay,
so scheduled retraining remained more reliable."* Do not drop these results or quote only the development
scenario.

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

### The main findings (after the audit fixes; 3 seeds, held-out scenarios, 95% CIs)
1. **Frozen models collapse after drift.** On ULB + drift, PR-AUC falls from ~0.64–0.78 to 0.005–0.16, while
   the model's pre-deployment test score still looks fine.
2. **ADWIN never raised a false alarm** and caught every drift on the tree models, one batch after it hit.
   **DDM** raises false alarms on logistic regression that cost real performance.
3. **Detection is limited by fraud counts.** The detectors get one value per fraud (~13 per batch). Detection
   took 1 batch on the development drift only because a large fraud burst followed it; elsewhere 3–9 batches.
   A moderate rise in the miss rate (logistic regression after 2- or 6-pair swaps) is never detected.
4. **Accumulate-after-drift reliably beats refitting on the alarm batch alone** (14 of 18 held-out comparisons
   significant, none worse).
5. **It does *not* beat periodic retraining** on held-out scenarios (12 not significant, 6 worse). Periodic
   retraining has the best held-out mean (0.649 vs 0.562). Accumulate looked best only on the development
   scenarios, because a large fraud burst right after that drift made detection unusually fast.
   - **Supplementary check** on 4 fresh scenarios (`run_supplementary.py`): accumulate is 0.13 behind periodic
     retraining.
   - **A hybrid fix** (periodic + ADWIN reset) looked slightly better on the scenarios it was designed from, but
     was level with periodic on the fresh ones (−0.003), so it was not adopted. **Periodic retraining is the
     recommendation.**
6. **Label delay** (labels arriving 1–3 batches late) costs every strategy 0.1–0.2 PR-AUC.
7. **Only the tree models hold the 90% precision target** when the threshold is carried over from the previous
   batch. Logistic regression slips to about 0.52–0.77.

Full analysis: `notebooks/02_results_analysis.ipynb`.

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
- **First found (superseded):** best in 5 of 6 tests under the original setup (single seed, 2-batch warm-up,
  development scenario only).
- **After the audit fixes** (equal warm-up, 3 seeds, held-out and fresh scenarios): it beats single-batch
  refits (14 of 18 held-out comparisons) but **not** periodic retraining, because it waits for ADWIN, which
  reacts 1–9 batches late. See section 1 and notebook 02 section 6.

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

### Post-audit fixes ✅
An internal project audit (kept locally, not in the repo) found the core correct but several claims unsupported. Everything it
listed was then fixed, except the report itself, the literature review (a report section), running the notebooks
in a real Jupyter kernel (needs a person in VS Code) and a natural-drift dataset (needs a Kaggle download):
- **Equal starting data:** every strategy now warms up on **6 batches** (one periodic-retrain window).
  Before, static models had 20,000 rows and periodic windows had 60,000, which mixed "adapting" with "more data".
- **Operational metrics:** besides the per-batch oracle recall at 90% precision, every run reports
  `op_recall`, `op_precision` and `op_fpr` at the threshold learnt on the *previous* labelled batch, which is
  what a live system can do.
- **Frozen protocol with held-out scenarios** (`run_protocol.py`): the settings were frozen after a sensitivity
  sweep on the development scenarios only (`run_sensitivity.py`). All strategies were then evaluated with
  3 seeds on 6 held-out drift scenarios: drift at 30% or 70%, 2 or 6 swapped pairs, random pairs, and
  gradual drift. Differences are judged with block-bootstrap 95% CIs (`src/evaluation/stats.py`).
- **Prior-art comparator:** `DDM warning window` (Gama et al., 2004, the scheme that accumulate-after-drift
  varies) and a **single-feature rule** baseline.
- **Label delay:** `--label-delay N`; labels reach the models and detectors N batches after scoring.
- **Drift severity study** (`run_severity.py`), **error analysis** (`run_error_analysis.py`) and **SHAP**
  (`run_shap.py`), plus a detector blind-spot check (`run_detector_blindspot.py`).
- **Smaller fixes:** gradient boosting without its hidden 10% early-stopping split; pinned
  `requirements-lock.txt`; unused packages removed; `PIPELINE.md` marked historical; a GitHub Actions test
  workflow.

### Infrastructure ✅
- GitHub repo: https://github.com/d1ya-m/fraud-drift-detection
- 86 tests (`pytest -q`, about 1 minute; no dataset needed); GitHub Actions runs them on every push
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
| 6 warm-up batches for every strategy | Same starting data as one periodic-retrain window, so strategies differ only in how they adapt | `--warmup-batches` |
| Report operational metrics beside the oracle ones | The per-batch 90%-precision threshold uses that batch's labels (an upper bound); `op_*` use the previous batch's threshold | `metrics.py`, `prequential.py` |
| Settings frozen: accumulate 6, retrain every 3 on the last 6, ADWIN δ 0.002 | Chosen on the development scenarios only (`sensitivity_summary.csv`); no value beat these on both | `run_experiment.py` defaults |
| Judge strategy differences with the held-out protocol and block-bootstrap CIs | Per-batch scores are noisy (sd ~0.2) and neighbouring batches correlated; one seed and one scenario can't rank strategies | `run_protocol.py`, `stats.py` |
| Gradient boosting `early_stopping=False`, `max_iter=100` | The default silently held out 10% of training data (~8 frauds) | `baselines.py` |
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
| **Abrupt / gradual drift** | The change happens at one moment / phases in over a period. The protocol tests both (`--drift-width`) |
| **Development vs held-out scenario** | Development scenarios were looked at while designing methods and choosing settings; held-out scenarios were not, so results on them are a fair test |
| **Operational recall / precision** | Recall and precision at a threshold fixed *before* the batch (learnt on the previous labelled batch), as a live system would use |
| **Oracle recall at 90% precision** | The same, but with the threshold chosen on the batch's own labels: an upper bound |
| **Label delay** | Fraud labels arrive some time after the transaction; `--label-delay N` = N batches later |
| **Block bootstrap CI** | A 95% confidence interval made by resampling runs of consecutive batches; a difference whose CI includes 0 is not established |
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
| **Accumulate after drift** | This project's variant: after an ADWIN alarm, refit on every new batch until 6 have been gathered, then freeze |
| **DDM warning window** | The established scheme it varies (Gama et al., 2004): buffer data from DDM's warning, retrain on it at the drift |
| **SHAP** | A method that splits a prediction into per-feature contributions, used for explaining models |

---

## 5. Getting set up

Requires Python 3.11+ and Git.

```powershell
git clone https://github.com/d1ya-m/fraud-drift-detection.git
cd fraud-drift-detection
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements-lock.txt   # exact versions used for the results (or requirements.txt, unpinned)
pytest -q                              # expect: 86 passed (about 1 minute)
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
# (also run run_imbalance.py with no --data, and with --data but no --inject-drift, for the other two datasets)

# post-audit experiments
python experiments/run_sensitivity.py --data data/raw/creditcard.csv      # ~25 min, development scenarios only
python experiments/run_protocol.py --data data/raw/creditcard.csv         # ~2 h, resumable; 10 scenarios x 3 seeds
python experiments/run_severity.py --data data/raw/creditcard.csv         # ~10 min
python experiments/run_error_analysis.py --data data/raw/creditcard.csv --batch-size 10000 --inject-drift 0.5
python experiments/run_shap.py --data data/raw/creditcard.csv --batch-size 10000 --inject-drift 0.5
python experiments/run_shap.py                                            # synthetic
python experiments/run_detector_blindspot.py --data data/raw/creditcard.csv
python experiments/run_supplementary.py --data data/raw/creditcard.csv       # supplementary, ~1.5 h
# everything in one notebook (presentation): notebooks/fraud_drift_project.ipynb
```

Outputs go to `results/` (CSVs; the protocol and sweeps in subfolders) and `results/figures/` (PNGs). They are
committed to Git (see "Infrastructure" in section 2), so commit reruns deliberately. The two notebooks in
`notebooks/` read them.

Tips: `--models logreg,online_logreg` and `--strategies base,accumulate` run a subset, which is handy for quick
checks. `python experiments/run_experiment.py --help` lists every option, including the drift scenarios
(`--swap-k`, `--pair-seed`, `--drift-width`) and `--label-delay`. The online adaptive random forest
(`online_arf`) is by far the slowest model.

---

## 6. How the code fits together

```
src/data/loader.py            load_dataset(path=None, **synthetic_kwargs) -> DataFrame
                              get_feature_columns(df) -> every column except Time and Class
src/data/synthetic.py         generate_synthetic_stream(...): 1% fraud; pattern flips halfway
                              (fraud on V1-V3 before, on V4-V6 after)
src/data/drift_injection.py   inject_feature_swap(df, at=0.5, k=4, pair_seed=None, width=0.0) -> (drifted_df, pairs)
src/data/stream.py            make_stream(df, batch_size) -> yields (X, y) in time order

src/models/base.py            BaseDetector: fit / partial_fit / predict_proba / refit
src/models/baselines.py       make_logreg / make_random_forest / make_gradient_boosting
                              (seed=42, imbalance="weights") -> StaticModel (fitted model in .model)
                              make_rule -> SingleFeatureRule (trivial one-feature baseline)
src/models/retrain.py         PeriodicRetrain(inner, every=3, window=6)
src/models/drift_retrain.py   AccumulateAfterDrift(inner, accumulate=6)   <- this project's variant
                              WarningWindowRetrain(inner)                 <- Gama et al. 2004 (prior art)
src/models/online.py          make_online_logreg / make_arf (River, learn every transaction)

src/drift/detectors.py        make_adwin() / make_ddm() / NoDriftDetector
                              .update(errors) -> bool   (errors = 1 per missed fraud)
src/imbalance/samplers.py     with_imbalance_handling(estimator, method), METHODS
src/evaluation/metrics.py     compute_metrics (pr_auc, roc_auc, oracle recall_at_p, fpr, threshold)
                              operational_metrics (op_recall, op_precision, op_fpr at a fixed threshold)
src/evaluation/prequential.py prequential_eval(model, stream, drift_detector=None, warmup_batches=1,
                              label_delay=0, score_sink=None) -> one row per batch
src/evaluation/drift_metrics.py drift_response(_table), trend_per_10_batches
src/evaluation/stats.py       block_bootstrap_mean, paired_difference (CIs for strategy comparisons)
src/explain/shap_analysis.py  shap_importance(model, X, y)
src/viz/plots.py              all figures + colour constants + apply_style()

experiments/run_experiment.py main experiment; exposes add_common_args, load_data, evaluate, run
experiments/run_imbalance.py  imbalance handling          experiments/run_reported_vs_deployed.py
experiments/run_sensitivity.py  settings sweep (dev only)  experiments/run_protocol.py  held-out protocol
experiments/run_severity.py   drift severity              experiments/run_error_analysis.py
experiments/run_shap.py       SHAP before/after drift
experiments/run_detector_blindspot.py  what the missed-fraud drift signal can and cannot see (diagnostic)
experiments/run_supplementary.py  SUPPLEMENTARY: periodic + ADWIN reset, on 4 fresh scenarios
tests/                        one test file per module (+ test_audit_fixes.py); synthetic or tiny fake data
```

**Every model has the same four methods** (`BaseDetector`), so the evaluation loop treats
them all alike. A new model only needs `fit`, `predict_proba`, and optionally `partial_fit`
(learn from a new batch) and `refit` (respond to a drift alarm).

**Results format** (every experiment writes this): one CSV row per batch with columns
`model_name, batch_idx, n, n_fraud, pr_auc, roc_auc, recall_at_p, fpr, threshold, op_recall, op_precision,
op_fpr, drift_flag`, plus `family`, `strategy`, `runtime_s`. Plots only ever read these columns, so a plot can
be built or changed without running any model.

**Model names in results:** `rule` (single-feature baseline), `logreg` (static), `logreg_retrain3` (periodic),
`logreg+adwin` (drift-triggered), `logreg_acc6+adwin` (accumulate after drift), `logreg_ddmwarn` (DDM warning
window), `logreg_smote` (SMOTE instead of class weights), `online_logreg`, `online_arf`.

---

## 7. Remaining tasks

Every task lists: **Goal**, **Steps**, **Done when** and **Pitfalls**. Rough effort is given
for someone new to the codebase.

---

### Task A: SHAP explainability (Stage 7)

> ✅ **Done** as specified below: `src/explain/shap_analysis.py`, `experiments/run_shap.py`, `plot_feature_importance`,
> tests in `tests/test_audit_fixes.py`. Results: `results/*_shap_importance.csv`, notebook 02 section 5.
> Kept for reference.

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
| Introduction | Fraud is rare and its patterns change (concept drift); an unmaintained model silently goes stale. Research question: which adaptive strategy keeps it reliable? | Section 1 here (title, research questions, storyline) |
| Related work | Active (detector-triggered) vs passive (scheduled/continuous) adaptation; ADWIN, DDM, DDM warning window; fraud-specific work (Dal Pozzolo et al.) | Literature review (Task to write) |
| Data | ULB + synthetic; class imbalance; how drift was injected (6 drift types) | `01_eda.ipynb`, `drift_injection.py` docstring |
| Method: evaluation | Prequential test-then-train; PR-AUC, oracle vs operational recall; held-out and fresh scenarios, 3 seeds, block-bootstrap CIs | Section 2 here, `prequential.py`, `metrics.py`, `stats.py`, `run_protocol.py` |
| Method: models and strategies | Static LR / RF / GB + rule; periodic retraining; online; ADWIN / DDM refit; DDM warning window; accumulate-after-drift | `src/models/`, notebook 02 Q3 intro |
| Results Q1: baselines | Baselines, oracle vs operational, imbalance handling | Notebook 02 section 1 |
| Results Q2: degradation | Reported vs deployed; degradation; severity | Notebook 02 section 2, `*_reported_vs_deployed.png`, `severity.png` |
| Results Q3: strategy comparison | Ranking table; forest plot; recall gain; recovery; label delay; cost. Periodic retraining is best; accumulate beats refit-on-alarm but not periodic | Notebook 02 section 3 + overall ranking, `protocol_comparisons.png` |
| Analysis: why detectors trail | Gap by phase, detection delay vs fraud count, supplementary fresh scenarios | Notebook 02 section 6 |
| Error analysis and explainability | Confusion at operational threshold; SHAP before/after drift | Notebook 02 sections 4–5 |
| Limitations and future work | Section 11 here; label-free drift signal | |
| Reproducibility | The exact commands from section 5 | |

**Figures to include:**
- `creditcard_drift50_reported_vs_deployed` (from notebook 02 section 2.1): the hidden problem
- `creditcard_drift50_pr_auc_over_time.png`: strategies over time on real data
- `figures/protocol_comparisons.png`: paired CIs across scenarios
- `severity.png` and `creditcard_drift50_shap.png`
- the overall ranking table from notebook 02

#### Done when
- [ ] Every number in the report can be traced to a CSV in `results/` produced by a listed command
- [ ] Limitations from section 11 are discussed
- [ ] A teammate other than the author has proofread it

---

### Task E (optional): Automatic tests on GitHub

> ✅ **Done:** `.github/workflows/tests.yml` (installs `requirements-lock.txt`, runs `pytest -q` on Python 3.13).

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

> ✅ **Mostly done** through `run_protocol.py` (3 seeds, held-out scenarios including gradual drift),
> `run_sensitivity.py` (accumulate 3/6/10, retrain interval 2/3/5, ADWIN δ 0.0002/0.002/0.02) and
> `run_severity.py`. Still open: more seeds (5+) if time allows, and gradual drift on the synthetic generator.

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

Numbers as of the post-audit commit (warm-up 6 batches, seed 42 unless stated). The full tables, figures
and interpretation are in `notebooks/02_results_analysis.ipynb`; rerunning experiments changes these numbers.
PR-AUC: 0 = useless, 1 = perfect; a random guess scores about the fraud rate (0.0017 on ULB).

**ULB as recorded (static models, mean PR-AUC over 23 batches):** random forest 0.762, gradient boosting 0.741,
adaptive random forest 0.728, logistic regression 0.598, single-feature rule 0.588. ADWIN never fires (correct:
no injected drift). DDM fires once on logistic regression and lowers its score to 0.480.

**ULB + injected drift (development scenario), mean PR-AUC after the drift:**

| Model | Frozen | Periodic retrain | + ADWIN | + DDM | DDM warning window | + ADWIN + accumulate |
|---|---|---|---|---|---|---|
| Logistic regression | 0.005 | 0.502 | 0.473 | 0.473 | 0.473 | **0.573** |
| Random forest | 0.156 | **0.762** | 0.680 | 0.708 | 0.758 | 0.736 |
| Gradient boosting | 0.016 | **0.698** | 0.484 | 0.543 | 0.644 | 0.662 |

**Held-out protocol (6 scenarios × 3 seeds, paired 95% CIs)** is the evidence to quote: accumulate beats plain
ADWIN in 14 of 18 comparisons (none worse), is never significantly better than periodic retraining (12 not
significant, 6 worse), and is mixed against DDM and the DDM warning window (5 better, 11 not significant, 2 worse).
Mean held-out post-drift PR-AUC: periodic 0.649, DDM warning window 0.596, DDM 0.591, accumulate 0.562, ADWIN 0.464,
frozen 0.161. Details: `results/protocol_summary.csv`, `results/protocol_comparisons.csv`.

**Imbalance handling:** see the notebook (section 1.3). Class weights are never far from the best and are the
default. Gradient boosting with no handling collapses (0.129 on ULB). Undersampling is cheapest but weakest after drift.

---

## 11. Known limitations

Discuss these in the report.

1. **The drift signal learns from missed frauds only** (~13 per batch). Detection is slow unless
   frauds are plentiful, and a moderate rise in the miss rate goes unnoticed (`results/detector_blindspot.csv`,
   `results/detector_signals.csv`). An offline check of richer signals found no free fix. Fraud scores help
   little. Label-free signals (all transactions) catch every drift on logistic regression within 0–3 batches,
   but they are slower for tree models on abrupt drift and raise 1–3 false alarms on undrifted data.
2. **The oracle recall is optimistic.** "Recall at 90% precision (oracle)" picks the threshold
   on each batch's own labels. The operational metrics (`op_*`) fix this, but their precision
   often misses the 90% target for linear models.
3. **Label delay is studied only at 1 and 3 batches** (10,000–30,000 transactions, a few hours of
   ULB). Real fraud labels take days to weeks.
4. **ULB's drift is injected**, not natural, and ULB covers only 48 hours from one issuer in 2013.
   A natural-drift dataset (Task B, IEEE-CIS) is still missing. Recurring drift is not tested.
5. **Small post-drift samples.** ~17 frauds per batch and ~14 post-drift batches. One batch of
   detection delay can decide a third of the post-drift frauds (see the error analysis).
6. **Three seeds only change model randomness**, not the data. The CIs mainly capture
   batch-to-batch variation.
7. **Imputation and encoding use the whole file** (a mild look-ahead; mainly IEEE-CIS).

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
- **Jupyter kernels from an AI-assistant sandbox:** command-line `jupyter nbconvert --execute`
  could not reach a kernel there, so saved notebook outputs were produced by executing the
  cells in-process. In VS Code, just pick the `.venv` kernel and Run All.
- **The protocol takes ~3 hours** (30 runs of ~6.5 min on ULB). It is resumable: finished
  scenario/seed folders are skipped. Use `--aggregate-only` to rebuild the summary tables.
- **Alarms in the drift batch:** the batch that contains the drift point is excluded from the
  "after drift" metrics (it is half clean), but an alarm in it counts as a detection, not a false alarm.
- **shap 0.53 output shapes:** random forest returns `(rows, features, 2)`, so take `[:, :, 1]`;
  gradient boosting returns `(rows, features)`.
