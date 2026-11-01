# Report Guide: everything needed to write the final report

This is the single source for the final report and the presentation. It gives the title, the storyline, the
content of every section with its key numbers, the figures and tables to use, wording rules and references.

- **Every number here comes from a CSV in `results/`** and is shown in `notebooks/02_results_analysis.ipynb`.
  When in doubt, the notebook is the authority.
- **Numbers are from the committed results.** If an experiment is rerun, recheck them against the notebook.

---

## 1. Title, framing and storyline

**Title:** *Keeping Credit-Card Fraud Detectors Reliable Under Concept Drift: A Comparison of Adaptive
Model-Maintenance Strategies*

**Framing in one sentence:**
- **The problem:** fraud models silently break when fraud patterns change.
- **The question:** which adaptive strategy best keeps a classical fraud model reliable, and why?

**Three research questions** (the report's and notebook's structure):
1. How well do classical models detect rare fraud, measured properly?
2. How badly does a model that is never updated degrade, and would a standard evaluation notice?
3. Which adaptive strategy keeps a fraud model working best, and why?

**Storyline** (use the same order for the report's results and the slides):
1. **The hidden problem:** a model's test score looks excellent, then it collapses after the drift.
2. **Strong baselines:** random forest is about 440× better than random.
3. **Adaptation works:** post-drift PR-AUC goes from 0.16 to 0.65.
4. **Ranking of strategies:** periodic retraining is most reliable.
5. **Why detector-based strategies trail:** they wait for an alarm that is slow when frauds are rare.
6. **Explainability:** what the model looks at before and after the drift.
7. **Rigour:** held-out and fresh scenarios, 3 seeds, confidence intervals.
8. **Recommendation and future work.**

---

## 2. Abstract (draft, about 200 words)

> Credit-card fraud models face two problems at once: fraud is extremely rare (0.17% of transactions), and
> fraud patterns change over time (concept drift). We study how to keep a classical fraud model reliable under
> drift.
>
> We compare a frozen model with the adaptive strategies a bank could use:
> - periodic retraining
> - online learning
> - retraining triggered by the ADWIN or DDM drift detectors
> - the DDM warning-window method
> - accumulate-after-drift, a variant we propose
>
> All are evaluated prequentially (test-then-train) on the ULB credit-card dataset with injected drift and on
> a synthetic stream. To avoid tuning to one scenario, settings were frozen on development scenarios and
> tested on held-out and fresh drift scenarios, with three seeds and block-bootstrap confidence intervals.
>
> Without maintenance, models lose 79–99% of their PR-AUC after a drift, while their pre-deployment test
> scores still look excellent. Periodic retraining is the most reliable strategy: post-drift PR-AUC 0.65 vs
> 0.16 for a frozen model, and 0.75 with random forest. Our accumulate-after-drift variant significantly
> improves standard detector-triggered retraining, but detector-based strategies trail periodic retraining.
> Their detectors learn from rare fraud labels (~13 per batch) and react 1–9 batches late.
>
> We recommend scheduled retraining, with drift detectors as monitoring alarms.

---

## 3. Section-by-section content

### 3.1 Introduction
- **Fraud is rare.** In the ULB data, 492 of 284,807 transactions (0.173%) are fraud. A model that never flags
  fraud is 99.83% accurate, so accuracy is meaningless here.
- **Fraud patterns change (concept drift).** Fraudsters adapt, so a model trained once goes stale. Nothing in
  its alert stream signals this: after the drift, our frozen model raised 1 alert in 134,807 transactions.
- **Gap:** standard evaluation (a shuffled train/test split) measures the model at one point in time and
  cannot reveal this decay.
- **Contributions:**
  1. A prequential comparison of six adaptive strategies against a frozen model, on three classical models,
     with imbalance-aware and operational metrics.
  2. A rigorous protocol: settings frozen on development scenarios; held-out and fresh drift scenarios;
     3 seeds; paired block-bootstrap 95% CIs; label delay.
  3. A proposed variant, accumulate-after-drift, and an honest evaluation of it.
  4. An analysis of *why* detector-triggered strategies trail periodic retraining (detection delay due to
     rare fraud labels), with error analysis and SHAP explanations.

### 3.2 Related work (write 1–2 pages; see the references in section 7)
- **Concept drift:** definitions and the abrupt, gradual and recurring types (Gama et al. 2014 survey).
- **Two families of adaptation** (Ditzler et al. 2015):
  - **Active** methods detect a change, then adapt: ADWIN (Bifet & Gavaldà 2007), DDM and its warning window
    (Gama et al. 2004).
  - **Passive** methods keep updating continuously or on a schedule: sliding-window retraining, online
    learners such as the adaptive random forest (Gomes et al. 2017).
- **Fraud-specific work:** Dal Pozzolo et al. (2015, 2018), on real bank streams with delayed labels and
  daily sliding-window retraining.
- **Imbalance:** SMOTE (Chawla et al. 2002), class weighting; why PR curves beat ROC on imbalanced data (Saito
  & Rehmsmeier 2015).
- **Evaluation of streams:** prequential evaluation (Gama et al. 2013).
- **Position our work:** a controlled comparison of active vs passive maintenance for rare-fraud streams. Our
  result (passive wins when labels are rare) agrees with recent comparisons (see the note in section 7).

### 3.3 Data
**ULB credit-card dataset.**
- 284,807 transactions over 48 hours (September 2013, European cardholders); 492 frauds (0.173%).
- Features V1–V28 are PCA components, anonymised by the data provider; Time and Amount are raw.
- `Time` is used only for ordering, never as a model feature.

**Synthetic stream.**
- 50,000 transactions, 1% fraud.
- The fraud signal moves from V1–V3 to V4–V6 halfway through (shift size 2.0), so the drift is known exactly.

**Drift injection on ULB** (`src/data/drift_injection.py`). From the drift point on, swap the values of the 4
most fraud-correlated features with the 4 least correlated ones (`Amount` excluded). This mimics fraudsters
changing tactics so that the learnt indicators stop working. Scenarios:

| Scenario | Drift at | Type | Role |
|---|---|---|---|
| dev | 50% | 4 strongest↔weakest pairs, abrupt | development |
| drift30 / drift70 | 30% / 70% | same, abrupt | held-out |
| k2 / k6 | 50% | 2 / 6 pairs | held-out |
| random_pairs | 50% | 4 random pairs | held-out |
| gradual | 40% → 60% | swap probability ramps up | held-out |
| delay1 / delay3 | 50% | dev, labels arrive 1 / 3 batches late | label delay |
| synthetic | 50% | generator's drift | development |
| 4 fresh scenarios | 40%, 60% (3 pairs), 35% (random pairs), 55% → 70% gradual | — | supplementary, never used before testing |

Use EDA figures and statements from `notebooks/01_eda.ipynb`: class imbalance, amount and time distributions,
and feature–fraud correlations.

### 3.4 Method
**Evaluation (prequential, test-then-train).**
- Transactions are processed in time order in batches: 10,000 on ULB (29 batches), 2,000 on synthetic
  (25 batches).
- Each batch is scored *before* the model may learn from it.
- Every strategy warms up on the same first 6 batches, which is one periodic-retraining window, so all
  strategies start with equal data.

**Metrics.**
- **PR-AUC (average precision):** the headline metric. A random model scores the fraud rate, 0.0017 on ULB.
- **Recall at 90% precision, oracle:** the threshold is picked on the batch's own labels, so it is an upper
  bound.
- **Operational recall, precision and FPR:** the threshold is learnt on the *previous* labelled batch, which is
  what a live system can do.
- **Drift response:** before/after means, drop, recovery time (batches until the 3-batch rolling PR-AUC is back
  at 90% of the pre-drift level) and alarm counts.

**Models.**
- Logistic regression (standardised, max_iter 1000).
- Random forest (200 trees, min_samples_leaf 2).
- Gradient boosting (scikit-learn HistGradientBoosting, 100 iterations, no early stopping).
- A single-feature rule baseline (the most fraud-correlated feature).
- Online: River logistic regression (SGD) and adaptive random forest (10 trees).
- Imbalance: class weights by default; SMOTE and undersampling compared, applied to training data only.

**Strategies.**

| Strategy | Behaviour |
|---|---|
| Frozen (reference) | trained once on the warm-up, never updated |
| Periodic retraining | every 3 batches, refit from scratch on the latest 6 batches |
| Online | update after every batch |
| + ADWIN / + DDM | frozen until the detector alarms, then refit on the alarm batch |
| DDM warning window (Gama et al. 2004) | buffer batches from DDM's warning; at the drift, refit on the buffer |
| + ADWIN + accumulate (ours) | after an ADWIN alarm, refit on every new batch until 6 post-alarm batches are gathered |

**Drift signal.**
- The detectors receive one value per fraud: 1 if it was missed at score < 0.5.
- **ADWIN** uses δ = 0.002 and checks after every value. **DDM** uses its standard settings.
- A detector restarts after each alarm. Alarms where the error *fell* are ignored, because that means the model
  improved.

**Protocol.**
- **Settings frozen on development scenarios:** a sensitivity sweep of accumulate size (3/6/10), retrain
  interval (2/3/5) and ADWIN δ (0.0002/0.002/0.02). The rule set beforehand: keep the default unless another
  value is clearly better on both development datasets. The frozen values are 6, 3 and 0.002.
- **Then evaluated:** 10 scenarios × 3 seeds.
- **Comparisons:** paired moving-block bootstrap (block 3, 5,000 resamples) of the per-batch PR-AUC difference
  after the drift; significant when the 95% CI excludes 0.
- **Supplementary:** 4 fresh scenarios, fixed before any run, for the follow-up analysis.

### 3.5 Results, Question 1: how well do classical models detect fraud?
ULB as recorded; models frozen after the warm-up; mean over 23 batches (notebook 02 §1.1).

| Model | PR-AUC | Operational recall | Operational precision |
|---|---|---|---|
| Random forest | **0.762** (~440× random) | 0.634 | 0.915 |
| Gradient boosting | 0.741 | 0.577 | 0.900 |
| Adaptive random forest (online) | 0.728 | 0.585 | 0.770 |
| Logistic regression | 0.598 | 0.401 | 0.722 |
| Online logistic regression | 0.590 | 0.577 | 0.641 |
| Single-feature rule | 0.588 | 0.444 | 0.768 |

- **The trees lead clearly**, and only they hold the 90% precision target at a carried-over threshold.
- **Logistic regression barely beats the one-feature rule.**
- **Imbalance handling** (§1.3) matters, but no method wins everywhere:
  - **ULB:** class weights are best for logistic regression (0.598 vs 0.432 with none). Gradient boosting with no
    handling collapses (0.129 vs 0.741 with weights).
  - **After the drift with accumulate (ULB + drift):** SMOTE is best for logistic regression (0.601) and gradient
    boosting (0.701); class weights are best for random forest (0.733).
  - **Undersampling** is cheapest but weakest after drift: random forest 0.535 vs 0.733.
  - **Class weights are the default** because they are never far from the best.

### 3.6 Results, Question 2: degradation without maintenance
- **Reported vs deployed** (§2.1, figure `creditcard_drift50_reported_vs_deployed.png`). Each model is trained
  once on the first 40% of ULB + drift:

  | Model | Reported, random split | Reported, time-ordered split | Deployed before drift | Deployed after drift |
  |---|---|---|---|---|
  | Random forest | 0.888 | 0.813 | 0.715 | **0.069** |
  | Gradient boosting | 0.883 | 0.768 | 0.789 | **0.006** |
  | Logistic regression | 0.673 | 0.710 | 0.734 | **0.004** |

  A shuffled split is also optimistic for the trees even before the drift (0.888 reported vs 0.715 deployed).
- **Degradation** (§2.3, development scenario, seed 42). Frozen models:
  - logistic regression 0.640 → 0.005 (−99%)
  - random forest 0.760 → 0.156 (−79%)
  - gradient boosting 0.777 → 0.016 (−98%)

  With periodic retraining the losses are 22%, none (−1%) and 10%.
- **Without injected drift** (§2.4): only mild decay, −0.03 to −0.07 PR-AUC per 10 batches for static models.
  ULB covers only 48 hours, which is why drift is injected.
- **Severity** (§2.5, figure `severity.png`): random forest's loss grows with the number of swapped pairs
  (0%, 9%, 79%, 90%, 95% for k = 1, 2, 4, 6, 8). Accumulate restores it to 0.70–0.76 at every level.

### 3.7 Results, Question 3: which adaptive strategy works best?
**Overall ranking**, held-out scenarios, mean of 3 seeds (notebook 02, "Overall ranking"):

| Strategy | Post-drift PR-AUC | Operational recall | False alarms per run |
|---|---|---|---|
| **Periodic retraining** | **0.649** | **0.504** | 0 |
| DDM warning window | 0.596 | 0.396 | 0.28 |
| + DDM | 0.591 | 0.404 | 0.22 |
| + ADWIN + accumulate (ours) | 0.562 | 0.431 | 0 |
| + ADWIN | 0.464 | 0.306 | 0 |
| Online logistic regression | 0.432 | 0.333 | 0 |
| Frozen (reference) | 0.161 | 0.070 | 0 |

- **Best combination: random forest + periodic retraining**, 0.748 held-out post-drift PR-AUC. Gradient
  boosting reaches 0.675 and logistic regression 0.522 with the same strategy.
- **Gain from adaptation:** up to +0.43 operational fraud recall over the frozen model (§3.3).
- **Paired comparisons of accumulate** (§3.2, figure `protocol_comparisons.png`), 18 held-out comparisons each:

  | Accumulate vs | Better | Not significant | Worse |
  |---|---|---|---|
  | + ADWIN (refit on alarm batch) | **14** | 4 | 0 |
  | Periodic retraining | 0 | 12 | **6** |
  | DDM | 5 | 11 | 2 |
  | DDM warning window | 5 | 11 | 2 |

- **Development vs held-out:** on the development scenarios accumulate had the best mean (0.665); on held-out
  scenarios it fell to 0.562. The held-out protocol is what revealed this.
- **Alarms** (§3.1): ADWIN raised **no false alarms in any scenario**. DDM raised false alarms on logistic
  regression (about 1 per run).
- **Label delay** (§3.5): every strategy loses 0.1–0.2 PR-AUC with a 3-batch delay; for example, random forest
  with periodic retraining drops from 0.77 to 0.61.
- **Recovery** (§3.4, single runs): random forest with ADWIN or accumulate is back within 1 batch; periodic
  retraining takes 5. Frozen models never recover.
- **Cost** (§3.6): periodic retraining is the most expensive (random forest 29.7 s vs 15.8 s for accumulate);
  for these models the difference is seconds.

### 3.8 Analysis: why do detector-triggered strategies trail? (notebook 02 §6)
**Where the gap lies.** Accumulate minus periodic retraining, per post-drift batch, split by phase:

| Phase | Batches | Difference |
|---|---|---|
| Waiting for the alarm | 210 | **−0.279** |
| ADWIN never fired | 84 | **−0.341** |
| Accumulating after the alarm | 375 | **+0.058** |
| After accumulating | 501 | 0.000 |

The whole deficit is the waiting time; once it reacts, accumulate is ahead.

**Why the wait is long.** ADWIN learns from one value per fraud, ~13 per batch.

| Frauds in the 3 batches after the drift | ADWIN delay |
|---|---|
| 96 (scenarios at 50%, a fraud burst) | ~1 batch |
| 35–43 (drift at 70% / 30%) | 3–4 batches |
| 25 (gradual) | ~8 batches |

A moderate rise in the miss rate (logistic regression, 0.16 → 0.34–0.37 after 2- or 6-pair swaps) is never
detected. Class-imbalance handling cannot fix this: it helps the model *learn*, but it cannot create more real
fraud labels for the detector.

**A tested fix that did not help.** "Periodic + ADWIN reset" retrains on schedule, and on an alarm restarts its
window on post-drift data. On the 4 fresh scenarios it was level with periodic retraining (−0.003 on average).
It looked slightly better (+0.015) only on the scenarios it was designed from, so it was not adopted.

**The fresh scenarios confirm the main result:** accumulate is 0.13 behind periodic retraining (5 of 12
comparisons significantly worse, none better).

**Would a different drift signal help?** Offline check on frozen models (§3.1, `detector_signals.csv`):
- **Fraud scores instead of caught/missed** help only slightly.
- **Label-free signals** (alert rate or mean score over all transactions):
  - they catch every drift on logistic regression within 0–3 batches
  - they are slower for trees on abrupt drift
  - they raise 1–3 false alarms on undrifted data
- **No signal is best everywhere.**

### 3.9 Error analysis and explainability (notebook 02 §4–5)
**Error analysis.** Random forest on the development scenario, at the operational threshold.
- **Before the drift:** 68 of 89 frauds caught (recall 0.76), precision 0.88, 11 alerts per 10,000
  transactions, 85% of the fraud amount.
- **After the drift, frozen:** 0 of 199 frauds caught and 1 alert in 134,807 transactions. It fails silently.
- **After the drift, accumulate:** 91 of 199 frauds (precision 0.85, 51% of the amount).
- **After the drift, DDM:** 106 of 199 frauds (precision 0.89, 56% of the amount).
- **Note:** one batch of detection delay can decide a third of the post-drift frauds (one batch held 63
  frauds).

**SHAP** (figure `creditcard_drift50_shap.png`).
- **Before the drift** the model relies on V10 (0.081), V14 (0.079), V12 (0.047) and V17 (0.032).
- **After retraining** it uses V22 (0.069), V25 (0.064) and V15 (0.044), exactly the columns these signals were
  swapped into.
- **The frozen model keeps relying on the old columns,** which now hold weak features. That explains its
  collapse.
- **On synthetic data** the importance moves from V1–V3 to V4–V6, matching the generator.

### 3.10 Discussion and recommendation
- **Retrain fraud models on a fixed schedule** on recent data. It was the most reliable strategy on held-out
  and fresh scenarios.
- **Use drift detectors as alarms.** ADWIN never raised a false alarm, so it is well suited to telling analysts
  that something changed, but it should not be the only trigger for retraining when labels are rare.
- **Our accumulate variant fixes refit-on-alarm,** so it is the better choice *if* retraining must be
  detector-triggered (for example, when retraining is expensive).

### 3.11 Limitations
- **The drift is artificial.** It is injected into one 48-hour dataset from 2013 (one issuer, PCA-anonymised).
  There is no natural drift.
- **Fraud is scarce in every batch:** ~17 per batch, so per-batch scores are noisy, and a single batch of delay
  matters.
- **Label delay was tested only at 1–3 batches** (hours). Real fraud labels arrive days to weeks later.
- **Seeds change only model randomness,** not the data. The CIs mainly capture batch-to-batch variation.
- **Some analyses are single runs** (seed 42): imbalance, severity, recovery, error analysis and SHAP.
- **The oracle metric is an upper bound;** the operational metrics are the realistic ones.

### 3.12 Future work
- A natural-drift dataset, such as IEEE-CIS (the loader supports it; see `IMPLEMENTATION_PLAN.md` Task B).
- Combining label-based and label-free drift signals to detect sooner without false alarms.
- Longer, realistic label delays; recurring drift.
- Adaptive retraining schedules (retrain more often when a detector warns).

### 3.13 Conclusion (draft)
> Without maintenance, fraud models fail silently after a drift while their test scores look healthy. Among
> the adaptive strategies compared, scheduled retraining on recent data was the most reliable, raising
> post-drift PR-AUC from 0.16 to 0.65 (0.75 with random forest). Detector-triggered strategies, including our
> improved accumulate-after-drift variant, trail it because drift detectors depend on rare fraud labels and
> react late. Drift detectors are best used as monitoring alarms alongside scheduled retraining.

---

## 4. Figures and tables

| Use for | File (in `results/figures/`) or notebook cell |
|---|---|
| The hidden problem | `creditcard_drift50_reported_vs_deployed.png` |
| Strategies over time | `creditcard_drift50_pr_auc_over_time.png`, `synthetic_pr_auc_over_time.png` |
| Ranking of strategies | notebook 02, "Overall ranking" table |
| Significance | `protocol_comparisons.png` (forest plot of paired 95% CIs) |
| Severity | `severity.png` |
| Why detectors trail | notebook 02 §6.1 tables (gap by phase, delay vs frauds) |
| Imbalance handling | `creditcard_drift50_imbalance.png` |
| Explainability | `creditcard_drift50_shap.png`, `synthetic_shap.png` |
| No-drift baseline | `creditcard_pr_auc_over_time.png` (ADWIN stays quiet) |

---

## 5. Wording rules

**Do:**
- **Lead with the strong results:** the hidden collapse, 440× random, 0.16 → 0.65, random forest +
  periodic 0.75, the rigour.
- **Call periodic retraining "the most reliable strategy"** and the detectors "useful monitoring alarms".
- **Use this sentence for our variant:** *"Accumulate-after-drift fixes the main weakness of detector-triggered
  retraining (refitting on a single batch) and improves it significantly (14 of 18 held-out comparisons), but it
  inherits the detector's delay, so scheduled retraining remained more reliable."*
- **Say "compared with a frozen model"** whenever you report a gain from a detector.

**Do not:**
- Claim accumulate (or any detector-based strategy) beats periodic retraining.
- Quote development-scenario numbers as the main result.
- Drop scenarios or results.
- Use accuracy as a metric.

---

## 6. Reproducibility (for the appendix)
- **Code:** https://github.com/d1ya-m/fraud-drift-detection
- **Environment:** Python 3.13; `pip install -r requirements-lock.txt`; 89 automated tests (`pytest -q`).
- **Commands:** `docs/IMPLEMENTATION_PLAN.md` section 5.
- **Main runtimes:** experiments ~20 min; protocol ~3 h; supplementary ~1.5 h; all others minutes.
- **Single presentation notebook:** `notebooks/fraud_drift_project.ipynb` contains the whole project: code,
  EDA, experiments (switched off by default; it loads the saved results) and the analysis.

---

## 7. References (verify each before citing; check the exact pages in the publisher's version)

**Peer-reviewed / standard references:**
1. Bifet, A., & Gavaldà, R. (2007). Learning from time-changing data with adaptive windowing. *Proc. SIAM
   International Conference on Data Mining (SDM)*. (ADWIN)
2. Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004). Learning with drift detection. *Advances in
   Artificial Intelligence, SBIA 2004*, LNCS 3171. (DDM and the warning window)
3. Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift
   adaptation. *ACM Computing Surveys*, 46(4).
4. Ditzler, G., Roveri, M., Alippi, C., & Polikar, R. (2015). Learning in nonstationary environments: A survey.
   *IEEE Computational Intelligence Magazine*, 10(4), 12–25. (active vs passive adaptation)
5. Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., & Bontempi, G. (2015). Credit card fraud detection and
   concept-drift adaptation with delayed supervised information. *IJCNN 2015*. DOI 10.1109/IJCNN.2015.7280527
6. Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., & Bontempi, G. (2018). Credit card fraud detection: A
   realistic modeling and a novel learning strategy. *IEEE Transactions on Neural Networks and Learning
   Systems*, 29(8).
7. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with
   undersampling for unbalanced classification. *IEEE SSCI 2015*. (the citation requested for the ULB dataset)
8. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority
   over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
9. Gomes, H. M., et al. (2017). Adaptive random forests for evolving data stream classification. *Machine
   Learning*, 106.
10. Gama, J., Sebastião, R., & Rodrigues, P. P. (2013). On evaluating stream learning algorithms. *Machine
    Learning*, 90. (prequential evaluation)
11. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when
    evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3).
12. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*;
    Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for
    trees. *Nature Machine Intelligence*, 2. (SHAP, TreeExplainer)
13. Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of
    Statistics*, 17(3). (moving-block bootstrap)
14. Software: Pedregosa et al. (2011), scikit-learn, *JMLR* 12; Montiel et al. (2021), River, *JMLR* 22;
    Lemaître et al. (2017), imbalanced-learn, *JMLR* 18.

**Recent comparisons that agree with our main result.** These are not peer-reviewed: read them first, and cite
them as preprints or software only.
- "When to Retrain: An Empirical Study of Retraining Policies…", arXiv 2608.19488. Periodic retraining was
  strongest under abrupt drift; ADWIN triggered no retrains under gradual drift.
- "Cost-Aware Adaptive Credit Card Fraud Detection Under Concept Drift", Research Square preprint
  rs-11398009. Frequent retraining had the best F1; ADWIN-only had the lowest modelled cost.
- MOLT benchmark (GitHub, ShashankScript/MOLT). Credit-card fraud with injected drift: periodic retraining
  recovered more than ADWIN-triggered retraining, and the detector took ~3.4 windows to fire.
