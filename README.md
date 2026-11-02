# Keeping Credit-Card Fraud Detectors Reliable Under Concept Drift: A Comparison of Adaptive Model-Maintenance Strategies
Classical (non-deep) ML for detecting fraudulent card transactions, and a comparison of the strategies that keep
such a model reliable as fraud patterns drift over time: periodic retraining, online learning and
drift-detector-triggered retraining (ADWIN, DDM, DDM warning window, and our accumulate-after-drift variant).

**Main result:** without maintenance, models lose 79–99% of PR-AUC after a drift while their test scores still
look excellent. Periodic retraining is the most reliable fix (held-out post-drift PR-AUC 0.65 vs 0.16 frozen;
random forest 0.75). Detector-triggered strategies trail it because the detectors learn from rare fraud labels
and react late.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run (works immediately on synthetic data, no download needed)
```bash
python experiments/run_experiment.py
```

## Notebooks
- **`notebooks/fraud_drift_project.ipynb`: the whole project in one notebook** (all code, EDA, experiments and
  analysis). Run All takes about a minute: experiments are switched off and load the saved results; switch
  them on to re-run (same code and seeds, same results)
- `notebooks/01_eda.ipynb`: exploratory analysis of the ULB data
- `notebooks/02_results_analysis.ipynb`: all results, organised by three research questions

Part F at the end of the single notebook is a live demo: two fresh runs with random forest (about 1 and 2
minutes) that show the model collapsing after drift and the adaptive strategies recovering. They write to
`demo_out/`, not `results/`.

**Results dashboard:** https://claude.ai/artifact/WrKbujCj6esZtvjbxrt3WZ, interactive charts of the saved results
(shared with the team).

## Test
```bash
pytest -q
```

## Interactive Web Dashboard
An interactive dashboard with two dedicated sections:
1. **Section 1: Live Inference Lab**: Test real-time transactions, tweak sliders/inputs for all 29 features (`Amount`, `V1`..`V28`), run preset scenarios, adjust decision thresholds, and see side-by-side diagnostic comparisons of how concept drift blinds unmaintained models while periodic adapted models catch it.
2. **Section 2: Research & Benchmark Results Hub**: Non-text-heavy presentation of the research results referenced in the project guide/artifact: executive KPI scorecards, interactive charts, strategy leaderboards, the waiting gap analysis, SHAP explainability, and error analysis.

### Launch the Dashboard
```bash
python run_dashboard.py
# Or on Windows, double-click: start_dashboard.bat
```
Navigate to `http://127.0.0.1:8050` in your web browser.

## Data
See [data/README.md](data/README.md) to download the real datasets into `data/raw/`.

<<<<<<< HEAD
See [PIPELINE.md](PIPELINE.md) for the full build plan.

=======
See [PIPELINE.md](PIPELINE.md) for the full build plan, and
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the remaining tasks and how to pick one up.
>>>>>>> 3d82e4a63c44558e0f4920b13ba0550bfc046782
