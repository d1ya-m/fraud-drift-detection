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
- `notebooks/01_eda.ipynb`: exploratory analysis of the ULB data
- `notebooks/02_results_analysis.ipynb`: all results, organised by three research questions

## Test
```bash
pytest -q
```

## Data
See [data/README.md](data/README.md) to download the real datasets into `data/raw/`.

See [PIPELINE.md](PIPELINE.md) for the full build plan, and
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the remaining tasks and how to pick one up.
