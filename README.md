# Concept-Drift-Aware Credit-Card Fraud Detection
Classical (non-deep) ML for detecting fraudulent card transactions that keeps
up as fraud patterns drift over time.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run (works immediately on synthetic data, no download needed)
```bash
python experiments/run_experiment.py
```

## Test
```bash
pytest -q
```

## Data
See [data/README.md](data/README.md) to download the real datasets into `data/raw/`.

See [PIPELINE.md](PIPELINE.md) for the full build plan.
