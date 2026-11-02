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

See [PIPELINE.md](PIPELINE.md) for the full build plan.

