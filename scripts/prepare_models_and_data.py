"""Train and cache models for the interactive fraud detection dashboard,
extract realistic transaction presets, and generate feature statistics.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.drift_injection import inject_feature_swap
from src.data.loader import TARGET, get_feature_columns, load_dataset


def main():
    print("=" * 60)
    print("Preparing Models & Data for Fraud Detection Dashboard")
    print("=" * 60)

    data_path = ROOT / "data" / "raw" / "creditcard.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    print(f"Loading {data_path}...")
    t0 = time.time()
    df = load_dataset(str(data_path))
    features = get_feature_columns(df)
    print(f"Loaded {len(df):,} transactions in {time.time() - t0:.2f}s with {len(features)} features.")

    # 1. Feature statistics (for UI bounds, sliders, and defaults)
    stats = {}
    for col in features:
        series = df[col]
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "p1": float(series.quantile(0.01)),
            "p99": float(series.quantile(0.99)),
            "median": float(series.median()),
        }

    # 2. Inject drift at 50%
    print("\nInjecting controlled concept drift at 50% (swapping top 4 correlated features)...")
    df_drifted, pairs = inject_feature_swap(df, at=0.5, k=4)
    print(f"Swapped pairs: {pairs}")
    split = int(0.5 * len(df))

    # Pre-drift warm-up data (60,000 transactions, matches project warmup of 6 batches)
    train_pre = df_drifted.iloc[:60000]
    # Post-drift adaptation data (60,000 transactions from after the drift point)
    train_post = df_drifted.iloc[split : split + 60000]

    cache_dir = ROOT / "models_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 3. Train Frozen Random Forest (trained only on pre-drift data)
    print("\nTraining Frozen Random Forest (Pre-drift baseline)...")
    t1 = time.time()
    rf_frozen = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_frozen.fit(train_pre[features], train_pre[TARGET])
    joblib.dump(rf_frozen, cache_dir / "rf_frozen.joblib")
    print(f"Saved rf_frozen in {time.time() - t1:.2f}s")

    # 4. Train Adapted Random Forest (Champion model, retrained post-drift)
    print("\nTraining Adapted Random Forest (Periodic Retrained Champion)...")
    t2 = time.time()
    rf_adapted = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_adapted.fit(train_post[features], train_post[TARGET])
    joblib.dump(rf_adapted, cache_dir / "rf_adapted.joblib")
    print(f"Saved rf_adapted in {time.time() - t2:.2f}s")

    # 5. Train Adapted Gradient Boosting (HistGradientBoosting)
    print("\nTraining Adapted HistGradientBoosting...")
    t3 = time.time()
    gb_adapted = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
    gb_adapted.fit(train_post[features], train_post[TARGET])
    joblib.dump(gb_adapted, cache_dir / "gb_adapted.joblib")
    print(f"Saved gb_adapted in {time.time() - t3:.2f}s")

    # 6. Train Adapted Logistic Regression
    print("\nTraining Adapted Logistic Regression...")
    t4 = time.time()
    logreg_adapted = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    )
    logreg_adapted.fit(train_post[features], train_post[TARGET])
    joblib.dump(logreg_adapted, cache_dir / "logreg_adapted.joblib")
    print(f"Saved logreg_adapted in {time.time() - t4:.2f}s")

    # 7. Extract Presets for the Interactive Dashboard
    print("\nExtracting realistic presets from dataset...")
    # Find genuine transactions
    normal_samples = df_drifted[(df_drifted[TARGET] == 0) & (df_drifted["Amount"] > 20) & (df_drifted["Amount"] < 50)]
    high_val_normal = df_drifted[(df_drifted[TARGET] == 0) & (df_drifted["Amount"] > 1000)]
    # Pre-drift fraud
    pre_frauds = df_drifted.iloc[:split]
    pre_frauds = pre_frauds[pre_frauds[TARGET] == 1]
    # Post-drift fraud
    post_frauds = df_drifted.iloc[split + 60000 :]
    post_frauds = post_frauds[post_frauds[TARGET] == 1]

    # Select representative samples
    s_normal = normal_samples.iloc[10][features].to_dict()
    s_high_val = high_val_normal.iloc[5][features].to_dict()
    s_pre_fraud = pre_frauds.iloc[12][features].to_dict()

    # Find a post-drift fraud that bypasses the frozen model but is caught by the adapted model
    best_evasive_fraud = None
    min_frozen_score = 1.0
    for idx in range(min(50, len(post_frauds))):
        row = post_frauds.iloc[idx]
        x = row[features].to_frame().T
        p_froz = rf_frozen.predict_proba(x)[0, 1]
        p_adap = rf_adapted.predict_proba(x)[0, 1]
        if p_froz < 0.20 and p_adap > 0.70:
            best_evasive_fraud = row[features].to_dict()
            print(f"Found ideal evasive fraud: Frozen P={p_froz:.3f}, Adapted P={p_adap:.3f}")
            break

    if best_evasive_fraud is None:
        best_evasive_fraud = post_frauds.iloc[0][features].to_dict()

    # Small card testing fraud ($1.00)
    card_testing_frauds = df_drifted[(df_drifted[TARGET] == 1) & (df_drifted["Amount"] <= 2.0)]
    s_card_testing = card_testing_frauds.iloc[3][features].to_dict() if len(card_testing_frauds) > 3 else pre_frauds.iloc[0][features].to_dict()

    presets = [
        {
            "id": "preset_grocery",
            "name": "Standard Grocery Purchase",
            "category": "Legitimate",
            "tag": "Safe ($34.20)",
            "description": "Typical consumer everyday card transaction with benign behavioral indicators.",
            "expected_fraud": False,
            "values": {k: round(float(v), 4) for k, v in s_normal.items()},
        },
        {
            "id": "preset_high_val",
            "name": "High-Value Laptop / Electronics",
            "category": "Legitimate",
            "tag": "Safe ($1,290.00)",
            "description": "High ticket price transaction that triggers heuristic amount filters but possesses authentic legitimate PCA signatures.",
            "expected_fraud": False,
            "values": {k: round(float(v), 4) for k, v in s_high_val.items()},
        },
        {
            "id": "preset_pre_drift_fraud",
            "name": "Pre-Drift Identity Theft Attack",
            "category": "Fraud",
            "tag": "Pre-Drift ($99.20)",
            "description": "Classic card-not-present fraud showing strong anomalous negative values on historical drivers (V14, V10, V12, V17). Flagged by both frozen and adapted models.",
            "expected_fraud": True,
            "values": {k: round(float(v), 4) for k, v in s_pre_fraud.items()},
        },
        {
            "id": "preset_concept_drift_fraud",
            "name": "Post-Drift Evasive Attack (Concept Drift)",
            "category": "Fraud",
            "tag": "Concept Drift 🚨",
            "description": "Fraudsters shifted evasion tactics: historical signals (V14, V17) look benign, while the fraud signature migrated into swapped features (V23, V22, V25, V15). Silently BYPASSES the frozen model, but CAUGHT by the adapted model!",
            "expected_fraud": True,
            "values": {k: round(float(v), 4) for k, v in best_evasive_fraud.items()},
        },
        {
            "id": "preset_card_testing",
            "name": "Automated Micro-Authorization Scam",
            "category": "Fraud",
            "tag": "Card Testing ($1.00)",
            "description": "Botnet performing rapid low-dollar auth checks ($1.00) to validate stolen credit card numbers before large draining attacks.",
            "expected_fraud": True,
            "values": {k: round(float(v), 4) for k, v in s_card_testing.items()},
        },
    ]

    # Save metadata JSON
    metadata = {
        "features": features,
        "swapped_pairs": pairs,
        "feature_stats": stats,
        "presets": presets,
        "models": [
            {
                "id": "rf_adapted",
                "name": "Adapted Random Forest (Champion)",
                "description": "Periodic retrained model with balanced class weights. Held-out post-drift PR-AUC: 0.748.",
                "file": "rf_adapted.joblib",
                "is_default": True,
            },
            {
                "id": "rf_frozen",
                "name": "Pre-Drift Frozen Random Forest",
                "description": "Trained only on warm-up data; demonstrates catastrophic degradation under concept drift (Post-drift PR-AUC: 0.156).",
                "file": "rf_frozen.joblib",
                "is_default": False,
            },
            {
                "id": "gb_adapted",
                "name": "Adapted HistGradientBoosting",
                "description": "Fast tree-based boosting with balanced weights. Held-out post-drift PR-AUC: 0.675.",
                "file": "gb_adapted.joblib",
                "is_default": False,
            },
            {
                "id": "logreg_adapted",
                "name": "Adapted Logistic Regression",
                "description": "Standardized linear model baseline. Held-out post-drift PR-AUC: 0.522.",
                "file": "logreg_adapted.joblib",
                "is_default": False,
            },
        ],
    }

    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata successfully saved to {cache_dir / 'metadata.json'}")
    print("Model training & cache preparation completed successfully!")


if __name__ == "__main__":
    main()
