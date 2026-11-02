"""Backend Server for the Interactive Concept-Drift-Aware Fraud Detection Dashboard.
Serves static web files, results benchmarks, and runs real-time inference on ML models.
"""
import http.server
import json
import os
import random
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

WEB_DIR = ROOT / "web"
FIGURES_DIR = ROOT / "results" / "figures"
CACHE_DIR = ROOT / "models_cache"

# Load metadata and models on server startup
print("Loading model metadata and cached models...")
with open(CACHE_DIR / "metadata.json", "r") as f:
    METADATA = json.load(f)

FEATURE_COLS = METADATA["features"]
FEATURE_STATS = METADATA["feature_stats"]
SWAPPED_PAIRS = METADATA["swapped_pairs"]

MODELS = {
    "rf_adapted": joblib.load(CACHE_DIR / "rf_adapted.joblib"),
    "rf_frozen": joblib.load(CACHE_DIR / "rf_frozen.joblib"),
    "gb_adapted": joblib.load(CACHE_DIR / "gb_adapted.joblib"),
    "logreg_adapted": joblib.load(CACHE_DIR / "logreg_adapted.joblib"),
}
print(f"Loaded {len(MODELS)} models into memory ready for real-time inference.")

# Load a compact pool of sample transactions from creditcard.csv for random sample button
SAMPLE_POOL = None
try:
    data_csv = ROOT / "data" / "raw" / "creditcard.csv"
    if data_csv.exists():
        # Load a quick sample
        df_sample = pd.read_csv(data_csv, nrows=10000)
        # Also grab some frauds
        df_all = pd.read_csv(data_csv)
        df_frauds = df_all[df_all["Class"] == 1]
        SAMPLE_POOL = {
            "legitimate": df_sample[df_sample["Class"] == 0][FEATURE_COLS].to_dict(orient="records")[:200],
            "fraud": df_frauds[FEATURE_COLS].to_dict(orient="records"),
        }
        print(f"Loaded sample pool: {len(SAMPLE_POOL['legitimate'])} legitimate and {len(SAMPLE_POOL['fraud'])} frauds.")
except Exception as e:
    print(f"Note: Could not preload sample pool ({e}). Presets will be used.")


def compute_feature_contributions(model_id, input_features, proba):
    """Estimate which features drove the score away from normal baseline."""
    contributions = []
    # Identify key features that deviate significantly from mean in standard deviation units
    for col in FEATURE_COLS:
        val = input_features.get(col, 0.0)
        mean_val = FEATURE_STATS[col]["mean"]
        std_val = max(FEATURE_STATS[col]["std"], 1e-4)
        z_score = (val - mean_val) / std_val
        
        # If this feature is in swapped pairs or known fraud drivers (V14, V10, V12, V17, V22, V23, V25, V15, Amount)
        weight = 1.0
        if col in ["V14", "V10", "V12", "V17"]:
            weight = 2.5
        elif col in ["V22", "V23", "V25", "V15"]:
            weight = 2.2
        elif col == "Amount":
            weight = 1.5

        impact = z_score * weight
        contributions.append({
            "feature": col,
            "value": round(val, 4),
            "z_score": round(z_score, 2),
            "impact": round(impact, 3),
            "direction": "push_fraud" if impact < -1.0 or (col in ["V22", "V23", "V25", "V15"] and abs(impact) > 1.5) else "push_legit"
        })

    # Sort by absolute impact
    contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return contributions[:8]


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler serving static files, figures, and inference API."""

    def end_headers(self):
        # Enable CORS and disable aggressive caching for local dashboard dev
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/styles.css":
            self._serve_file(WEB_DIR / "styles.css", "text/css; charset=utf-8")
        elif path == "/app.js":
            self._serve_file(WEB_DIR / "app.js", "application/javascript; charset=utf-8")
        elif path == "/data.json":
            self._serve_file(WEB_DIR / "data.json", "application/json; charset=utf-8")
        elif path.startswith("/figures/"):
            fig_name = path[len("/figures/") :]
            fig_path = FIGURES_DIR / fig_name
            if fig_path.exists() and fig_path.is_file():
                self._serve_file(fig_path, "image/png")
            else:
                self.send_error(404, f"Figure not found: {fig_name}")
        elif path == "/api/metadata":
            self._send_json(METADATA)
        elif path == "/api/results":
            results_path = WEB_DIR / "data.json"
            with open(results_path, "r") as f:
                self._send_json(json.load(f))
        elif path == "/api/random_sample":
            query = parse_qs(parsed.query)
            category = query.get("category", ["any"])[0]
            sample = self._get_random_sample(category)
            self._send_json(sample)
        else:
            # Fallback to web directory static files
            file_path = WEB_DIR / path.lstrip("/")
            if file_path.exists() and file_path.is_file():
                self._serve_file(file_path)
            else:
                self.send_error(404, f"Path not found: {path}")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)
            try:
                body = json.loads(raw_body.decode("utf-8"))
            except Exception as e:
                self._send_json({"error": f"Invalid JSON payload: {e}"}, status=400)
                return

            response = self._run_inference(body)
            self._send_json(response)
        else:
            self.send_error(404, f"Unknown API endpoint: {parsed.path}")

    def _run_inference(self, payload):
        t0 = time.time()
        model_id = payload.get("model", "rf_adapted")
        threshold = float(payload.get("threshold", 0.50))
        input_data = payload.get("features", {})

        # Ensure all required features are present
        features_vector = []
        for col in FEATURE_COLS:
            val = input_data.get(col, FEATURE_STATS[col]["mean"])
            features_vector.append(float(val))

        X = np.array(features_vector).reshape(1, -1)
        df_input = pd.DataFrame([features_vector], columns=FEATURE_COLS)

        # Primary selected model prediction
        model = MODELS.get(model_id, MODELS["rf_adapted"])
        proba = float(model.predict_proba(df_input)[0, 1])
        is_fraud = bool(proba >= threshold)

        # Concept drift comparison: always compute both Frozen and Adapted probabilities
        p_frozen = float(MODELS["rf_frozen"].predict_proba(df_input)[0, 1])
        p_adapted = float(MODELS["rf_adapted"].predict_proba(df_input)[0, 1])

        # Risk classification
        if proba >= 0.80:
            risk_level = "CRITICAL"
            risk_color = "#ef4444"
        elif proba >= 0.50:
            risk_level = "HIGH RISK"
            risk_color = "#f97316"
        elif proba >= 0.20:
            risk_level = "MODERATE"
            risk_color = "#f59e0b"
        else:
            risk_level = "LOW / SAFE"
            risk_color = "#10b981"

        # Concept drift detection status for this specific transaction
        drift_evasion_detected = bool(p_frozen < 0.30 and p_adapted >= 0.60)
        historical_caught_both = bool(p_frozen >= 0.50 and p_adapted >= 0.50)

        top_contributions = compute_feature_contributions(model_id, input_data, proba)

        latency_ms = round((time.time() - t0) * 1000, 2)

        return {
            "success": True,
            "model_id": model_id,
            "threshold": threshold,
            "probability": round(proba, 4),
            "probability_pct": round(proba * 100, 1),
            "is_fraud": is_fraud,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "latency_ms": latency_ms,
            "comparison": {
                "frozen_model": {
                    "name": "Pre-Drift Frozen RF",
                    "probability": round(p_frozen, 4),
                    "probability_pct": round(p_frozen * 100, 1),
                    "is_fraud": bool(p_frozen >= threshold),
                },
                "adapted_model": {
                    "name": "Periodic Adapted RF (Champion)",
                    "probability": round(p_adapted, 4),
                    "probability_pct": round(p_adapted * 100, 1),
                    "is_fraud": bool(p_adapted >= threshold),
                },
                "drift_evasion_detected": drift_evasion_detected,
                "historical_caught_both": historical_caught_both,
            },
            "top_contributions": top_contributions,
        }

    def _get_random_sample(self, category):
        if not SAMPLE_POOL:
            # Fallback to presets
            preset = random.choice(METADATA["presets"])
            return {"category": preset["category"], "features": preset["values"], "source": "preset"}

        if category == "fraud" or (category == "any" and random.random() < 0.5):
            chosen = random.choice(SAMPLE_POOL["fraud"])
            cat = "Fraud"
        else:
            chosen = random.choice(SAMPLE_POOL["legitimate"])
            cat = "Legitimate"

        return {
            "category": cat,
            "features": {k: round(float(v), 4) for k, v in chosen.items()},
            "source": "live_stream_slice",
        }

    def _serve_file(self, file_path, content_type=None):
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            if content_type:
                self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file {file_path.name}: {e}")

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(port=8050):
    server_address = ("127.0.0.1", port)
    httpd = http.server.ThreadingHTTPServer(server_address, DashboardRequestHandler)
    print(f"\n========================================================")
    print(f"🚀 Concept-Drift Fraud Detection Dashboard LIVE at:")
    print(f"👉 http://127.0.0.1:{port}")
    print(f"========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    run_server(port)
