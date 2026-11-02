#!/usr/bin/env python3
"""Launcher for the Interactive Fraud Detection & Concept Drift Dashboard."""
import sys
import webbrowser
from server import run_server

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    print(f"\n========================================================")
    print(f"🚀 Concept-Drift-Aware Fraud Detection Dashboard")
    print(f"🌐 Access URL: http://127.0.0.1:{port}")
    print(f"========================================================\n")
    try:
        webbrowser.open(f"http://127.0.0.1:{port}")
    except Exception:
        pass
    run_server(port)
