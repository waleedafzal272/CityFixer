"""
The City Fixer — Road Damage Detection & Mapping System (Root Entrypoint)
========================================================================
Main application entrypoint. Boots the production Flask backend, initializes
the SQLite database store, and serves the high-contrast Light Green
operations dashboard.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app import app, init_db

if __name__ == "__main__":
    init_db()
    print("=" * 60)
    print("  🏗️  The City Fixer — Road Damage Surveillance System")
    print("  📍  Dashboard : http://127.0.0.1:5000/")
    print("  📡  API       : http://127.0.0.1:5000/api/detections")
    print("  📊  Clusters  : http://127.0.0.1:5000/api/clusters")
    print("  📑  PDF Brief : http://127.0.0.1:5000/api/generate-report")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
