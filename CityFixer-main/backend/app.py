"""
The City Fixer — Road Damage Detection & Mapping System
========================================================
Flask backend providing REST API endpoints for road damage detections,
automatic severity ranking, and report generation.

Author : AI/ML Workspace
Version: 1.0.0
"""

import os
import csv
import io
import json
import uuid
import sqlite3
from datetime import datetime, timezone

from flask import Flask, jsonify, request, render_template, Response
from flask_cors import CORS

# Import DBSCAN clustering engine
try:
    from backend.clustering import cluster_detections
except ImportError:
    from clustering import cluster_detections

# Import PDF Report Generator
try:
    from backend.report_generator import generate_pdf_report
except ImportError:
    from report_generator import generate_pdf_report

# ---------------------------------------------------------------------------
# App Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(os.path.dirname(BASE_DIR), "templates")
DB_PATH = os.path.join(BASE_DIR, "city_fixer.db")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)  # Allow cross-origin requests from any frontend client


# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------

def get_db():
    """Return a new SQLite connection with Row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")  # Better concurrent read perf
    return conn


def init_db():
    """Create the detections table if it does not already exist."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            detection_id   TEXT PRIMARY KEY,
            class_name     TEXT NOT NULL,
            bbox_x1        REAL NOT NULL,
            bbox_y1        REAL NOT NULL,
            bbox_x2        REAL NOT NULL,
            bbox_y2        REAL NOT NULL,
            confidence     REAL NOT NULL,
            latitude       REAL NOT NULL,
            longitude      REAL NOT NULL,
            timestamp      TEXT NOT NULL,
            severity       TEXT NOT NULL,
            raw_json       TEXT
        );
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Severity Ranking Algorithm
# ---------------------------------------------------------------------------
# Tier 1 (Critical)  — Large potholes / D40 damage with high confidence
# Tier 2 (Moderate)  — Smaller potholes or large cracks
# Tier 3 (Minor)     — Small cracks and surface-level wear

# Class names that are treated as "pothole-class" (critical by nature)
POTHOLE_CLASSES = {"pothole", "d40", "d-40", "d40-pothole"}

# Bbox-area thresholds (in px²) — tuned for 640×640 YOLO input images
LARGE_AREA_THRESHOLD = 5000
MEDIUM_AREA_THRESHOLD = 2000


def calculate_severity(class_name: str, bbox: list, confidence: float) -> str:
    """
    Determine a severity tier based on damage class, bounding-box area,
    and model confidence.

    Parameters
    ----------
    class_name  : str   – YOLO class label (e.g. "pothole", "D40", "crack")
    bbox        : list  – [x1, y1, x2, y2] bounding box coordinates
    confidence  : float – Model confidence score (0.0 – 1.0)

    Returns
    -------
    str – One of "Tier 1", "Tier 2", "Tier 3"
    """
    x1, y1, x2, y2 = bbox
    area = abs((x2 - x1) * (y2 - y1))
    is_pothole = class_name.lower().strip() in POTHOLE_CLASSES

    if is_pothole:
        # Potholes are inherently dangerous
        if area >= LARGE_AREA_THRESHOLD or confidence >= 0.80:
            return "Tier 1"   # Critical
        elif area >= MEDIUM_AREA_THRESHOLD or confidence >= 0.60:
            return "Tier 2"   # Moderate
        else:
            return "Tier 3"   # Minor (small / low-confidence pothole)
    else:
        # Cracks and other surface damage
        if area >= LARGE_AREA_THRESHOLD and confidence >= 0.75:
            return "Tier 2"   # Large cracks are still moderate
        else:
            return "Tier 3"   # Minor surface damage


# ---------------------------------------------------------------------------
# Helper: Row → dict
# ---------------------------------------------------------------------------

def row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a DB row back into the JSON contract the frontend expects."""
    return {
        "detection_id": row["detection_id"],
        "class_name": row["class_name"],
        "bbox": [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]],
        "confidence": round(row["confidence"], 4),
        "gps": {
            "latitude": row["latitude"],
            "longitude": row["longitude"],
        },
        "timestamp": row["timestamp"],
        "severity": row["severity"],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    """Serve the Leaflet / Bootstrap dashboard."""
    return render_template("index.html")


@app.route("/api/detections", methods=["GET"])
def get_detections():
    """
    GET /api/detections
    -------------------
    Return all stored detections ordered by most recent first.

    Query parameters (optional):
        severity  – Filter by tier, e.g. ?severity=Tier 1
        limit     – Max number of results, e.g. ?limit=50
    """
    severity_filter = request.args.get("severity")
    limit = request.args.get("limit", default=500, type=int)

    conn = get_db()
    query = "SELECT * FROM detections"
    params = []

    if severity_filter:
        query += " WHERE severity = ?"
        params.append(severity_filter)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    detections = [row_to_dict(r) for r in rows]
    return jsonify({"count": len(detections), "detections": detections}), 200


@app.route("/api/detections", methods=["POST"])
def post_detections():
    """
    POST /api/detections
    --------------------
    Accept one or more detection payloads, auto-calculate severity if
    missing, and persist to the database.

    Body (JSON):
        Single object  – { class_name, bbox, confidence, gps, ... }
        Array          – [ { ... }, { ... } ]
    """
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    # Normalise to a list
    items = payload if isinstance(payload, list) else [payload]
    inserted = []
    errors = []

    conn = get_db()

    for idx, item in enumerate(items):
        try:
            # --- Extract & validate required fields -----------------------
            class_name = item.get("class_name") or item.get("className", "")
            bbox = item.get("bbox", [0, 0, 0, 0])
            confidence = float(item.get("confidence", 0.0))
            gps = item.get("gps", {})
            latitude = float(gps.get("latitude", 0.0))
            longitude = float(gps.get("longitude", 0.0))
            timestamp = item.get("timestamp", datetime.now(timezone.utc).isoformat())
            detection_id = item.get("detection_id", str(uuid.uuid4()))

            if not class_name:
                raise ValueError("class_name is required")
            if len(bbox) != 4:
                raise ValueError("bbox must be [x1, y1, x2, y2]")

            # --- Auto-calculate severity if not provided ------------------
            severity = item.get("severity")
            if not severity:
                severity = calculate_severity(class_name, bbox, confidence)

            # --- Persist --------------------------------------------------
            conn.execute(
                """INSERT OR REPLACE INTO detections
                   (detection_id, class_name,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    confidence, latitude, longitude,
                    timestamp, severity, raw_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    detection_id, class_name,
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    confidence, latitude, longitude,
                    timestamp, severity,
                    json.dumps(item),
                ),
            )
            inserted.append({
                "detection_id": detection_id,
                "severity": severity,
            })

        except Exception as exc:
            errors.append({"index": idx, "error": str(exc)})

    conn.commit()
    conn.close()

    status = 201 if not errors else 207  # Multi-Status if partial errors
    return jsonify({
        "inserted": len(inserted),
        "errors": errors,
        "results": inserted,
    }), status


@app.route("/api/report", methods=["GET"])
def generate_report():
    """
    GET /api/report
    ---------------
    Generate a road-health summary report.

    Query parameters:
        format – "json" (default) or "csv"
    """
    fmt = request.args.get("format", "json").lower()
    conn = get_db()

    # Aggregate stats
    total = conn.execute("SELECT COUNT(*) AS n FROM detections").fetchone()["n"]
    tier_counts = conn.execute(
        "SELECT severity, COUNT(*) AS n FROM detections GROUP BY severity"
    ).fetchall()
    class_counts = conn.execute(
        "SELECT class_name, COUNT(*) AS n FROM detections GROUP BY class_name ORDER BY n DESC"
    ).fetchall()
    avg_conf = conn.execute(
        "SELECT AVG(confidence) AS avg_conf FROM detections"
    ).fetchone()["avg_conf"]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_detections": total,
        "average_confidence": round(avg_conf, 4) if avg_conf else 0.0,
        "severity_breakdown": {r["severity"]: r["n"] for r in tier_counts},
        "class_breakdown": {r["class_name"]: r["n"] for r in class_counts},
    }

    if fmt == "csv":
        # Return all detections as a downloadable CSV
        rows = conn.execute("SELECT * FROM detections ORDER BY timestamp DESC").fetchall()
        conn.close()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "detection_id", "class_name", "bbox_x1", "bbox_y1",
            "bbox_x2", "bbox_y2", "confidence", "latitude",
            "longitude", "timestamp", "severity",
        ])
        for r in rows:
            writer.writerow([
                r["detection_id"], r["class_name"],
                r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"],
                r["confidence"], r["latitude"], r["longitude"],
                r["timestamp"], r["severity"],
            ])

        csv_bytes = output.getvalue()
        return Response(
            csv_bytes,
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=city_fixer_report.csv"
            },
        )

    conn.close()
    return jsonify(report), 200


@app.route("/api/clusters", methods=["GET"])
def get_clusters():
    """
    GET /api/clusters
    -----------------
    Compute and return DBSCAN GPS hotspot clusters for municipal road surveillance.

    Query parameters:
        eps – maximum distance in meters to cluster nearby hazards (default: 120.0)
        min_samples – minimum hazard count to declare a hotspot cluster (default: 2)
    """
    eps = request.args.get("eps", default=120.0, type=float)
    min_samples = request.args.get("min_samples", default=2, type=int)

    conn = get_db()
    rows = conn.execute("SELECT * FROM detections").fetchall()
    conn.close()

    detections = []
    for r in rows:
        detections.append({
            "detection_id": r["detection_id"],
            "class_name": r["class_name"],
            "confidence": r["confidence"],
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "severity": r["severity"],
            "timestamp": r["timestamp"],
        })

    cluster_result = cluster_detections(detections, eps_meters=eps, min_samples=min_samples)
    return jsonify(cluster_result), 200


@app.route("/api/generate-report", methods=["GET"])
def download_pdf_report():
    """
    GET /api/generate-report
    ------------------------
    Generate and stream the official Municipal Road Health & Maintenance Brief (PDF).
    """
    pdf_bytes = generate_pdf_report(db_path=DB_PATH)
    filename = f"Municipal_Road_Health_Brief_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf"
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename={filename}",
            "Content-Type": "application/pdf",
        },
    )


@app.route("/api/seed", methods=["POST"])
def seed_demo_data():
    """
    POST /api/seed
    --------------
    Insert sample detections for demo / testing purposes.
    Generates points around Lahore, Punjab.
    """
    import random

    demo_classes = [
        "pothole", "D40", "longitudinal_crack",
        "transverse_crack", "alligator_crack", "pothole",
    ]
    base_lat, base_lon = 31.5497, 74.3436  # Lahore Metropolitan center (matches frontend map)

    samples = []
    for i in range(25):
        cls = random.choice(demo_classes)
        w = random.randint(30, 200)
        h = random.randint(30, 200)
        x1 = random.randint(0, 440)
        y1 = random.randint(0, 440)
        bbox = [x1, y1, x1 + w, y1 + h]
        conf = round(random.uniform(0.35, 0.97), 4)
        lat = base_lat + random.uniform(-0.08, 0.08)
        lon = base_lon + random.uniform(-0.08, 0.08)

        samples.append({
            "class_name": cls,
            "bbox": bbox,
            "confidence": conf,
            "gps": {"latitude": round(lat, 6), "longitude": round(lon, 6)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # Re-use the POST logic
    with app.test_request_context(
        "/api/detections", method="POST",
        content_type="application/json",
        data=json.dumps(samples),
    ):
        return post_detections()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    print("=" * 60)
    print("  🏗️  The City Fixer — Road Damage Detection System")
    print("  📍  Dashboard : http://127.0.0.1:5000/")
    print("  📡  API       : http://127.0.0.1:5000/api/detections")
    print("  📊  Report    : http://127.0.0.1:5000/api/report")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)

