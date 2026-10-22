# 🛣️ The City Fixer — Intelligent Road Damage Surveillance & Mapping System

> **An End-to-End AI-Powered Municipal Infrastructure Platform** combining YOLOv8 computer vision, DBSCAN geospatial hotspot analytics, real-time dashcam telemetry processing, and interactive GIS mapping to automate municipal road maintenance and civil repair prioritization.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-blueviolet)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9.4-199900?logo=leaflet&logoColor=white)](https://leafletjs.com/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-DBSCAN-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![ReportLab](https://img.shields.io/badge/ReportLab-PDF_Engine-CC0000)](https://www.reportlab.com/)

---

## 📌 Project Overview

Traditional municipal road condition monitoring is manual, slow, and reactive—often relying on citizen complaints after vehicular damage or accidents occur. **The City Fixer** automates this lifecycle end-to-end:

1. **Computer Vision Inference (YOLOv8)**: Detects road cavities, transverse cracks, and alligator cracking from vehicle dashcam streams.
2. **Dynamic Severity Ranking**: Ranks hazards automatically into **Tier 1 (Severe)**, **Tier 2 (Moderate)**, and **Tier 3 (Minor)** based on class damage severity and bounding box physical footprint.
3. **Geospatial Hotspot Clustering (DBSCAN)**: Uses spherical Haversine distance clustering to locate high-density hazard zones requiring immediate civil resurfacing.
4. **Operations Dashboard**: Full-featured Leaflet.js and Tailwind CSS dark dashboard displaying live telemetry, cluster radii, metrics, and inspection modals.
5. **Municipal PDF Brief**: Automated generation of official, downloadable briefs for city officials and maintenance crews.

---

## 🏛️ System Architecture

```
                                  +---------------------------+
                                  | Dashcam Video / Camera    |
                                  +-------------+-------------+
                                                |
                                                v
                                  +---------------------------+
                                  | video_processor.py        |
                                  |  - OpenCV Frame Extractor |
                                  |  - YOLOv8 Object Detector |
                                  |  - GPS Telemetry Embedder |
                                  +-------------+-------------+
                                                | HTTP POST JSON
                                                v
+-----------------------------------------------------------------------------------------+
|                               Flask Backend (backend/app.py)                           |
|                                                                                         |
|   +-----------------------+     +------------------------+    +-----------------------+ |
|   | Severity Classifier   |     | SQLite Database Store  |    | DBSCAN Hotspot Engine | |
|   | (Area & Class Heuristics)   | (city_fixer.db)        |    | (clustering.py)       | |
|   +-----------------------+     +------------------------+    +-----------------------+ |
|                                                                                         |
|   Endpoints:                                                                            |
|   * GET  /api/detections       * POST /api/detections       * GET  /api/clusters        |
|   * GET  /api/report           * GET  /api/generate-report   * POST /api/seed            |
+-------------------------------------------+---------------------------------------------+
                                            |
                    +-----------------------+-----------------------+
                    |                                               |
                    v                                               v
+---------------------------------------+       +---------------------------------------+
| Interactive Web Dashboard             |       | Municipal PDF Brief                   |
| (templates/index.html)                |       | (backend/report_generator.py)         |
|  - Leaflet GIS OpenStreetMap Map      |       |  - Executive Summary Statistics       |
|  - DBSCAN Hotspot Radii Zones         |       |  - High-Density Cluster Hotspot Table |
|  - Real-Time Telemetry Feed & Filters |       |  - Road Health Index & Budget Plans   |
|  - Damage Inspection & Crew Dispatch  |       |  - Downloadable Vector PDF            |
+---------------------------------------+       +---------------------------------------+
```

---

## 🛠️ Tech Stack & Dependencies

| Layer | Technologies | Role in System |
| :--- | :--- | :--- |
| **AI / Computer Vision** | YOLOv8 (Ultralytics), OpenCV, PyTorch | Object detection on dashcam footage (potholes, cracks) |
| **Backend & REST API** | Python, Flask, Flask-CORS, SQLite3 | REST endpoints, database persistence, severity classifier |
| **Spatial Analytics** | scikit-learn (`DBSCAN`), NumPy | Haversine spherical geospatial clustering & density hotspots |
| **Report Generation** | ReportLab | Executive PDF report compilation & statistical brief export |
| **Frontend UI** | Leaflet.js, Tailwind CSS, FontAwesome 6 | Interactive GIS mapping, telemetry feeds, and metrics |

---

## 📁 Repository Structure

```
ai_ml_workspace/
├── backend/
│   ├── app.py                  # Production Flask REST API server
│   ├── clustering.py           # DBSCAN Haversine hotspot clustering module
│   ├── report_generator.py     # ReportLab municipal PDF brief engine
│   └── city_fixer.db           # SQLite database (auto-generated)
├── templates/
│   └── index.html              # Leaflet.js + Tailwind CSS operations dashboard
├── datasets/                   # RDD2022 dataset directory (train/val images & labels)
├── data.yaml                   # YOLOv8 custom dataset configuration
├── train.py                    # Model training script with GPU/CPU toggle
├── infer_and_send.py           # Single-frame inference & REST transmission script
├── video_processor.py          # Continuous dashcam video processor (1 FPS sampling)
├── requirements.txt            # Pinned dependency definitions
└── README.md                   # Complete project documentation
```

---

## ⚡ Installation & Environment Setup

### Option A: Using Conda (Recommended)

```bash
# 1. Create a clean Conda environment with Python 3.10
conda create -n cityfixer python=3.10 -y
conda activate cityfixer

# 2. Clone or navigate to the workspace
cd ai_ml_workspace

# 3. Install core dependencies
pip install -r requirements.txt
```

### Option B: Using Standard Python Virtual Environment

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt
```

---

## 🚦 Execution Guide

### 1. Launch the Backend Server & Dashboard

Start the Flask application:
```bash
python backend/app.py
```

Console Output:
```
============================================================
  🏗️  The City Fixer — Road Damage Detection System
  📍  Dashboard : http://127.0.0.1:5000/
  📡  API       : http://127.0.0.1:5000/api/detections
  📊  Clusters  : http://127.0.0.1:5000/api/clusters
  📑  PDF Brief : http://127.0.0.1:5000/api/generate-report
============================================================
```

Open your browser to [**http://127.0.0.1:5000/**](http://127.0.0.1:5000/) to access the dashboard.

---

### 2. Train Custom YOLOv8 Model (Week 2)

Train a custom detector on the RDD2022 dataset:
```bash
# Configure dataset in data.yaml and run:
python train.py
```
*Note: Configured for 15 epochs at 640x640 resolution. Automatically utilizes CUDA GPU if detected; otherwise falls back gracefully to CPU.*

---

### 3. Run Single Image Inference & Send Telemetry (Week 2)

Run inference on an arbitrary road frame and stream the structured JSON payload to the backend:
```bash
python infer_and_send.py --image path/to/sample_road.jpg --weights best.pt
```

---

### 4. Process Dashcam Video Streams (Week 3)

Run the OpenCV video processor at a controlled interval (1 frame per second):

```bash
# Process a real dashcam .mp4 file:
python video_processor.py --video dashcam_punjab.mp4 --weights best.pt --interval 1.0

# Or run the built-in virtual dashcam route simulator (no video file required):
python video_processor.py --simulate
```

---

### 5. Generate DBSCAN Geospatial Clusters (Week 3)

Test the clustering module independently from the command line:
```bash
python backend/clustering.py
```

---

### 6. Generate & Export Municipal Reports (Week 4 Stretch)

1. **Download Official PDF Report**:
   - Navigate to [http://127.0.0.1:5000/api/generate-report](http://127.0.0.1:5000/api/generate-report) in your browser.
   - Or click the red **"PDF Brief"** button in the dashboard top navigation bar.
2. **Download CSV Telemetry**:
   - Navigate to [http://127.0.0.1:5000/api/report?format=csv](http://127.0.0.1:5000/api/report?format=csv).
3. **CLI Standalone PDF Generation**:
   ```bash
   python backend/report_generator.py
   # Generates: Municipal_Road_Health_Brief.pdf
   ```

---

## 📡 REST API Reference

### 1. `GET /api/detections`
Retrieve all stored road damage detections.
* **Query Parameters:**
  * `severity` (*optional*): Filter by tier (`Tier 1`, `Tier 2`, `Tier 3`).
  * `limit` (*optional*): Maximum results to retrieve (default: `500`).
* **Response Example:**
  ```json
  {
    "count": 33,
    "detections": [
      {
        "detection_id": "8a97ac42-6564-4170-a6d1-27c6eafbd63e",
        "class_name": "pothole",
        "bbox": [100.0, 150.0, 320.0, 380.0],
        "confidence": 0.94,
        "gps": { "latitude": 32.5735, "longitude": 74.0762 },
        "severity": "Tier 1",
        "timestamp": "2026-10-17T01:00:00Z"
      }
    ]
  }
  ```

---

### 2. `POST /api/detections`
Ingest new detection telemetry from dashcams or edge inference devices.
* **JSON Payload (Single or Array):**
  ```json
  {
    "class_name": "pothole",
    "bbox": [120, 200, 310, 360],
    "confidence": 0.92,
    "gps": { "latitude": 32.5741, "longitude": 74.0769 },
    "timestamp": "2026-10-17T01:12:00Z"
  }
  ```
* **Status:** `201 Created`

---

### 3. `GET /api/clusters`
Compute and return DBSCAN spatial clusters (hotspots) from all stored GPS coordinates.
* **Query Parameters:**
  * `eps` (*optional*): Distance threshold in meters (default: `120.0`).
  * `min_samples` (*optional*): Minimum hazard count to form a cluster (default: `2`).
* **Response Example:**
  ```json
  {
    "total_detections": 33,
    "total_clusters": 1,
    "clustered_hazards_count": 8,
    "isolated_hazards_count": 25,
    "clusters": [
      {
        "cluster_id": 1,
        "center": { "latitude": 32.57385, "longitude": 74.07662 },
        "radius_meters": 97.7,
        "total_hazards": 8,
        "dominant_severity": "Tier 1",
        "average_severity_score": 2.75,
        "priority": "CRITICAL (Immediate Maintenance)",
        "severity_breakdown": { "Tier 1": 6, "Tier 2": 2, "Tier 3": 0 }
      }
    ]
  }
  ```

---

### 4. `GET /api/generate-report`
Generates and streams the official vector PDF brief (*Municipal Road Health & Maintenance Brief*) formatted for city commissioners.

---

### 5. `GET /api/report`
Returns high-level degradation statistics in JSON or CSV format (`?format=csv`).

---

## 👥 Team Contributions Breakdown

| Member / Role | Focus Area | Deliverables |
| :--- | :--- | :--- |
| **Lead AI & Computer Vision Engineer** | Detection Model & Video Pipeline | Dataset curation (`data.yaml`), YOLOv8 fine-tuning (`train.py`), frame sampler & dashcam inference (`video_processor.py`). |
| **Backend & Systems Architect** | API, Persistence & Severity Engine | Flask REST API (`app.py`), SQLite schema, severity heuristic algorithms, automated telemetry ingestion. |
| **Data Scientist & Spatial Engineer** | Geospatial Clustering & Analytics | Haversine DBSCAN algorithm (`clustering.py`), spatial centroid calculations, density rankings, ReportLab PDF synthesis (`report_generator.py`). |
| **Full-Stack & Frontend Engineer** | Dashboard UI/UX & Mapping | Tailwind CSS dark operations dashboard (`index.html`), Leaflet.js GIS map with hotspot rings, inspection modal, telemetry tables. |

---

## 📜 License

This project is licensed under the **MIT License**. Free for academic, municipal, and commercial research.
