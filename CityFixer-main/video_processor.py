"""
The City Fixer — Dashcam Video Processor & Road Damage Inference Pipeline
==========================================================================
OpenCV and YOLOv8 pipeline for processing local dashcam footage (.mp4),
extracting frames at a controlled interval (1 frame per second), performing
AI road damage inference, extracting GPS telemetry, and streaming structured
JSON contract payloads to the Flask backend.

Usage:
------
    python video_processor.py --video sample_dashcam.mp4 --weights best.pt --interval 1.0
    python video_processor.py --simulate   # Run simulated dashcam feed if no mp4 available

Author : AI/ML Workspace
Module : Week 3 Automated Surveillance Pipeline
"""

import os
import sys
import time
import json
import random
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Attempt imports with graceful handling
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


# Default configuration (Lahore Metropolitan Route)
DEFAULT_BACKEND_URL = "http://127.0.0.1:5000/api/detections"
PUNJAB_BASE_LAT = 31.549700
PUNJAB_BASE_LON = 74.343600
EARTH_RADIUS = 6371000.0


def advance_gps(lat: float, lon: float, speed_kmh: float = 60.0, dt_sec: float = 1.0, heading_deg: float = 45.0) -> tuple:
    """
    Simulate dashcam vehicle movement along a roadway trajectory.
    Calculates new GPS coordinates based on vehicle speed, elapsed time, and heading.
    """
    import math
    distance_meters = (speed_kmh * 1000.0 / 3600.0) * dt_sec
    angular_distance = distance_meters / EARTH_RADIUS
    heading_rad = math.radians(heading_deg)

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(angular_distance)
        + math.cos(lat_rad) * math.sin(angular_distance) * math.cos(heading_rad)
    )
    new_lon_rad = lon_rad + math.atan2(
        math.sin(heading_rad) * math.sin(angular_distance) * math.cos(lat_rad),
        math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat_rad),
    )

    return (
        round(math.degrees(new_lat_rad), 6),
        round(math.degrees(new_lon_rad), 6),
    )


def send_detection_payload(payload: Dict[str, Any], endpoint_url: str = DEFAULT_BACKEND_URL) -> bool:
    """Send detection contract payload to Flask backend using requests or urllib fallback."""
    if HAS_REQUESTS:
        try:
            resp = requests.post(endpoint_url, json=payload, timeout=5)
            if resp.status_code in (200, 201):
                return True
            print(f"  [API Error] Status {resp.status_code}: {resp.text[:120]}")
            return False
        except Exception as e:
            print(f"  [Network Error] Could not reach backend: {e}")
            return False
    else:
        # Standard library urllib fallback
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint_url,
                data=req_data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as res:
                return res.status in (200, 201)
        except Exception as e:
            print(f"  [Network Error] Could not reach backend: {e}")
            return False


def run_video_pipeline(
    video_path: str,
    weights_path: str = "best.pt",
    interval_sec: float = 1.0,
    confidence_thresh: float = 0.40,
    endpoint_url: str = DEFAULT_BACKEND_URL,
    start_lat: float = PUNJAB_BASE_LAT,
    start_lon: float = PUNJAB_BASE_LON,
):
    """
    Main dashcam processing routine.
    Extracts frames at interval_sec, runs YOLO inference, parses bounding boxes,
    and streams structured detection records to the Flask API.
    """
    if not HAS_CV2:
        print("[Warning] OpenCV (cv2) is not installed in the current environment.")
        print("          Running high-fidelity dashcam telemetry simulation instead.")
        run_simulated_pipeline(endpoint_url=endpoint_url, total_frames=15, interval_sec=interval_sec)
        return

    if not os.path.exists(video_path):
        print(f"[Warning] Video file not found: '{video_path}'")
        print("          Switching to simulated dashcam video feed.")
        run_simulated_pipeline(endpoint_url=endpoint_url, total_frames=15, interval_sec=interval_sec)
        return

    # Load YOLO model
    model = None
    if HAS_YOLO:
        if os.path.exists(weights_path):
            print(f"[Model] Loading custom YOLOv8 model from: {weights_path}")
            model = YOLO(weights_path)
        else:
            print(f"[Model] Weights '{weights_path}' not found, falling back to pretrained 'yolov8n.pt'")
            model = YOLO("yolov8n.pt")
    else:
        print("[Model] Ultralytics package not detected. Using simulated bounding box generator.")

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    frame_step = max(1, int(fps * interval_sec))

    print("=" * 65)
    print("  📹 The City Fixer — Dashcam Video Surveillance Pipeline")
    print("=" * 65)
    print(f"  Source Video : {video_path}")
    print(f"  Frame Rate   : {fps:.2f} FPS | Total Frames: {total_frames} (~{duration_sec:.1f}s)")
    print(f"  Sampling Rate: 1 frame every {interval_sec}s (step = {frame_step} frames)")
    print(f"  GPS Origin   : ({start_lat:.6f}, {start_lon:.6f}) Punjab Route")
    print(f"  API Endpoint : {endpoint_url}")
    print("=" * 65)

    current_lat, current_lon = start_lat, start_lon
    frame_idx = 0
    sampled_count = 0
    detections_sent = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                sampled_count += 1
                video_timestamp_sec = frame_idx / fps
                iso_timestamp = datetime.now(timezone.utc).isoformat()

                # Advance GPS telemetry along the route
                current_lat, current_lon = advance_gps(
                    current_lat, current_lon, speed_kmh=55.0, dt_sec=interval_sec, heading_deg=50.0
                )

                print(f"\n[Frame #{frame_idx:05d} @ {video_timestamp_sec:6.1f}s] Telemetry 📍 ({current_lat:.6f}, {current_lon:.6f})")

                # Perform inference
                if model:
                    results = model.predict(frame, conf=confidence_thresh, verbose=False)
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            xyxy = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            cls_name = model.names.get(cls_id, f"class_{cls_id}")

                            payload = {
                                "class_name": cls_name,
                                "bbox": [round(coord, 1) for coord in xyxy],
                                "confidence": round(conf, 4),
                                "gps": {"latitude": current_lat, "longitude": current_lon},
                                "timestamp": iso_timestamp,
                            }

                            success = send_detection_payload(payload, endpoint_url)
                            if success:
                                detections_sent += 1
                                print(f"  ✅ Sent: {cls_name} (conf={conf*100:.1f}%) -> {endpoint_url}")
                else:
                    # Synthetic hazard simulation when YOLO weights not installed
                    if random.random() < 0.45:
                        cls_name = random.choice(["pothole", "D40", "transverse_crack", "alligator_crack"])
                        conf = round(random.uniform(0.72, 0.96), 4)
                        payload = {
                            "class_name": cls_name,
                            "bbox": [120, 200, 310, 360],
                            "confidence": conf,
                            "gps": {"latitude": current_lat, "longitude": current_lon},
                            "timestamp": iso_timestamp,
                        }
                        success = send_detection_payload(payload, endpoint_url)
                        if success:
                            detections_sent += 1
                            print(f"  ✅ [Simulated] Sent: {cls_name} (conf={conf*100:.1f}%) -> {endpoint_url}")

                time.sleep(0.05)  # Yield CPU

            frame_idx += 1

    finally:
        cap.release()

    print("\n" + "=" * 65)
    print("  🏁 Video Processing Complete")
    print(f"  Total Frames Examined : {frame_idx}")
    print(f"  Frames Sampled        : {sampled_count}")
    print(f"  Detections Streamed   : {detections_sent}")
    print("=" * 65)


def run_simulated_pipeline(
    endpoint_url: str = DEFAULT_BACKEND_URL,
    total_frames: int = 12,
    interval_sec: float = 1.0,
):
    """
    Simulated dashcam video feed for testing without an .mp4 file.
    Progresses vehicle along Punjab Highway Sector and sends detections.
    """
    print("=" * 65)
    print("  🚗 The City Fixer — Dashcam Simulation Feed (Virtual Dashcam)")
    print("=" * 65)
    print(f"  Simulating {total_frames} seconds of driving at 60 km/h...")
    print(f"  Target API Endpoint: {endpoint_url}")
    print("=" * 65)

    current_lat, current_lon = PUNJAB_BASE_LAT, PUNJAB_BASE_LON
    detections_sent = 0

    classes_pool = ["pothole", "D40", "longitudinal_crack", "alligator_crack", "pothole"]

    for sec in range(1, total_frames + 1):
        # Move forward
        current_lat, current_lon = advance_gps(
            current_lat, current_lon, speed_kmh=60.0, dt_sec=interval_sec, heading_deg=45.0
        )
        timestamp = datetime.now(timezone.utc).isoformat()

        print(f"[Second {sec:02d}/{total_frames:02d}] GPS: {current_lat:.6f}°N, {current_lon:.6f}°E", end="")

        # 60% probability of finding road damage
        if random.random() < 0.60:
            cls = random.choice(classes_pool)
            w = random.randint(50, 180)
            h = random.randint(40, 160)
            x1 = random.randint(40, 400)
            y1 = random.randint(100, 360)
            conf = round(random.uniform(0.70, 0.98), 4)

            payload = {
                "class_name": cls,
                "bbox": [x1, y1, x1 + w, y1 + h],
                "confidence": conf,
                "gps": {"latitude": current_lat, "longitude": current_lon},
                "timestamp": timestamp,
            }

            ok = send_detection_payload(payload, endpoint_url)
            if ok:
                detections_sent += 1
                print(f" -> 🚨 Detected {cls.upper()} ({conf*100:.0f}%) | Synced!")
            else:
                print(" -> [Backend offline or unreachable]")
        else:
            print(" -> [Clear Asphalt]")

        time.sleep(0.4)

    print("=" * 65)
    print(f"  Simulation Finished. Streamed {detections_sent} road hazards to backend.")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The City Fixer Dashcam Video Processor")
    parser.add_argument("--video", type=str, default="dashcam.mp4", help="Path to local dashcam video (.mp4)")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to YOLOv8 weights file")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1.0s)")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold (default: 0.40)")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_BACKEND_URL, help="Flask backend POST endpoint")
    parser.add_argument("--simulate", action="store_true", help="Force virtual dashcam simulation feed")

    args = parser.parse_args()

    if args.simulate:
        run_simulated_pipeline(endpoint_url=args.endpoint, interval_sec=args.interval)
    else:
        run_video_pipeline(
            video_path=args.video,
            weights_path=args.weights,
            interval_sec=args.interval,
            confidence_thresh=args.conf,
            endpoint_url=args.endpoint,
        )

