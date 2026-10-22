"""
The City Fixer — Inference & Backend Integration Script
========================================================
Run YOLOv8 object detection on road images/video frames,
parse the results, and POST structured detection payloads
to the Flask backend at http://127.0.0.1:5000/api/detections.

Usage
-----
    # Single image
    python infer_and_send.py --source road_photo.jpg

    # Entire folder of images
    python infer_and_send.py --source images/

    # Webcam / live camera feed (press 'q' to stop)
    python infer_and_send.py --source 0

    # Custom weights + custom GPS
    python infer_and_send.py \\
        --weights runs/road_damage/city_fixer_v1/weights/best.pt \\
        --source road.jpg \\
        --lat 31.52 --lon 74.35

Prerequisites
-------------
    pip install ultralytics opencv-python requests numpy

Author : AI/ML Workspace
Version: 1.0.0
"""

import os
import sys
import uuid
import json
import glob
import argparse
from datetime import datetime, timezone

import cv2
import numpy as np
import requests
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

# Flask backend URL
API_ENDPOINT = "http://127.0.0.1:5000/api/detections"

# Model weights — will try custom weights first, then fall back
# to pretrained YOLOv8n (won't know your custom classes but
# demonstrates the pipeline end-to-end).
DEFAULT_WEIGHTS = "runs/road_damage/city_fixer_v1/weights/best.pt"
FALLBACK_WEIGHTS = "yolov8n.pt"

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.30

# Default GPS coordinates (Lahore, Punjab) — override via CLI
# In production these would come from a GPS module or phone API
DEFAULT_LAT = 31.5204
DEFAULT_LON = 74.3587

# How much random jitter to add to GPS for simulating movement
# (only used when --gps-jitter is enabled)
GPS_JITTER = 0.002


# ─────────────────────────────────────────────────────────────
# Class name mapping
# ─────────────────────────────────────────────────────────────
# This maps YOLO class indices to the names your backend expects.
# If using a custom-trained model, these come from data.yaml.
# If using the pretrained fallback (COCO), we only care about
# the pipeline flow — no road-damage classes will be detected.

CUSTOM_CLASS_MAP = {
    0: "pothole",
    1: "crack",
}


# ─────────────────────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────────────────────

def load_model(weights_path: str) -> YOLO:
    """
    Load a YOLOv8 model from the given weights file.
    Falls back to pretrained yolov8n.pt if the custom
    weights are not found.
    """
    if os.path.isfile(weights_path):
        print(f"  ✅ Loading custom weights: {weights_path}")
        return YOLO(weights_path)
    else:
        print(f"  ⚠️  Custom weights not found: {weights_path}")
        print(f"  📦 Falling back to pretrained: {FALLBACK_WEIGHTS}")
        return YOLO(FALLBACK_WEIGHTS)


def parse_detections(results, model, base_lat: float, base_lon: float,
                     jitter: bool = False) -> list[dict]:
    """
    Parse YOLOv8 Results objects into the JSON payload contract
    expected by the Flask backend.

    Parameters
    ----------
    results   : ultralytics Results list from model.predict()
    model     : The YOLO model (for class name lookup)
    base_lat  : Base GPS latitude
    base_lon  : Base GPS longitude
    jitter    : If True, add small random offset to GPS coords

    Returns
    -------
    list[dict] — Detection payloads ready to POST
    """
    detections = []

    for result in results:
        boxes = result.boxes  # ultralytics Boxes object

        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            # ── Bounding box (xyxy format: x1, y1, x2, y2) ───
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            x1, y1, x2, y2 = [round(v, 2) for v in xyxy]

            # ── Confidence score ──────────────────────────────
            conf = float(boxes.conf[i].cpu().numpy())

            # Skip low-confidence detections
            if conf < CONFIDENCE_THRESHOLD:
                continue

            # ── Class ID → class name ─────────────────────────
            cls_id = int(boxes.cls[i].cpu().numpy())

            # Try model's own class names first, then our map
            if hasattr(model, "names") and cls_id in model.names:
                class_name = model.names[cls_id]
            elif cls_id in CUSTOM_CLASS_MAP:
                class_name = CUSTOM_CLASS_MAP[cls_id]
            else:
                class_name = f"class_{cls_id}"

            # ── GPS coordinates ───────────────────────────────
            # In production: read from GPS hardware, EXIF, or
            # phone sensor API. Here we use defaults + jitter.
            lat = base_lat
            lon = base_lon
            if jitter:
                lat += np.random.uniform(-GPS_JITTER, GPS_JITTER)
                lon += np.random.uniform(-GPS_JITTER, GPS_JITTER)

            # ── Assemble the detection payload ────────────────
            detection = {
                "detection_id": str(uuid.uuid4()),
                "class_name": class_name,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 4),
                "gps": {
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # severity is auto-calculated by the backend
            }

            detections.append(detection)

    return detections


def send_to_backend(detections: list[dict], endpoint: str = API_ENDPOINT) -> bool:
    """
    POST a list of detection payloads to the Flask backend.

    Returns True on success, False on failure.
    """
    if not detections:
        print("  ℹ️  No detections to send.")
        return True

    try:
        response = requests.post(
            endpoint,
            json=detections,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        if response.status_code in (200, 201, 207):
            data = response.json()
            inserted = data.get("inserted", 0)
            errors = data.get("errors", [])
            print(f"  ✅ Sent {inserted} detection(s) to backend")
            if errors:
                print(f"  ⚠️  {len(errors)} error(s): {errors}")
            return True
        else:
            print(f"  ❌ Backend returned HTTP {response.status_code}")
            print(f"     {response.text[:300]}")
            return False

    except requests.ConnectionError:
        print("  ❌ Cannot connect to backend — is the Flask server running?")
        print(f"     Expected at: {endpoint}")
        return False
    except requests.Timeout:
        print("  ❌ Request timed out")
        return False


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image for visual
    verification. Returns the annotated frame.
    """
    TIER_COLORS = {
        "pothole": (0, 0, 255),      # Red  (BGR)
        "d40":     (0, 0, 255),
        "crack":   (255, 165, 0),     # Orange-ish
    }
    DEFAULT_COLOR = (255, 200, 0)     # Cyan fallback

    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det["class_name"]
        conf = det["confidence"]
        color = TIER_COLORS.get(label.lower(), DEFAULT_COLOR)

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return annotated


# ─────────────────────────────────────────────────────────────
# Inference Modes
# ─────────────────────────────────────────────────────────────

def infer_images(model, sources: list[str], args):
    """
    Run inference on one or more image files.
    """
    for img_path in sources:
        print(f"\n  🔍 Processing: {img_path}")

        # Read image
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  ⚠️  Could not read image: {img_path}")
            continue

        # Run YOLO inference
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        # Parse detections
        detections = parse_detections(
            results, model,
            base_lat=args.lat, base_lon=args.lon,
            jitter=args.gps_jitter,
        )

        print(f"  📊 Found {len(detections)} detection(s)")
        for det in detections:
            print(f"     • {det['class_name']:20s}  "
                  f"conf={det['confidence']:.2%}  "
                  f"bbox={det['bbox']}")

        # Send to backend
        if not args.dry_run:
            send_to_backend(detections, endpoint=args.endpoint)
        else:
            print("  🧪 Dry-run mode — skipping backend POST")
            print(f"     Payload preview:\n{json.dumps(detections[:2], indent=2)}")

        # Show annotated image (if display is available)
        if args.show:
            annotated = draw_detections(frame, detections)
            cv2.imshow("City Fixer — Detections", annotated)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()


def infer_webcam(model, camera_id: int, args):
    """
    Run continuous inference on a webcam / video feed.
    Press 'q' to stop.
    """
    print(f"\n  📹 Opening camera {camera_id} — press 'q' to quit")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"  ❌ Cannot open camera {camera_id}")
        return

    frame_count = 0
    send_every = 10  # Send detections every N frames to avoid flooding

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run inference
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        detections = parse_detections(
            results, model,
            base_lat=args.lat, base_lon=args.lon,
            jitter=True,  # Always jitter for video (simulates driving)
        )

        # Send every N frames
        if detections and frame_count % send_every == 0:
            if not args.dry_run:
                send_to_backend(detections, endpoint=args.endpoint)
            print(f"  📤 Frame {frame_count}: sent {len(detections)} detection(s)")

        # Draw and display
        annotated = draw_detections(frame, detections)
        cv2.imshow("City Fixer — Live Feed", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  🛑 Stopped after {frame_count} frames")


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="City Fixer — Detect road damage and send to backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer_and_send.py --source road.jpg
  python infer_and_send.py --source images/ --gps-jitter
  python infer_and_send.py --source 0                     # webcam
  python infer_and_send.py --source road.jpg --dry-run    # no POST
        """,
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Image path, directory of images, or camera index (0, 1, ...)",
    )
    parser.add_argument(
        "--weights", type=str, default=DEFAULT_WEIGHTS,
        help=f"Path to YOLOv8 weights file (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--lat", type=float, default=DEFAULT_LAT,
        help=f"GPS latitude for detections (default: {DEFAULT_LAT})",
    )
    parser.add_argument(
        "--lon", type=float, default=DEFAULT_LON,
        help=f"GPS longitude for detections (default: {DEFAULT_LON})",
    )
    parser.add_argument(
        "--gps-jitter", action="store_true",
        help="Add small random offset to GPS coords (simulates movement)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse detections but don't POST to backend",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display annotated images with OpenCV windows",
    )
    parser.add_argument(
        "--endpoint", type=str, default=API_ENDPOINT,
        help=f"Backend API URL (default: {API_ENDPOINT})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  🏗️  The City Fixer — Inference Pipeline")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────
    model = load_model(args.weights)

    # ── Determine source type ─────────────────────────────────
    source = args.source

    # Check if source is a camera index (integer)
    if source.isdigit():
        infer_webcam(model, int(source), args)
        return

    # Check if source is a directory
    if os.path.isdir(source):
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(source, ext)))
        image_files.sort()

        if not image_files:
            print(f"  ❌ No images found in: {source}")
            return

        print(f"  📂 Found {len(image_files)} image(s) in {source}")
        infer_images(model, image_files, args)
        return

    # Single file
    if os.path.isfile(source):
        infer_images(model, [source], args)
        return

    print(f"  ❌ Source not found: {source}")
    print("     Provide a valid image path, directory, or camera index.")


if __name__ == "__main__":
    main()
