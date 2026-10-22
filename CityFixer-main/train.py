"""
The City Fixer — YOLOv8 Training Script
=========================================
Train a YOLOv8-Nano model on a custom road-damage dataset
(pothole + crack) defined in data.yaml.

Usage
-----
    python train.py                 # Train with default settings
    python train.py --epochs 30     # Override epochs via CLI (optional)

Prerequisites
-------------
    pip install ultralytics
    Ensure datasets/road_damage/{train,val}/{images,labels}/ are populated.

Author : AI/ML Workspace
Version: 1.0.0
"""

import os
import argparse
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
# Hyperparameters & Configuration
# ─────────────────────────────────────────────────────────────

# Pretrained checkpoint — YOLOv8 Nano is fast & lightweight.
# Swap to 'yolov8s.pt' (small) or 'yolov8m.pt' (medium) for
# higher accuracy at the cost of training speed.
BASELINE_MODEL = "yolov8n.pt"

# Path to the dataset configuration file
DATA_CONFIG = "data.yaml"

# Training hyperparameters
DEFAULT_EPOCHS = 15      # Number of full passes over the training set
IMAGE_SIZE     = 640     # Input resolution (640×640 px)
BATCH_SIZE     = 8       # Images per batch — lower this if you hit OOM

# Output directory for training runs
PROJECT_DIR = "runs/road_damage"
RUN_NAME    = "city_fixer_v1"


# ─────────────────────────────────────────────────────────────
# Device Selection — CPU vs GPU
# ─────────────────────────────────────────────────────────────
#
# By default, Ultralytics auto-detects a CUDA GPU and uses it.
# You can force a specific device:
#
#   device="cpu"       → Force CPU training (slow but always works)
#   device="0"         → Use GPU index 0 (NVIDIA CUDA)
#   device="0,1"       → Multi-GPU training on GPUs 0 and 1
#   device="mps"       → Apple Silicon (M1/M2) GPU acceleration
#
# If you have no NVIDIA GPU, the library will fall back to CPU
# automatically — no changes needed.
# ─────────────────────────────────────────────────────────────

def get_device() -> str:
    """
    Detect whether a CUDA GPU is available and return the
    appropriate device string for Ultralytics.
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✅ CUDA GPU detected: {gpu_name}")
            return "0"                       # Use first GPU
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  ✅ Apple MPS backend detected")
            return "mps"
        else:
            print("  ⚠️  No GPU detected — training will use CPU (slower)")
            return "cpu"
    except ImportError:
        print("  ⚠️  PyTorch not found — defaulting to CPU")
        return "cpu"


def train(epochs: int = DEFAULT_EPOCHS):
    """
    Load a pretrained YOLOv8 model and fine-tune it on the
    road-damage dataset.
    """
    print("=" * 60)
    print("  🏗️  The City Fixer — Model Training")
    print("=" * 60)

    # ── 1. Detect device ──────────────────────────────────────
    device = get_device()

    # ── 2. Verify dataset config exists ───────────────────────
    if not os.path.isfile(DATA_CONFIG):
        print(f"\n  ❌ Dataset config not found: {DATA_CONFIG}")
        print("     Make sure data.yaml is in your working directory")
        print("     and that datasets/road_damage/ contains your images.")
        return

    # ── 3. Load pretrained model ──────────────────────────────
    print(f"\n  📦 Loading baseline model: {BASELINE_MODEL}")
    model = YOLO(BASELINE_MODEL)

    # ── 4. Train ──────────────────────────────────────────────
    print(f"  🚀 Starting training — {epochs} epochs, {IMAGE_SIZE}px, batch {BATCH_SIZE}")
    print(f"  📂 Dataset config   : {DATA_CONFIG}")
    print(f"  💾 Output directory  : {PROJECT_DIR}/{RUN_NAME}")
    print(f"  🖥️  Device           : {device}\n")

    results = model.train(
        data=DATA_CONFIG,
        epochs=epochs,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,          # Overwrite if re-running

        # ── Optional advanced hyperparameters ─────────────────
        # Uncomment and tweak as needed:
        #
        # lr0=0.01,             # Initial learning rate
        # lrf=0.001,            # Final learning rate (cosine decay target)
        # momentum=0.937,       # SGD momentum / Adam beta1
        # weight_decay=0.0005,  # L2 regularisation
        # warmup_epochs=3.0,    # LR warm-up epochs
        # augment=True,         # Enable built-in augmentations
        # mosaic=1.0,           # Mosaic augmentation probability
        # flipud=0.5,           # Vertical flip probability
        # fliplr=0.5,           # Horizontal flip probability
        # hsv_h=0.015,          # HSV-Hue augmentation range
        # hsv_s=0.7,            # HSV-Saturation augmentation range
        # hsv_v=0.4,            # HSV-Value augmentation range
        # patience=5,           # Early-stopping patience (0 = disabled)
        # save_period=5,        # Save checkpoint every N epochs
        # workers=4,            # DataLoader worker processes
    )

    # ── 5. Summary ────────────────────────────────────────────
    best_weights = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    last_weights = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "last.pt")

    print("\n" + "=" * 60)
    print("  ✅ Training complete!")
    print(f"  🏆 Best weights : {best_weights}")
    print(f"  📄 Last weights : {last_weights}")
    print("=" * 60)
    print("\n  Next step: run inference with infer_and_send.py")
    print(f"    python infer_and_send.py --weights {best_weights} --source <image>")

    return results


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on the City Fixer road-damage dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--batch", type=int, default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--imgsz", type=int, default=IMAGE_SIZE,
        help=f"Input image size (default: {IMAGE_SIZE})"
    )
    parser.add_argument(
        "--model", type=str, default=BASELINE_MODEL,
        help=f"Pretrained model checkpoint (default: {BASELINE_MODEL})"
    )
    args = parser.parse_args()

    # Allow CLI overrides
    BASELINE_MODEL = args.model
    BATCH_SIZE = args.batch
    IMAGE_SIZE = args.imgsz

    train(epochs=args.epochs)

