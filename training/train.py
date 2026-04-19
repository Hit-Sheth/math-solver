"""
train.py — Handwritten Math Symbol Classifier Training Script

Downloads the Kaggle "Handwritten Math Symbols" dataset, trains a CNN
to classify digits (0-9) and operators (+, -, *, /), and saves the
model as .keras along with a label mapping JSON.

Usage:
    python train.py --username YOUR_KAGGLE_USERNAME --key YOUR_KAGGLE_KEY

Requirements:
    pip install -r requirements.txt
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ── Configuration ──────────────────────────────────────────────────────────────

DATASET_SLUG = "sagyamthapa/handwritten-math-symbols"
IMG_SIZE = 28
NUM_EPOCHS = 20
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 5

# Mapping from folder names in the dataset to our canonical labels.
# The dataset uses descriptive folder names for operators.
FOLDER_TO_LABEL = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "+": "+", "plus": "+", "add": "+",
    "-": "-", "minus": "-", "sub": "-",
    "times": "*", "x": "*", "mul": "*", "multiplication": "*",
    "div": "/", "division": "/", "obelus": "/",
}

# The 14 classes we care about
TARGET_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/"]

# Directories
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODEL_DIR = SCRIPT_DIR.parent / "models"


# ── Kaggle Credentials Setup ──────────────────────────────────────────────────

def setup_kaggle_credentials(username: str, key: str):
    """Create the kaggle.json credentials file programmatically."""
    if platform.system() == "Windows":
        kaggle_dir = Path.home() / ".kaggle"
    else:
        kaggle_dir = Path.home() / ".kaggle"

    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"

    creds = {"username": username, "key": key}
    kaggle_json.write_text(json.dumps(creds))

    # Set permissions on non-Windows systems
    if platform.system() != "Windows":
        os.chmod(kaggle_json, 0o600)

    print(f"[✓] Kaggle credentials saved to {kaggle_json}")


# ── Dataset Download ──────────────────────────────────────────────────────────

def download_dataset():
    """Download and extract the Kaggle dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = DATA_DIR / "handwritten-math-symbols.zip"

    # Check if already extracted
    extracted_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if extracted_dirs:
        print(f"[✓] Dataset already extracted in {DATA_DIR}")
        return

    print(f"[...] Downloading dataset: {DATASET_SLUG}")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET_SLUG,
            "-p", str(DATA_DIR),
        ],
        shell=True,
        check=True,
    )

    # Extract
    print("[...] Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    # Cleanup zip
    zip_path.unlink()
    print(f"[✓] Dataset extracted to {DATA_DIR}")


# ── Data Loading ──────────────────────────────────────────────────────────────

def find_class_folders(root: Path) -> dict[str, Path]:
    """
    Walk the extracted dataset directory to find folders whose names
    match our target labels. Returns {canonical_label: folder_path}.
    """
    found = {}

    for item in root.rglob("*"):
        if not item.is_dir():
            continue

        folder_name = item.name.strip().lower()

        if folder_name in FOLDER_TO_LABEL:
            canonical = FOLDER_TO_LABEL[folder_name]
            if canonical in TARGET_LABELS and canonical not in found:
                # Verify it actually contains images
                images = list(item.glob("*.png")) + list(item.glob("*.jpg")) + list(item.glob("*.jpeg"))
                if images:
                    found[canonical] = item
                    print(f"    Found class '{canonical}' → {item.name}/ ({len(images)} images)")

    return found


def load_images(class_folders: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load all images from class folders, resize to IMG_SIZE x IMG_SIZE
    grayscale, normalize to [0, 1].

    Returns (X, y, label_map) where label_map is {int_index: string_label}.
    """
    # Create integer label mapping
    label_map = {i: label for i, label in enumerate(TARGET_LABELS)}
    label_to_int = {label: i for i, label in label_map.items()}

    images = []
    labels = []
    skipped = 0

    for label, folder in class_folders.items():
        int_label = label_to_int[label]
        image_files = (
            list(folder.glob("*.png"))
            + list(folder.glob("*.jpg"))
            + list(folder.glob("*.jpeg"))
        )

        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("L")  # Grayscale
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0
                images.append(arr)
                labels.append(int_label)
            except Exception:
                skipped += 1

    if skipped > 0:
        print(f"    Skipped {skipped} unreadable images")

    X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(labels)

    print(f"[✓] Loaded {len(X)} images across {len(class_folders)} classes")
    return X, y, label_map


# ── Model Architecture ────────────────────────────────────────────────────────

def build_model(num_classes: int):
    """Build a CNN for 28x28 grayscale image classification."""
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers

    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(X_train, y_train, X_val, y_val, num_classes: int, epochs: int = NUM_EPOCHS):
    """Train the model and return it along with training history."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    model = build_model(num_classes)
    model.summary()

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    datagen.fit(X_train)

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_val, y_val, label_map):
    """Print evaluation metrics."""
    y_pred = model.predict(X_val).argmax(axis=1)

    labels_idx = list(range(len(label_map)))
    target_names = [label_map[i] for i in range(len(label_map))]
    report = classification_report(
        y_val, y_pred, labels=labels_idx, target_names=target_names, zero_division=0
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Loss:     {val_loss:.4f}")

    return val_acc


# ── Save ──────────────────────────────────────────────────────────────────────

def save_model(model, label_map):
    """Save the trained model and label mapping."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "math_symbol_model.keras"
    label_map_path = MODEL_DIR / "label_map.json"

    model.save(model_path)
    label_map_path.write_text(json.dumps(label_map, indent=2))

    print(f"\n[✓] Model saved to:     {model_path}")
    print(f"[✓] Label map saved to: {label_map_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a handwritten math symbol classifier"
    )
    parser.add_argument("--username", required=True, help="Kaggle username")
    parser.add_argument("--key", required=True, help="Kaggle API key")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    args = parser.parse_args()

    print("=" * 60)
    print("  HANDWRITTEN MATH SYMBOL CLASSIFIER — TRAINING")
    print("=" * 60)

    # Step 1: Setup Kaggle credentials
    print("\n[1/6] Setting up Kaggle credentials...")
    setup_kaggle_credentials(args.username, args.key)

    # Step 2: Download dataset
    print("\n[2/6] Downloading dataset...")
    download_dataset()

    # Step 3: Find class folders
    print("\n[3/6] Scanning for class folders...")
    class_folders = find_class_folders(DATA_DIR)

    missing = [lbl for lbl in TARGET_LABELS if lbl not in class_folders]
    if missing:
        print(f"\n[!] WARNING: Missing classes: {missing}")
        print("    The model will be trained on available classes only.")

    if len(class_folders) < 2:
        print("[✗] Not enough classes found. Check the dataset structure.")
        sys.exit(1)

    # Step 4: Load images
    print("\n[4/6] Loading and preprocessing images...")
    X, y, label_map = load_images(class_folders)

    # Step 5: Split and train
    print("\n[5/6] Training model...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train: {len(X_train)} | Val: {len(X_val)}")

    model, history = train(X_train, y_train, X_val, y_val, len(label_map), args.epochs)

    # Step 6: Evaluate and save
    print("\n[6/6] Evaluating and saving...")
    val_acc = evaluate(model, X_val, y_val, label_map)
    save_model(model, label_map)

    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE — Val Accuracy: {val_acc:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
