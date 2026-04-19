"""
main.py — FastAPI Backend for Handwritten Math Solver

Loads a trained Keras CNN model and label map, accepts 3 images
(operand1, operator, operand2) via POST, predicts symbols,
computes the result, and returns the equation + answer.

Usage:
    uvicorn main:app --reload --port 8000
"""

import json
import io
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "math_symbol_model.keras"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
IMG_SIZE = 28

DIGIT_LABELS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
OPERATOR_LABELS = {"+", "-", "*", "/"}

# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Handwritten Math Solver API",
    description="Predict handwritten digits and operators, compute the result.",
    version="1.0.0",
)

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Loading ─────────────────────────────────────────────────────────────

model = None
label_map = None


@app.on_event("startup")
def load_model():
    """Load the trained model and label mapping on server startup."""
    global model, label_map

    import tensorflow as tf

    if not MODEL_PATH.exists():
        print(f"[!] Model not found at {MODEL_PATH}")
        print("    Run training first: python training/train.py --username ... --key ...")
        return

    if not LABEL_MAP_PATH.exists():
        print(f"[!] Label map not found at {LABEL_MAP_PATH}")
        return

    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)

    # Convert string keys to int keys
    label_map = {int(k): v for k, v in label_map.items()}

    print(f"[✓] Model loaded from {MODEL_PATH}")
    print(f"[✓] Label map: {label_map}")


# ── Image Preprocessing ───────────────────────────────────────────────────────

def segment_image(image_bytes: bytes, combine_all: bool = False) -> list[np.ndarray]:
    """
    Segment the canvas into individual symbols from left to right.
    Returns a list of preprocessed 28x28 arrays ready for prediction.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(img, dtype=np.uint8)

    # Binarize. Canvas is white bg, black ink.
    # Inverse threshold so ink is white (required for findContours)
    _, thresh = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out noise (lower threshold to catch division dots)
        if w > 3 and h > 3:
            bboxes.append((x, y, w, h))

    if not bboxes:
        return []

    if combine_all:
        x_min = min(b[0] for b in bboxes)
        y_min = min(b[1] for b in bboxes)
        x_max = max(b[0] + b[2] for b in bboxes)
        y_max = max(b[1] + b[3] for b in bboxes)
        bboxes = [(x_min, y_min, x_max - x_min, y_max - y_min)]
    else:
        # Sort bounding boxes left-to-right (by x coordinate)
        bboxes.sort(key=lambda b: b[0])

    crops = []
    for (x, y, w, h) in bboxes:
        crop = arr[y:y+h, x:x+w]

        # Make crop square with padding
        max_dim = max(w, h)
        padding = int(max_dim * 0.2)
        target_size = max_dim + (padding * 2)

        # White square
        square = np.full((target_size, target_size), 255, dtype=np.uint8)

        # Center crop in square
        y_off = (target_size - h) // 2
        x_off = (target_size - w) // 2
        square[y_off:y_off+h, x_off:x_off+w] = crop

        # Resize to 28x28
        pil_sq = Image.fromarray(square).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        res_arr = np.array(pil_sq, dtype=np.float32)

        # Normalize [0, 1]
        res_arr = res_arr / 255.0
        crops.append(res_arr.reshape(1, IMG_SIZE, IMG_SIZE, 1))

    return crops


def is_canvas_empty(image_bytes: bytes) -> bool:
    """Check if the canvas image is essentially empty (all black or all white)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(img, dtype=np.float32)

    # If std dev is very low, the canvas has no meaningful content
    if arr.std() < 10:
        return True

    # If the number of non-background pixels is very small
    threshold = 30
    non_bg_pixels = np.sum(arr > threshold) if arr.mean() < 127 else np.sum(arr < (255 - threshold))
    total_pixels = arr.shape[0] * arr.shape[1]

    return (non_bg_pixels / total_pixels) < 0.005


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_symbols(image_bytes: bytes, combine_all: bool = False) -> tuple[str, float]:
    """Segment image and predict all symbols. Returns (combined_label, average_confidence)."""
    if model is None or label_map is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    crops = segment_image(image_bytes, combine_all=combine_all)
    if not crops:
        return "", 0.0

    labels = []
    confidences = []

    for crop in crops:
        predictions = model.predict(crop, verbose=0)
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx])
        
        labels.append(label_map[class_idx])
        confidences.append(confidence)

    combined_label = "".join(labels)
    avg_conf = sum(confidences) / len(confidences)

    return combined_label, avg_conf


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "message": "Handwritten Math Solver API",
    }


@app.post("/solve")
async def solve(
    operand1: UploadFile = File(..., description="Image of the first digit"),
    operator: UploadFile = File(..., description="Image of the operator"),
    operand2: UploadFile = File(..., description="Image of the second digit"),
):
    """
    Accept 3 images (operand1, operator, operand2), predict symbols,
    compute the result, and return the equation + answer.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first by running: python training/train.py",
        )

    # Read image bytes
    img1_bytes = await operand1.read()
    op_bytes = await operator.read()
    img2_bytes = await operand2.read()

    # Check for empty canvases
    if is_canvas_empty(img1_bytes):
        raise HTTPException(status_code=400, detail="Operand 1 canvas is empty. Please draw a digit.")
    if is_canvas_empty(op_bytes):
        raise HTTPException(status_code=400, detail="Operator canvas is empty. Please draw an operator.")
    if is_canvas_empty(img2_bytes):
        raise HTTPException(status_code=400, detail="Operand 2 canvas is empty. Please draw a digit.")

    # Predict symbols
    pred1, conf1 = predict_symbols(img1_bytes)
    pred_op, conf_op = predict_symbols(op_bytes, combine_all=True)
    pred2, conf2 = predict_symbols(img2_bytes)

    # Validate predictions (allowing multi-digit logic)
    if not pred1 or not all(c in DIGIT_LABELS for c in pred1):
        raise HTTPException(
            status_code=422,
            detail=f"Operand 1 was recognized as '{pred1}', but expected only digits (0-9). Try drawing more clearly.",
        )

    if not pred_op or pred_op not in OPERATOR_LABELS:
        raise HTTPException(
            status_code=422,
            detail=f"Operator was recognized as '{pred_op}', but expected exactly one operator (+, -, *, /). Try drawing more clearly.",
        )

    if not pred2 or not all(c in DIGIT_LABELS for c in pred2):
        raise HTTPException(
            status_code=422,
            detail=f"Operand 2 was recognized as '{pred2}', but expected only digits (0-9). Try drawing more clearly.",
        )

    # Compute result
    num1 = int(pred1)
    num2 = int(pred2)
    equation = f"{num1} {pred_op} {num2}"

    try:
        if pred_op == "+":
            result = num1 + num2
        elif pred_op == "-":
            result = num1 - num2
        elif pred_op == "*":
            result = num1 * num2
        elif pred_op == "/":
            if num2 == 0:
                raise HTTPException(
                    status_code=422,
                    detail="Division by zero! Cannot divide by 0.",
                )
            result = round(num1 / num2, 4)
        else:
            raise HTTPException(status_code=422, detail=f"Unknown operator: {pred_op}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Computation error: {str(e)}")

    return {
        "operand1": {"value": pred1, "confidence": round(conf1, 4)},
        "operator": {"value": pred_op, "confidence": round(conf_op, 4)},
        "operand2": {"value": pred2, "confidence": round(conf2, 4)},
        "equation": equation,
        "result": result,
    }
