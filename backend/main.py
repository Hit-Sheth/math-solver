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

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess a canvas image for prediction:
    - Convert to grayscale
    - Resize to 28x28
    - The canvas sends white strokes on black background
    - Normalize to [0, 1]
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Resize to 28x28
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)

    # Check if image is mostly white (inverted canvas)
    # Canvas: white stroke on black bg → we want white-on-black (like the dataset)
    # If the mean is high, the image is mostly white → invert it
    if arr.mean() > 127:
        arr = 255.0 - arr

    # Normalize to [0, 1]
    arr = arr / 255.0

    # Reshape for model input: (1, 28, 28, 1)
    arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return arr


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

def predict_symbol(image_bytes: bytes) -> tuple[str, float]:
    """Predict the symbol from image bytes. Returns (label, confidence)."""
    if model is None or label_map is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    processed = preprocess_image(image_bytes)
    predictions = model.predict(processed, verbose=0)
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_idx])

    return label_map[class_idx], confidence


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

    # Predict each symbol
    pred1, conf1 = predict_symbol(img1_bytes)
    pred_op, conf_op = predict_symbol(op_bytes)
    pred2, conf2 = predict_symbol(img2_bytes)

    # Validate predictions
    if pred1 not in DIGIT_LABELS:
        raise HTTPException(
            status_code=422,
            detail=f"Operand 1 was recognized as '{pred1}' (confidence: {conf1:.0%}), but expected a digit (0-9). Try drawing more clearly.",
        )

    if pred_op not in OPERATOR_LABELS:
        raise HTTPException(
            status_code=422,
            detail=f"Operator was recognized as '{pred_op}' (confidence: {conf_op:.0%}), but expected an operator (+, -, *, /). Try drawing more clearly.",
        )

    if pred2 not in DIGIT_LABELS:
        raise HTTPException(
            status_code=422,
            detail=f"Operand 2 was recognized as '{pred2}' (confidence: {conf2:.0%}), but expected a digit (0-9). Try drawing more clearly.",
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
