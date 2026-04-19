# Handwritten Math Solver

A full-stack application that recognizes handwritten digits and operators (+, -, *, /) drawn on canvases, predicts each symbol using a trained CNN model, and returns the computed equation.

## Architecture
- **Frontend**: Next.js with React Hooks for custom Canvas drawing.
- **Backend**: FastAPI server that preprocesses images (scaling, color inversion) and runs inference.
- **Model**: Custom Keras CNN trained on Kaggle's "Handwritten Math Symbols" dataset.

---

## 1. Environment Setup

It is recommended to use a Python virtual environment.

```bash
python -m venv venv
.\venv\Scripts\Activate  # On Windows
# source venv/bin/activate # On Mac/Linux
```

### Install Dependencies
Backend and Training dependencies:
```bash
pip install -r training/requirements.txt
pip install -r backend/requirements.txt
```

---

## 2. Model Training

The `train.py` script automatically downloads the "Handwritten Math Symbols" dataset from Kaggle, preprocesses the images (resizes to 28x28 grayscale), trains the CNN model, and saves it in the `models/` directory.

**You must provide your Kaggle credentials:**
1. Go to your [Kaggle Account Settings](https://www.kaggle.com/settings).
2. Scroll to the "API" section and click "Create New Token". This downloads `kaggle.json`.
3. Open `kaggle.json` to see your `"username"` and `"key"`.

Run the training script (replace with your actual username and key):
```bash
python training/train.py --username YOUR_USERNAME --key YOUR_KEY
```
*Depending on your hardware, this will take 5-10 minutes. It saves `math_symbol_model.keras` and `label_map.json` into the `models/` folder.*

---

## 3. Starting the Backend

The FastAPI backend runs on port `8000`.

```bash
uvicorn backend.main:app --reload --port 8000
```
*(Ensure you run this from the root directory `d:\Project\math-solver`)*

The backend will log `[✓] Model loaded from ...` if it successfully loads the model created in Step 2.

---

## 4. Starting the Frontend

The Next.js frontend runs on port `3000`.

Open a new terminal window:
```bash
cd frontend
npm run dev
```

Navigate to [http://localhost:3000](http://localhost:3000) in your browser.

---

## 5. Usage
1. Draw a digit (0-9) in the **Operand 1** box.
2. Draw an operator (+, -, *, /) in the **Operator** box.
3. Draw a digit (0-9) in the **Operand 2** box.
4. Click **Solve ⚡**.
5. The backend will parse the drawing, predict the digits/operators, calculate the result, and stream the response to the UI.

## Error Handling
The application handles various edge cases gracefully:
- **Empty Canvases:** Shows warning if you forget to draw in one of the boxes.
- **Division by Zero:** Detects and prevents `/ 0`.
- **Misclassification Guard:** Notifies you if an operator is expected but a digit is detected (or vice versa), asking you to redraw clearly.
