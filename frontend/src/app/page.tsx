"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import "./globals.css";

/* ── Types ───────────────────────────────────────────────────────────────── */

interface SymbolResult {
  value: string;
  confidence: number;
}

interface SolveResult {
  operand1: SymbolResult;
  operator: SymbolResult;
  operand2: SymbolResult;
  equation: string;
  result: number;
}

interface ErrorResult {
  detail: string;
}

/* ── Drawing Hook ────────────────────────────────────────────────────────── */

function useCanvas(canvasRef: React.RefObject<HTMLCanvasElement | null>) {
  const isDrawing = useRef(false);
  const lastPos = useRef<{ x: number; y: number } | null>(null);

  const getPos = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      if ("touches" in e) {
        const touch = e.touches[0];
        return {
          x: (touch.clientX - rect.left) * scaleX,
          y: (touch.clientY - rect.top) * scaleY,
        };
      }
      return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
      };
    },
    [canvasRef]
  );

  const startDrawing = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      e.preventDefault();
      isDrawing.current = true;
      lastPos.current = getPos(e);
    },
    [getPos]
  );

  const draw = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      e.preventDefault();
      if (!isDrawing.current) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const pos = getPos(e);
      const prev = lastPos.current;
      if (!prev) return;

      ctx.beginPath();
      ctx.moveTo(prev.x, prev.y);
      ctx.lineTo(pos.x, pos.y);
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 14;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke();

      lastPos.current = pos;
    },
    [canvasRef, getPos]
  );

  const stopDrawing = useCallback(() => {
    isDrawing.current = false;
    lastPos.current = null;
  }, []);

  const clear = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, [canvasRef]);

  // Initialize canvas background
  useEffect(() => {
    clear();
  }, [clear]);

  return { startDrawing, draw, stopDrawing, clear };
}

/* ── Canvas Card Component ───────────────────────────────────────────────── */

function CanvasCard({
  label,
  type,
  canvasRef,
  width,
}: {
  label: string;
  type: "digit" | "operator";
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  width: number;
}) {
  const { startDrawing, draw, stopDrawing, clear } = useCanvas(canvasRef);

  return (
    <div className="canvas-card">
      <span
        className={`canvas-card__label canvas-card__label--${type}`}
      >
        {label}
      </span>
      <div className="canvas-wrapper">
        <canvas
          ref={canvasRef}
          width={width}
          height={250}
          style={{ width: width, height: 250 }}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
        />
      </div>
      <button className="btn-clear" onClick={clear} type="button">
        ✕ Clear
      </button>
    </div>
  );
}

/* ── Main Page ───────────────────────────────────────────────────────────── */

export default function Home() {
  const canvas1Ref = useRef<HTMLCanvasElement | null>(null);
  const canvasOpRef = useRef<HTMLCanvasElement | null>(null);
  const canvas2Ref = useRef<HTMLCanvasElement | null>(null);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SolveResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canvasToBlob = (canvas: HTMLCanvasElement): Promise<Blob> => {
    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => {
          if (blob) resolve(blob);
          else reject(new Error("Failed to export canvas"));
        },
        "image/png"
      );
    });
  };

  const handleSolve = async () => {
    setError(null);
    setResult(null);

    const c1 = canvas1Ref.current;
    const cOp = canvasOpRef.current;
    const c2 = canvas2Ref.current;

    if (!c1 || !cOp || !c2) {
      setError("Canvas elements not found.");
      return;
    }

    setLoading(true);

    try {
      const [blob1, blobOp, blob2] = await Promise.all([
        canvasToBlob(c1),
        canvasToBlob(cOp),
        canvasToBlob(c2),
      ]);

      const formData = new FormData();
      formData.append("operand1", blob1, "operand1.png");
      formData.append("operator", blobOp, "operator.png");
      formData.append("operand2", blob2, "operand2.png");

      const res = await fetch("http://localhost:8000/solve", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData: ErrorResult = await res.json();
        setError(errData.detail || `Server error: ${res.status}`);
        return;
      }

      const data: SolveResult = await res.json();
      setResult(data);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(`Connection error: ${err.message}. Is the backend running?`);
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header__badge">
          AI-Powered Recognition
        </div>
        <h1 className="header__title">Math Solver</h1>
        <p className="header__subtitle">
          Draw a digit, an operator, and another digit — let AI do the math.
        </p>
      </header>

      {/* Canvas Area */}
      <section className="canvas-area">
        <CanvasCard
          label="Operand 1"
          type="digit"
          canvasRef={canvas1Ref}
          width={350}
        />
        <CanvasCard
          label="Operator"
          type="operator"
          canvasRef={canvasOpRef}
          width={200}
        />
        <CanvasCard
          label="Operand 2"
          type="digit"
          canvasRef={canvas2Ref}
          width={350}
        />
      </section>

      {/* Solve Button */}
      <div className="actions">
        <button
          className="btn-solve"
          onClick={handleSolve}
          disabled={loading}
          id="solve-button"
        >
          {loading ? (
            <>
              <div className="spinner" />
              Solving...
            </>
          ) : (
            <>
              <span className="btn-solve__icon">⚡</span>
              Solve
            </>
          )}
        </button>
      </div>

      {/* Result */}
      {result && (
        <div className="result-area">
          <div className="result-card result-card--success">
            <div className="result-card__label">Result</div>

            <div className="result-equation">
              <div className="result-symbol">
                <span className="result-symbol__value">
                  {result.operand1.value}
                </span>
                <span className="result-symbol__conf">
                  {(result.operand1.confidence * 100).toFixed(1)}%
                </span>
              </div>

              <div className="result-symbol">
                <span className="result-symbol__value">
                  {result.operator.value}
                </span>
                <span className="result-symbol__conf">
                  {(result.operator.confidence * 100).toFixed(1)}%
                </span>
              </div>

              <div className="result-symbol">
                <span className="result-symbol__value">
                  {result.operand2.value}
                </span>
                <span className="result-symbol__conf">
                  {(result.operand2.confidence * 100).toFixed(1)}%
                </span>
              </div>

              <span className="result-equals">=</span>

              <div className="result-answer">{result.result}</div>
            </div>

            {/* Confidence Bars */}
            <div className="confidence-section">
              <div className="confidence-bars">
                <div className="confidence-bar">
                  <span className="confidence-bar__label">Operand 1</span>
                  <div className="confidence-bar__track">
                    <div
                      className="confidence-bar__fill"
                      style={{
                        width: `${result.operand1.confidence * 100}%`,
                      }}
                    />
                  </div>
                  <span className="confidence-bar__value">
                    {(result.operand1.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="confidence-bar">
                  <span className="confidence-bar__label">Operator</span>
                  <div className="confidence-bar__track">
                    <div
                      className="confidence-bar__fill"
                      style={{
                        width: `${result.operator.confidence * 100}%`,
                      }}
                    />
                  </div>
                  <span className="confidence-bar__value">
                    {(result.operator.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="confidence-bar">
                  <span className="confidence-bar__label">Operand 2</span>
                  <div className="confidence-bar__track">
                    <div
                      className="confidence-bar__fill"
                      style={{
                        width: `${result.operand2.confidence * 100}%`,
                      }}
                    />
                  </div>
                  <span className="confidence-bar__value">
                    {(result.operand2.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="result-area">
          <div className="result-card result-card--error">
            <div className="result-card__label">Error</div>
            <p className="result-error-msg">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
