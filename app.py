"""
app.py — Flask Web Application
================================
Serves the fraud detection dashboard.
Production-ready: gunicorn-compatible, proper error handling, logging.

Local dev  : python app.py
Production : gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT
"""

import logging
import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify

from predictor import FraudPredictor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path("logs/app.log")),
    ],
)
log = logging.getLogger(__name__)

# ── App init ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Load model once at startup — not per request
try:
    predictor = FraudPredictor()
    log.info("FraudPredictor loaded successfully.")
except RuntimeError as e:
    log.error(f"Failed to load model: {e}")
    predictor = None  # handled in routes


# ── V-feature columns expected by the model ───────────────────────────────────
V_COLS = [f"V{i}" for i in range(1, 29)]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Render the main dashboard."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: { Time, Amount, V1..V28 }
    Returns JSON: { label, probability, risk_score, risk_level }
    """
    if predictor is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body received."}), 400

        # Validate required fields
        required = ["Time", "Amount"] + V_COLS
        missing  = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Cast all values to float
        transaction = {k: float(data[k]) for k in required}

        result = predictor.predict(transaction)
        log.info(
            f"Prediction → {result['label']} | "
            f"prob={result['probability']:.4f} | "
            f"Amount={transaction['Amount']:.2f}"
        )
        return jsonify(result), 200

    except ValueError as e:
        log.warning(f"Invalid input: {e}")
        return jsonify({"error": f"Invalid input value: {e}"}), 422

    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error."}), 500


@app.route("/health")
def health():
    """Health-check endpoint for Render / Heroku."""
    return jsonify({"status": "ok", "model_loaded": predictor is not None}), 200


@app.route("/model-info")
def model_info():
    """Return model metadata (metrics, threshold)."""
    if predictor is None:
        return jsonify({"error": "Model not loaded."}), 503
    return jsonify({
        "threshold": predictor.threshold,
        "metrics"  : predictor.metrics,
    }), 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    log.info(f"Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
