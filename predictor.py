"""
predictor.py — Reusable inference module
=========================================
Loads trained artefacts once at startup and exposes a single predict()
function that mirrors the exact preprocessing applied during training.

Usage:
    from predictor import FraudPredictor
    predictor = FraudPredictor()
    result = predictor.predict(transaction_dict)
"""

import logging
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import joblib

log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"


class FraudPredictor:
    """
    Production inference class for credit card fraud detection.

    Loads model, scaler, and metadata once at init (not per-request).
    Thread-safe for Flask/FastAPI: models are read-only after loading.
    """

    def __init__(self):
        self._load_artefacts()

    def _load_artefacts(self):
        """Load all saved model artefacts from the models/ directory."""
        try:
            self.model        = joblib.load(MODELS_DIR / "rf_model.joblib")
            self.scaler       = joblib.load(MODELS_DIR / "scaler.joblib")
            self.feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
            self.scale_cols   = joblib.load(MODELS_DIR / "scale_cols.joblib")

            with open(MODELS_DIR / "metadata.json") as f:
                meta = json.load(f)

            self.threshold = meta["threshold"]
            self.metrics   = meta.get("metrics", {})
            log.info(f"Model loaded | Threshold: {self.threshold:.4f}")

        except FileNotFoundError as e:
            raise RuntimeError(
                f"Model artefacts not found. Run `python train.py` first.\n{e}"
            )

    # ── Feature engineering (must match train.py exactly) ────────────────────
    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hour_of_day"] = (df["Time"] % 86400) // 3600
        df["log_amount"]  = np.log1p(df["Amount"])
        df["amount_bin"]  = pd.cut(
            df["Amount"],
            bins=[-1, 10, 50, 200, 500, np.inf],
            labels=[0, 1, 2, 3, 4],
        ).astype(int)
        return df

    def _preprocess(self, raw: dict) -> pd.DataFrame:
        """
        Accept a single transaction dict, apply feature engineering
        and scaling, return a 1-row DataFrame ready for inference.
        """
        df = pd.DataFrame([raw])
        df = self._engineer_features(df)

        # Select and order features to match training
        df = df[self.feature_cols]

        # Scale the same columns the scaler was fit on
        df[self.scale_cols] = self.scaler.transform(df[self.scale_cols])
        return df

    def predict(self, transaction: dict) -> dict:
        """
        Predict fraud probability for a single transaction.

        Args:
            transaction: dict with keys Time, Amount, V1–V28

        Returns:
            {
                "label":       "Fraud" | "Legitimate",
                "probability": float (0–1),
                "risk_score":  int (0–100),
                "risk_level":  "Low" | "Medium" | "High" | "Critical",
            }
        """
        try:
            X = self._preprocess(transaction)
            prob  = float(self.model.predict_proba(X)[0, 1])
            label = "Fraud" if prob >= self.threshold else "Legitimate"

            # Map probability to a 0–100 risk score
            risk_score = int(prob * 100)

            if risk_score < 25:
                risk_level = "Low"
            elif risk_score < 50:
                risk_level = "Medium"
            elif risk_score < self.threshold * 100:
                risk_level = "High"
            else:
                risk_level = "Critical"

            return {
                "label"      : label,
                "probability": round(prob, 4),
                "risk_score" : risk_score,
                "risk_level" : risk_level,
            }

        except Exception as e:
            log.error(f"Prediction error: {e}")
            raise

    def batch_predict(self, transactions: list[dict]) -> list[dict]:
        """Predict fraud for a list of transactions (more efficient than looping)."""
        try:
            rows = [pd.DataFrame([t]) for t in transactions]
            df   = pd.concat(rows, ignore_index=True)
            df   = self._engineer_features(df)
            df   = df[self.feature_cols]
            df[self.scale_cols] = self.scaler.transform(df[self.scale_cols])

            probs  = self.model.predict_proba(df)[:, 1]
            labels = ["Fraud" if p >= self.threshold else "Legitimate" for p in probs]

            return [
                {
                    "label"      : lbl,
                    "probability": round(float(p), 4),
                    "risk_score" : int(p * 100),
                }
                for lbl, p in zip(labels, probs)
            ]
        except Exception as e:
            log.error(f"Batch prediction error: {e}")
            raise
