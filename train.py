"""
Credit Card Fraud Detection — Production Training Pipeline
==========================================================
Author : Kunal Kongari
Model  : Random Forest (selected after full benchmark — best PR-AUC)
Dataset: Kaggle Credit Card Fraud Detection (creditcard.csv)

Why Random Forest?
  • Highest PR-AUC (0.8067) on held-out test set
  • Balanced Precision (0.8169) / Recall (0.7733) — unlike GBM which collapses precision
  • Bagging averages out SMOTE noise across 200+ trees
  • Native handling of orthogonal PCA features (V1–V28) via sqrt-feature subsampling
  • Better temporal generalisation vs gradient-boosted models

Run:
    python train.py                      # train with defaults
    python train.py --data path/to/csv   # custom data path
"""

import argparse
import logging
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "train.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three lightweight features that meaningfully boost fraud signal:
      - hour_of_day : fraud clusters in specific hours (e.g. late-night)
      - log_amount  : raw Amount is right-skewed; log compresses outliers
      - amount_bin  : ordinal bucket captures non-linear amount effects
    """
    df = df.copy()
    df["hour_of_day"] = (df["Time"] % 86400) // 3600          # 0–23
    df["log_amount"]  = np.log1p(df["Amount"])                 # log(1 + x)
    df["amount_bin"]  = pd.cut(
        df["Amount"],
        bins=[-1, 10, 50, 200, 500, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(data_path: str):
    """
    Load CSV → engineer features → scale Time/Amount → split into
    train/val/test with stratification.

    Scaling rationale:
      V1–V28 are already PCA-transformed and unit-normalised.
      Only Time, Amount, and the engineered features need RobustScaler
      (robust to outliers from high-value transactions).
    """
    log.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    log.info(f"Dataset shape: {df.shape} | Fraud rate: {df['Class'].mean()*100:.4f}%")

    # Validate expected columns
    expected_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = engineer_features(df)

    # ── Feature matrix ────────────────────────────────────────────────────────
    SCALE_COLS = ["Time", "Amount", "hour_of_day", "log_amount", "amount_bin"]
    V_COLS     = [f"V{i}" for i in range(1, 29)]
    FEATURE_COLS = V_COLS + SCALE_COLS

    X = df[FEATURE_COLS].copy()
    y = df["Class"].values

    # ── Train / Val / Test split (60 / 20 / 20) ───────────────────────────────
    # Stratified to preserve 0.17% fraud rate in every split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=SEED  # 0.25 × 0.80 = 0.20
    )

    log.info(
        f"Split → Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}"
    )

    # ── Scale only the non-PCA columns ────────────────────────────────────────
    scaler = RobustScaler()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_val[SCALE_COLS]   = scaler.transform(X_val[SCALE_COLS])
    X_test[SCALE_COLS]  = scaler.transform(X_test[SCALE_COLS])

    # Save scaler + feature list for inference
    joblib.dump(scaler,       MODELS_DIR / "scaler.joblib")
    joblib.dump(FEATURE_COLS, MODELS_DIR / "feature_cols.joblib")
    joblib.dump(SCALE_COLS,   MODELS_DIR / "scale_cols.joblib")
    log.info("Scaler saved.")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 3. SMOTE RESAMPLING
# ─────────────────────────────────────────────────────────────────────────────
def apply_smote(X_train, y_train):
    """
    SMOTE on training data only — never on val or test.
    sampling_strategy=0.15 → fraud becomes 15% of training set,
    which avoids over-synthetic noise while giving RF enough positive examples.
    """
    log.info("Applying SMOTE to training data...")
    before = np.bincount(y_train)
    smote = SMOTE(sampling_strategy=0.15, random_state=SEED, n_jobs=-1)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    after = np.bincount(y_res)
    log.info(f"Before SMOTE → Legit: {before[0]:,} | Fraud: {before[1]:,}")
    log.info(f"After  SMOTE → Legit: {after[0]:,}  | Fraud: {after[1]:,}")
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
def tune_random_forest(X_train, y_train):
    """
    RandomizedSearchCV over 20 iterations with StratifiedKFold(5).
    Scored on average_precision (PR-AUC) — the correct metric for imbalanced
    datasets where the positive class (fraud) is what matters.
    """
    param_dist = {
        "n_estimators"     : [200, 300, 400, 500],
        "max_depth"        : [8, 10, 12, None],
        "min_samples_leaf" : [2, 4, 5, 8],
        "min_samples_split": [2, 5, 10],
        "max_features"     : ["sqrt", "log2", 0.3],
        "class_weight"     : ["balanced", "balanced_subsample"],
    }

    base_rf = RandomForestClassifier(n_jobs=-1, random_state=SEED)
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    search = RandomizedSearchCV(
        estimator           = base_rf,
        param_distributions = param_dist,
        n_iter              = 20,
        scoring             = "average_precision",
        cv                  = cv,
        n_jobs              = -1,
        random_state        = SEED,
        verbose             = 1,
        refit               = True,
    )

    log.info("Running RandomizedSearchCV (20 iterations × 5-fold CV)...")
    search.fit(X_train, y_train)

    log.info(f"Best CV PR-AUC : {search.best_score_:.4f}")
    log.info(f"Best params    : {search.best_params_}")
    return search.best_estimator_, search.best_params_


# ─────────────────────────────────────────────────────────────────────────────
# 5. THRESHOLD CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_threshold(model, X_val, y_val):
    """
    Business-cost-aware threshold selection on the validation set.
    We maximise F1 as a proxy for balanced precision/recall.
    In production you'd replace this with an actual cost matrix
    (e.g. false-negative cost >> false-positive cost).
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx  = np.argmax(f1_scores[:-1])  # last entry has no matching threshold
    threshold = float(thresholds[best_idx])

    log.info(
        f"Optimal threshold: {threshold:.4f}  "
        f"(Val F1={f1_scores[best_idx]:.4f})"
    )
    return threshold


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, threshold: float):
    """Print and return a metrics dict for the test set."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold" : round(threshold, 4),
        "auc_roc"   : round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc"    : round(average_precision_score(y_test, y_prob), 4),
        "f1"        : round(f1_score(y_test, y_pred), 4),
        "precision" : round(precision_score(y_test, y_pred), 4),
        "recall"    : round(recall_score(y_test, y_pred), 4),
    }

    log.info("=" * 55)
    log.info("TEST SET RESULTS")
    log.info("=" * 55)
    for k, v in metrics.items():
        log.info(f"  {k:<12}: {v}")
    log.info("\n" + classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    log.info("=" * 55)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(data_path: str):
    log.info("=" * 55)
    log.info("CREDIT CARD FRAUD DETECTION — TRAINING PIPELINE")
    log.info("=" * 55)

    # Step 1 — Load & preprocess
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess(data_path)

    # Step 2 — SMOTE on train only
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # Step 3 — Hyperparameter tuning (fit on SMOTE-resampled train)
    best_model, best_params = tune_random_forest(X_train_res, y_train_res)

    # Step 4 — Threshold calibration on clean validation set
    threshold = find_optimal_threshold(best_model, X_val, y_val)

    # Step 5 — Final evaluation on held-out test set
    metrics = evaluate(best_model, X_test, y_test, threshold)

    # Step 6 — Persist model artefacts
    joblib.dump(best_model, MODELS_DIR / "rf_model.joblib")

    metadata = {
        "threshold"   : threshold,
        "best_params" : best_params,
        "metrics"     : metrics,
        "feature_cols": list(X_train.columns),
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Model saved → models/rf_model.joblib")
    log.info("Metadata    → models/metadata.json")
    log.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RF fraud detector")
    parser.add_argument("--data", default="creditcard.csv", help="Path to creditcard.csv")
    args = parser.parse_args()
    main(args.data)
