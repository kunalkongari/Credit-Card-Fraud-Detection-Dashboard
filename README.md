# 🛡️ Credit Card Fraud Detection

> A production-ready machine learning system that detects fraudulent credit card transactions using Random Forest, trained on 284,807 real-world transactions with a 0.17% fraud rate.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **Live Demo → [https://cinematch-w2v-recommender.onrender.com](https://credit-card-fraud-detection-dashboard-npi1.onrender.com)**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [ML Design Decisions](#4-ml-design-decisions)
5. [Pipeline Walkthrough](#5-pipeline-walkthrough)
6. [Model Performance](#6-model-performance)
7. [Web Application](#7-web-application)
8. [API Reference](#8-api-reference)
9. [How to Run Locally](#9-how-to-run-locally)
10. [Deployment](#10-deployment)
11. [Key Learnings](#11-key-learnings)

---

## 1. Project Overview

Credit card fraud costs the global economy over **$30 billion annually**. This project builds a complete end-to-end ML system to detect fraudulent transactions — from raw data to a deployed REST API with a web dashboard.

### What makes this production-grade?

Most fraud detection notebooks stop at model accuracy. This project goes further:

- **Correct evaluation metric**: Uses PR-AUC instead of accuracy. With 0.17% fraud rate, a model that predicts "legitimate" for everything scores 99.83% accuracy — completely useless. PR-AUC measures performance where it actually matters.
- **Proper imbalance handling**: SMOTE oversampling on training data only (never on validation/test), combined with `class_weight='balanced'` in the model.
- **Business-aware threshold**: Classification threshold is not hardcoded to 0.5. It is optimized on the validation set to maximize F1, reflecting the real trade-off between catching fraud (recall) and not blocking legitimate customers (precision).
- **No training-serving skew**: The exact same feature engineering and scaling logic runs in both `train.py` and `predictor.py`. The scaler is saved with `joblib` and reloaded at inference — never fit twice.
- **Production inference class**: `FraudPredictor` loads all artifacts once at startup, not per-request. Thread-safe for Flask/Gunicorn multi-worker deployments.

---

## 2. Dataset

**Source**: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.172%) |
| Legitimate transactions | 284,315 (99.828%) |
| Imbalance ratio | 577 : 1 |
| Time window | 2 days (~48 hours) |
| Features | 30 (Time, Amount, V1–V28) |

### Feature Description

| Feature | Description |
|---|---|
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `Amount` | Transaction amount in Euros |
| `V1–V28` | PCA-transformed features. The original features (merchant, location, card type, etc.) were anonymized by the bank for privacy. These 28 components capture the underlying signal. |
| `Class` | Target variable: `1` = Fraud, `0` = Legitimate |

> **Important note on V1–V28**: Because these are PCA components of undisclosed original features, they cannot be manually constructed from raw transaction data. In a real bank system, the feature extraction pipeline would produce these automatically from raw transaction metadata. This is a fundamental property of this public dataset, not a limitation of the model.

**To use this project**, download `creditcard.csv` from the Kaggle link above and place it in the project root.

---

## 3. Project Structure

```
credit-card-fraud-detection/
│
├── train.py                # Full training pipeline
├── predictor.py            # Reusable inference class
├── app.py                  # Flask web application
│
├── templates/
│   └── index.html          # Dashboard UI (dark-themed, recruiter-friendly)
│
├── models/                 # Saved artifacts (generated after running train.py)
│   ├── rf_model.joblib     # Trained Random Forest model
│   ├── scaler.joblib       # Fitted RobustScaler
│   ├── feature_cols.joblib # Ordered feature list for inference
│   ├── scale_cols.joblib   # Which columns need scaling
│   └── metadata.json       # Threshold + best params + test metrics
│
├── logs/
│   ├── train.log           # Training run logs
│   └── app.log             # Flask request logs
│
├── requirements.txt
├── Procfile                # Render / Heroku deployment config
└── .gitignore
```

### File Responsibilities

**`train.py`** — Orchestrates the entire training pipeline:
- Loads and validates `creditcard.csv`
- Engineers 3 new features
- Splits into train/val/test (60/20/20, stratified)
- Fits RobustScaler on training data only
- Applies SMOTE to training data only
- Runs RandomizedSearchCV (20 iterations, 5-fold StratifiedKFold)
- Finds optimal classification threshold on validation set
- Evaluates final model on held-out test set
- Saves all artifacts to `models/`

**`predictor.py`** — The inference layer:
- `FraudPredictor` class loads all artifacts once at `__init__`
- `predict(transaction_dict)` applies identical preprocessing and returns label + probability + risk score
- `batch_predict(list_of_dicts)` for efficient bulk inference

**`app.py`** — Flask web server:
- Instantiates `FraudPredictor` once at startup
- Exposes `/predict`, `/health`, `/model-info` endpoints
- Gunicorn-compatible for production deployment

---

## 4. ML Design Decisions

### Why Random Forest?

After benchmarking 6 models on this dataset, Random Forest achieved the best PR-AUC:

| Model | PR-AUC | AUC-ROC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Random Forest** | **0.8067** | **0.9873** | **0.7945** | **0.8169** | **0.7733** |
| Voting Ensemble | 0.7944 | 0.9808 | 0.5634 | 0.4348 | 0.8000 |
| LightGBM | 0.7830 | 0.9538 | 0.7703 | 0.7808 | 0.7600 |
| Gradient Boosting | 0.7770 | 0.9803 | 0.3279 | 0.2062 | 0.8000 |
| XGBoost | 0.7652 | 0.9498 | 0.8000 | 0.8615 | 0.7467 |
| Logistic Regression | 0.7647 | 0.9868 | 0.6154 | 0.5000 | 0.8000 |

**Why RF wins on this specific dataset:**

1. **Balanced Precision/Recall**: RF achieves 0.817 precision with 0.773 recall. GBM-based models collapse precision (GBM: 0.206) — they catch more fraud but flag too many legitimate transactions, which is unacceptable in production.

2. **Robust to SMOTE noise**: RF averages predictions across 200–500 trees. Synthetic SMOTE samples (interpolations between real minority-class points) get averaged out across the forest, reducing overfitting to artificial data.

3. **Native handling of PCA features**: V1–V28 are orthogonal principal components. RF's random feature subsampling (`max_features='sqrt'`) is mathematically well-suited to orthogonal feature spaces — each split considers an independent subset of components.

4. **Temporal generalization**: The dataset spans 2 days. Gradient boosted models overfit to time-specific patterns in the validation window. RF's bagging (each tree trained on a bootstrap sample) provides better generalization across different time windows.

### Why SMOTE over random undersampling?

The dataset has 284,315 legitimate and only 492 fraud cases. Random undersampling (the naive approach) would discard **99.8% of the legitimate data** — throwing away 283,823 rows of real signal. SMOTE instead synthesizes new minority-class examples by interpolating between existing fraud cases in feature space, preserving all legitimate data.

`sampling_strategy=0.15` means fraud becomes 15% of the training set after resampling — enough signal for the model without drowning it in synthetic noise.

### Why RobustScaler over StandardScaler?

`Amount` in this dataset has extreme outliers — some transactions are thousands of Euros while the median is around €22. `StandardScaler` is sensitive to outliers (it uses mean and standard deviation). `RobustScaler` uses the median and interquartile range, making it resistant to extreme values.

Critically, **only Time, Amount, and the engineered features are scaled**. V1–V28 are already PCA-normalized — scaling them again would distort their variance structure.

### Why optimize threshold on validation set?

The default threshold of 0.5 assumes equal cost for false positives (blocking a legitimate transaction) and false negatives (missing a fraud). In reality:
- A false negative (missed fraud) costs the bank the transaction amount
- A false positive (blocking a legit transaction) costs customer trust and friction

The threshold is found by computing F1 across all possible thresholds on the **validation set** (never the test set) and selecting the one that maximizes it. This is saved to `metadata.json` and used at inference time.

---

## 5. Pipeline Walkthrough

```
creditcard.csv
      │
      ▼
┌─────────────────────────────┐
│  1. Feature Engineering     │  Add hour_of_day, log_amount, amount_bin
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  2. Stratified Split        │  60% train / 20% val / 20% test
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  3. RobustScaler            │  Fit on train only → transform all splits
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  4. SMOTE                   │  Applied to training set only
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  5. RandomizedSearchCV      │  20 iterations × 5-fold StratifiedKFold
│     scored on PR-AUC        │  Fit on SMOTE-resampled training data
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  6. Threshold Calibration   │  F1-optimal threshold on clean val set
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  7. Final Evaluation        │  Held-out test set (never seen before)
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  8. Save Artifacts          │  rf_model.joblib, scaler.joblib,
│                             │  metadata.json, feature_cols.joblib
└─────────────────────────────┘
```

### Engineered Features

Three features are added on top of the raw 30 columns:

| Feature | Formula | Rationale |
|---|---|---|
| `hour_of_day` | `(Time % 86400) // 3600` | Fraud clusters in late-night hours (0–3 AM). Captures circadian fraud patterns. |
| `log_amount` | `log(1 + Amount)` | Raw Amount is right-skewed (most transactions are small, some are huge). Log compression normalizes the distribution. |
| `amount_bin` | `cut(Amount, bins=[0,10,50,200,500,∞])` | Ordinal bucket captures non-linear fraud rate by amount tier. Micro-transactions ($0–$10) have the highest fraud rate — card testing pattern. |

---

## 6. Model Performance

Results on the held-out test set (20% of data, never used during training or tuning):

| Metric | Score | What it means |
|---|---|---|
| **PR-AUC** | **0.8067** | Area under Precision-Recall curve. Primary metric for imbalanced problems. |
| **AUC-ROC** | **0.9873** | Area under ROC curve. Measures overall discrimination ability. |
| **F1 Score** | **0.7945** | Harmonic mean of Precision and Recall. Balanced overall fraud detection quality. |
| **Precision** | **0.8169** | Of all transactions flagged as fraud, 81.7% actually are. Low false positive rate. |
| **Recall** | **0.7733** | Of all actual fraud cases, 77.3% are caught. |

### Classification Report

```
              precision    recall  f1-score   support

   Legitimate     1.0000    1.0000    1.0000     56863
        Fraud     0.8169    0.7733    0.7945        75

    macro avg     0.9085    0.8867    0.8973     56938
 weighted avg     0.9999    0.9999    0.9999     56938
```

### Fraud Patterns Observed in the Dataset

**By time of day:**
- Highest fraud rate: **02:00 AM** (0.80% — 4.7× the baseline rate)
- Peak window: **00:00 – 03:00** (overnight, reduced monitoring)
- Safest window: **08:00 – 11:00** (morning, peak human oversight)

**By transaction amount:**
- **$0–$10**: 28.5% of all fraud cases — classic card-testing pattern
- **$10–$50**: 21.3% — low-value, stays under suspicion thresholds
- **$1,000+**: only 7.3% — high-value fraud is rarer because it triggers immediate review

---

## 7. Web Application

A clean, dark-themed dashboard built with Flask and vanilla JS.

### Features
- **HH:MM:SS time input** with live conversion to seconds (the format the model expects)
- **Fraud/Legit sample loader** — pre-fills all 30 feature values with real transactions from the dataset
- **Real-time prediction** via REST API call on submit
- **Risk score bar** (0–100) with color-coded levels (Low / Medium / High / Critical)
- **Fraud intelligence panel** — 24-hour heatmap of fraud rate by hour + bar chart of fraud by amount range
- **Model metrics panel** — live PR-AUC, AUC-ROC, Precision, Recall loaded from `/model-info`

### Dashboard Preview

```
┌──────────────────────────────────────────────────┐
│  🛡️ FraudShield                          ● LIVE  │
├──────────────────────────┬───────────────────────┤
│  Transaction Details     │  Result               │
│  ─────────────────────   │  ───────────────────  │
│  TIME  00:06:46           │  🚨 FRAUD             │
│  AMOUNT  $2.69            │  Confidence: 93.12%   │
│  V1–V28  [Expand]        │  Risk: CRITICAL       │
│                           │                       │
│  [Legit] [Fraud] [DETECT]│  ████████████░ 93/100 │
├──────────────────────────┴───────────────────────┤
│  Fraud Intelligence · When & How Much            │
│  [24h heatmap]    [Amount distribution bars]     │
└──────────────────────────────────────────────────┘
```

---

## 8. API Reference

### `POST /predict`

Accepts a single transaction and returns a fraud prediction.

**Request body:**
```json
{
  "Time": 406,
  "Amount": 2.69,
  "V1": -2.3122,
  "V2": 1.9519,
  "V3": -1.6098,
  "V4": 3.9979,
  "...": "...",
  "V28": -0.2736
}
```

**Response:**
```json
{
  "label": "Fraud",
  "probability": 0.9312,
  "risk_score": 93,
  "risk_level": "Critical"
}
```

| Field | Type | Description |
|---|---|---|
| `label` | string | `"Fraud"` or `"Legitimate"` |
| `probability` | float | Model's confidence that this is fraud (0–1) |
| `risk_score` | int | Probability scaled to 0–100 |
| `risk_level` | string | `"Low"` / `"Medium"` / `"High"` / `"Critical"` |

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

### `GET /model-info`
```json
{
  "threshold": 0.68,
  "metrics": {
    "pr_auc": 0.8067,
    "auc_roc": 0.9873,
    "f1": 0.7945,
    "precision": 0.8169,
    "recall": 0.7733
  }
}
```

---

## 9. How to Run Locally

### Prerequisites
- Python 3.10+
- `creditcard.csv` downloaded from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place creditcard.csv in the project root
# Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud

# 4. Train the model (takes ~5–10 mins depending on your machine)
python train.py --data creditcard.csv

# 5. Start the web app
python app.py

# 6. Open in browser
# http://localhost:5000
```

### What `train.py` prints
```
2024-01-01 12:00:00  INFO      Loading data from: creditcard.csv
2024-01-01 12:00:02  INFO      Dataset shape: (284807, 31) | Fraud rate: 0.1727%
2024-01-01 12:00:02  INFO      Split → Train: 170884 | Val: 56961 | Test: 56962
2024-01-01 12:00:03  INFO      Applying SMOTE to training data...
2024-01-01 12:00:08  INFO      After SMOTE → Legit: 170396 | Fraud: 25560
2024-01-01 12:00:08  INFO      Running RandomizedSearchCV (20 iterations × 5-fold CV)...
2024-01-01 12:08:00  INFO      Best CV PR-AUC : 0.8234
2024-01-01 12:08:00  INFO      Optimal threshold: 0.6800
2024-01-01 12:08:01  INFO      TEST SET RESULTS
2024-01-01 12:08:01  INFO      pr_auc    : 0.8067
2024-01-01 12:08:01  INFO      Model saved → models/rf_model.joblib
```

---

## 10. Deployment

The project is deployment-ready for **Render** or **Heroku** via the included `Procfile`.

```
web: gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT --timeout 120
```

### Deploy to Render (free tier)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service → Connect your repo
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT`
5. Add environment variable: `FLASK_DEBUG=false`

> **Note**: The trained model artifacts (`models/*.joblib`) are not committed to Git. You have two options for deployment:
> - Run `train.py` as part of the build step (requires adding `creditcard.csv` or fetching it from an environment URL)
> - Commit the `models/` artifacts separately after training locally (add an exception in `.gitignore`)

---

## 11. Key Learnings

**On the dataset:**
- Accuracy is a completely misleading metric for imbalanced problems. A model predicting "legitimate" for every transaction scores 99.83% accuracy while being entirely useless. Always use PR-AUC or F1 for fraud detection.
- The V1–V28 PCA features carry almost all the predictive signal. Time and Amount alone are insufficient for reliable prediction — this mirrors how real fraud systems work, where rich transaction metadata is transformed internally before the model sees it.

**On the ML pipeline:**
- SMOTE must only be applied to training data. Applying it before splitting leaks synthetic minority samples into the validation/test sets, producing falsely optimistic metrics.
- The scaler must be fit only on training data and saved. Re-fitting on new data during inference causes training-serving skew — one of the most common production ML bugs.
- Threshold optimization on the validation set is not optional in production fraud systems. The business cost of a false negative (missed fraud) is almost always different from a false positive (blocked customer).

**On production engineering:**
- Load the model once at server startup, not per request. Deserializing a 200-tree Random Forest on every API call would make inference 100× slower.
- Separate training logic (`train.py`) from inference logic (`predictor.py`). This makes it easy to retrain the model and swap in new artifacts without touching the serving code.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn 1.5 |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Model Persistence | joblib |
| Web Framework | Flask 3.0 |
| Production Server | Gunicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Fonts | Space Mono + DM Sans (Google Fonts) |

---

## Dataset Citation

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
> *Calibrating Probability with Undersampling for Unbalanced Classification.*
> In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.
