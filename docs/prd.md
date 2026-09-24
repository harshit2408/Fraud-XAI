# PRD: Explainable Fraud & Anomaly Detection on Transactional Data

**Document Version:** 2.0 (Final)  
**Status:** Active  
**Document Type:** Product Requirements Document + Technical Architecture + Sprint Plan  
**Intended Audience:** Senior ML Engineer / AI Architect (individual contributor building this as a portfolio project)  
**Last Updated:** June 2026  
**Changelog:** v2.0 — Reduced Docker stack from 8 services to 4. Eliminated MLflow container (local-only), Zookeeper (replaced by Kafka KRaft), kafka-producer container (one-shot script), and kafka-consumer container (folded into fraud-api as asyncio background task).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Business Context](#2-problem-statement--business-context)
3. [Goals & Objectives](#3-goals--objectives)
4. [Success Metrics](#4-success-metrics)
5. [Functional Requirements](#5-functional-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Technical Architecture](#7-technical-architecture)
8. [Tech Stack & Justification](#8-tech-stack--justification)
9. [Dataset Specification](#9-dataset-specification)
10. [Development Phases & Sprints](#10-development-phases--sprints)
    - [Phase 0 — Foundation & Environment Setup](#phase-0--foundation--environment-setup)
    - [Phase 1 — Data Ingestion, EDA & Feature Engineering](#phase-1--data-ingestion-eda--feature-engineering)
    - [Phase 2 — Baseline Model (XGBoost/LightGBM)](#phase-2--baseline-model-xgboostlightgbm)
    - [Phase 3 — Advanced Sequential Model (TFT / Transformer)](#phase-3--advanced-sequential-model-tft--transformer)
    - [Phase 4 — Explainability Layer (SHAP + Dashboards)](#phase-4--explainability-layer-shap--dashboards)
    - [Phase 5 — Model Serving via FastAPI](#phase-5--model-serving-via-fastapi)
    - [Phase 6 — Real-Time Inference Simulation (Kafka)](#phase-6--real-time-inference-simulation-kafka)
    - [Phase 7 — Monitoring & Drift Detection](#phase-7--monitoring--drift-detection)
    - [Phase 8 — Business Impact Quantification](#phase-8--business-impact-quantification)
    - [Phase 9 — Precision Improvement (Recall-Preserving)](#phase-9--precision-improvement-recall-preserving)
    - [Phase 10 — Testing Strategy](#phase-10--testing-strategy)
    - [Phase 11 — Packaging, Documentation & Portfolio Polish](#phase-11--packaging-documentation--portfolio-polish)
    - [Phase 12 — Model Architecture Exploration: Graph Neural Network (GNN-GraphSAGE)](#phase-12--model-architecture-exploration-graph-neural-network-gnn-graphsage)
11. [Sprint Plan Summary](#11-sprint-plan-summary)
12. [Repository & Folder Structure](#12-repository--folder-structure)
13. [API Contracts](#13-api-contracts)
14. [Configuration & Environment Management](#14-configuration--environment-management)
15. [Risks & Mitigations](#15-risks--mitigations)
16. [Definition of Done](#16-definition-of-done)
17. [Glossary](#17-glossary)

---

## 1. Executive Summary

This document defines the complete build plan for a production-grade, explainable fraud detection system on financial transaction data. The project is designed as a senior-level ML engineering portfolio artifact that demonstrates the following capabilities in a single cohesive system:

- **Sequential / temporal modeling** using Temporal Fusion Transformer (TFT) for transaction history, bridging deep learning expertise into structured tabular data
- **Tree-based modeling** using XGBoost as a calibrated baseline, reflecting standard fintech practice
- **Regulatory-grade explainability** using SHAP values, compliant with RBI / FCA / SEC model interpretability mandates
- **Class imbalance mastery** using Focal Loss, SMOTE, and class-weighted training with detailed Precision-Recall analysis
- **Production serving** via a FastAPI inference service with request/response logging
- **Real-time inference simulation** using Apache Kafka to mimic a live transaction stream
- **Model monitoring** using Evidently AI for data drift and prediction distribution tracking
- **Business impact framing** expressed in cost-benefit terms meaningful to finance stakeholders

The goal is not just a trained model. The goal is a deployable, explainable, monitored, production-aware ML system that demonstrates every signal a fintech technical head looks for when hiring an AI/ML engineer.

---

## 2. Problem Statement & Business Context

### 2.1 The Business Problem

Financial fraud costs the global economy over $5 trillion annually. At every bank, NBFC, payments processor, and fintech platform, fraud detection is the single largest applied ML investment. The challenge is three-dimensional:

1. **Scale**: Millions of transactions per minute. A model that takes 500ms per prediction is unusable.
2. **Imbalance**: Fraud represents 0.1%–0.3% of transactions. A naive model predicting "not fraud" achieves 99.7% accuracy and catches zero fraud.
3. **Regulatory constraint**: Regulators (RBI in India, FCA in UK, SEC in US, GDPR in EU) require that any automated decision affecting a customer's financial standing must be explainable. A black-box deep learning model cannot legally be deployed for lending, credit, or fraud decisioning in most jurisdictions.

### 2.2 Why Existing Approaches Fall Short

- **Notebook-only models**: Most ML portfolios demonstrate model training but not deployment, monitoring, or explainability — the exact skills finance companies need.
- **Tree models without explainability**: XGBoost is the industry standard, but deploying it without SHAP-based explanations is non-compliant and practically useless for business users.
- **Deep learning without sequence modeling**: Applying a flat MLP to tabular transaction data ignores the temporal dimension — the same transaction looks very different when preceded by 10 normal vs. 10 suspicious transactions.
- **No drift monitoring**: Models trained on Q1 data degrade by Q3 due to fraud pattern evolution. Detecting and alerting on this is production hygiene, not a bonus.

### 2.3 The Opportunity

This project builds a system that solves all three dimensions — scale-awareness via streaming, imbalance handling via principled techniques, and regulatory compliance via deep explainability — presented in a way that is immediately understandable to both technical and business stakeholders.

---

## 3. Goals & Objectives

### 3.1 Primary Goals

| ID | Goal |
|----|------|
| G1 | Build a fraud detection model that outperforms a naive baseline on PR-AUC, not just ROC-AUC |
| G2 | Generate per-prediction SHAP explanations that can be surfaced to a fraud analyst |
| G3 | Serve predictions via a REST API with sub-100ms P95 latency |
| G4 | Simulate real-time transaction stream ingestion via Kafka |
| G5 | Detect data drift and model degradation using Evidently AI |
| G6 | Express model value in business cost terms ($X saved annually) |

### 3.2 Secondary Goals

| ID | Goal |
|----|------|
| S1 | Compare tree-based baseline vs. TFT on the same dataset with a clean ablation |
| S2 | Implement reproducible training pipeline with MLflow experiment tracking |
| S3 | Containerize the full stack with Docker Compose for one-command local deployment |
| S4 | Produce a public-facing README and demo video suitable for a GitHub portfolio |

### 3.3 Non-Goals (Out of Scope)

- Real-money transaction processing or integration with actual payment rails
- Multi-tenant architecture or SaaS product features
- Graph neural networks (valuable but out of scope for this project's focus)
- AutoML or hyperparameter search beyond standard Optuna sweeps
- Mobile application or browser extension

---

## 4. Success Metrics

### 4.1 Model Performance Metrics

These are measured on the held-out test set. Note: **PR-AUC is the primary metric**, not ROC-AUC, because ROC-AUC is misleading on heavily imbalanced datasets.

| Metric | Minimum Threshold | Stretch Target |
|--------|-------------------|----------------|
| PR-AUC (XGBoost baseline) | 0.70 | 0.80 |
| PR-AUC (TFT / sequential model) | 0.75 | 0.85 |
| Recall at 5% FPR | ≥ 0.70 | ≥ 0.80 |
| F1-Score (fraud class, threshold-optimized) | ≥ 0.65 | ≥ 0.75 |
| SHAP explanation coverage | 100% of API responses | — |

> **Note on threshold selection**: The optimal classification threshold is NOT 0.5. It must be selected based on the business cost matrix defined in Phase 8. This is a key differentiator from amateur implementations.

> **Note on precision at the deployed threshold (added after Phase 8 closed)**:
> the metrics above are dataset-level (PR-AUC, recall at a fixed FPR) and were
> met by the Phase 3/8 model. They do not by themselves guarantee acceptable
> precision at whatever threshold the cost model selects — Phase 8 found the
> deployed operating point at **95.5% recall / 6.2% precision**, which the
> table above does not flag as a problem because none of its rows measure
> precision at the actual deployed threshold. Phase 9 adds that measurement
> explicitly: **precision ≥ 30% at recall ≥ 80%, at the deployed operating
> threshold** — a business-usability target, not a ranking-quality target, and
> deliberately separate from PR-AUC/F1 above.

### 4.2 System Performance Metrics

| Metric | Target |
|--------|--------|
| P95 API inference latency | < 100ms |
| Kafka consumer lag | < 500 events under simulated load |
| Evidently drift report generation | < 30 seconds per batch |
| Docker Compose cold start | < 2 minutes |

### 4.3 Code Quality Metrics

| Metric | Target |
|--------|--------|
| Unit test coverage (core modules) | ≥ 80% |
| All notebooks convert cleanly to scripts | Yes |
| No hardcoded credentials or paths | Yes |
| All configs externalized to YAML | Yes |

---

## 5. Functional Requirements

### FR-01: Data Pipeline
- The system must load the IEEE-CIS Fraud Detection dataset (or PaySim as fallback) from a local or cloud path
- The pipeline must produce a clean, feature-engineered dataframe with no data leakage between train and test splits
- Time-based train/test split must be used, not random split, to reflect production realism

### FR-02: Baseline Model
- Train an XGBoost classifier with class imbalance handling
- Expose calibrated probability scores (not just binary labels)
- Log all experiments to MLflow including parameters, metrics, and model artifacts

### FR-03: Sequential Model
- Train a Temporal Fusion Transformer (TFT) using PyTorch Forecasting or a custom implementation
- Input sequences: last N transactions per card/account as temporal context
- Output: fraud probability for the most recent transaction

### FR-04: Explainability
- Generate SHAP TreeExplainer values for XGBoost predictions
- Generate SHAP DeepExplainer or KernelExplainer values for TFT predictions
- Surface top-5 contributing features per prediction in API response
- Provide a standalone SHAP summary/beeswarm dashboard (static HTML export)

### FR-05: Class Imbalance Handling
- Implement SMOTE oversampling on training data
- Implement Focal Loss for the neural model
- Implement class-weighted XGBoost with `scale_pos_weight`
- Compare all three approaches in an ablation notebook

### FR-06: API Serving
- FastAPI endpoint: `POST /predict` — accepts transaction JSON, returns fraud probability + SHAP explanation
- FastAPI endpoint: `GET /health` — liveness check
- FastAPI endpoint: `GET /metrics` — Prometheus-compatible metrics endpoint
- All requests and responses logged to a structured JSON log file

### FR-07: Streaming Simulation
- Kafka producer **script** (`src/streaming/producer.py`) that reads test transactions and publishes to a `transactions` topic at a configurable rate — runs as a one-shot CLI command, not a Docker service
- Kafka consumer loop runs as an **asyncio background task inside `fraud-api`**, started via FastAPI's `lifespan` event — no separate container required
- Consumer polls the `transactions` topic, calls the model directly in-process (no HTTP hop), and publishes fraud alerts to the `fraud_alerts` topic

### FR-08: Monitoring
- Evidently AI drift report comparing reference data (training) vs. current window (sliding 24h batch)
- Prometheus metrics for: prediction latency, request count, fraud rate, model version
- Grafana dashboard with panels for all four Prometheus metrics above

### FR-09: Business Impact
- A Jupyter notebook titled `business_impact.ipynb` that:
  - Defines a cost matrix (cost of false negative, cost of false positive, value of true positive)
  - Computes expected annual savings at the model's achieved performance level
  - Plots threshold vs. net business value curve

---

## 6. Non-Functional Requirements

### 6.1 Performance
- Inference latency: < 100ms P95 per single transaction prediction (XGBoost)
- Batch inference: 10,000 transactions processed in < 60 seconds
- Kafka consumer throughput: ≥ 500 transactions/second on consumer side

### 6.2 Reproducibility
- All random seeds parameterized in `config/config.yaml`
- Dataset download/preparation step is idempotent and scripted
- `make reproduce` command re-runs the full training pipeline from raw data to saved model

### 6.3 Portability
- Full stack runs via `docker compose up` on any machine with Docker installed
- No hard dependency on GPU — CPU-capable training for TFT (GPU optional via device flag in config)
- Python 3.10+ with `requirements.txt` and `requirements-dev.txt` pinned

### 6.4 Maintainability
- All configuration in `config/config.yaml`, never hardcoded in source files
- Logging via Python `logging` module with structured JSON formatter
- Type hints on all function signatures in `src/` modules
- Docstrings on all public functions

### 6.5 Security (Portfolio Context)
- No real PII or financial data — public Kaggle datasets only
- `.env` file for any API keys (MLflow remote tracking URI if used), never committed
- `.gitignore` covers all data files, model artifacts, and environment files

---

## 7. Technical Architecture

### 7.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
│  IEEE-CIS Dataset (CSV) → Feature Engineering → Train/Test Split   │
│  (Time-based split, no leakage)                                     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │      TRAINING LAYER        │
              │  ┌──────────────────────┐  │
              │  │  XGBoost Baseline    │  │
              │  │  + SMOTE / class wt  │  │
              │  └──────────────────────┘  │
              │  ┌──────────────────────┐  │
              │  │  TFT Sequential Model│  │
              │  │  + Focal Loss        │  │
              │  └──────────────────────┘  │
              │  MLflow (local, cli only)  │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   EXPLAINABILITY LAYER     │
              │  SHAP TreeExplainer (XGB)  │
              │  SHAP DeepExplainer (TFT)  │
              │  Feature Importance Plots  │
              │  SHAP Dashboard (HTML)     │
              └─────────────┬─────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                 SERVING LAYER  [Docker: fraud-api]                   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  FastAPI Inference Service                                  │    │
│  │  POST /predict  GET /health  GET /metrics                   │    │
│  │  → Model: XGB + TFT loaded at startup                       │    │
│  │  → SHAP values computed per request                         │    │
│  │  → Logs to structured JSON                                  │    │
│  │                                                             │    │
│  │  [asyncio background task — same process]                   │    │
│  │  Kafka Consumer Loop                                        │    │
│  │  → Polls 'transactions' topic from Kafka                    │    │
│  │  → Calls model directly (no HTTP hop)                       │    │
│  │  → Publishes alerts to 'fraud_alerts' topic                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ▲                                       │
│                              │                                       │
│  python src/streaming/producer.py   ← CLI script, not a service     │
│  (reads test parquet, publishes to Kafka at configured rate)         │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │  [Docker: kafka]           │
              │  Apache Kafka (KRaft mode) │
              │  No Zookeeper required     │
              └─────────────┬─────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                     MONITORING LAYER                                 │
│                                                                      │
│  Evidently AI          [Docker: prometheus]   [Docker: grafana]     │
│  (drift reports,       scrapes /metrics        dashboards over      │
│   run as script)       from fraud-api          prometheus           │
└──────────────────────────────────────────────────────────────────────┘

Docker Compose services (4 total):
  1. fraud-api     — FastAPI + Kafka consumer background task
  2. kafka         — KRaft mode, no Zookeeper
  3. prometheus    — metrics scraping
  4. grafana       — dashboards

MLflow runs locally during training only: `mlflow ui` — not a container.
Kafka producer runs as a one-shot script: `python src/streaming/producer.py`
```

### 7.2 Data Flow

```
Raw CSV
  │
  ├──► data_loader.py        → loads raw CSVs, validates schema
  │
  ├──► feature_engineering.py → creates temporal features, aggregates,
  │                              encodes categoricals
  │
  ├──► data_splitter.py       → time-based 80/20 split
  │                              (no random shuffle)
  │
  ├──► imbalance_handler.py   → SMOTE on train set only
  │                              (never on test set)
  │
  ├──► xgb_trainer.py         → trains XGBoost, logs to MLflow
  │
  ├──► tft_trainer.py         → builds sequences, trains TFT
  │
  ├──► shap_explainer.py      → generates SHAP values, plots
  │
  └──► evaluator.py           → PR-AUC, F1, cost-matrix threshold
                                 selection, comparison report
```

### 7.3 Inference Flow (per API request)

```
POST /predict
  { transaction_json }
        │
        ▼
  validate_request()   ← pydantic schema validation
        │
        ▼
  feature_transform()  ← apply same preprocessing as training
        │
        ├──► xgb_model.predict_proba()  → p_fraud_xgb
        │
        └──► tft_model.predict()        → p_fraud_tft
                │
                ▼
         ensemble_score()    → weighted average or max
                │
                ▼
         shap_explain()      → top 5 feature contributions
                │
                ▼
         log_to_file()       → structured JSON log
                │
                ▼
  {
    "transaction_id": "...",
    "fraud_probability": 0.87,
    "model": "ensemble",
    "threshold": 0.42,
    "decision": "FRAUD",
    "explanation": [
      {"feature": "amount_zscore", "contribution": +0.31},
      {"feature": "hour_of_day", "contribution": +0.18},
      ...
    ]
  }
```

---

## 8. Tech Stack & Justification

### 8.1 Core ML

| Tool | Version | Justification |
|------|---------|---------------|
| **XGBoost** | ≥ 1.7 | Industry standard in fintech for tabular data. Natively supports `scale_pos_weight` for imbalance. Fast SHAP TreeExplainer support. |
| **LightGBM** | ≥ 4.0 | Secondary baseline for speed comparison. Histogram-based splits make it faster than XGBoost on large datasets. |
| **PyTorch** | ≥ 2.0 | TFT implementation backbone. GPU optional. |
| **PyTorch Forecasting** | ≥ 1.0 | Provides `TemporalFusionTransformer` class directly. Handles sequence batching, attention, gating internally. |
| **PyTorch Lightning** | ≥ 2.0 | Training loop abstraction for TFT. Handles checkpointing, early stopping, LR scheduling. |
| **scikit-learn** | ≥ 1.3 | Preprocessing, metrics, pipeline utilities. |
| **imbalanced-learn** | ≥ 0.11 | SMOTE, ADASYN, RandomOverSampler implementations. |

### 8.2 Explainability

| Tool | Version | Justification |
|------|---------|---------------|
| **SHAP** | ≥ 0.44 | De facto standard for model explainability in finance. TreeExplainer is exact (not approximate) for XGBoost. |

### 8.3 Experiment Tracking

| Tool | Version | Justification |
|------|---------|---------------|
| **MLflow** | ≥ 2.10 | Open source, self-hosted experiment tracking. Runs locally during training only (`mlflow ui`). Not deployed as a Docker service — training is a one-time activity, not a runtime concern. |

### 8.4 Serving & Streaming

| Tool | Version | Justification |
|------|---------|---------------|
| **FastAPI** | ≥ 0.110 | Async-capable, auto-generates OpenAPI docs, Pydantic validation built in. |
| **Uvicorn** | ≥ 0.27 | ASGI server for FastAPI. Production-grade with worker configuration. |
| **Apache Kafka** | 3.5+ KRaft (via Docker) | Industry-standard event streaming. KRaft mode eliminates the Zookeeper dependency — single container, no coordination service needed. |
| **kafka-python** | ≥ 2.0 | Python Kafka client for producer/consumer scripts. |

### 8.5 Monitoring

| Tool | Version | Justification |
|------|---------|---------------|
| **Evidently AI** | ≥ 0.4 | Purpose-built ML monitoring. Drift reports, prediction distribution, data quality — all with one `Report` call. |
| **Prometheus** | Latest (Docker) | Pull-based metrics collection. FastAPI exposes `/metrics` via `prometheus-fastapi-instrumentator`. |
| **Grafana** | Latest (Docker) | Visualization layer over Prometheus. Pre-built dashboards importable via JSON. |

### 8.6 Data & Storage

| Tool | Justification |
|------|---------------|
| **pandas** | Primary dataframe manipulation |
| **numpy** | Numerical operations |
| **pyarrow / parquet** | Intermediate storage of processed features (faster than CSV for reloading) |

### 8.7 Development & DevOps

| Tool | Justification |
|------|---------------|
| **Docker + Docker Compose** | One-command reproducible environment for all services |
| **Make** | `Makefile` for common commands: `make train`, `make serve`, `make test`, `make reproduce` |
| **pytest** | Unit and integration testing |
| **black + isort + flake8** | Code formatting and linting |
| **pre-commit** | Runs linters on every commit |
| **python-dotenv** | `.env` file loading for secrets |
| **pyyaml** | Configuration loading |
| **Optuna** | Hyperparameter optimization (Phase 2 stretch goal) |

---

## 9. Dataset Specification

### 9.1 Primary Dataset: IEEE-CIS Fraud Detection

**Source:** https://www.kaggle.com/competitions/ieee-fraud-detection/data  
**License:** Competition use — suitable for portfolio projects  
**Size:** ~590MB (train_transaction.csv + train_identity.csv)

**Key files:**
```
data/raw/
├── train_transaction.csv   # 590,540 rows × 394 columns
└── train_identity.csv      # 144,233 rows × 41 columns (joinable on TransactionID)
```

**Target column:** `isFraud` (binary: 0 = legitimate, 1 = fraud)  
**Fraud rate:** ~3.5% (higher than real-world ~0.1%, but still highly imbalanced)  
**Key temporal column:** `TransactionDT` (seconds offset from reference point, not absolute timestamp)

### 9.2 Fallback Dataset: PaySim

**Source:** https://www.kaggle.com/datasets/ealaxi/paysim1  
**Use if:** IEEE-CIS cannot be downloaded (requires Kaggle account + competition acceptance)  
**Size:** ~470MB  
**Fraud rate:** 0.13%  
**Advantage:** Has explicit transaction type column (TRANSFER, PAYMENT, etc.) which enables richer feature engineering

### 9.3 Critical Data Notes

- **Do NOT use random train/test split**. IEEE-CIS has a temporal dimension (`TransactionDT`). Use the first 80% of time-sorted data for training, last 20% for testing. This prevents data leakage from future fraud patterns.
- **Do NOT apply SMOTE to the test set**. SMOTE is a training-time augmentation only.
- **Do NOT impute test set using training set statistics computed post-split**. Fit all scalers/encoders on training data only, then transform test data.
- The identity table join introduces NaN for ~75% of transactions. This is expected and must be handled explicitly.
- `TransactionDT` must be converted to cyclical time features (hour_sin, hour_cos, day_of_week_sin, etc.) — not used as raw integer.

### 9.4 Feature Groups (Post-Engineering)

After feature engineering, features should fall into these semantic groups (important for SHAP grouping and dashboard organization):

| Group | Examples |
|-------|---------|
| **Transaction Amount** | `TransactionAmt`, `amount_log`, `amount_zscore`, `amount_percentile` |
| **Temporal** | `hour_sin`, `hour_cos`, `day_of_week`, `days_since_first_tx` |
| **Card / Account** | `card1`, `card2`, `card_type`, `card_category` |
| **Aggregated History** | `card_tx_count_7d`, `card_amt_mean_7d`, `card_fraud_rate_30d` |
| **Product** | `ProductCD`, `P_emaildomain`, `R_emaildomain` |
| **Device / Identity** | `DeviceType`, `DeviceInfo` (from identity join) |
| **V-features** | `V1`–`V339` (masked Vesta features — treat as numerical, do PCA or keep top-k) |

---

## 10. Development Phases & Sprints

---

### Phase 0 — Foundation & Environment Setup

**Sprint:** 0  
**Duration:** 1–2 days  
**Objective:** Working local environment, repo structure, dataset downloaded, Docker services verify.

#### Action Items

**0.1 — Repository Initialization**
```bash
git init fraud-detection-explainable
cd fraud-detection-explainable
# Create .gitignore (Python, data, models, .env)
# Create directory structure (see Section 12)
```

**0.2 — Python Environment**
```bash
conda activate fraudx
# Create requirements.txt with pinned versions
pip install -r requirements.txt
```

`requirements.txt` must include:
```
xgboost==2.0.3
lightgbm==4.3.0
torch==2.2.0
pytorch-forecasting==1.0.0
pytorch-lightning==2.2.0
scikit-learn==1.4.0
imbalanced-learn==0.12.0
shap==0.44.0
mlflow==2.11.0
fastapi==0.110.0
uvicorn[standard]==0.27.0
kafka-python==2.0.2
evidently==0.4.22
pandas==2.2.0
numpy==1.26.0
pyarrow==15.0.0
pydantic==2.6.0
pyyaml==6.0.1
python-dotenv==1.0.0
optuna==3.6.1
prometheus-fastapi-instrumentator==6.4.0
matplotlib==3.8.0
seaborn==0.13.2
plotly==5.19.0
pytest==8.0.0
black==24.2.0
isort==5.13.0
flake8==7.0.0
pre-commit==3.6.0
```

**0.3 — Configuration File**

Create `config/config.yaml`:
```yaml
project:
  name: "fraud-detection-explainable"
  version: "1.0.0"
  random_seed: 42

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  train_file: "train_transaction.csv"
  identity_file: "train_identity.csv"
  target_col: "isFraud"
  temporal_col: "TransactionDT"
  train_split_ratio: 0.80
  sequence_length: 10  # for TFT: last N transactions per card

features:
  drop_cols: ["TransactionID", "TransactionDT"]
  v_features_pca_components: 30  # PCA on V1-V339
  categorical_cols: ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]
  numerical_cols: []  # auto-detected after drop and cat separation

model:
  xgboost:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    scale_pos_weight: 29  # set to (neg_count / pos_count) at runtime
    eval_metric: "aucpr"
    early_stopping_rounds: 50
  tft:
    max_encoder_length: 10
    max_prediction_length: 1
    hidden_size: 64
    attention_head_size: 4
    dropout: 0.1
    hidden_continuous_size: 32
    learning_rate: 0.001
    max_epochs: 30
    batch_size: 128

imbalance:
  strategy: "smote"  # options: smote, class_weight, focal_loss, none
  smote_k_neighbors: 5
  focal_loss_gamma: 2.0
  focal_loss_alpha: 0.25

thresholds:
  default: 0.5
  cost_fn: 500    # cost of false negative ($): fraud goes undetected
  cost_fp: 5      # cost of false positive ($): legitimate tx blocked, customer friction
  revenue_tp: 480 # value recovered from true positive: money saved + investigation avoided

mlflow:
  tracking_uri: "http://localhost:5000"  # local only — run `make mlflow` during training
  experiment_name: "fraud_detection"

serving:
  host: "0.0.0.0"
  port: 8000
  model_path: "models/xgb_model.pkl"
  tft_model_path: "models/tft_model.pt"
  log_file: "logs/predictions.jsonl"

kafka:
  bootstrap_servers: "localhost:9092"
  input_topic: "transactions"
  output_topic: "fraud_alerts"
  consumer_group: "fraud_detector"
  producer_rate_per_second: 100

monitoring:
  reference_data_path: "data/processed/train_features.parquet"
  evidently_report_dir: "monitoring/reports"
  drift_check_interval_hours: 24
  prometheus_port: 8001
```

**0.4 — Docker Compose Base**

Create `docker-compose.yml` with the following **4 services**. This is the complete set — no more, no less.

- `fraud-api` — FastAPI inference service with Kafka consumer running as an asyncio background task inside the same process
- `kafka` — Apache Kafka in **KRaft mode** (no Zookeeper required; single-node, single container)
- `prometheus` — metrics scraping from `fraud-api /metrics`
- `grafana` — dashboard visualization over Prometheus

**MLflow is NOT a Docker service.** It runs locally during training only:
```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```
Stop it when training is done. It serves no purpose in the runtime serving stack.

**Kafka producer is NOT a Docker service.** It is a one-shot Python script run from the terminal when you want to simulate a transaction stream:
```bash
python src/streaming/producer.py --rate 200 --limit 5000
```

**0.5 — Makefile**

```makefile
.PHONY: setup data train serve mlflow test monitor reproduce stream docker-up docker-down

setup:
	pip install -r requirements.txt
	pre-commit install

data:
	python src/data/download_data.py
	python src/data/preprocess.py

train:
	python src/training/train_xgb.py
	python src/training/train_tft.py

mlflow:
	mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

stream:
	python src/streaming/producer.py --rate 200 --limit 5000

test:
	pytest tests/ -v --cov=src --cov-report=html

monitor:
	python src/monitoring/drift_reporter.py

reproduce:
	make data && make train

docker-up:
	docker compose up -d

docker-down:
	docker compose down
```

**0.6 — Dataset Download Script**

`src/data/download_data.py` — this script should:
1. Check if Kaggle credentials exist in `~/.kaggle/kaggle.json`
2. Download the IEEE-CIS dataset using `kaggle competitions download ieee-fraud-detection`
3. Extract CSV files to `data/raw/`
4. Print SHA256 checksums for reproducibility verification

**Phase 0 Done When:**
- [ ] `make setup` completes without errors
- [ ] `docker compose up -d` starts all **4 services** (fraud-api, kafka, prometheus, grafana)
- [ ] Raw data CSVs present in `data/raw/`
- [ ] `config/config.yaml` validated (loaded by test: `tests/test_config.py`)

---

### Phase 1 — Data Ingestion, EDA & Feature Engineering

**Sprint:** 1  
**Duration:** 3–5 days  
**Objective:** Clean, feature-rich, properly split dataset ready for model training.

#### Action Items

**1.1 — Exploratory Data Analysis Notebook**

Create `notebooks/01_eda.ipynb`. This notebook documents findings but is NOT part of the production pipeline. It must cover:

- Target distribution (fraud rate, counts)
- Missing value heatmap (especially identity join NaNs)
- Transaction amount distribution (raw vs. log-transformed) by class
- Correlation of V-features with target (to justify PCA reduction)
- Temporal distribution: fraud rate by hour of day, day of week
- Card-level aggregation: transaction count and amount distributions
- Key insight: plot fraud rate by ProductCD — this is a strong signal

**1.2 — Data Loader Module**

File: `src/data/data_loader.py`

```python
class DataLoader:
    def __init__(self, config: dict):
        ...
    
    def load_raw(self) -> pd.DataFrame:
        """Load and merge train_transaction + train_identity."""
        ...
    
    def validate_schema(self, df: pd.DataFrame) -> None:
        """Assert expected columns exist, target col is binary, no future leakage cols."""
        ...
    
    def sort_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by TransactionDT ascending. Critical for time-based split."""
        ...
```

**1.3 — Feature Engineering Module**

File: `src/data/feature_engineering.py`

Implement these features in order. Each feature group is a separate method for testability:

```python
class FeatureEngineer:
    def create_temporal_features(self, df) -> pd.DataFrame:
        """
        Convert TransactionDT to:
        - hour_of_day (0-23)
        - day_of_week (0-6)  
        - hour_sin = sin(2π * hour / 24)
        - hour_cos = cos(2π * hour / 24)
        - day_sin = sin(2π * day / 7)
        - day_cos = cos(2π * day / 7)
        """
    
    def create_amount_features(self, df) -> pd.DataFrame:
        """
        - amount_log = log1p(TransactionAmt)
        - amount_cents = TransactionAmt mod 1 (transactions ending in .00 vs .99)
        """
    
    def create_card_aggregates(self, df) -> pd.DataFrame:
        """
        Group by card1 (proxy for card number), compute rolling stats.
        CRITICAL: Must be computed on sorted data, with no leakage from test rows.
        - tx_count_per_card: total transactions on this card seen so far
        - mean_amount_per_card: mean amount on this card
        - max_amount_per_card: historical max
        - std_amount_per_card: historical std
        - amount_vs_mean_ratio: current tx / card mean (anomaly signal)
        """
    
    def encode_categoricals(self, df, fit=True) -> pd.DataFrame:
        """
        Frequency encode high-cardinality categoricals (P_emaildomain, card1).
        Label encode low-cardinality categoricals (ProductCD, card4, card6).
        fit=True: fit encoders (train). fit=False: transform only (test/inference).
        """
    
    def reduce_v_features(self, df, fit=True) -> pd.DataFrame:
        """
        PCA on V1-V339 columns → 30 components.
        fit=True: fit PCA (train). fit=False: transform only (test/inference).
        """
    
    def handle_missing_values(self, df, fit=True) -> pd.DataFrame:
        """
        Numerical NaN: fill with -999 (tree models handle this well; signals missingness).
        Categorical NaN: fill with 'MISSING' string before encoding.
        fit=True: fit imputers on train. fit=False: use fitted imputers on test.
        """
    
    def save_transformers(self, path: str) -> None:
        """Pickle all fitted transformers (scalers, encoders, PCA) to disk."""
    
    def load_transformers(self, path: str) -> None:
        """Load transformers for inference-time use."""
```

**1.4 — Data Splitter Module**

File: `src/data/data_splitter.py`

```python
def time_based_split(df: pd.DataFrame, temporal_col: str, ratio: float) -> tuple:
    """
    Sort by temporal_col, take first `ratio` fraction as train, remainder as test.
    MUST NOT shuffle. Returns (X_train, X_test, y_train, y_test).
    Logs: train shape, test shape, fraud rate in each split.
    """
```

**1.5 — Imbalance Handler Module**

File: `src/data/imbalance_handler.py`

```python
class ImbalanceHandler:
    def apply_smote(self, X_train, y_train) -> tuple:
        """SMOTE with k_neighbors from config. Log class counts before/after."""
    
    def get_scale_pos_weight(self, y_train) -> float:
        """Return neg_count / pos_count for XGBoost scale_pos_weight."""
    
    def get_class_weights(self, y_train) -> dict:
        """Compute sklearn-style class weight dict for neural model."""
```

**1.6 — Save Processed Features**

Save output of feature engineering as parquet files:
```
data/processed/
├── train_features.parquet     # X_train (features only)
├── test_features.parquet      # X_test
├── train_labels.parquet       # y_train  
├── test_labels.parquet        # y_test
└── transformers/
    ├── pca.pkl
    ├── label_encoders.pkl
    ├── freq_encoders.pkl
    └── imputer.pkl
```

**Phase 1 Done When:**
- [ ] `notebooks/01_eda.ipynb` runs end-to-end and is committed (cleared outputs)
- [ ] `python src/data/preprocess.py` produces all parquet files
- [ ] `tests/test_feature_engineering.py` passes (no leakage, correct NaN handling, correct split sizes)
- [ ] Fraud rate in train split is documented and matches EDA findings
- [ ] All transformers serialized and loadable

---

### Phase 2 — Baseline Model (XGBoost/LightGBM)

**Sprint:** 2  
**Duration:** 3–4 days  
**Objective:** A well-tuned, calibrated XGBoost model with MLflow tracking and threshold optimization.

#### Action Items

**2.1 — XGBoost Trainer**

File: `src/training/train_xgb.py`

```python
class XGBTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.feature_names = None
    
    def build_model(self, scale_pos_weight: float) -> xgb.XGBClassifier:
        """Instantiate XGBClassifier with config params + scale_pos_weight."""
    
    def train(self, X_train, y_train, X_val, y_val) -> None:
        """
        Train with early stopping on validation set.
        Use eval_metric='aucpr' (PR-AUC, not ROC-AUC).
        Log to MLflow: all hyperparams, eval metrics per round.
        """
    
    def predict_proba(self, X) -> np.ndarray:
        """Return fraud probability scores (column 1)."""
    
    def save(self, path: str) -> None:
        """Save model + feature_names to pickle."""
    
    @classmethod
    def load(cls, path: str) -> 'XGBTrainer':
        """Load model from pickle."""
```

**2.2 — Evaluator Module**

File: `src/evaluation/evaluator.py`

This module is the most important for the portfolio signal. It must produce:

```python
class ModelEvaluator:
    def compute_pr_auc(self, y_true, y_prob) -> float:
        """Compute area under Precision-Recall curve."""
    
    def compute_roc_auc(self, y_true, y_prob) -> float:
        """For completeness but NOT primary metric."""
    
    def find_optimal_threshold(self, y_true, y_prob, cost_fn, cost_fp) -> float:
        """
        For each threshold t in [0.01, 0.99]:
            cost = FP_count(t) * cost_fp + FN_count(t) * cost_fn
        Return threshold that minimizes total cost.
        This is the threshold used for binary decisions.
        """
    
    def compute_metrics_at_threshold(self, y_true, y_prob, threshold) -> dict:
        """Return: precision, recall, f1, accuracy, FP, FN, TP, TN."""
    
    def plot_pr_curve(self, y_true, y_prob, model_name, save_path) -> None:
        """PR curve with AUC in legend."""
    
    def plot_roc_curve(self, y_true, y_prob, model_name, save_path) -> None:
        """ROC curve."""
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path) -> None:
        """Normalized confusion matrix heatmap."""
    
    def plot_threshold_vs_business_value(self, y_true, y_prob, cost_fn, cost_fp, revenue_tp, save_path) -> None:
        """
        X-axis: threshold (0.01 to 0.99)
        Y-axis: net business value (TP*revenue_tp - FP*cost_fp - FN*cost_fn)
        Mark optimal threshold with vertical line.
        """
    
    def generate_classification_report(self, y_true, y_pred) -> str:
        """Full sklearn classification report as string."""
```

**2.3 — MLflow Integration**

Each training run logs:
- **Parameters**: all hyperparams from config
- **Metrics**: pr_auc, roc_auc, f1, precision, recall (all at optimal threshold)
- **Artifacts**: model pickle, PR curve PNG, confusion matrix PNG, feature importance PNG
- **Tags**: model_type, dataset, imbalance_strategy, random_seed

**2.4 — Imbalance Ablation Notebook**

Create `notebooks/02_imbalance_ablation.ipynb`:

Compare these three setups on the same XGBoost architecture:
1. `scale_pos_weight` only (no oversampling)
2. SMOTE on training data
3. `scale_pos_weight` + SMOTE

Compare by: PR-AUC, F1 at optimal threshold, training time.  
**Conclusion cell must declare winner and reasoning.**

**2.5 — LightGBM Variant (Optional but Recommended)**

File: `src/training/train_lgbm.py` — Mirror of `train_xgb.py` but using LightGBM.
Compare speed vs. XGBoost on the same dataset. 
Document which is faster and whether performance differs meaningfully.

**Phase 2 Done When:**
- [ ] XGBoost trains, evaluates, and saves to `models/xgb_model.pkl`
- [ ] MLflow experiment shows logged run at `http://localhost:5000`
- [ ] PR-AUC ≥ 0.70 on test set (IEEE-CIS)
- [ ] Optimal threshold identified using cost matrix from config
- [ ] Imbalance ablation notebook committed with clear conclusion
- [ ] `tests/test_xgb_trainer.py` passes (model loads, predicts correct shape)

---

### Phase 3 — Advanced Sequential Model (TFT / Transformer)

**Sprint:** 3  
**Duration:** 5–7 days (most complex phase)  
**Objective:** Temporal Fusion Transformer that uses transaction history context for improved fraud detection.

#### Action Items

**3.1 — Sequence Builder Module**

File: `src/data/sequence_builder.py`

This is the most nuanced data engineering step. The TFT requires sequences — for each transaction, we need to provide the last N transactions on the same card as context.

```python
class SequenceBuilder:
    def __init__(self, sequence_length: int, group_col: str = 'card1'):
        """
        sequence_length: number of historical transactions to include
        group_col: column that identifies "same account" (card1 is proxy for card number)
        """
    
    def build_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each transaction row:
        1. Look back sequence_length transactions on the same card1 value
        2. If fewer than sequence_length exist, pad with zeros (or -999 for numericals)
        3. Produce a 'time_idx' column (0, 1, 2, ... per card group)
        
        Output format must match PyTorch Forecasting's TimeSeriesDataSet expectations:
        - 'group_id': string identifier for each card (card1 as string)
        - 'time_idx': integer, monotonically increasing within each group
        - All feature columns
        - Target column: isFraud
        """
    
    def create_pytorch_dataset(self, df: pd.DataFrame, train: bool) -> TimeSeriesDataSet:
        """
        Wrap the sequence DataFrame into PyTorch Forecasting TimeSeriesDataSet.
        - time_varying_known_reals: temporal features (hour_sin, hour_cos, ...)
        - time_varying_unknown_reals: transaction features (amount, V-features, ...)
        - static_categoricals: card4, card6, ProductCD
        - target: isFraud
        """
```

**3.2 — TFT Trainer Module**

File: `src/training/train_tft.py`

```python
class TFTTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.trainer = None
    
    def build_model(self, training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
        """
        Instantiate TFT from PyTorch Forecasting.
        Key params from config:
        - hidden_size, attention_head_size, dropout
        - output_size: 1 (binary fraud probability)
        - loss: BinaryMSELoss or custom FocalLoss
        - learning_rate from config
        """
    
    def build_trainer(self) -> pl.Trainer:
        """
        PyTorch Lightning Trainer with:
        - EarlyStopping callback (patience=5, monitor='val_loss')
        - ModelCheckpoint callback (save best by val_loss)
        - LearningRateMonitor
        - max_epochs from config
        - GPU if available, else CPU
        """
    
    def train(self, train_dataset, val_dataset) -> None:
        """
        Create DataLoaders, run trainer.fit().
        Log to MLflow: val_loss per epoch, final pr_auc.
        """
    
    def predict(self, dataset) -> np.ndarray:
        """Return fraud probability for each item in dataset."""
    
    def save(self, path: str) -> None:
        """Save PyTorch Lightning checkpoint."""
    
    @classmethod
    def load(cls, path: str, config: dict) -> 'TFTTrainer':
        """Load from checkpoint."""
```

**3.3 — Focal Loss Implementation**

File: `src/training/losses.py`

```python
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    gamma: focusing parameter (2.0 default). Higher = more focus on hard examples.
    alpha: class balance weight (0.25 default for fraud class).
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...
```

**3.4 — Model Comparison Notebook**

Create `notebooks/03_model_comparison.ipynb`:

Side-by-side comparison of XGBoost vs. TFT:
- PR-AUC
- F1 at optimal threshold
- Precision-Recall curve on same axes
- Training time
- Inference time per prediction
- Top features (XGB feature importance vs. TFT attention weights)

**Conclusion must state which model you would deploy in production and why** (hint: XGBoost for latency + explainability, TFT as ensemble contributor for complex accounts with long history).

**3.5 — Ensemble (Optional Stretch)**

File: `src/models/ensemble.py`

Simple weighted average of XGBoost and TFT probabilities:
```
p_final = w_xgb * p_xgb + w_tft * p_tft
```
Find optimal weights via grid search on validation set.

**Phase 3 Done When:**
- [ ] `python src/training/train_tft.py` completes without error
- [ ] TFT PR-AUC ≥ 0.75 on test set
- [ ] TFT model checkpoint saved to `models/tft_model.ckpt`
- [ ] `notebooks/03_model_comparison.ipynb` committed with clear conclusion
- [ ] `tests/test_tft_trainer.py` passes (model loads, predicts correct shape for sequence input)

---

### Phase 4 — Explainability Layer (SHAP + Dashboards)

**Sprint:** 4  
**Duration:** 3–4 days  
**Objective:** Full SHAP-based explainability for both models, with a production-ready explanation format.

#### Action Items

**4.1 — SHAP Explainer Module**

File: `src/explainability/shap_explainer.py`

```python
class FraudExplainer:
    def __init__(self, model, model_type: str, feature_names: list, background_data=None):
        """
        model_type: 'xgboost' or 'tft'
        background_data: required for KernelExplainer (subset of training data, ~100 rows)
        """
    
    def build_explainer(self):
        """
        XGBoost → shap.TreeExplainer(model)  [exact, fast]
        TFT → shap.DeepExplainer(model, background_data)  [approximate]
        """
    
    def explain_single(self, X_single: pd.DataFrame) -> dict:
        """
        Compute SHAP values for one transaction.
        Returns:
        {
            "base_value": float,     # expected model output (mean training prediction)
            "features": [
                {"name": "amount_log", "value": 8.3, "shap_value": 0.31},
                {"name": "hour_sin", "value": -0.8, "shap_value": 0.18},
                ...
            ],
            "top_positive": [...],   # top 3 features pushing toward fraud
            "top_negative": [...],   # top 3 features pushing away from fraud
        }
        """
    
    def explain_batch(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for a batch. Returns (n_samples, n_features) array.
        Used for summary plots and drift analysis.
        """
    
    def plot_summary(self, X_test: pd.DataFrame, save_path: str) -> None:
        """
        Beeswarm / summary plot: feature importance + direction of effect.
        Save as PNG and HTML (interactive via plotly).
        """
    
    def plot_waterfall_single(self, X_single: pd.DataFrame, y_true: int, save_path: str) -> None:
        """
        Waterfall plot for one prediction showing how each feature
        contributes from base_value to final prediction.
        """
    
    def plot_dependence(self, X: pd.DataFrame, feature_name: str, save_path: str) -> None:
        """
        SHAP dependence plot for a specific feature.
        Shows how feature value affects prediction.
        """
    
    def plot_feature_importance(self, X: pd.DataFrame, top_n: int, save_path: str) -> None:
        """
        Bar chart of mean |SHAP value| per feature.
        Top `top_n` features.
        """
```

**4.2 — SHAP Analysis Notebook**

Create `notebooks/04_shap_analysis.ipynb`:

This notebook is the portfolio showcase notebook. It must include:

1. **Global Feature Importance** — Top 20 features by mean |SHAP value|. Annotate each with business meaning ("amount_log: high-value transactions are riskier").

2. **Beeswarm Summary Plot** — Full SHAP summary with color for feature value direction.

3. **Waterfall Plots for 3 example transactions:**
   - A correctly caught fraud case (TP)
   - A missed fraud case (FN) — why did the model fail?
   - A false alarm (FP) — what made this look suspicious?

4. **Dependence Plot for `TransactionAmt`** — Show the non-linear relationship.

5. **Interaction Plot** — SHAP interaction between `TransactionAmt` and `card_tx_count_7d`.

6. **Regulatory compliance note (markdown cell)** — Explicitly state: "Under RBI Model Risk Management guidelines and equivalent FCA/SEC frameworks, automated fraud decisions must be accompanied by a reason code. The SHAP values above provide the legal basis for explanation."

**4.3 — SHAP Dashboard Export**

File: `src/explainability/dashboard_generator.py`

```python
def generate_shap_dashboard(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
    output_path: str
) -> None:
    """
    Generate a standalone HTML file (no server required) that contains:
    - Feature importance bar chart (Plotly)
    - Beeswarm plot (SHAP JS library)
    - Sample-level waterfall plots for top 5 fraud predictions
    Saved to monitoring/shap_dashboard.html
    """
```

**4.4 — Explanation Format for API**

Define the `ExplanationResponse` Pydantic model now (used in Phase 5):

```python
class FeatureContribution(BaseModel):
    feature_name: str
    feature_value: float
    shap_contribution: float
    direction: str  # "increases_risk" or "decreases_risk"

class PredictionExplanation(BaseModel):
    base_fraud_rate: float
    top_risk_factors: list[FeatureContribution]  # top 5 positive SHAP
    top_mitigating_factors: list[FeatureContribution]  # top 5 negative SHAP
    explanation_confidence: str  # "high" (tree-based exact) or "approximate" (deep)
```

**Phase 4 Done When:**
- [ ] `notebooks/04_shap_analysis.ipynb` committed with all plots visible (PNG outputs committed to `reports/figures/`)
- [ ] `monitoring/shap_dashboard.html` generated and viewable in browser
- [ ] `explain_single()` returns correct format tested in `tests/test_shap_explainer.py`
- [ ] Waterfall plot for at least one TP, FN, FP documented

---

### Phase 5 — Model Serving via FastAPI

**Sprint:** 5  
**Duration:** 3–4 days  
**Objective:** Production-ready REST API that serves fraud predictions with SHAP explanations.

#### Action Items

**5.1 — Application Structure**

```
src/api/
├── main.py              # FastAPI app factory, router registration
├── routes/
│   ├── predict.py       # POST /predict
│   ├── health.py        # GET /health
│   └── metrics.py       # GET /metrics (Prometheus)
├── schemas/
│   ├── request.py       # TransactionRequest Pydantic model
│   └── response.py      # PredictionResponse Pydantic model
├── middleware/
│   ├── logging.py       # Request/response structured JSON logging
│   └── timing.py        # Latency measurement middleware
└── model_loader.py      # Singleton model loading on startup
```

**5.2 — Request Schema**

File: `src/api/schemas/request.py`

```python
class TransactionRequest(BaseModel):
    transaction_id: str
    transaction_amount: float
    product_code: str  # A, B, C, D, W
    card1: int
    card4: Optional[str]  # visa, mastercard, etc.
    card6: Optional[str]  # credit, debit
    p_email_domain: Optional[str]
    r_email_domain: Optional[str]
    # ... all input features needed by the model
    # Note: temporal features (hour_sin etc.) computed server-side from timestamp
    request_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TX_12345",
                "transaction_amount": 299.99,
                "product_code": "W",
                ...
            }
        }
    )
```

**5.3 — Response Schema**

File: `src/api/schemas/response.py`

```python
class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(ge=0.0, le=1.0)
    decision: str  # "FRAUD" or "LEGITIMATE"
    threshold_used: float
    model_version: str
    latency_ms: float
    explanation: PredictionExplanation  # from Phase 4
    timestamp: datetime
    
class HealthResponse(BaseModel):
    status: str  # "healthy" or "degraded"
    model_loaded: bool
    uptime_seconds: float
    total_predictions: int
    fraud_rate_last_1000: float
```

**5.4 — Prediction Route**

File: `src/api/routes/predict.py`

```python
@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(request: TransactionRequest) -> PredictionResponse:
    """
    1. Validate request (Pydantic handles this automatically)
    2. Transform features (apply loaded transformers)
    3. Run XGBoost inference
    4. Run TFT inference if transaction history available
    5. Compute ensemble score (or XGB only if no history)
    6. Apply SHAP explainer
    7. Apply threshold from config
    8. Log to predictions.jsonl
    9. Return PredictionResponse
    
    Error handling:
    - 422: validation error (bad input)
    - 503: model not loaded
    - 500: internal inference error (logged, generic message returned)
    """
```

**5.5 — Model Loader (Singleton)**

File: `src/api/model_loader.py`

```python
class ModelRegistry:
    """
    Loaded once on startup via FastAPI lifespan event.
    Holds: xgb_model, tft_model, feature_transformer, shap_explainer, threshold.
    Thread-safe read (models are stateless after loading).
    """
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'ModelRegistry':
        if cls._instance is None:
            cls._instance = cls._load_all_models()
        return cls._instance
```

**5.6 — Structured Logging Middleware**

Every request must log:
```json
{
  "timestamp": "2026-01-15T10:23:45.123Z",
  "transaction_id": "TX_12345",
  "fraud_probability": 0.87,
  "decision": "FRAUD",
  "latency_ms": 23.4,
  "model_version": "xgb_v1.0",
  "top_feature": "amount_log",
  "top_shap_value": 0.31
}
```
Written to `logs/predictions.jsonl` (one JSON object per line — queryable with `jq`).

**5.7 — Prometheus Metrics**

Using `prometheus-fastapi-instrumentator`, expose:
- `fraud_prediction_total` (counter) — total predictions made
- `fraud_rate_gauge` (gauge) — rolling fraud rate over last 1000 predictions
- `http_request_duration_seconds` (histogram) — latency distribution
- `model_version_info` (info) — current loaded model version

**5.8 — API Integration Test**

File: `tests/test_api.py`

```python
# Uses FastAPI TestClient
def test_predict_legitimate_transaction():
    """Send known-legitimate transaction, expect fraud_probability < threshold."""

def test_predict_fraud_transaction():
    """Send known-fraud transaction (from test set), expect fraud_probability > threshold."""

def test_predict_returns_explanation():
    """Verify response contains explanation with at least 3 features."""

def test_health_endpoint():
    """GET /health returns status=healthy and model_loaded=True."""

def test_invalid_request_returns_422():
    """Send request missing required fields, expect 422."""

def test_latency_under_100ms():
    """Time 100 predictions, assert P95 < 100ms."""
```

**Phase 5 Done When:**
- [ ] `make serve` starts API, swagger docs accessible at `http://localhost:8000/docs`
- [ ] `tests/test_api.py` passes all 6 tests
- [ ] P95 latency < 100ms verified in test output
- [ ] `logs/predictions.jsonl` populates on every request
- [ ] Prometheus metrics visible at `http://localhost:8000/metrics`

---

### Phase 6 — Real-Time Inference Simulation (Kafka)

**Sprint:** 6  
**Duration:** 2–3 days  
**Objective:** Simulate live transaction stream, demonstrate the model operating in a streaming context.

#### Action Items

**6.1 — Kafka Docker Setup (KRaft Mode)**

Kafka 3.5+ supports KRaft — its own internal consensus protocol — which eliminates the Zookeeper dependency entirely. One container, zero coordination overhead.

Add to `docker-compose.yml` (this is the only Kafka-related service needed):
```yaml
kafka:
  image: confluentinc/cp-kafka:7.5.0
  ports:
    - "9092:9092"
  environment:
    KAFKA_NODE_ID: 1
    KAFKA_PROCESS_ROLES: broker,controller
    KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
    KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
    KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT
    KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
    KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
    KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
    KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
    KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
    KAFKA_LOG_DIRS: /var/lib/kafka/data
    CLUSTER_ID: "fraud-detection-cluster-01"
```

No `zookeeper` service. No `depends_on: [zookeeper]`. No `KAFKA_ZOOKEEPER_CONNECT`. This is intentional.

**6.2 — Kafka Producer**

File: `src/streaming/producer.py`

```python
class TransactionProducer:
    """
    Reads test transactions from test_features.parquet.
    Publishes to Kafka 'transactions' topic at configurable rate.
    
    Rate limiting: sleep(1 / rate_per_second) between publishes.
    Message format: JSON-serialized transaction row.
    Message key: card1 value (ensures transactions from same card go to same partition).
    """
    
    def __init__(self, config: dict):
        self.producer = KafkaProducer(
            bootstrap_servers=config['kafka']['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8')
        )
    
    def produce_from_file(self, file_path: str, rate: int = 100, limit: int = None):
        """
        Stream transactions from parquet file to Kafka topic.
        rate: transactions per second
        limit: stop after N transactions (None = stream all)
        Log every 1000 messages published.
        """
    
    def produce_single(self, transaction: dict):
        """Publish one transaction. Used for testing."""
```

**6.3 — Kafka Consumer (Background Task inside fraud-api)**

The consumer does not get its own container. It runs as an `asyncio` background coroutine started inside FastAPI's `lifespan` event — same process, same memory space as the model, no HTTP hop for predictions.

File: `src/streaming/consumer.py`

```python
import asyncio
import json
import logging
from kafka import KafkaConsumer, KafkaProducer
from src.api.model_loader import ModelRegistry

logger = logging.getLogger(__name__)

class FraudDetectionConsumer:
    """
    Kafka consumer that runs as an asyncio background task inside fraud-api.
    Calls the model directly in-process (no HTTP round-trip).
    Publishes fraud alerts to 'fraud_alerts' topic.
    """

    def __init__(self, config: dict):
        self.config = config
        self.consumer = KafkaConsumer(
            config['kafka']['input_topic'],
            bootstrap_servers=config['kafka']['bootstrap_servers'],
            group_id=config['kafka']['consumer_group'],
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='earliest',
            consumer_timeout_ms=1000  # non-blocking poll
        )
        self.alert_producer = KafkaProducer(
            bootstrap_servers=config['kafka']['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self._running = False

    def process_message(self, message: dict) -> dict:
        """
        Call model directly via ModelRegistry (no HTTP).
        Apply feature transform, run XGB inference, compute SHAP.
        Return prediction dict.
        If fraud_probability > threshold: publish to fraud_alerts topic.
        """
        registry = ModelRegistry.get_instance()
        features = registry.transformer.transform(message)
        prob = registry.xgb_model.predict_proba(features)[0][1]
        explanation = registry.explainer.explain_single(features)

        result = {
            "transaction_id": message.get("transaction_id"),
            "fraud_probability": float(prob),
            "decision": "FRAUD" if prob > registry.threshold else "LEGITIMATE",
            "explanation": explanation
        }

        if result["decision"] == "FRAUD":
            self.alert_producer.send(
                self.config['kafka']['output_topic'],
                value=result
            )
            logger.warning(f"FRAUD ALERT: {result['transaction_id']} p={prob:.3f}")

        return result

    async def run_async(self):
        """
        Asyncio-compatible consumer loop.
        Runs in background — does not block the FastAPI event loop.
        Uses asyncio.to_thread() so the blocking Kafka poll runs in a thread pool.
        """
        self._running = True
        logger.info("Kafka consumer background task started")
        msg_count = 0

        while self._running:
            # Offload blocking poll to thread pool — keeps event loop free
            messages = await asyncio.to_thread(self._poll_batch)
            for msg in messages:
                try:
                    self.process_message(msg.value)
                    msg_count += 1
                    if msg_count % 100 == 0:
                        logger.info(f"Consumer processed {msg_count} messages")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

    def _poll_batch(self):
        """Synchronous Kafka poll — called from thread pool."""
        records = self.consumer.poll(timeout_ms=500)
        return [msg for batch in records.values() for msg in batch]

    def stop(self):
        self._running = False
        self.consumer.close()
        self.alert_producer.close()
```

Register the consumer in `src/api/main.py` using FastAPI's `lifespan`:

```python
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from src.streaming.consumer import FraudDetectionConsumer
from src.api.model_loader import ModelRegistry

consumer_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models, start Kafka consumer background task
    ModelRegistry.get_instance()
    consumer = FraudDetectionConsumer(config=load_config())
    consumer_task = asyncio.create_task(consumer.run_async())

    yield  # app is running

    # Shutdown: stop consumer cleanly
    consumer.stop()
    consumer_task.cancel()

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)
```

**Why this design is correct:**
- No extra container, no extra network hop, no separate process to manage
- Model is called directly in memory — eliminates ~20ms HTTP overhead per message
- `asyncio.to_thread()` ensures the blocking Kafka poll doesn't freeze the event loop
- Consumer stops cleanly on `docker compose down` via the lifespan shutdown path

**6.4 — Streaming Demo Script**

File: `scripts/run_streaming_demo.sh`

```bash
#!/bin/bash
# Bring up the 4-service stack (kafka + fraud-api + prometheus + grafana)
# The Kafka consumer loop starts automatically inside fraud-api on startup.
# Run the producer script to simulate a live transaction stream.
# Print a summary after the run.

set -e

echo "Starting services..."
docker compose up -d
echo "Waiting for fraud-api to be healthy..."
until curl -sf http://localhost:8000/health > /dev/null; do sleep 2; done
echo "fraud-api is ready."

echo "Starting transaction producer: 5000 transactions at 200 tx/sec"
python src/streaming/producer.py --rate 200 --limit 5000

echo "Producer finished. Waiting 5s for consumer to drain..."
sleep 5

echo "Fetching stats..."
python scripts/streaming_stats.py
```

`scripts/streaming_stats.py` fetches `http://localhost:8000/metrics` and prints:
- Total predictions processed
- Fraud alert count
- Average prediction latency
- Consumer messages processed (from Prometheus counters)

**Phase 6 Done When:**
- [ ] `docker compose up` starts all 4 services cleanly (kafka, fraud-api, prometheus, grafana)
- [ ] `GET /health` confirms `kafka_consumer_running: true` in the response
- [ ] `make stream` (producer script) publishes 5000 test transactions at 200 tx/sec
- [ ] Fraud alerts appear in the `fraud_alerts` Kafka topic during the producer run
- [ ] `scripts/run_streaming_demo.sh` completes end-to-end in < 90 seconds
- [ ] Consumer lag stays < 500 messages throughout the demo (verify via Prometheus metric)

---

### Phase 7 — Monitoring & Drift Detection

**Sprint:** 7  
**Duration:** 2–3 days  
**Objective:** Automated drift detection, Prometheus metrics, Grafana dashboard.

#### Action Items

**7.1 — Evidently Drift Report**

File: `src/monitoring/drift_reporter.py`

```python
class DriftReporter:
    def __init__(self, reference_data_path: str, report_dir: str):
        self.reference_data = pd.read_parquet(reference_data_path)
        self.report_dir = report_dir
    
    def generate_data_drift_report(self, current_data: pd.DataFrame, report_name: str) -> None:
        """
        Evidently DataDriftPreset: compare reference (training) vs current window.
        Features monitored: all input features used by model.
        Output: HTML report saved to monitoring/reports/{report_name}.html
        Alert if: > 30% of features show drift (configurable threshold).
        """
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.reference_data, current_data=current_data)
        report.save_html(...)
    
    def generate_model_performance_report(
        self,
        current_data: pd.DataFrame,
        y_true: pd.Series,
        y_prob: pd.Series,
        report_name: str
    ) -> None:
        """
        Evidently ClassificationPreset: track PR-AUC, F1, fraud rate over time.
        """
    
    def check_drift_alert(self, report_path: str) -> bool:
        """
        Parse JSON summary of Evidently report.
        Return True if drift detected above threshold.
        Log alert with feature names that drifted.
        """
```

**7.2 — Prometheus Configuration**

Create `monitoring/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fraud-api'
    static_configs:
      - targets: ['fraud-api:8000']
    metrics_path: '/metrics'
```

**7.3 — Grafana Dashboard**

Create `monitoring/grafana/dashboards/fraud_detection.json`

Panels to include:
1. **Fraud Rate (%) — Time Series** — Rolling fraud rate over time. Alert if > 2x historical baseline.
2. **Prediction Latency P95 — Gauge** — Target line at 100ms.
3. **Request Volume — Bar Chart** — Predictions per minute.
4. **Model Version — Stat Panel** — Currently deployed model version.
5. **Feature Drift Indicator — Stat Panel** — Green/Red based on latest Evidently report.

Grafana datasource configured as Prometheus at `http://prometheus:9090`.

**7.4 — Automated Drift Check Schedule**

File: `src/monitoring/drift_scheduler.py`

Simple loop that:
1. Reads last N predictions from `logs/predictions.jsonl`
2. Reconstructs feature distribution from logged values
3. Calls `DriftReporter.generate_data_drift_report()`
4. If drift detected: writes to `monitoring/alerts/drift_alert_{timestamp}.json`
5. Sleeps for `config.monitoring.drift_check_interval_hours * 3600` seconds

**Phase 7 Done When:**
- [ ] Evidently HTML drift report generates cleanly at `monitoring/reports/`
- [ ] Prometheus scrapes FastAPI metrics (verify at `http://localhost:9090`)
- [ ] Grafana dashboard loads with all 5 panels (access at `http://localhost:3000`)
- [ ] `drift_scheduler.py` runs for 5 minutes and produces at least one report
- [ ] `tests/test_monitoring.py` tests that drift report generates without error

---

### Phase 8 — Business Impact Quantification

**Sprint:** 8  
**Duration:** 1–2 days  
**Objective:** Translate model metrics into financial terms that resonate with business stakeholders.

#### Action Items

**8.1 — Business Impact Notebook**

Create `notebooks/05_business_impact.ipynb`

This notebook is the **most important portfolio document** — it demonstrates that you think like a product-aware ML engineer, not just a data scientist.

**Section 1: Cost Matrix Definition**
```python
cost_false_negative = 500   # Average fraud loss that goes undetected ($)
cost_false_positive = 5     # Cost of blocking a legitimate transaction:
                             #   customer friction, support call, potential churn
revenue_true_positive = 480 # Value captured: fraud prevented minus investigation cost

# Context for IEEE-CIS dataset scaling:
daily_transactions = 590540 / 180   # dataset spans ~180 days
fraud_rate = 0.035                  # 3.5% fraud rate
daily_fraud_count = daily_transactions * fraud_rate
annual_scale_factor = 365
```

**Section 2: Model Performance Translation**
```python
# At optimal threshold (from Phase 2/evaluator):
tp_rate = recall_at_optimal  # fraud correctly caught
fp_rate = fpr_at_optimal     # legitimate tx incorrectly blocked

annual_tx = daily_transactions * 365
annual_fraud = annual_tx * fraud_rate

annual_tp = annual_fraud * tp_rate
annual_fn = annual_fraud * (1 - tp_rate)
annual_fp = (annual_tx - annual_fraud) * fp_rate

gross_benefit = annual_tp * revenue_true_positive
false_negative_loss = annual_fn * cost_false_negative
false_positive_cost = annual_fp * cost_false_positive
net_annual_value = gross_benefit - false_negative_loss - false_positive_cost
```

**Section 3: Threshold Sensitivity Analysis**
Plot net_annual_value vs. threshold from 0.01 to 0.99.  
Mark current optimal threshold and the business value at that point.  
This is the chart you show in an interview when asked "what's the business value of your model?".

**Section 4: Comparison Against Naive Baseline**
What is the business value of a model that predicts "no fraud" for everything?  
`naive_value = 0 - (annual_fraud * cost_false_negative)` — every fraud is missed.  
Show: `model_improvement = net_annual_value - naive_value`

**Section 5: Executive Summary Cell (markdown)**
```
## Model Business Summary

At the computed optimal threshold of {threshold}:

| Metric | Value |
|--------|-------|
| Annual transactions modeled | {annual_tx:,.0f} |
| Fraud cases detected | {annual_tp:,.0f} ({tp_rate:.1%} recall) |
| Legitimate transactions blocked | {annual_fp:,.0f} ({fp_rate:.2%} FPR) |
| Gross value from fraud prevention | ${gross_benefit:,.0f} |
| Cost of missed fraud | ${false_negative_loss:,.0f} |
| Cost of false alarms | ${false_positive_cost:,.0f} |
| **Net annual business value** | **${net_annual_value:,.0f}** |
| Improvement over "flag nothing" baseline | ${model_improvement:,.0f} |

Model interpretability provided via SHAP values on 100% of predictions,
compliant with RBI Model Risk Management and FCA Model Risk frameworks.
```

**Phase 8 Done When:**
- [ ] `notebooks/05_business_impact.ipynb` committed with all outputs visible
- [ ] Net annual business value computed and documented
- [ ] Threshold sensitivity plot generated and saved to `reports/figures/`
- [ ] Executive summary cell contains specific dollar figures

---

### Phase 9 — Precision Improvement (Recall-Preserving)

**Sprint:** 9
**Duration:** 3–6 days (sequential — see gate below)
**Status: CLOSED (gate not met, 2026-09-09).** Steps 9.0–9.4 ran; the best
result (9.4) reached **19.17% precision at recall ≥80%** — 10.8 points short
of the ≥30% target band. [ADR-004](adr/ADR-004-operating-point-selection.md)
shows in closed form that step 9.5 (the two-stage cascade) cannot close the
remaining gap and formally accepts the 9.4 operating point as final. See §9.5
below and `docs/IMPLEMENTATION_PLAN.md` PRD Phase 9 for the full step log.
**Objective:** Raise precision at the deployed operating point without giving up
recall, using changes evidenced in the fraud-detection literature rather than
guessed at.

#### Why this phase exists

At the deployed threshold (0.006123, blended ensemble) the model achieves
**95.5% recall at 6.2% precision** — 15.2 false positives per fraud caught,
597,480 legitimate customers blocked per year at dataset scale (Phase 8
figures). That ratio is too high for production use: most of what the model
flags is not fraud.

**Target band:** precision in the 30–40% range while keeping recall as close to
current as the literature supports. Research conducted 2026-09-08 (see
`docs/IMPLEMENTATION_PLAN.md` Phase 9 section for full citations) found the
best comparable published IEEE-CIS result at **P=0.3447, R=0.8275**
(AUPRC 0.7373) — i.e. the realistic frontier trades some recall for a
5–6x precision gain, not the exact "no recall loss" ideal. This phase is
scoped to get as close to that frontier as evidence supports, not to promise
95% recall at 30% precision simultaneously — no published IEEE-CIS result
combines those two numbers, and the arithmetic (§4.1) shows why: 95% recall at
30% precision implies a ~4.5% FPR, an order of magnitude below what this model
currently needs to reach 95% recall.

**Conflict with FR-05, stated explicitly.** FR-05 (§5) lists SMOTE oversampling
as a required imbalance-handling technique. The 2026-09-08 research found
published evidence that SMOTE/ADASYN *degrade* PR-AUC and precision for
gradient-boosted trees on fraud data while improving recall — the opposite of
what this phase needs — and that resampling broadly showed no reliable
improvement on a real (non-benchmark) imbalanced card dataset. **P9-2 below
therefore treats SMOTE as a candidate to test and likely reject, not a
requirement to implement.** FR-05's ablation notebook already exists
(`notebooks/02_imbalance_ablation.ipynb` — verify path) and can carry this
finding; FR-05 itself is not rewritten by this phase.

#### Sequencing and the early-stop gate

**Steps run in order, not in parallel.** 9.0 is measurement tooling, not a
retraining step — build it first so every later step has `precision_at_recall`
available for its evaluation. From 9.1 onward, each step is: (1) implement the
change, (2) retrain the affected model(s), (3) re-run the full evaluation
(PR-AUC, threshold sweep, precision-at-recall via 9.0, confusion matrix at the
deployed and re-derived thresholds, business-impact recompute via
`src/evaluation/business_impact.py`), (4) check the change against the target
band.

> **Stop condition: if a step's post-retrain metrics land in the target
> band (≥ 30% precision at ≥ 80% recall, or better), do not proceed to the
> next step.** Record the result, freeze that configuration, and move to
> Phase 10. Each further step is there because the previous one was
> insufficient, not because all steps are mandatory. Skipping straight to a
> later step out of order is not supported — later steps assume the earlier
> ones' features/weights are already in place.

**9.0 — Precision-at-fixed-recall reporting (evaluation tooling, no retraining)**

Add `ModelEvaluator.precision_at_recall(y_true, y_prob, target_recall)` to
`src/evaluation/evaluator.py`: walk `sklearn.metrics.precision_recall_curve`'s
output and return the precision at **the point with the largest threshold whose
recall is still ≥ `target_recall`** (the tightest constraint-satisfying
operating point — precision at the threshold you would deploy if recall were
pinned at exactly this floor). The naive "first point at or above
`target_recall`" reading is rejected: on the sklearn curve that is the
recall-1.0 corner for every floor, which collapses all recall targets to one
number and answers nothing. precision along the PR curve is sawtoothed, so the
returned value is not a guaranteed lower bound — it is precision at one
specific operating point. Validated as a real, currently-missing capability on
2026-09-08 — the
evaluator has `compute_metrics_at_threshold` (precision/recall at a given
*threshold*) and `find_optimal_threshold` (cost-optimal threshold), but nothing
answers "what is our precision if we require ≥70% recall", which is the form
the business actually asks the question in (per the research conducted this
session). This is independent of, and runs before, the retraining sequence —
report it once against the current model as the Phase 9 baseline, then again
after every step in 9.1–9.5 as a standard line in that step's evaluation, not
just the target-band pass/fail. Add a corresponding row to the Phase 8
sensitivity table style output and to `reports/RESULTS.md`.

**9.1 — Cost model resolution (config only, no retraining)**

Settle whether `revenue_tp` (config `thresholds.revenue_tp`) means recovery net
of principal or the principal itself (the double-count identified in the
2026-09-08 business-impact review — `revenue_tp=480` + `cost_fn=500` implies a
$980 swing per fraud, roughly double the $500 actually at stake). This is a
business decision, not a modeling change, and it changes the *threshold*, not
the model — re-derive the operating threshold from
`ModelEvaluator.find_optimal_threshold` (or the `net_of_principal` convention
in `business_impact.py`) against the existing model before touching any
feature or training code. **Check the stop condition here first** — thresholds
0.05–0.10 already reach 20–33% precision on the *current* model (Phase 8
sensitivity table), so this step alone may be enough to approach the target
band, at a real recall cost (65–75%).

**9.2 — Client/UID entity features (feature engineering, XGBoost + LightGBM retrain)**

Construct a client identifier the way the top Kaggle solutions for this exact
dataset did: `UID = card1_addr1 + "_" + floor(day - D1)`, where
`day = TransactionDT / 86400`. Build aggregation features on `UID` (frequency
encoding, `TransactionAmt` mean/std, `D`-column mean/std, `M`-column match-rate
means, categorical count features) — do **not** feed `UID` itself into the
model, only its aggregates, to avoid overfitting to an identifier absent at
serving time for a new card. This is additive to `create_card_aggregates`
(currently keyed on `card1` alone) in `src/data/feature_engineering.py`.
Retrain XGBoost and LightGBM (the TFT's existing `card1`-keyed sequence
grouping is a separate, already-present mechanism — evaluate whether it needs
the richer `UID` grouping too, but do not assume it does before measuring).

**9.3 — Imbalance-handling re-tuning (training config change, all 3 models retrain)**

Reduce `scale_pos_weight` (XGBoost/LightGBM) and the focal-loss `alpha`/`gamma`
(TFT) from their current recall-favoring settings. Sweep a small grid (e.g.
`scale_pos_weight` at 50%, 75%, 100% of current) and re-run the ablation
notebook comparing against class-weighting alone (no resampling) per the FR-05
conflict noted above. Do not apply SMOTE/ADASYN as the default path; if tested,
report the PR-AUC/precision delta explicitly so the ablation notebook shows
the evidence rather than assuming it helps.

**9.4 — Multi-window RFM/velocity aggregates (feature engineering, XGBoost + LightGBM retrain)**

Extend `create_velocity_features` and `create_card_aggregates`
(`src/data/feature_engineering.py`) with two things the pipeline does not have
today, both validated against this dataset on 2026-09-08 (see the
`docs/IMPLEMENTATION_PLAN.md` Phase 9 research table for the source of the
distinction):

- **Fixed rolling-window counts per card** — e.g. transaction count in the
  trailing 10 minutes / 1 hour / 24 hours (Whitrow, Bahnsen). Distinct from the
  existing `time_since_last_tx` (a single point-in-time gap) and from
  `create_card_aggregates`'s `tx_count_per_card` (an *expanding* window from
  the card's first seen transaction, not a fixed trailing window) — a card
  with 200 transactions over 6 months and one with 20 in the last hour can
  have the same expanding count but very different fixed-window counts, and
  only the second pattern reads as a burst.
- **Short-window vs. long-window comparison** — e.g. std of `TransactionAmt`
  over the trailing 24h against the card's 30-day mean/std. `mean_amount_per_card`
  already tracks the cumulative mean; this adds the deviation-from-recent-normal
  signal, which the cumulative mean cannot express once a card has enough
  history to dilute a recent spike.
- **`dist1` engineering, conditional on `ProductCD == 'W'`.** Verified against
  `data/raw/train_transaction.csv` on 2026-09-08: `dist1` is non-null for
  **`ProductCD == 'W'` only** (0% populated for C/H/R/S) — its apparent overall
  presence/fraud correlation is confounded by `ProductCD` (product C alone is
  11.7% fraud and never has `dist1`) and vanishes once measured within `W`
  (2.00% fraud present vs. 2.09% absent). The real signal is a **value
  gradient within W**: fraud rate is flat (~1.6–1.8%) across the bottom three
  quintiles but roughly doubles in the top quintile (2.95% at `dist1 > 36`).
  Add `log1p(dist1)` and a `dist1_high` flag (top-quintile threshold, fit on
  train only) scoped to `W` rows; leave other products' rows at the existing
  null-fill, since there is nothing to condition on there. Currently `dist1`
  passes into the model as a raw, un-engineered numeric column (only
  `dist_present` is derived, in `create_null_count_features`) — this is
  additive to that, not a replacement.
- **`dist2` — investigated, not adding a presence feature.** Same check for
  `dist2`: non-null rates vary by `ProductCD` (0% for W, 39% for C, 38% for S,
  15% for R, 3% for H) and the presence→fraud direction **flips by product**
  (H: 8.3% present vs. 4.7% absent; R: 1.8% present vs. 4.1% absent) — the
  apparent aggregate 3.2x lift is a `ProductCD` mix effect, not a per-product
  signal. No clean, non-confounded feature to extract here beyond what
  `ProductCD` already gives the model; do not add a `dist2_present` feature on
  the strength of this analysis.

**Not implementable on this dataset, ruled out 2026-09-08:** IP-based velocity
(count by IP in a time window, IP→geolocation distance/time) — IEEE-CIS has no
IP address or latitude/longitude columns. Vesta never documented what `dist1`/
`dist2` are distances *between*; the common "billing↔shipping" or
"billing↔device" reading is community speculation, not a confirmed label —
treated here as unconfirmed framing, not fact. Do not attempt to backfill
IP/geo columns from `DeviceInfo` or the `id_*` fields — none of them carry
that information.

Grouping stays on `card1` (or the P9-2 `UID`, once available) since that is
the finest-grained entity identifier this dataset provides — there is no
per-account or per-IP key more specific to window against.

**9.5 — Two-stage cascade (architecture change) — EVALUATED AND REJECTED 2026-09-09**

> **Closed by [ADR-004](adr/ADR-004-operating-point-selection.md): not built.**
> A cascade is a monotone filter, so its reachable points are subsets of the
> stage-1 flagged set. Measured on the promoted 9.4 model: the flagged set is
> 36,816 rows / 3,677 fraud (**9.99%**, not the ~6% assumed below). Holding
> end-to-end recall at 80% lets stage 2 keep 88.4% of the frauds it receives,
> which at 30% precision allows 7,586 FPs — an **FPR of 0.229** on the flagged
> negatives. The incumbent single-stage ensemble already achieves **0.414** at
> that same recall, so the cascade would need a **44.6% relative FPR reduction**
> from the same features on 69% fewer negatives, in the region the incumbent was
> optimised for. Additionally, the prevalence-lift rationale below does not
> produce precision at a *fixed recall* (that is a ranking-quality property), and
> the one genuinely new mechanism — affording expensive features — has **no
> candidate feature identified**. The original specification is kept below for
> the record.


Train a second-stage model on only the transactions the stage-one (current)
ensemble flags at its operating threshold. Stage two sees a much less
imbalanced problem (flagged set is ~6% fraud vs. 3.5% overall) and can afford
more expensive features. This is the largest change in the sequence — an
additional model, an additional serving hop, and a new ADR describing the
two-stage decision path (extending ADR-001) — so it is scoped last and is
conditional on the earlier, cheaper steps not reaching the target band.

**Explicitly out of scope for this phase:** graph neural networks (2026-09-08
research found comparable GNN benchmarks at AUROC 0.86, below this ensemble's
current 0.9066 — not a demonstrated win here **— superseded 2026-09-09, see
Phase 12: a newer source reports AUC-ROC 0.9248 on the same IEEE-CIS dataset,
above this ensemble's current ROC-AUC. Phase 12 evaluates a GNN-GraphSAGE
attempt on that basis**) and precision-at-fixed-recall as
a direct training objective (Eban et al.) — the latter is a plausible future
direction for the TFT specifically but has no established GBDT formulation and
is not evidenced enough yet to sequence into this phase.

**Phase 9 Done When:**
- [x] `ModelEvaluator.precision_at_recall` implemented and unit-tested; reported for the current (pre-Phase-9) model as the baseline before any other step runs — **done 2026-09-08** (`src/evaluation/evaluator.py`; baseline in `reports/RESULTS.md` "PRD Phase 9 — 9.0": 27.97% / 15.95% / 9.46% precision at recall 70/80/90%; band NOT MET, sequence continues)
- [x] Cost-model convention resolved and documented (revenue_tp meaning settled) — `docs/adr/` gets a short ADR or an update to ADR-001 §6.3 recording the decision — **done 2026-09-08** (adopted `net_of_principal`: each recovered fraud counted once; recorded in ADR-001 §6.3 resolution; threshold re-derived on val `0.006123 → 0.014740`, test P=10.14% R=88.88%, band NOT MET at both the proposed and deployed thresholds, sequence continues)
- [~] For each step executed (9.1 through the last step run): retrain completed, full evaluation re-run — including precision-at-recall for at least recall targets 70%, 80%, 90% — results recorded in `reports/RESULTS.md` with model version/run id — **9.1 (no retrain), 9.2 (XGB `dec764c1…` + LGBM `40f8a65f…`), 9.3 (all-3 retrain, `dataset_hash 431cda76…`, ensemble run `0692d881…`) all recorded in RESULTS.md "PRD Phase 9"; 9.4+ pending**
- [~] The stop condition was checked after every step and the reason for stopping (target met) or continuing (target not yet met, by how much) is recorded — **done for 9.0–9.3, each with an explicit points-short figure (9.0 −14.05pt on the P@R≥80% curve; 9.1 −19.86pt; 9.2 −21.43pt; 9.3 −19.69pt at the deployed threshold); continues while 9.4 runs**
- [x] Final configuration's precision and recall at its operating threshold are documented against the ≥30% precision / ≥80% recall target band, with an explicit statement if the band was not fully reached — **done 2026-09-09 (9.4 final + ADR-004).** Deployed: **P 9.99% @ R 90.5%** (threshold 0.012251, 3-way xgb .608/lgbm .384/tft .008). **The band was NOT reached.** Measured frontier: R70→29.17%, R75→24.12%, R80→19.18%, R85→14.56%, R90→10.38%. At R≥80% the best available precision is 19.18% — **10.8 points short**. [ADR-004](adr/ADR-004-operating-point-selection.md) records the band as **infeasible on this dataset with the current feature set**, with the closed-form argument
- [ ] `src/evaluation/business_impact.py` re-run against the final model; `reports/ensemble_test_probabilities.npz` and the Phase 8 notebook outputs regenerated to match
- [x] FR-05's SMOTE requirement is either satisfied with evidence it helps, or the ablation notebook documents why it was rejected in favor of class-weighting — **done 2026-09-08 (9.3): rejected on measured evidence, −0.057 test PR-AUC vs class-weighting alone; recorded in `reports/imbalance_ablation_results.json` and RESULTS.md "9.3"**
- [x] `models/ensemble.json` (or per-model artifacts) updated only if a later step's model actually replaces the deployed one — do not update it for steps that were tested and rejected — **held throughout.** 9.4 was promoted only after it was shown to dominate 9.3 **at every matched recall** (R70 +2.84pt, R75 +2.43pt, R80 +0.89pt, R90 +0.10pt); its apparent deployed-precision dip vs 9.3 was threshold placement, not a worse model. The `--promote` guard kept `models/` untouched on the gate-fail runs

---

### Phase 10 — Testing Strategy

**Sprint:** 10  
**Duration:** 2–3 days (runs in parallel with Phase 11)  
**Objective:** Sufficient test coverage to demonstrate production engineering discipline.

#### Test Structure

```
tests/
├── unit/
│   ├── test_feature_engineering.py
│   ├── test_imbalance_handler.py
│   ├── test_sequence_builder.py
│   ├── test_shap_explainer.py
│   ├── test_losses.py
│   └── test_evaluator.py
├── integration/
│   ├── test_api.py              # FastAPI TestClient tests
│   ├── test_training_pipeline.py # End-to-end train on small subset
│   └── test_monitoring.py
├── performance/
│   └── test_latency.py          # 100 predictions, assert P95 < 100ms
└── conftest.py                  # Shared fixtures: small dataset, loaded models
```

#### Critical Test Cases

**test_feature_engineering.py**
```python
def test_no_data_leakage_in_split():
    """Assert train max TransactionDT < test min TransactionDT."""

def test_smote_does_not_touch_test_set():
    """Assert test set size unchanged after SMOTE applied to train."""

def test_transformers_fit_only_on_train():
    """Assert encoder state set only from train data."""

def test_missing_value_handling():
    """All NaN values replaced, no NaN in output."""

def test_pca_variance_retention():
    """30 PCA components retain >= 80% variance."""
```

**test_evaluator.py**
```python
def test_optimal_threshold_minimizes_total_cost():
    """Given cost_fn=500, cost_fp=5: optimal threshold is lower than 0.5."""

def test_pr_auc_above_baseline():
    """PR-AUC of trained model > random baseline (= fraud_rate)."""
```

**test_api.py**
```python
def test_prediction_response_schema():
    """Response matches PredictionResponse schema exactly."""

def test_explanation_has_top_features():
    """explanation.top_risk_factors has at least 3 features."""

def test_fraud_decision_consistent_with_threshold():
    """If fraud_probability > threshold: decision == 'FRAUD'."""
```

**conftest.py**
```python
@pytest.fixture(scope="session")
def small_dataset():
    """Load first 1000 rows of test_features.parquet."""

@pytest.fixture(scope="session")
def loaded_xgb_model():
    """Load model from models/xgb_model.pkl."""

@pytest.fixture(scope="session")
def api_client():
    """FastAPI TestClient with models pre-loaded."""
```

#### Test Execution

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run only unit tests (fast, no model loading)
pytest tests/unit/ -v

# Run performance tests
pytest tests/performance/ -v -s
```

**Status (2026-09-24): all three previously-missing test files now exist**
(`tests/integration/test_api.py`, `tests/integration/test_training_pipeline.py`,
`tests/performance/test_latency.py` — see `docs/IMPLEMENTATION_PLAN.md` "PRD
Phase 10" for the full record). The phase is not closed: the latency test
measures the real deployed service against real artifacts and **fails** —
P95 501.5ms vs. the 100ms budget below, left failing rather than adjusted, per
an explicit decision to document the gap instead of hiding it.

**Phase 10 Done When:**
- [x] `pytest tests/unit/` passes with 0 failures — verified 2026-09-24
- [x] `pytest tests/integration/` passes (requires services running) — verified 2026-09-24; the real-artifact tests in this directory (`test_real_ensemble_artifact_loads.py`, `test_train_serve_equivalence.py`) pass against the actual `models/` artifacts
- [x] Coverage report shows ≥ 75% coverage on `src/` modules — **generated 2026-09-24: 78% overall**, meeting the >75% bar (`pytest tests/ --ignore=tests/performance --cov=src --cov-report=term-missing`). Below the PRD's original ≥80% figure by 2 points; the user's stated bar for this checkbox is 75%, which is met. See `docs/IMPLEMENTATION_PLAN.md` "PRD Phase 10" for the per-module breakdown; the remaining gap to 80% concentrates almost entirely in the `main()`/CLI/argparse blocks of the training and tuning scripts (`train_xgb.py` 47%, `train_tft.py` 44%, `train_lgbm.py` 43%, `tune_tft.py` 19%, `data_loader.py` 18%), which are exercised through their classes and functions by the existing unit tests but not through their `if __name__ == "__main__":` orchestration, since that only runs meaningfully against the real 590k-row dataset
- [ ] `pytest tests/performance/test_latency.py` passes (P95 < 100ms) — **fails**: measured P95 500-560ms / p50 454-475ms / max 699-970ms across repeated runs over 100 real predictions against the real deployed ensemble, 2026-09-24. Known limitation, not investigated further in this pass — see `docs/IMPLEMENTATION_PLAN.md` "PRD Phase 10" for the measurement and its likely contributors

**Full suite (2026-09-24): 756 passed, 1 known failure** (the latency
assertion above) — `pytest tests/ -q`, confirmed by three consecutive runs.

---

### Phase 11 — Packaging, Documentation & Portfolio Polish

**Sprint:** 10  
**Duration:** 2–3 days  
**Objective:** Public-facing repository ready for portfolio presentation.

#### Action Items

**10.1 — Final README.md**

The public README must include:
- **One-line description** with key technologies named
- **Architecture diagram** (text-based or linked image)
- **Quick start**: `git clone → make setup → make data → make train → make serve`
- **API usage example** with `curl` command and response
- **Key results table**: PR-AUC, F1, net annual business value
- **Project structure** tree
- **Technical decisions** section: why XGBoost + TFT, why SHAP, why time-based split
- **Screenshots**: SHAP beeswarm, Grafana dashboard, Swagger UI
- **Dataset instructions** with Kaggle download steps

**10.2 — Docker Compose Final**

This is the complete and final `docker-compose.yml`. **4 services only.** Every service below has a distinct, non-overlapping responsibility that cannot be merged without a real architectural trade-off.

```yaml
version: '3.8'

services:

  # ── 1. Inference API + Kafka Consumer ───────────────────────────────
  # FastAPI serves HTTP predictions. Kafka consumer runs as an asyncio
  # background task inside the same process — no separate container needed.
  fraud-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      kafka:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ── 2. Kafka (KRaft mode — no Zookeeper) ────────────────────────────
  # Single-node Kafka using its built-in KRaft consensus protocol.
  # Zookeeper is not required from Kafka 3.3+.
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
      KAFKA_LOG_DIRS: /var/lib/kafka/data
      CLUSTER_ID: "fraud-detection-cluster-01"
    healthcheck:
      test: ["CMD", "kafka-broker-api-versions", "--bootstrap-server", "localhost:9092"]
      interval: 10s
      timeout: 10s
      retries: 10

  # ── 3. Prometheus ────────────────────────────────────────────────────
  # Scrapes /metrics from fraud-api every 15s.
  # Stores time-series data for Grafana to query.
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      - fraud-api

  # ── 4. Grafana ───────────────────────────────────────────────────────
  # Dashboard UI over Prometheus. Auto-provisions datasource and dashboards
  # from the mounted config directory on startup.
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    depends_on:
      - prometheus

# MLflow: NOT a service. Run locally during training only:
#   make mlflow   →   mlflow ui --backend-store-uri ./mlruns --port 5000
#
# Kafka Producer: NOT a service. Run as a one-shot script:
#   make stream   →   python src/streaming/producer.py --rate 200 --limit 5000
```

**10.3 — Dockerfile**

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**10.4 — Notebook Cleanup**

All notebooks must:
- Have cleared outputs committed (clean diff in git)
- Include a markdown cell at the top with: purpose, inputs, outputs, key findings
- Run end-to-end with `jupyter nbconvert --to notebook --execute` without error

**10.5 — Results Documentation**

Create `reports/RESULTS.md`:
```markdown
# Model Results Summary

## Dataset
- IEEE-CIS Fraud Detection
- 590,540 transactions, 3.5% fraud rate
- Time-based 80/20 split

## Model Performance

| Model | PR-AUC | ROC-AUC | F1 (optimal threshold) | Optimal Threshold |
|-------|--------|---------|------------------------|-------------------|
| XGBoost (scale_pos_weight) | TBD | TBD | TBD | TBD |
| XGBoost (SMOTE) | TBD | TBD | TBD | TBD |
| TFT (Focal Loss) | TBD | TBD | TBD | TBD |
| Ensemble | TBD | TBD | TBD | TBD |

## Business Impact
- Net annual value: $TBD
- Improvement vs. naive baseline: $TBD

## Key SHAP Insights
- Top 3 fraud signals: TBD
```
(Fill in actual values once training is complete)

**Phase 11 Done When:**
- [ ] `docker compose up` brings all **4 services** online from cold start in < 2 minutes
- [ ] `docker compose ps` shows exactly 4 running containers: fraud-api, kafka, prometheus, grafana
- [ ] Public README renders correctly on GitHub
- [ ] All notebooks execute cleanly
- [ ] `reports/RESULTS.md` has actual numbers filled in
- [ ] Repository has no committed data files, model artifacts, or `.env` files
- [ ] Pre-commit hooks pass on all files (`pre-commit run --all-files`)

---

### Phase 12 — Model Architecture Exploration: Graph Neural Network (GNN-GraphSAGE)

**Sprint:** 12 (new; not in the original §11 Sprint Plan Summary table —
opened 2026-09-09 as a follow-on to Phase 9, not a renumbering of it)
**Duration:** TBD — planning only at this stage; no code has been written
**Status:** Planned, not started. **Documentation-only per explicit instruction
— no implementation in this pass.**
**Objective:** Evaluate whether a graph-based model (transactions as nodes,
edges from shared `card1` / `addr1`+`ProductCD`) can out-rank the current
XGBoost+LightGBM(+TFT) ensemble on this project's own temporal, leakage-audited
IEEE-CIS split, using a specific published result as the working hypothesis
rather than a vague "try graphs" direction.

#### 12.0 — Why this phase exists, and why now

Phase 9 (§10, above) explicitly excluded GNNs from scope, citing 2026-09-08
research that found a comparable GNN benchmark at **AUROC 0.86**
(arXiv:2503.22681, "detectGNN") — below this ensemble's ROC-AUC at the time
(0.9066) and therefore not a demonstrated win. That exclusion is **superseded
here on new evidence**, not overturned by opinion:

> Uddin & Aziz, *"Shapley Value-Guided Adaptive Ensemble Learning for
> Explainable Financial Fraud Detection with U.S. Regulatory Compliance
> Validation"* (arXiv:2604.14231, submitted 2026-04-14). Table II reports a
> **GNN-GraphSAGE** model, evaluated on a held-out 20% split of the **same
> IEEE-CIS dataset** (590,540 transactions, 3.5% fraud rate, 118,108-row
> held-out test set) this project uses:
>
> | Metric | Reported value |
> |---|---|
> | AUC-ROC | **0.9248** |
> | PR-AUC | **0.6334** |
> | F1 (at τ\*=0.86) | **0.6013** |
> | Precision / Recall (at τ\*) | 70.6% / 52.4% |
>
> Verified first-hand from the paper (not taken on the citation alone) —
> see §12.1 for the caveats that verification surfaced.

This project's own current, freshly-retrained ensemble (2026-09-09, this
session — `reports/ensemble_results.json`, mlflow run `69d3a4d5…`,
`dataset_hash d20f03c0…`) scores **test PR-AUC 0.5502, ROC-AUC 0.9154**. The
paper's reported GNN-GraphSAGE PR-AUC (0.6334) would be a **+0.083 absolute**
improvement if it transferred cleanly to this project's pipeline — enough to
justify a dedicated exploration phase, but the gap between "reported on their
pipeline" and "reproducible on ours" is exactly what §12.2's steps exist to
close.

**TFT is not being discarded because it failed** — record this plainly, since
this phase is opened in the same session that finished repairing it. This
session found and fixed two real defects in the TFT training path (a val-set
scaler leak, and a since-reverted regularisation change that had made things
worse) and, once fixed, produced the **best TFT result in this project's
history**: test PR-AUC 0.4670, overfit gap 0.164 (down from ≥0.31 in every
prior TFT run — see `reports/RESULTS.md` §6 and the model manifests for the
full trajectory). TFT is being set aside because, even at its best, it
contributes a marginal-and-unstable ensemble weight (~0.008–0.05, inside the
flat region of a search dominated by its 0.82 correlation with XGBoost — see
`reports/ensemble_results.json` `diagnostics.bootstrap_weight_ci.tft`), not
because the repair work was wasted. **TFT is kept in the codebase, its
manifests and figures stay in `reports/`, and this phase's evidence trail
documents it as a completed, evaluated trial** — a newer architecture is being
tried next because the evidence for it (a higher PR-AUC on this same dataset,
from an architecture this ensemble has never included) is stronger than the
evidence for continuing to invest in TFT's marginal contribution, not because
TFT "didn't work."

#### 12.1 — Caveats surfaced by reading the paper directly (not just the abstract)

Before treating 0.6334 PR-AUC as a target this pipeline should hit, four
differences from this project's own methodology must be accounted for — each
is a plausible reason the reported number would not transfer as-is:

1. **Split methodology is not confirmed temporal.** The paper describes
   5-fold stratified cross-validation plus an 80/20 held-out split with
   SMOTE-Tomek applied within training folds. It does not state that the
   held-out split is time-ordered on `TransactionDT`. This project's entire
   Phase A rework (`docs/RESULTS.md` §1) exists because a non-temporal split
   on this exact dataset was found to leak distributional information and
   inflate PR-AUC (measured delta: 0.0076 absolute at the time, on a fix that
   was mostly *unsupervised* leakage — a random/stratified split leaks more).
   **If the paper's split is not temporal, its 0.6334 is not directly
   comparable to this project's 0.5502** and the honest expectation for a
   temporal-split reproduction is lower.
2. **New infrastructure, not a drop-in trainer.** GraphSAGE requires an
   explicit transaction-to-transaction graph: nodes = transactions, edges from
   shared `card1` (up to 10 neighbors) or the composite key `addr1+ProductCD`
   (up to 5 neighbors), built via PyTorch Geometric's `NeighborLoader` with
   2-hop mini-batch sampling (385,018 edges across 590,540 nodes in the
   paper). This is a new data-engineering component (`src/data/` gains a
   graph-construction module), not a fourth `train_*.py` following the
   existing `XGBTrainer`/`LGBMTrainer`/`TFTTrainer` pattern — the code-
   architect's Build Sequence conventions (types → core logic → integration →
   tests) still apply, but "core logic" here is materially larger than adding
   a trainer.
3. **Class-weighted loss, not SMOTE, for the GNN specifically** — the paper
   applies SMOTE-Tomek to its other four architectures but uses class-weighted
   cross-entropy (`pos_weight≈27.6`) for GNN-GraphSAGE, reasoning that
   synthetic-node generation would produce nodes without meaningful graph
   connectivity. This is **consistent with this project's own finding**
   (`reports/imbalance_ablation_results.json`, PRD Phase 9 §9.3: SMOTE
   degrades val PR-AUC by 0.057 vs. class-weighting alone on the GBDTs) —
   one fewer disagreement to resolve, and a reason to expect the GNN
   direction may transfer better than a naive reading suggests.
4. **The paper's own authors flag the attribution as unresolved**: *"Whether
   GNN's strong performance reflects genuine topological signal from the
   transaction graph or benefits of neighborhood aggregation acting as an
   implicit feature-smoothing mechanism on correlated tabular inputs remains
   an open question for future investigation."* Carrying this caveat forward
   here rather than dropping it protects against overselling the result
   internally before this project's own reproduction confirms it — if the
   gain is smoothing rather than topology, a much cheaper feature-engineering
   change might capture most of it without a graph pipeline at all.

Not disqualifying, but recorded for completeness: the paper's own comparison
table used two different GPUs for different architectures (RTX 5090 for
LSTM/Transformer/XGBoost/SGAE; RTX 3090 Ti for GNN-GraphSAGE), a minor rigor
wrinkle in their own methodology, not something this project inherits.

#### 12.2 — Planned steps (sequential; **none started, no code written**)

Following this project's own Feature Implementation Workflow (research →
plan → TDD → review → commit) and the code-architect's pattern-analysis
process:

1. **12.2.0 — Confirm split methodology.** Re-derive or directly ask whether
   the paper's 80/20 held-out split is temporal. If confirmed non-temporal,
   revise the +0.083 expectation downward before writing any code, using this
   project's own A7/Phase-A leakage-fix delta as the reference magnitude for
   how much a split-methodology difference can move PR-AUC on this dataset.
2. **12.2.1 — Graph construction module (design only in this phase).**
   Design `src/data/graph_builder.py` (name chosen to match the existing
   `sequence_builder.py` convention): builds a `card1`- and
   `addr1+ProductCD`-edged transaction graph from this project's **existing**
   train-only-fitted feature frame — the fit/apply split discipline
   (`src/data/preprocess.py`, `feature_engineering.py`) must carry over
   unchanged so the GNN inherits the same leakage guarantees every other
   model here has. Output: a design doc / ADR, not code.
3. **12.2.2 — Model module (design only).** Design `src/models/gnn_model.py`
   (two `SAGEConv` layers, matching the paper's 128/64-unit sizing as a
   starting point, plus a 3-layer MLP head) and a `GNNTrainer` class matching
   the existing `XGBTrainer`/`LGBMTrainer`/`TFTTrainer` interface contract
   (`build_model`/`train`/`predict_proba`/`predict_proba_calibrated`/
   `predict`/`save`/`load`) so it slots into `scripts/run_ensemble_eval.py`'s
   existing per-model loading pattern rather than requiring a parallel
   evaluation path. Requires `torch-geometric` — a new dependency, not yet in
   `requirements.txt`.
4. **12.2.3 — Standalone train + evaluate on THIS project's temporal split.**
   The actual test: does GNN-GraphSAGE beat 0.5502 test PR-AUC on this
   project's own train/val/test split, with this project's own leakage
   guarantees intact? This is the step that resolves §12.1's caveats with
   evidence instead of speculation. Record the result in `reports/RESULTS.md`
   regardless of outcome — a negative result here (GNN underperforms once the
   temporal split and this project's leakage fixes are respected) is exactly
   as valuable to record as a positive one, per this project's existing norm
   of documenting rejected approaches (LightGBM's 9.2 rejection-then-9.3
   reinstatement; TFT's own trajectory above).
5. **12.2.4 — Ensemble integration, conditional on 12.2.3.** Only if
   12.2.3 shows a standalone lift: re-run the pairwise-correlation diversity
   check and the (now validation-gated, per this session's item-1 fix)
   min-lift gate with GNN as a candidate 4th ensemble member, replacing or
   supplementing TFT depending on which correlates less with XGBoost.
6. **12.2.5 — ADR.** Write `docs/adr/ADR-005-gnn-architecture-evaluation.md`
   recording the outcome (adopted / rejected-with-evidence) regardless of
   which way 12.2.3 goes, following the ADR-004 precedent of writing the
   ADR even for a rejected direction.

**Phase 12 Done When (documentation checkpoint for the planning pass):**
- [x] Prior GNN exclusion (Phase 9) reviewed and the new evidence that
  supersedes it is recorded with a citation, not just a claim
- [x] The cited paper was read directly (not summarized from title/abstract
  alone) and its methodology compared against this project's own pipeline
  before any number was treated as a target
- [x] TFT's disposition (completed trial, kept in the repo, its best-ever
  result recorded) is stated explicitly so it reads as "moving on," not
  "TFT failed"
- [x] Planned steps are sequenced and each is scoped to fit this project's
  existing conventions (trainer interface, config-driven hyperparameters,
  manifest/checksum artifact discipline) rather than introducing new patterns

---

#### 12.3 — Implementation and result (2026-09-10)

Phase 12 was subsequently **implemented end-to-end** (12.2.0–12.2.5) and the
GNN was **rejected on evidence**. Full record: `docs/adr/ADR-005`
(Status: Accepted — rejected direction) and `reports/RESULTS.md` §11.

| Step | Outcome |
|---|---|
| **12.2.0** split methodology | Confirmed: the paper's split is **not** stated to be temporal (5-fold stratified CV + SMOTE-Tomek in-fold). The +0.083 expectation was revised down before any code, using Phase A's 0.0076 leakage-fix delta as the reference magnitude. |
| **12.2.1** `src/data/graph_builder.py` | Built. Reviewed by `ecc:mle-reviewer` **before implementation** for the leakage question (does `NeighborLoader` 2-hop sampling pull future-split neighbours into train batches?). Verdict: APPROVE WITH CHANGES; findings R1–R9. Resolution: one static undirected `edge_index` + **phase-scoped edge masks** — a train batch samples only on edges with both endpoints in the train span, and neighbour caps are computed within each phase's node prefix so a test row's position cannot reshape train↔train edges. `card1` (raw int) and `(addr1, ProductCD)` edge keys; the `-999.0` `addr1` imputation sentinel (~11% of rows) is excluded from composite edges. `QuantileTransformer` fit on the train node span only. |
| **12.2.2** `src/models/gnn_model.py` + `GNNTrainer` | Built. 2×`SAGEConv` (128→64) + 3-layer MLP head → 1 logit, **no normalization layers** (a hard leakage constraint — the eval-mode full-graph forward spans val/test nodes). `GNNTrainer` in `src/training/train_gnn.py` matches the `XGBTrainer`/`LGBMTrainer`/`TFTTrainer` interface exactly (`build_model`/`train`/`predict_proba`/`predict_proba_calibrated`/`predict`/`save`/`load`), 3-file checksummed artifact, `build_manifest` lineage. `torch-geometric==2.5.3` + `torch-scatter` + `torch-sparse` added to `requirements.txt`. |
| **12.2.3** standalone train + evaluate | **Test PR-AUC 0.4410** (val 0.5260, train 0.6564, ROC-AUC 0.8849, overfit gap 0.2155), mlflow `6615a0fc…`, seed 42, unchanged `dataset_hash`. **0.109 below the 0.5502 ensemble baseline**, 0.026 below TFT's best (0.4670), 0.192 below the paper's 0.6334. The **0.085 val→test drop** confirms the split-methodology caveat: a large part of the gap to the paper is a non-temporal-split artifact. 32 new unit tests (leakage regression tests included); full suite passing. |
| **12.2.4** ensemble integration | **NOT triggered.** Pre-registered gate (proceed iff standalone test PR-AUC > `0.5502 + 0.005`): `0.4410 < 0.5552`. `models/ensemble.json` untouched; TFT not replaced. |
| **12.2.5** ADR | `docs/adr/ADR-005-gnn-architecture-evaluation.md` written, Status **Accepted** (rejected direction, per the ADR-004 precedent). |

**Disposition:** all Phase 12 code, tests, config, artifacts (`models/gnn_model.*`,
`reports/gnn_results.json`), the graph cache, and `make train-gnn` stay in the
repo as a completed, evaluated trial. No follow-up is scheduled; a cheap probe
(add `card1`/`uid` mean-aggregate features to the GBDTs to see if they capture
the non-leaking part of any graph signal) is noted in ADR-005 §6.

---

#### 12.4 — Revisit: richer edges, deeper architecture, hyperparameter search (2026-09-12 to 2026-09-23)

User-directed follow-on to §12.3, combining three hypotheses named in
ADR-005 §6: richer edge types beyond `card1`/`(addr1,ProductCD)`, a
deeper/regularized architecture to address the 0.216 overfit gap, and a
proper Optuna hyperparameter search rather than paper-frozen values. Full
record: [ADR-006](adr/ADR-006-gnn-revisit-edges-depth-hpo.md) (Status:
**Rejected**).

Reviewed before implementation by `architect` and `ecc:mle-reviewer`. Both
rated the odds of success as poor: the only edge keys that survive leakage
scrutiny (`card2`/`card3`/`card5`, `(addr1,card1)`) are refinements of the
existing `card1` key, not a new linking modality — the columns that would add
one (`P_emaildomain`, `R_emaildomain`, `DeviceInfo`, `id_31`, `id_33`) are
already frequency-encoded and non-injective upstream. The user directed
proceeding anyway; both reviews' mitigations (arm-separated attribution,
val-search/val-confirm with zero test access during search) were adopted as
preconditions.

| Arm | Scope | Outcome |
|---|---|---|
| A — richer edges, frozen architecture | `card_full = (card1,card2,card3,card5)`, `addr_card = (addr1,card1)`, each with its own sentinel exclusion | No individual run cleared the incumbent (val PR-AUC 0.5260) by a margin worth searching around alone; `addr_card` was the strongest single direction |
| B — deeper/regularized architecture, frozen (legacy) graph | `weight_decay` up to `1e-3`, `dropout` up to 0.6, input dropout, `F.normalize` (no `BatchNorm`/`LayerNorm` at any depth), 3-layer variant | No individual run cleared the incumbent alone |
| C — hyperparameter search (30-trial Optuna, `TPESampler`+`MedianPruner`, val-only objective, test never opened during search) | Combined search space over both arms' axes plus `learning_rate`, `aggr`, `pos_weight_scale` | Trial 14 (`addr_card` edges, 3-layer residual GraphSAGE) reached **val PR-AUC 0.5505** — the only trial across both ADRs to clear the ≥0.5502 stop-before-test gate — before crashing; a crash-resume continuation never improved on that value |

**Gate 2 confirmatory run** (trial 14's hyperparameters, re-verified directly
against the Optuna study database, full epoch budget, read on test exactly
once, mlflow `d6133250654241c7bf833e9f2d8f14e3`):

| Metric | Value |
|---|---|
| Val PR-AUC | 0.5411 |
| **Test PR-AUC** | **0.4630** |
| Test ROC-AUC | 0.8905 |
| Overfit gap (train−test) | 0.2549 |

**Result: 0.4630, short of the pre-registered adoption threshold (0.5552) by
0.092, and below the un-margined 0.5502 baseline.** The val→test drop
(0.078) is consistent with two prior independent runs (0.070–0.085 across
ADR-005 and both ADR-006 attempts), confirming a genuine temporal-distribution-shift
component that further tuning on the validation side does not close. The
overfit gap did not shrink across the revisit (0.2155 → 0.2549) despite Arm
B's regularization changes. Ensemble integration (Gate 3) and the diversity
check (Gate 4) were not triggered — both are conditional on Gate 2 passing.

A separate, non-gating diagnostic (same session) checked whether a **3-way
swap** — {XGBoost, LightGBM, GNN} replacing TFT in the deployed blend, rather
than a 4-way add — did better, read once on test: test PR-AUC 0.5495 versus
the deployed blend's 0.5502, a statistically negligible step backward. The
weight search assigned the GNN a blend weight of 0.006, essentially the same
near-zero weight TFT already holds (0.008).

**Disposition:** ADR-006 closes as Rejected. All three hypotheses (richer
edges, deeper/regularized architecture, hyperparameter search) were tested
and none reversed ADR-005's verdict. The deployed 3-way XGBoost + TFT +
LightGBM ensemble (test PR-AUC 0.5502) remains the production configuration.
All Phase 12/ADR-006 code, tests, config, and artifacts stay in the repo as a
completed, evaluated trial.

---

## 11. Sprint Plan Summary

| Sprint | Phase | Duration | Key Deliverable |
|--------|-------|----------|-----------------|
| 0 | Foundation | 2 days | Repo structure, env, Docker base, config |
| 1 | Data Pipeline | 4 days | Clean parquet files, feature engineering, time-based split |
| 2 | Baseline Model | 3 days | XGBoost trained, PR-AUC ≥ 0.70, MLflow logged |
| 3 | TFT Model | 6 days | TFT trained on sequences, PR-AUC ≥ 0.75, comparison notebook |
| 4 | Explainability | 3 days | SHAP values, dashboard HTML, explanation API format |
| 5 | API Serving | 3 days | FastAPI live, < 100ms P95, Prometheus metrics |
| 6 | Streaming | 2 days | Kafka (KRaft), consumer background task in fraud-api, producer script demo |
| 7 | Monitoring | 2 days | Evidently reports, Grafana dashboard |
| 8 | Business Impact | 1 day | Cost-matrix notebook, dollar value figures |
| 9 | Precision Improvement | 3–6 days (sequential, gated) | Precision ≥ 30% at recall ≥ 80% at the deployed threshold, or documented stop point |
| 10 | Testing | 2 days | ≥ 80% unit test coverage, all tests passing |
| 11 | Polish | 2 days | Public README, Docker Compose final, clean notebooks |
| 12 | GNN Architecture Exploration | ~1 day (added 2026-09-09; implemented + closed 2026-09-10); revisited 2026-09-12 to 2026-09-23 | **Closed, twice.** Paper-frozen GraphSAGE: test PR-AUC **0.4410** vs. 0.5502 ensemble baseline → rejected (ADR-005). Revisit with richer edges + deeper architecture + 30-trial HPO (ADR-006): best confirmatory result test PR-AUC **0.4630**, short of the 0.5552 adoption threshold → rejected. Deployed 3-way ensemble (XGBoost+TFT+LightGBM, test PR-AUC 0.5502) unchanged. Infrastructure from both trials kept. See §12.3, §12.4. |

**Total estimated calendar time: ~33–36 developer-days (7 weeks at full time, 10–12 weeks part-time), excluding Phase 12 — added after this estimate and not yet scoped to a duration.**

Critical path: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5.  
Phases 6, 7, 8 can run in parallel after Phase 5.  
Phase 9 depends on Phase 8 (needs `business_impact.py` and the exported test
probabilities) and is internally sequential with an early-stop gate — its
actual duration depends on which step reaches the target band, so 3–6 days is
a range, not a fixed estimate.  
Phases 10 and 11 are continuous but formally completed last, after Phase 9's
final model is frozen.  
Phase 12 depends on Phase 9's final ensemble (needs a frozen baseline PR-AUC
to beat) but not on Phases 10/11 — it can run in parallel with testing/polish
work once step 12.2.0 is unblocked.

---

## 12. Repository & Folder Structure

```
fraud-detection-explainable/
│
├── config/
│   └── config.yaml                # All configuration
│
├── data/
│   ├── raw/                       # .gitignored — downloaded CSVs
│   │   ├── train_transaction.csv
│   │   └── train_identity.csv
│   └── processed/                 # .gitignored — generated parquet files
│       ├── train_features.parquet
│       ├── test_features.parquet
│       ├── train_labels.parquet
│       ├── test_labels.parquet
│       └── transformers/
│           ├── pca.pkl
│           ├── label_encoders.pkl
│           ├── freq_encoders.pkl
│           └── imputer.pkl
│
├── models/                        # .gitignored — trained model artifacts
│   ├── xgb_model.pkl
│   ├── lgbm_model.pkl
│   └── tft_model.ckpt
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_imbalance_ablation.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_shap_analysis.ipynb
│   └── 05_business_impact.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_data.py
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   ├── data_splitter.py
│   │   ├── imbalance_handler.py
│   │   └── sequence_builder.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_xgb.py
│   │   ├── train_lgbm.py
│   │   ├── train_tft.py
│   │   └── losses.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── ensemble.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py
│   │   └── dashboard_generator.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── model_loader.py
│   │   ├── routes/
│   │   │   ├── predict.py
│   │   │   ├── health.py
│   │   │   └── metrics.py
│   │   ├── schemas/
│   │   │   ├── request.py
│   │   │   └── response.py
│   │   └── middleware/
│   │       ├── logging.py
│   │       └── timing.py
│   ├── streaming/
│   │   ├── __init__.py
│   │   └── producer.py          # CLI script: python producer.py --rate N --limit N
│   │   # consumer.py logic lives in src/api/main.py lifespan as asyncio background task
│   └── monitoring/
│       ├── __init__.py
│       ├── drift_reporter.py
│       └── drift_scheduler.py
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_feature_engineering.py
│   │   ├── test_imbalance_handler.py
│   │   ├── test_sequence_builder.py
│   │   ├── test_shap_explainer.py
│   │   ├── test_losses.py
│   │   └── test_evaluator.py
│   ├── integration/
│   │   ├── test_api.py
│   │   ├── test_training_pipeline.py
│   │   └── test_monitoring.py
│   └── performance/
│       └── test_latency.py
│
├── monitoring/
│   ├── prometheus.yml
│   ├── reports/                   # .gitignored — generated Evidently reports
│   ├── alerts/                    # .gitignored — drift alert JSONs
│   └── grafana/
│       ├── dashboards/
│       │   └── fraud_detection.json
│       └── datasources/
│           └── prometheus.yaml
│
├── reports/
│   ├── RESULTS.md
│   └── figures/                   # Committed PNG exports from notebooks
│       ├── pr_curve_xgb.png
│       ├── pr_curve_tft.png
│       ├── shap_beeswarm.png
│       ├── shap_waterfall_tp.png
│       ├── shap_waterfall_fn.png
│       ├── shap_waterfall_fp.png
│       └── threshold_vs_business_value.png
│
├── logs/                          # .gitignored — runtime logs
│   └── predictions.jsonl
│
├── scripts/
│   └── run_streaming_demo.sh
│
├── .env.example                   # Template for .env (committed, no secrets)
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 13. API Contracts

### POST /predict

**Request:**
```json
{
  "transaction_id": "TX_999123",
  "transaction_amount": 1250.00,
  "product_code": "W",
  "card1": 9500,
  "card4": "visa",
  "card6": "credit",
  "p_email_domain": "gmail.com",
  "r_email_domain": "yahoo.com",
  "device_type": "desktop",
  "addr1": 315,
  "dist1": 0.0,
  "request_timestamp": "2026-01-15T10:23:45Z"
}
```

**Response (200):**
```json
{
  "transaction_id": "TX_999123",
  "fraud_probability": 0.873,
  "decision": "FRAUD",
  "threshold_used": 0.42,
  "model_version": "xgb_v1.0_2026-01",
  "latency_ms": 23.4,
  "timestamp": "2026-01-15T10:23:45.234Z",
  "explanation": {
    "base_fraud_rate": 0.035,
    "top_risk_factors": [
      {
        "feature_name": "amount_vs_mean_ratio",
        "feature_value": 8.3,
        "shap_contribution": 0.31,
        "direction": "increases_risk"
      },
      {
        "feature_name": "r_email_domain_freq",
        "feature_value": 0.003,
        "shap_contribution": 0.22,
        "direction": "increases_risk"
      },
      {
        "feature_name": "hour_sin",
        "feature_value": -0.98,
        "shap_contribution": 0.18,
        "direction": "increases_risk"
      }
    ],
    "top_mitigating_factors": [
      {
        "feature_name": "card_tx_count_7d",
        "feature_value": 24.0,
        "shap_contribution": -0.09,
        "direction": "decreases_risk"
      }
    ],
    "explanation_confidence": "high"
  }
}
```

**Response (422) — Validation Error:**
```json
{
  "detail": [
    {
      "loc": ["body", "transaction_amount"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### GET /health

**Response (200):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "xgb_v1.0_2026-01",
  "uptime_seconds": 3600.4,
  "total_predictions": 15243,
  "fraud_rate_last_1000": 0.038
}
```

### GET /metrics

Returns Prometheus text format (standard):
```
# HELP fraud_prediction_total Total fraud predictions made
# TYPE fraud_prediction_total counter
fraud_prediction_total 15243.0

# HELP fraud_rate_gauge Rolling fraud rate over last 1000 predictions
# TYPE fraud_rate_gauge gauge
fraud_rate_gauge 0.038

# HELP http_request_duration_seconds Request duration in seconds
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.01"} 8234.0
http_request_duration_seconds_bucket{le="0.05"} 14102.0
http_request_duration_seconds_bucket{le="0.1"} 15201.0
...
```

---

## 14. Configuration & Environment Management

### Environment Variables (.env file — not committed)

```bash
# .env (copy from .env.example)
# MLflow tracking URI is NOT here — it's a local process, not a secret.
# Only true secrets belong in .env.
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

### .env.example (committed to repo)

```bash
# Copy this file to .env and fill in your values
KAGGLE_USERNAME=REPLACE_ME
KAGGLE_KEY=REPLACE_ME
```

### Config Loading Pattern

Every module that needs configuration must follow this pattern:
```python
import yaml
from pathlib import Path

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(Path(config_path)) as f:
        return yaml.safe_load(f)

# Never hardcode paths, thresholds, or model params in source files
# Always: config = load_config(); threshold = config['thresholds']['default']
```

---

## 15. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| IEEE-CIS dataset download blocked (Kaggle account/competition acceptance) | Medium | High | Document PaySim as tested fallback. Provide `download_data.py` that handles both. |
| TFT training too slow on CPU | High | Medium | Use `max_epochs=5` for initial testing; document GPU command. Provide pre-trained TFT checkpoint in releases. |
| SHAP `DeepExplainer` for TFT produces poor explanations | Medium | Medium | Fall back to `KernelExplainer` with 100-sample background. Document trade-off. |
| Kafka KRaft container fails to start (missing CLUSTER_ID or misconfigured env) | Low | Medium | CLUSTER_ID must be a valid base64 UUID. Use `kafka-storage format` or set it explicitly in docker-compose env. Provide `docker compose logs kafka` diagnostic instructions in README. |
| V1-V339 features (masked by Vesta) reduce interpretability | Low | Low | PCA reduces these to 30 components. Name them `vesta_pca_1...30`. Add note in SHAP dashboard that these are proprietary signals. |
| Data leakage in aggregate features | High | High | `test_no_data_leakage_in_split()` test. Feature engineering must sort data first. Review aggregation window logic carefully. |
| Overfit TFT model (high train PR-AUC, low test PR-AUC) | Medium | High | Early stopping, dropout, weight decay. Monitor val_loss per epoch. Compare train/val curves in MLflow. |

---

## 16. Definition of Done

A phase is considered complete when ALL of the following are true:

1. **Code complete**: All files listed in the phase exist with no `TODO` or `pass` stubs in production code paths
2. **Tests pass**: All tests in the relevant test module pass with 0 failures
3. **No hardcoded values**: Config loaded from `config.yaml` for all parameters
4. **Logged**: Training runs logged to MLflow; API requests logged to `predictions.jsonl`
5. **Notebook executed**: Any notebook for the phase runs end-to-end without error
6. **Committed**: Code committed to git with a meaningful commit message

The overall project is complete when:
- [ ] `docker compose up` brings all **4 services** online (fraud-api, kafka, prometheus, grafana)
- [ ] `docker compose ps` shows exactly 4 running containers — no more
- [ ] `make reproduce` (data + train) runs from raw CSV to saved models
- [ ] `pytest tests/` passes with ≥ 80% coverage
- [ ] All 5 notebooks run cleanly
- [ ] `reports/RESULTS.md` has actual performance numbers
- [ ] Public README contains architecture diagram, quick start, and results table
- [ ] No data, model, or credential files committed to git

---

## 17. Glossary

| Term | Definition |
|------|-----------|
| **PR-AUC** | Area Under the Precision-Recall Curve. Primary metric for imbalanced classification. Unlike ROC-AUC, it is sensitive to the minority class performance. |
| **SMOTE** | Synthetic Minority Oversampling Technique. Creates synthetic minority-class samples by interpolating between existing ones. Applied only to training data. |
| **Focal Loss** | Modified cross-entropy that down-weights easy (correctly classified) examples and focuses on hard (misclassified) examples. Parameterized by gamma (focusing strength) and alpha (class balance). |
| **SHAP** | SHapley Additive exPlanations. Game-theory-based method for explaining individual model predictions. `shap_value[i]` = contribution of feature i to the deviation from base prediction. |
| **TreeExplainer** | SHAP explainer optimized for tree-based models (XGBoost, LightGBM). Computes exact SHAP values (not approximate). |
| **TFT** | Temporal Fusion Transformer. A transformer architecture designed for multi-step forecasting on tabular time series. Uses attention mechanisms to handle variable-length history windows. |
| **TimeSeriesDataSet** | PyTorch Forecasting class that handles sequence construction, normalization, and DataLoader creation for temporal models. |
| **Data Drift** | When the statistical distribution of input features at inference time diverges from the distribution at training time. Causes model degradation without any visible error. |
| **Optimal Threshold** | The classification threshold that minimizes total business cost (FP * cost_fp + FN * cost_fn), not necessarily 0.5. |
| **scale_pos_weight** | XGBoost parameter = (count of negative class) / (count of positive class). Adjusts the weight of the positive class during training to compensate for imbalance. |
| **Time-Based Split** | Train/test split using temporal order rather than random shuffle. Prevents future information leaking into training, which would overestimate real-world performance. |
| **RBI** | Reserve Bank of India. Issues model risk management guidelines for financial institutions operating in India. |
| **FCA** | Financial Conduct Authority (UK). Requires explainability for automated financial decisions under GDPR and Consumer Duty regulations. |
| **GNN** | Graph Neural Network. A model architecture that operates on graph-structured data (nodes + edges) rather than flat feature vectors or sequences, propagating information between connected nodes. See Phase 12. |
| **GraphSAGE** | "Graph SAmple and aggreGatE" — a GNN variant that learns node embeddings by sampling and aggregating features from a node's local neighborhood, enabling mini-batch training on graphs too large to fit in memory at once. See Phase 12. |

---

*End of Document*

*Version 2.0 (Final) — June 2026*  
*This document serves as the authoritative specification for the Explainable Fraud Detection portfolio project.*