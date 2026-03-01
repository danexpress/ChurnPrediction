# ChurnGuard 🛡️
### Automated ML Pipeline for Customer Churn Prediction & Model Registry Promotion



---

## Overview

**ChurnGuard** is a production-grade, end-to-end machine learning pipeline that automates the full customer churn prediction lifecycle — from raw e-commerce data ingestion through feature engineering, multi-model training, evaluation, and MLflow registry promotion. Built on Apache Airflow with a weekly schedule, it eliminates manual intervention across the ML lifecycle by enforcing ROC-AUC performance thresholds before any model reaches production.

The pipeline is designed around real-world constraints: messy input data with missing values and outliers, reproducible feature transformers, conditional model promotion gates, and full lineage tracking through MLflow. A model only reaches production if it earns it.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      ChurnGuard Pipeline (Airflow DAG)               │
│                         Schedule: @weekly                            │
└──────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────┐
  │  load_ecommerce_data │  ← Reads Excel, imputes missing values,
  │                      │    removes duplicates & outliers (IQR)
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │    validate_data     │  ← Schema checks, null assertions,
  │                      │    column presence validation
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  engineer_features   │  ← Encoding, scaling, feature creation,
  │                      │    transformer serialization (joblib)
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │     train_models     │  ← Multi-model training, cross-validation,
  │                      │    ROC-AUC evaluation & best model selection
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │    register_model    │  ← MLflow registry logging, conditional
  │                      │    promotion to production (ROC-AUC > 0.75)
  └─────────────────────┘
             │
             ▼
  ┌─────────────────────┐
  │  PostgreSQL + MLflow │  ← Model metadata, metrics, versioning,
  │                      │    artifact storage & lineage tracking
  └─────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | Apache Airflow 2.8+ (TaskFlow API) |
| **ML Training & Registry** | scikit-learn + MLflow |
| **Data Processing** | Pandas, NumPy |
| **Model Serialization** | joblib |
| **Metadata & Versioning** | PostgreSQL + MLflow Tracking Server |
| **Validation** | Custom schema validation (data_utils) |
| **Language** | Python 3.11+ |

---

## Pipeline Stages

**Stage 1 — Data Ingestion & Cleaning (`load_ecommerce_data`)**
Reads the raw e-commerce Excel dataset and applies a multi-strategy cleaning pass before any downstream task sees the data. Numerical columns use median or mode imputation depending on their distribution characteristics — tenure and order counts use median; satisfaction scores use mode with a fallback default. Categorical fields like payment mode, login device, and marital status use mode imputation. Duplicate `CustomerID` rows are dropped keeping the first occurrence. Outliers in high-variance columns (`Tenure`, `CashbackAmount`, `OrderCount`) are removed using a conservative 3×IQR fence to preserve the bulk of the distribution while eliminating genuine anomalies. The stage logs retention rate and remaining null counts before passing data downstream.

**Stage 2 — Data Validation (`validate_data`)**
Runs a structured validation suite against the cleaned dataframe before any transformation occurs. Validation failures raise hard exceptions that halt the DAG — bad data never reaches the feature engineering stage silently. Passes a validation-confirmed payload downstream via XCom.

**Stage 3 — Feature Engineering (`engineer_features`)**
Delegates to `MLPipeline.engineer_features()` which applies encoding, scaling, and derived feature creation in a single fitted pass. Transformers are serialized to disk via joblib so they can be reloaded at inference time without re-fitting — ensuring training and serving transformations are identical. Feature metadata is passed downstream for model interpretability.

**Stage 4 — Model Training & Selection (`train_models`)**
Trains multiple candidate models, evaluates each on a held-out test set, and selects the best performer by ROC-AUC score. The winning model and its transformers are saved to disk. Metrics are sanitized before XCom serialization — NaN and Inf values are handled explicitly to prevent silent JSON serialization failures downstream.

**Stage 5 — Registry & Promotion (`register_model`)**
Logs the winning model to MLflow with full metadata: model type, metrics, feature importance, and transformer path. Applies a hard performance gate — models with ROC-AUC above 0.75 are promoted to the `production` stage in the MLflow registry. Models below threshold are registered but held in `staging` for review. This gate ensures production is never downgraded by a weaker model on a bad data week.

---

## Project Structure

```
churnguard/
├── dags/
│   ├── dag_pipeline.py              # Main Airflow DAG (TaskFlow API)
│   ├── data_utils.py                # Validation, config, MLflow helpers
│   ├── ml_pipeline.py               # MLPipeline class: features + training
│   └── data/
│       └── E Commerce Dataset.xlsx  # Source dataset
├── models/
│   ├── production/
│   │   └── churn_model.pkl          # Best model artifact
│   └── transformers/
│       └── feature_transformers.pkl # Fitted feature transformers
├── mlflow/
│   └── mlruns/                      # MLflow experiment tracking store
├── docker-compose.yml               # Airflow + PostgreSQL + MLflow stack
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- 4GB RAM minimum

### 1. Clone the repository

```bash
git clone https://github.com/danexpress/ChurnPrediction
cd ChrunPrediction
```

### 2. Start the full stack

```bash
astro dev start
*** if error occur use this "docker network create churn-prediction_4031ac_airflow" and run astro dev start
```

This starts Airflow (webserver + scheduler), PostgreSQL (metadata DB), and the MLflow tracking server.

### 3. Access the interfaces

| Service | URL | Credentials |
|---|---|---|
| Airflow UI | http://localhost:8080 | admin / admin |
| MLflow UI | http://localhost:5001 | — |

### 4. Trigger the pipeline


via the UI — navigate to **DAGs → churn_prediction_pipeline → Trigger DAG**.

### 5. Monitor the run

Watch task-by-task progress in the Airflow Graph View. Each stage logs key metrics — rows retained, features created, model scores, and promotion decisions — directly to Airflow task logs.

### 6. Inspect registered models

Open the MLflow UI at `http://localhost:5001` and navigate to **Models → churn_prediction** to see all registered versions, their metrics, and production promotion status.

---

## Data Cleaning Strategy

The cleaning stage applies column-aware imputation rather than blanket median-fill — a deliberate choice that better preserves the statistical properties of each feature:

| Column | Strategy | Rationale |
|---|---|---|
| `Tenure`, `OrderCount`, `CouponUsed` | Median | Right-skewed distributions; median is more robust than mean |
| `SatisfactionScore` | Mode | Ordinal scale; most frequent score is the appropriate central tendency |
| `CashbackAmount`, `OrderAmountHike` | Zero-fill | Missing likely indicates no cashback event, not unknown |
| `DaySinceLastOrder` | Median | Skewed; median prevents outlier inflation |
| `PreferredPaymentMode`, `Gender`, etc. | Mode | Categorical; most frequent category as default |
| Outliers | 3×IQR removal | Conservative fence preserves ~99%+ of valid distribution |

---

## Model Selection & Promotion Logic

```python
# Best model selected by ROC-AUC on held-out test set
best_model_name = max(
    model_results.keys(),
    key=lambda x: model_results[x].test_scores.get("roc_auc", 0),
)

# Hard promotion gate — production requires ROC-AUC > 0.75
if roc_auc > 0.75:
    promote_model("churn_prediction", model_version, "production")
else:
    # Registered in staging for manual review
    logging.info(f"Model held in staging (ROC-AUC: {roc_auc:.4f})")
```

This means production is never silently overwritten by a degraded model — a common failure mode in naive retraining pipelines.

---

## MLflow Tracking

Every pipeline run logs the following to MLflow:

- **Model type** (algorithm name)
- **Test metrics** — ROC-AUC, precision, recall, F1, accuracy
- **Feature importance** — ranked feature contributions from the winning model
- **Artifact paths** — model pickle and transformer pickle locations
- **Registry stage** — `staging` or `production` based on performance gate

This gives full lineage across every weekly run: which model ran, on what data shape, with what performance, and whether it reached production.

---

## Design Decisions & Tradeoffs

**Why TaskFlow API over classic Airflow operators?**
TaskFlow (`@task` decorators) dramatically reduces boilerplate for Python-heavy pipelines. XCom passing is handled implicitly via return values rather than explicit `xcom_push/pull` calls, making the DAG code read like a linear Python script rather than a collection of loosely coupled operators. For a pipeline this Python-native, TaskFlow is the right abstraction.

**Why joblib for transformer serialization instead of passing via XCom?**
Fitted sklearn transformers can be large — serializing them through XCom (which is backed by the Airflow metadata database) risks hitting size limits and slowing the scheduler. Saving to a temp directory and passing only the file path via XCom is the correct pattern for any artifact above a few kilobytes.

**Why ROC-AUC as the promotion threshold metric?**
Churn datasets are almost always class-imbalanced — churned customers are a minority. Accuracy is misleading in this setting (a model predicting "no churn" for everyone can be 90%+ accurate). ROC-AUC evaluates the model's ability to rank churners above non-churners regardless of threshold, making it the right primary metric for this problem.

**Why a hard 0.75 threshold?**
0.75 ROC-AUC is a reasonable minimum for a churn model to be actionable — below that, the model's ranking signal is too weak to reliably prioritize retention outreach. In a production setting, this threshold would be calibrated against the cost of false positives (unnecessary retention spend) vs. false negatives (missed churners).

---

## Future Improvements

- Add Great Expectations data quality checks as a formal pipeline stage
- Implement hyperparameter tuning with Optuna before final model selection
- Add retraining drift detection — trigger pipeline if feature distributions shift beyond a KL-divergence threshold
- Expose model predictions via a FastAPI inference endpoint backed by the MLflow-registered model
- Add email/Slack alerting via Airflow callbacks on pipeline failure or model demotion

---


## License

MIT License. See `LICENSE` for details.
