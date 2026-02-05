# 🧠 Churn Prediction Pipeline (Airflow + MLflow + PostgreSQL + MLOps)

## 📌 Overview
This repository contains a **production-grade churn prediction pipeline** built using:

- **Apache Airflow** (orchestration)
- **MLflow** (experiment tracking + model registry)
- **PostgreSQL** (MLflow backend store)
- **Pandas / Scikit-learn / Joblib** (feature engineering + model training)
- **Custom data validation & ML pipeline modules**

The system runs on a weekly schedule and automates the full MLOps lifecycle:

1. Load raw e-commerce customer data  
2. Clean, impute, validate, and deduplicate records  
3. Engineer ML features  
4. Train multiple ML models  
5. Select the best model based on ROC-AUC  
6. Register & optionally promote the model in MLflow  
7. Persist feature transformers and artifacts  

This project demonstrates **real-world MLOps patterns** including reproducibility, automated data checks, 
experiment tracking, outlier handling, and production model governance.

---

## 🚀 Key Features

### 🔹 **Automated Data Ingestion**
- Reads Excel data stored in Airflow DAGs directory  
- Logs row count, column summaries, and churn rate  
- Robust error handling and missing data detection  

### 🔹 **Advanced Data Cleaning Pipeline**
Includes:
- Missing value imputation (median, mode, zero-fill, conditional strategies)  
- Outlier removal using IQR thresholds  
- Duplicate removal based on `CustomerID`  
- Critical column enforcement (`Churn`, `Tenure`, `SatisfactionScore`)  
- Column-level logging for traceability  

### 🔹 **Validation Layer**
Before ML training, the pipeline:
- Converts cleaned dict data into DataFrame  
- Runs `validate_data()` for schema and quality verification  
- Fails early on structural issues  

### 🔹 **Feature Engineering**
Implements:
- Config-driven transformations (scaling, encoding, derived features, etc.)  
- Transformer persistence using Joblib  
- Separation of predictors, target, and metadata  
- Full logging of feature counts and names  

### 🔹 **Model Training & Selection**
- Trains multiple models using the custom `MLPipeline` class  
- Computes metrics including ROC-AUC  
- Selects the best-performing model  
- Saves:
  - Trained model
  - Preprocessing transformers
  - Feature importances
  - Model metadata  

### 🔹 **MLflow Model Registry Automation**
- Registers each trained model version  
- Auto-promotes to production if ROC-AUC ≥ 0.75  
- Logs metadata, metrics, and transformer paths  

---

## 📂 Project Structure

├── dags/
│ ├── churn_prediction_pipeline.py # Main Airflow DAG
│ ├── data/ # Input dataset
│ ├── data_utils.py # Validation + config helpers
│ └── ml_pipeline.py # ML workflow implementation
├── models/
│ ├── production/ # Production model artifacts
│ └── transformers/ # Feature transformers
└── README.md


---

## ▶️ How the DAG Works

### **1️⃣ Load Data**


load_ecommerce_data()

- Reads Excel file  
- Logs structure, missingness, and churn distribution  

### **2️⃣ Validate Data**


validate_data()

- Checks schema, types, and completeness  

### **3️⃣ Engineer Features**


engineer_features()

- Applies transformations  
- Saves `X`, `y`, and transformers  

### **4️⃣ Train Models**


train_models()

- Trains all configured models  
- Selects best based on ROC-AUC  
- Saves artifacts  

### **5️⃣ Register & Promote**


register_model()

- Registers model to MLflow registry  
- Auto-promotes if threshold met  

---

## 🧪 Example Output (Logs)
- Rows before/after cleaning  
- Outlier counts per feature  
- Feature importance  
- Best model name  
- ROC-AUC score  
- Model version + promotion status  

---

## 🛠 Requirements

- Docker Compose environment with:
  - Airflow
  - MLflow server
  - PostgreSQL
- Python 3.10+  
- Required Python libraries:
  - pandas  
  - numpy  
  - scikit-learn  
  - mlflow  
  - joblib  

---

## 🎯 Future Enhancements
- Add drift detection  
- Add SHAP explainability  
- Add batch inference DAG  
- Add S3/GCS artifact storage  

---

## 📄 License
MIT License
