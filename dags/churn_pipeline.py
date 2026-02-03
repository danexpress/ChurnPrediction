import sys

sys.path.insert(0, "/usr/local/airflow/dags")

from airflow.sdk import dag, task
from pendulum import datetime
import pandas as pd
from typing import Dict, Any
import logging


@dag(
    start_date=datetime(2026, 2, 3),
    schedule="@weekly",
    catchup=False,
    tags=["ml", "churn", "prediction", "ecommerce"],
    description="Production customer churn prediction pipeline with ML features",
)
def churn_prediction_pipeline():
    @task()
    def load_ecommerce_data() -> Dict[str, Any]:
        try:
            excel_path = "/usr/local/airflow/dags/data/E Commerce Dataset.xlsx"
            sheet_name = "E Comm"

            logging.info(f"Loading data from {excel_path}...")
            df = pd.read_excel(excel_path, sheet_name=sheet_name)

            logging.info(f"Loaded {len(df)} rows of data with shape: {df.shape}")
            logging.info(f"Columns: {df.columns.tolist()}")

            df = _clean_and_impute_data(df)
            final_rows = len(df)

            logging.info(
                f"Data cleaning completed. {final_rows} rows remain after cleaning"
            )

            # ensure Churn column exists and properly formatted
            if "Churn" in df.columns:
                df["Churn"] = df["Churn"].astype(int)
                churn_rate = df["Churn"].mean()
                logging.info(
                    f"Dataset loaded: {final_rows} rows, {churn_rate:.2%} churn rate"
                )
            else:
                logging.error("No 'Churn' column found in dataset")
                return {"error": "No 'Churn' column found in dataset"}

            return {
                "data": df.to_dict(orient="records"),
                "n_customers": len(df),
                "churn_rate": float(churn_rate),
            }

        except Exception as e:
            logging.error(f"Error loading data: {e}")

            raise RuntimeError(f"Failed to load e-commerce dataset: {e}")

    def _clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
        import numpy as np

        initial_rows = len(df)
        logging.info(f"Starting data cleaning for {initial_rows} rows...")

        missing_summary = df.isnull().sum()
        missing_percentage = (missing_summary / initial_rows * 100).round(2)

        for col in missing_summary[missing_summary > 0].index:
            logging.info(
                f'Column "{col}" has {missing_summary[col]} missing values ({missing_percentage[col]}%)'
            )

        # Remove rows where target variable is missing
        if "Churn" in df.columns:
            before_drop = len(df)
            df = df.dropna(subset=["Churn"])
            # FIX #3: Corrected log message
            logging.info(
                f"Removed {before_drop - len(df)} rows with missing Churn values"
            )

        # Handle ID columns - remove duplicates but keep the data
        if "CustomerID" in df.columns:
            duplicated_ids = df.duplicated(subset=["CustomerID"], keep="first").sum()
            if duplicated_ids > 0:
                # FIX #4: Corrected method name
                df = df.drop_duplicates(subset=["CustomerID"], keep="first")
                logging.info(
                    f"Removed {duplicated_ids} rows with duplicate CustomerID values"
                )

        # Numerical imputation strategies
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "Churn" in numerical_cols:
            numerical_cols.remove("Churn")

        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                if col in [
                    "Tenure",
                    "OrderCount",
                    "CouponUsed",
                    "NumberOfDeviceRegistered",
                ]:
                    df[col] = df[col].fillna(df[col].median())
                elif col in ["SatisfactionScore"]:
                    # FIX #5: Added missing 'not'
                    df[col] = df[col].fillna(
                        df[col].mode()[0] if not df[col].mode().empty else 3
                    )

                elif col in ["CashbackAmount", "OrderAmountHikeFromlastYear"]:
                    df[col] = df[col].fillna(0)
                elif col in ["DaySinceLastOrder"]:

                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                else:
                    df[col] = df[col].fillna(df[col].median())

                logging.info(f"Imputed {col} (numerical)")

        # Categorical imputation strategies
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if "CustomerID" in categorical_cols:
            categorical_cols.remove("CustomerID")

        for col in categorical_cols:
            if df[col].isnull().sum() > 0:

                if col in [
                    "PreferredPaymentMode",
                    "PreferredLoginDevice",
                    "PreferedOrderCat",
                ]:
                    mode_val = (
                        df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    )
                    df[col] = df[col].fillna(mode_val)
                elif col in ["Gender", "MaritalStatus"]:
                    mode_val = (
                        df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    )
                    df[col] = df[col].fillna(mode_val)
                else:
                    mode_val = (
                        df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    )
                    df[col] = df[col].fillna(mode_val)

                logging.info(f"Imputed {col} (categorical)")

        # Check for any remaining missing values in critical columns
        critical_columns = ["Churn", "Tenure", "SatisfactionScore"]
        critical_missing = df[critical_columns].isnull().sum()

        if critical_missing.sum() > 0:
            before_critical_drop = len(df)
            df = df.dropna(subset=critical_columns)
            logging.info(
                f"Removed {before_critical_drop - len(df)} rows with missing critical values"
            )

        # Data validation and cleaning - remove outliers

        outlier_columns = ["Tenure", "CashbackAmount", "OrderCount"]
        for col in outlier_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    logging.info(f"Removed {outliers} outliers from {col}")

        final_rows = len(df)
        retention_rate = (final_rows / initial_rows) * 100

        logging.info(
            f"Data cleaning completed: {final_rows}/{initial_rows} rows retained ({retention_rate:.1f}%)"
        )
        logging.info(f"Remaining missing values: {df.isnull().sum().sum()}")

        return df

    @task()
    def validate_data(data_dict: Dict[str, Any]) -> Dict[str, Any]:
        from data_utils import validate_data

        df = pd.DataFrame(data_dict["data"])

        validation_result = validate_data(df)

        if not validation_result.is_valid:
            raise ValueError(f"Data validation failed: {validation_result.issues}")

        logging.info("Data validation passed successfully")

        return {
            "data": df.to_dict(orient="records"),
            "n_customers": data_dict["n_customers"],
            "validation_passed": True,
        }

    @task()
    def engineer_features(df_dict: Dict[str, Any]) -> Dict[str, Any]:
        from data_utils import get_config
        from ml_pipeline import MLPipeline
        import joblib
        from pathlib import Path

        df = pd.DataFrame(df_dict["data"])

        config = get_config()
        ml_pipeline = MLPipeline(config)

        target_col = "Churn"
        feature_set = ml_pipeline.engineer_features(df, fit=True, target_col=target_col)

        X = feature_set.features.drop(
            columns=[target_col, "CustomerID"], errors="ignore"
        )
        y = feature_set.features[target_col]

        logging.info(
            f"Feature engineering completed: {len(X.columns)} features created"
        )

        temp_dir = Path("/tmp/airflow_features")
        temp_dir.mkdir(exist_ok=True)

        X_path = temp_dir / "X_features.pkl"
        y_path = temp_dir / "y_target.pkl"
        transformers_path = temp_dir / "transformers.pkl"

        joblib.dump(X, X_path)
        joblib.dump(y, y_path)
        joblib.dump(feature_set.transformers, transformers_path)

        return {
            "X_path": str(X_path),
            "y_path": str(y_path),
            "transformers_path": str(transformers_path),
            "feature_names": list(X.columns),
            "feature_metadata": feature_set.feature_metadata,
            "n_samples": len(X),
            "n_features": len(X.columns),
        }

    @task()
    def train_models(feature_data: Dict[str, Any]) -> Dict[str, Any]:
        from ml_pipeline import MLPipeline
        from data_utils import get_config
        import joblib
        from pathlib import Path
        import math

        config = get_config()
        ml_pipeline = MLPipeline(config)

        X = joblib.load(feature_data["X_path"])
        y = joblib.load(feature_data["y_path"])

        feature_names = feature_data["feature_names"]

        logging.info(
            f"Loaded training data: {len(X)} samples, {len(X.columns)} features"
        )

        # Load transformers (optional)
        try:
            transformers = joblib.load(feature_data["transformers_path"])
            ml_pipeline.transformers = transformers
        except Exception as e:
            logging.warning(f"Could not load transformers: {e}")

        # Train models
        model_results = ml_pipeline.train_models(X, y, feature_names)

        # Select best model
        best_model_name = max(
            model_results.keys(),
            key=lambda x: model_results[x].test_scores.get("roc_auc", 0),
        )

        best_result = model_results[best_model_name]

        # Save model
        production_model_path = Path("models/production/churn_model.pkl")
        transformers_path = Path("models/transformers/feature_transformers.pkl")

        ml_pipeline.save_model(
            best_result, str(production_model_path), str(transformers_path)
        )

        logging.info(
            f"Best model ({best_model_name}) saved with ROC-AUC: {best_result.test_scores['roc_auc']:.4f}"
        )

        # Clean feature importance for JSON serialization
        feature_importance = best_result.feature_importance.copy()
        feature_importance = feature_importance.replace(
            {float("nan"): None, float("inf"): None, float("-inf"): None}
        )

        # Clean model metrics for JSON serialization
        clean_metrics = {}
        for k, v in best_result.test_scores.items():
            if hasattr(v, "item"):
                clean_metrics[k] = float(v)
            elif isinstance(v, (int, float)):
                if math.isnan(v) or math.isinf(v):
                    clean_metrics[k] = None
                else:
                    clean_metrics[k] = float(v)
            else:
                clean_metrics[k] = v

        return {
            "best_model_name": best_model_name,
            "model_path": str(production_model_path),
            "transformers_path": str(transformers_path),
            "model_metrics": clean_metrics,
            "feature_importance": feature_importance.to_dict("records"),
        }

    @task()
    def register_model(training_result: Dict[str, Any]) -> Dict[str, Any]:
        from data_utils import register_model, promote_model

        model_version = register_model(
            model_path=training_result["model_path"],
            model_name="churn_prediction",
            model_type=training_result["best_model_name"],
            metrics=training_result["model_metrics"],
            metadata={
                "feature_importance": training_result["feature_importance"],
                "transformers_path": training_result["transformers_path"],
            },
        )

        # Promote to production if performance threshold is met
        roc_auc = training_result["model_metrics"].get("roc_auc", 0)
        if roc_auc > 0.75:
            promote_model("churn_prediction", model_version, "production")
            logging.info(f"Model version {model_version} promoted to production")
        else:
            logging.info(
                f"Model version {model_version} registered but not promoted (ROC-AUC: {roc_auc:.4f})"
            )

        return {
            "model_version": int(model_version),
            "promoted_to_production": bool(roc_auc > 0.75),
            "model_metrics": training_result["model_metrics"],
        }

    # Task dependencies
    raw_data = load_ecommerce_data()
    validated_data = validate_data(raw_data)
    features = engineer_features(validated_data)
    training_result = train_models(features)
    model_info = register_model(training_result)


churn_prediction_pipeline()
