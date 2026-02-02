import sys

sys.path.insert(0, "/usr/local/airflow/dags")

from airflow.decorators import dag, task
from pendulum import datetime
import pandas as pd
from typing import Dict, Any
import logging


@dag(
    start_date=datetime(2026, 1, 30),
    schedule_interval="@weekly",
    catchup=False,
    tags=["ml", "churn", "prediction", "ecommerce"],
    description="Production customer churn prediction pipeline with ML features",
)
def churn_prediction_pipeline():
    @task()
    def load_ecommerce_data():
        try:
            excel_path = "/usr/local/airflow/data/E commerce Dataset.xlsx"
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
            if "Churn" not in df.columns:
                df["Churn"] = df["Churn"].astype(int)
                churn_rate = df["Churn"].mean()
                logging.info(
                    f"Dataset loaded: {final_rows} rows, {churn_rate:.2f}% churn rate"
                )
            else:
                logging.info("No 'Churn' column found in dataset")
                return {"error": "No 'Churn' column found in dataset"}

            return {
                "data": df.to_dict(orient="records"),
                "n_customers": len(df),
                "churn_rate": float(churn_rate),
            }

        except Exception as e:
            logging.error(f"Error loading data: {e}")

    def _clean_and_inpute_data(df: pd.DataFrame):
        import numpy as np

        initial_rows = len(df)
        logging.info(f"Starting data cleaning for {initial_rows} rows...")

        missing_summary = df.isnull().sum()
        missing_percentage = (missing_summary / initial_rows * 100).round(2)

        for col in missing_summary[missing_summary > 0].index:
            logging.info(f'Column "{col}" has {missing_summary[col]} missing values')

        # remove columns for target variable
        if "Churn" in df.columns:
            df = df.drop(subset=["Churn"])
            logging.info(f'Dropped column "Churn" because it is the target variable.')

        # Handle ID columns - remove the duplicates but keep the data
        if "CustomerID" in df.columns:
            duplicated_ids = df.duplicated(subset=["CustomerID"], keep="first").sum()
            if duplicated_ids > 0:
                df = df.drop_duplicates(subset=["CustomerID"], keep="first")
                logging.info(
                    f"Dropped {duplicated_ids} rows with duplicate CustomerID values."
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
                    df[col] = df[col].fillna(
                        df[col].mode()[0] if df[col].mode().empty else 3
                    )
                elif col in ["CashbackAmout", "OrderAmountHikeFromlastYear"]:
                    df[col] = df[col].fillna(0)
                elif col in ["DaySinceLastOrder"]:
                    median = df[col].median()
                    df[col] = df[col].fillna(median if median < 10 else 10)
                else:
                    df[col] = df[col].fillna(df[col].median())

                logging.info(f"Imputed {col} numerical")

        # 4. Categorical imputation strategies
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if "CustomerID" in categorical_cols:
            categorical_cols.remove("CustomerID")

        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                if col in [
                    "PreferredPayment",
                    "PreferredLoginDevice",
                    "PreferredOrderCategory",
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

                logging.info(f"Imputed {col} categorical")

        # chek for any remaining values in critical columns
        critical_columns = ["Churn", "Tenure", "SatisfactionScore"]
        critical_missing = df[critical_columns].isnull().sum()

        if critical_missing.sum() > 0:
            before_critical_drop = len(df)
            df = df.dropna(subset=critical_columns)
            logging.info(
                f"Dropped {before_critical_drop - len(df)} rows with missing critical values."
            )

        # Data validation and cleaning
        outlier_columns = ["Tenure", "Cashback", "OrderCount"]
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
                    logging.info(f"Dropped {outliers} outliers from column {col}.")

        final_rows = len(df)
        retention_rate = (final_rows / initial_rows) * 100

        logging.info(f"Data cleaning completed. {len(df)} rows remain after cleaning.")
        logging.info(f"Missing values summary: \n{missing_percentage}")

        return df

    @task()
    def validate_data(data_dict):
        from data_utils import validate_data

        df = pd.DataFrame(data_dict["data"])

        validation_result = validate_data(df)

        if not validation_result.is_valid:
            raise ValueError(f"Data validation failed: {validation_result.issues}")

        logging.info(f"Data validation complete: {validation_result.summary}")

        return {
            "data": df.to_dict(orient="records"),
            "n_customers": data_dict["n_customers"],
            "validation_passed": True,
        }
