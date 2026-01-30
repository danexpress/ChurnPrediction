import sys

sys.path.insert(0, "/usr/local/airflow/dags")

from airflow.decorators import dag, task
from pendulum import datetime
import pandas as pd
from typing import Dict, Any
import logging

@dag(
    start_date=datetime(2026, 1, 30),
    schedule_interval='@weekly',
    catchup=False,
    tags=['ml', 'churn', 'prediction', 'ecommerce'],
    description='Production customer churn prediction pipeline with ML features'
)
def churn_prediction_pipeline():
    @task()
    def load_ecommerce_data():
        try:
            excel_path = '/usr/local/airflow/data/E commerce Dataset.xlsx'
            sheet_name = 'E Comm'

        logging.info(f"Loading data from {excel_path}...")
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        logging.info(f"Loaded {len(df)} rows of data with shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")

        df = _clean_and_inpute_data(df)

        return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
        

    def _clean_and_inpute_data(df: pd.DataFrame):
        import numpy as np

        initial_rows = len(df)
        logging.info(f'Starting data cleaning for {initial_rows} rows...')

        missing_summary = df.isnull().sum()
        missing_percentage = (missing_summary / initial_rows * 100).round(2)

        for col in missing_summary[missing_summary > 0].index:
            logging.info(f'Column "{col}" has {missing_summary[col]} missing values')
        
        if 'Churn' in df.columns:
            df = df.drop(subset=['Churn'])
            logging.info(f'Dropped column "Churn" because it is the target variable.')

        if 'CustomerID' in df.columns:
            duplicated_ids = df.duplicated(subset=['CustomerID'], keep='first').sum()
            if duplicated_ids > 0:
                df = df.drop_duplicates(subset=['CustomerID'], keep='first')
                logging.info(f'Dropped {duplicated_ids} rows with duplicate CustomerID values.')

        logging.info(f'Data cleaning completed. {len(df)} rows remain after cleaning.')
        logging.info(f'Missing values summary: \n{missing_percentage}')

        return df
