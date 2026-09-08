"""
DAG: Pipeline de riesgo crediticio
1. Carga el dataset crudo a PostgreSQL (staging)
2. Entrena y compara modelos (Random Forest vs Regresión Logística)
3. Publica métricas y feature importance en PostgreSQL
4. Genera predicciones sobre el batch actual
"""
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

DATA_PATH = "/opt/airflow/data/credit_default.csv"
MODEL_DIR = "/opt/airflow/models"

default_args = {
    "owner": "credit-risk-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def load_raw_data(**context):
    df = pd.read_csv(DATA_PATH)
    hook = PostgresHook(postgres_conn_id="credit_postgres")
    engine = hook.get_sqlalchemy_engine()

    records = df[
        ["credit_limit", "gender", "education", "marital_status", "age", "avg_balance", "avg_payment"]
    ].copy()
    records["raw_payload"] = df.to_dict(orient="records")
    records.to_sql("credit_applications", engine, if_exists="append", index=False)
    context["ti"].xcom_push(key="row_count", value=len(df))


def train_and_compare_models(**context):
    import sys

    sys.path.insert(0, "/opt/airflow/ml")
    from train_model import train  # noqa: E402

    results = train()
    context["ti"].xcom_push(key="metrics", value=results)


def publish_metrics(**context):
    import json

    metrics = context["ti"].xcom_pull(key="metrics", task_ids="train_and_compare_models")
    hook = PostgresHook(postgres_conn_id="credit_postgres")

    for model_name in ["random_forest", "logistic_regression"]:
        model_metrics = metrics[model_name]
        run_id = hook.get_first(
            """
            INSERT INTO model_runs (model_name, auc_roc, n_records, default_rate, metrics_json)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
            """,
            parameters=(
                model_name,
                model_metrics["auc_roc"],
                metrics["n_records"],
                metrics["default_rate"],
                json.dumps(model_metrics),
            ),
        )[0]

        if model_name == metrics["best_model"]:
            for fi in metrics["feature_importance"]:
                hook.run(
                    """
                    INSERT INTO feature_importance (model_run_id, feature_name, importance)
                    VALUES (%s, %s, %s)
                    """,
                    parameters=(run_id, fi["feature"], fi["importance"]),
                )


with DAG(
    dag_id="credit_risk_pipeline",
    default_args=default_args,
    description="Pipeline de scoring de riesgo crediticio (RF vs LogReg)",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "credit-risk"],
) as dag:

    t1 = PythonOperator(task_id="load_raw_data", python_callable=load_raw_data)
    t2 = PythonOperator(task_id="train_and_compare_models", python_callable=train_and_compare_models)
    t3 = PythonOperator(task_id="publish_metrics", python_callable=publish_metrics)

    t1 >> t2 >> t3
