"""Airflow DAG connecting all three assignment stages."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON = PROJECT_ROOT / ".venv/bin/python"
COMPOSE_FILE = PROJECT_ROOT / "code/deployment/docker-compose.yml"

with DAG(
    dag_id="pmldl_assignment_1_pipeline",
    description="Preprocess, train and deploy the Palmer Penguins classifier",
    start_date=datetime(2026, 1, 1),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["pmldl", "assignment-1", "mlops"],
) as dag:
    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command=f"'{PYTHON}' '{PROJECT_ROOT / 'code/datasets/preprocess.py'}'",
    )
    train_and_evaluate = BashOperator(
        task_id="train_and_evaluate",
        bash_command=f"'{PYTHON}' '{PROJECT_ROOT / 'code/models/train.py'}'",
    )
    deploy = BashOperator(
        task_id="deploy",
        bash_command=f"docker compose -f '{COMPOSE_FILE}' up -d --build --force-recreate --remove-orphans",
    )

    preprocess_data >> train_and_evaluate >> deploy
