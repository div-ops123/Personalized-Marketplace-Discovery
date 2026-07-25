"""Airflow DAG 1 -- Daily Feature Pipeline.

Backfills docs/data-flow.md's User Daily Features and Candidate Daily
Features tables from the raw event log, one task instance per simulated
day. Trigger a historical backfill with:

    docker compose -f infra/docker-compose.pipeline.yml exec airflow-scheduler \
        airflow dags backfill daily_feature_pipeline -s 2025-01-01 -e 2025-04-01

start_date/end_date below must match data_gen/config.py's
SIMULATION_START_DATE (2025-01-01, 90-day SIMULATION_WINDOW_DAYS -> last
raw-event day is 2025-03-31). end_date is one day past that so the final
backfilled snapshot_date (2025-04-01) reflects the complete 90-day history
-- snapshot_date D aggregates events strictly before day D (see
pipelines/spark_jobs/daily_features.py), so 2025-04-01's snapshot is the
first one to see all of 2025-03-31's events.

Each task shells out to the isolated pipeline-venv (see
infra/airflow/Dockerfile) rather than importing pipelines.spark_jobs
directly -- Airflow's own environment doesn't have pandas/pyspark
installed, only the pipeline-venv does. This file itself must stay light
(only Airflow's own dependencies) since it's parsed by the scheduler
process directly.
"""

import datetime

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

PIPELINE_PYTHON = "/home/airflow/pipeline-venv/bin/python"
RUN_SCRIPT = "/opt/airflow/project/pipelines/spark_jobs/run_daily_features.py"

with DAG(
    dag_id="daily_feature_pipeline",
    description="DAG 1: compute User/Candidate Daily Features from raw events for one day.",
    schedule_interval="@daily",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    end_date=pendulum.datetime(2025, 4, 1, tz="UTC"),
    catchup=False,
    max_active_runs=4,
    default_args={"retries": 1, "retry_delay": datetime.timedelta(minutes=2)},
    tags=["daily-features", "phase-3"],
) as dag:
    compute_daily_features = BashOperator(
        task_id="compute_daily_features",
        bash_command=(f"{PIPELINE_PYTHON} {RUN_SCRIPT} --snapshot-date " "{{ ds }}"),
    )
