FROM apache/airflow:2.9.3-python3.11

COPY requirements.txt /requirements.txt
RUN python -m venv /home/airflow/pipeline-venv \
    && /home/airflow/pipeline-venv/bin/pip install --no-cache-dir -r /requirements.txt
