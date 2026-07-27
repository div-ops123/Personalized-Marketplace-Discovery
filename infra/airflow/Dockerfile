FROM apache/airflow:2.9.3-python3.11

# PySpark's local mode launches a JVM gateway subprocess -- the base Airflow
# image has no JDK at all. default-jdk-headless resolves to a Debian-tracked
# openjdk build; default-java is the stable update-alternatives symlink so
# JAVA_HOME doesn't need to hardcode a version-specific path.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jdk-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java
USER airflow

COPY requirements-pipeline.txt /requirements.txt
RUN python -m venv /home/airflow/pipeline-venv \
    && /home/airflow/pipeline-venv/bin/pip install --no-cache-dir -r /requirements.txt
