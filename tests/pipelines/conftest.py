"""Shared fixtures for pipelines/ tests."""

import pytest
from pyspark.sql import SparkSession

from pipelines.spark_jobs.spark_session import get_spark_session


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """A local, single-threaded SparkSession reused across the whole test session."""
    session = get_spark_session(app_name="pipelines-tests", master="local[1]")
    yield session
    session.stop()
