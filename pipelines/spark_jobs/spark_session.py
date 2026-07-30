"""Local-mode SparkSession construction.

No dedicated Spark cluster exists in this project's infra -- PySpark runs
in local mode (master="local[*]") inside whichever process executes the
job, matching docs/build-phases.md step 2's reasoning against standing up
a master/worker cluster at this data scale.
"""

import os
import sys

from pyspark.sql import SparkSession

# Without this, Spark's Python worker subprocess resolves "python" via
# PATH -- on Windows that hits the Microsoft Store alias stub instead of
# the actual (venv) interpreter, and every worker task fails with
# "Timed out while waiting for the Python worker to connect back".
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def get_spark_session(app_name: str = "daily_feature_pipeline", master: str = "local[*]") -> SparkSession:
    """Builds a local-mode SparkSession with quiet logging.

    Args:
        app_name: Spark application name, shown in the (unused) Spark UI.
        master: Spark master URL. Tests pass "local[1]" to keep a single
            worker thread for lighter, more deterministic test runs.

    Returns:
        SparkSession: A local-mode session.
    """
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.ui.enabled", "false")
        # Default driver heap (1g) OOMs on run_dataset_builders.py's full-history
        # collect()/toPandas() calls once real event volume (not tiny unit-test
        # fixtures) is involved. local[*] mode runs driver+executors in one JVM,
        # so only spark.driver.memory needs raising here.
        .config("spark.driver.memory", "8g")
        # Default cap (1g) on how much a toPandas()/collect() may return to
        # the driver -- the retrieval dataset alone serializes to several GB
        # once per-row anchor+candidate text_embedding arrays are included
        # across 1.6M+ impression rows.
        .config("spark.driver.maxResultSize", "6g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
