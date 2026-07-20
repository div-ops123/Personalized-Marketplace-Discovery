"""Warehouse connection layer with a Postgres/Snowflake backend switch.

The backend is selected via the WAREHOUSE_BACKEND environment variable so
the same pipeline code runs against local Postgres (docker-compose) or
real Snowflake (AWS demo session) with no code change.
"""

import os
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


class UnsupportedWarehouseBackendError(Exception):
    """Raised when WAREHOUSE_BACKEND is not a recognized value."""


def _require_env(name: str) -> str:
    """Returns the value of a required environment variable.

    Args:
        name: Environment variable name.

    Returns:
        str: The variable's value.

    Raises:
        ValueError: If the variable is unset or empty.
    """
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _build_postgres_url() -> str:
    """Builds a SQLAlchemy connection URL for the local Postgres backend.

    Returns:
        str: A postgresql+psycopg2:// connection URL.
    """
    user = _require_env("POSTGRES_USER")
    password = quote_plus(_require_env("POSTGRES_PASSWORD"))
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = _require_env("POSTGRES_DB")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def _build_snowflake_url() -> str:
    """Builds a SQLAlchemy connection URL for the Snowflake backend.

    Returns:
        str: A snowflake:// connection URL.
    """
    user = _require_env("SNOWFLAKE_USER")
    password = quote_plus(_require_env("SNOWFLAKE_PASSWORD"))
    account = _require_env("SNOWFLAKE_ACCOUNT")
    db = _require_env("SNOWFLAKE_DATABASE")
    schema = _require_env("SNOWFLAKE_SCHEMA")
    warehouse = _require_env("SNOWFLAKE_WAREHOUSE")
    return (
        f"snowflake://{user}:{password}@{account}/{db}/{schema}"
        f"?warehouse={warehouse}"
    )


def build_connection_url(backend: str) -> str:
    """Builds the SQLAlchemy connection URL for the given backend.

    Args:
        backend: Either "postgres" or "snowflake".

    Returns:
        str: A SQLAlchemy-compatible connection URL.

    Raises:
        UnsupportedWarehouseBackendError: If backend is not recognized.
        ValueError: If a required connection env var is missing.
    """
    if backend == "postgres":
        return _build_postgres_url()
    if backend == "snowflake":
        return _build_snowflake_url()
    raise UnsupportedWarehouseBackendError(
        f"Unknown WAREHOUSE_BACKEND: {backend!r}. Expected 'postgres' or 'snowflake'."
    )


def get_engine() -> Engine:
    """Creates a SQLAlchemy engine for the configured warehouse backend.

    Backend is selected by the WAREHOUSE_BACKEND environment variable so
    pipeline code never branches on environment itself — set it to
    "postgres" for local/docker-compose runs or "snowflake" for the AWS
    demo session.

    Returns:
        Engine: A SQLAlchemy engine ready for use.

    Raises:
        UnsupportedWarehouseBackendError: If WAREHOUSE_BACKEND is not
            "postgres" or "snowflake".
        ValueError: If a required connection env var is missing.

    Examples:
        >>> os.environ["WAREHOUSE_BACKEND"] = "postgres"
        >>> engine = get_engine()
    """
    backend = os.environ.get("WAREHOUSE_BACKEND", "postgres")
    url = build_connection_url(backend)
    return create_engine(url)
