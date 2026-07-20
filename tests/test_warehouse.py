"""Unit tests for common.warehouse connection URL construction."""

import pytest

from common.warehouse import UnsupportedWarehouseBackendError, build_connection_url


def test_postgres_url_uses_env_vars(monkeypatch):
    """Postgres URL is built from POSTGRES_* env vars."""
    monkeypatch.setenv("POSTGRES_USER", "dev")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "marketplace_dw")

    url = build_connection_url("postgres")

    assert url == "postgresql+psycopg2://dev:secret@localhost:5432/marketplace_dw"


def test_postgres_url_defaults_host_and_port(monkeypatch):
    """Host/port fall back to sensible defaults when unset."""
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.setenv("POSTGRES_USER", "dev")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_DB", "marketplace_dw")

    url = build_connection_url("postgres")

    assert "@localhost:5432/" in url


def test_postgres_url_missing_required_var_raises(monkeypatch):
    """A missing required Postgres env var raises ValueError."""
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_DB", "marketplace_dw")

    with pytest.raises(ValueError):
        build_connection_url("postgres")


def test_snowflake_url_uses_env_vars(monkeypatch):
    """Snowflake URL is built from SNOWFLAKE_* env vars."""
    monkeypatch.setenv("SNOWFLAKE_USER", "dev")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "secret")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "abc123")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "marketplace_dw")
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "public")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "compute_wh")

    url = build_connection_url("snowflake")

    assert url == (
        "snowflake://dev:secret@abc123/marketplace_dw/public?warehouse=compute_wh"
    )


def test_unsupported_backend_raises():
    """An unrecognized backend name raises a clear, typed error."""
    with pytest.raises(UnsupportedWarehouseBackendError):
        build_connection_url("mysql")
