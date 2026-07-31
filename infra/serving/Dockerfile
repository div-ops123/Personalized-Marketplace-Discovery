# Shared image for all three Phase 6 serving containers (API Gateway,
# Retrieval Service, Ranking Service) -- they differ only by the uvicorn
# entrypoint each docker-compose.serving.yml service passes as its
# command. Plain uv, this repo's actual package manager -- unlike
# infra/airflow/Dockerfile's pip+hand-built-venv split (needed there to
# isolate PySpark's JVM/JDK requirement from Airflow's own managed
# Python env), serving is pure Python/FastAPI with no such isolation
# need.
FROM python:3.11-slim

# COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependencies before source: lets Docker cache the (slow) dependency
# install layer across rebuilds that only touch serving/training code.
COPY pyproject.toml uv.lock ./
RUN uv sync --extra serving --no-install-project --no-dev

# Only what the serving containers actually import at runtime -- no
# Postgres/Airflow/Spark/MLflow/torch, per docs/build-phases.md Phase 6's
# "clone and click around without touching Postgres/Airflow/Spark at
# all" (the static exports mounted at runtime cover the rest).
COPY common ./common
COPY training/ranking_preprocessing.py training/vocab.py training/__init__.py ./training/
COPY serving ./serving

ENV PATH="/app/.venv/bin:$PATH"
