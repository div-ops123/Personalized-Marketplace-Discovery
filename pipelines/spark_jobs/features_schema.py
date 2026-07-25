"""SQLAlchemy Core table definitions for the historized daily feature tables.

Columns match data-flow.md's "Offline Feature Store Shape" table exactly.
Both tables are historized (one row per entity per snapshot_date,
append-only in steady state) -- see db_io.py for the delete-then-insert
write that makes a backfill rerun of a single day idempotent.
"""

from sqlalchemy import Column, Date, Float, Integer, MetaData, PrimaryKeyConstraint, String, Table
from sqlalchemy.dialects.postgresql import ARRAY

features_metadata = MetaData()

user_daily_features_table = Table(
    "user_daily_features",
    features_metadata,
    Column("snapshot_date", Date, nullable=False),
    Column("user_id", String, nullable=False),
    Column("preferred_brands", ARRAY(String), nullable=False),
    Column("avg_purchase_price", Float, nullable=False),
    Column("historical_category_affinity", ARRAY(String), nullable=False),
    PrimaryKeyConstraint("snapshot_date", "user_id"),
)

candidate_daily_features_table = Table(
    "candidate_daily_features",
    features_metadata,
    Column("snapshot_date", Date, nullable=False),
    Column("candidate_id", String, nullable=False),
    Column("recommendation_ctr", Float, nullable=False),
    Column("recommendation_cvr", Float, nullable=False),
    Column("recommendation_impressions", Integer, nullable=False),
    PrimaryKeyConstraint("snapshot_date", "candidate_id"),
)
