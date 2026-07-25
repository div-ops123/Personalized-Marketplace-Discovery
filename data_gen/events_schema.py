"""SQLAlchemy Core table definitions for the raw event log.

Columns match data-flow.md's "What Gets Logged" section exactly. No engine
or side effects here -- see db_writer.py for table creation and writes.
"""

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, PrimaryKeyConstraint, String, Table

events_metadata = MetaData()

impression_events_table = Table(
    "impression_events",
    events_metadata,
    Column("timestamp", DateTime, nullable=False),
    Column("user_id", String, nullable=False),
    Column("anchor_item_id", String, nullable=False),
    Column("recommendation_impression_id", String, nullable=False),
    Column("candidate_item_id", String, nullable=False),
    Column("position", Integer, nullable=False),
    Column("retrieval_similarity_score", Float, nullable=False),
    Column("device", String, nullable=False),
    Column("country", String, nullable=False),
    PrimaryKeyConstraint("recommendation_impression_id", "candidate_item_id"),
)

click_events_table = Table(
    "click_events",
    events_metadata,
    Column("click_time", DateTime, nullable=False),
    Column("user_id", String, nullable=False),
    Column("candidate_item_id", String, nullable=False),
    Column("recommendation_impression_id", String, nullable=False),
    PrimaryKeyConstraint("recommendation_impression_id", "candidate_item_id"),
)

purchase_events_table = Table(
    "purchase_events",
    events_metadata,
    Column("purchase_time", DateTime, nullable=False),
    Column("user_id", String, nullable=False),
    Column("item_id", String, nullable=False),
    Column("order_id", String, primary_key=True),
)
