"""SQLAlchemy Core table definitions for the two materialized training
dataset tables (data-flow.md's "Two Independent Dataset Builders").

Unlike the daily feature tables, these are not historized -- each run of
the dataset builders rebuilds the whole table from current history (see
dataset_db_io.py's truncate-then-insert write), so there's no snapshot_date
partition key.
"""

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, PrimaryKeyConstraint, String, Table
from sqlalchemy.dialects.postgresql import ARRAY

dataset_metadata = MetaData()

retrieval_training_examples_table = Table(
    "retrieval_training_examples",
    dataset_metadata,
    Column("recommendation_impression_id", String, nullable=False),
    Column("candidate_item_id", String, nullable=False),
    Column("anchor_item_id", String, nullable=False),
    Column("anchor_category", String, nullable=False),
    Column("anchor_subcategory", String, nullable=False),
    Column("anchor_brand", String, nullable=False),
    Column("anchor_price_tier", String, nullable=False),
    Column("anchor_tags", ARRAY(String), nullable=False),
    Column("anchor_image_embedding", ARRAY(Float), nullable=True),
    Column("anchor_text_embedding", ARRAY(Float), nullable=False),
    Column("candidate_category", String, nullable=False),
    Column("candidate_subcategory", String, nullable=False),
    Column("candidate_brand", String, nullable=False),
    Column("candidate_price_tier", String, nullable=False),
    Column("candidate_tags", ARRAY(String), nullable=False),
    Column("candidate_image_embedding", ARRAY(Float), nullable=True),
    Column("candidate_text_embedding", ARRAY(Float), nullable=False),
    Column("label", Integer, nullable=False),
    PrimaryKeyConstraint("recommendation_impression_id", "candidate_item_id"),
)

ranking_training_examples_table = Table(
    "ranking_training_examples",
    dataset_metadata,
    Column("recommendation_impression_id", String, nullable=False),
    Column("candidate_item_id", String, nullable=False),
    Column("user_id", String, nullable=False),
    Column("anchor_item_id", String, nullable=False),
    Column("timestamp", DateTime, nullable=False),
    Column("device", String, nullable=False),
    Column("country", String, nullable=False),
    Column("position", Integer, nullable=False),
    Column("retrieval_similarity_score", Float, nullable=False),
    # User features -- nullable: no User Daily Features snapshot exists yet
    # before this impression for new/cold-start users (LLD.md's documented
    # "user-level features simply unavailable" behavior).
    Column("user_preferred_brands", ARRAY(String), nullable=True),
    Column("user_avg_purchase_price", Float, nullable=True),
    Column("user_historical_category_affinity", ARRAY(String), nullable=True),
    Column("anchor_category", String, nullable=False),
    Column("anchor_subcategory", String, nullable=False),
    Column("anchor_brand", String, nullable=False),
    Column("anchor_price", Float, nullable=False),
    Column("candidate_category", String, nullable=False),
    Column("candidate_subcategory", String, nullable=False),
    Column("candidate_brand", String, nullable=False),
    Column("candidate_price", Float, nullable=False),
    # Candidate historical rates -- nullable: a candidate's own first-ever
    # impression predates any Candidate Daily Features snapshot for it.
    Column("candidate_recommendation_ctr", Float, nullable=True),
    Column("candidate_recommendation_cvr", Float, nullable=True),
    Column("candidate_recommendation_impressions", Integer, nullable=True),
    Column("same_brand", Integer, nullable=False),
    Column("same_category", Integer, nullable=False),
    Column("same_subcategory", Integer, nullable=False),
    Column("price_ratio", Float, nullable=False),
    Column("price_diff", Float, nullable=False),
    Column("label", Integer, nullable=False),
    PrimaryKeyConstraint("recommendation_impression_id", "candidate_item_id"),
)
