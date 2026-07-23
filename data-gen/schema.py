"""SQLAlchemy Core table definitions for the item catalog and users tables.

Uses the Postgres-specific ARRAY type for embedding/tag columns since
this reference data is generated and loaded against the local
docker-compose Postgres warehouse. Snowflake compatibility for these
tables is out of scope until an actual AWS demo session needs it.
"""

from sqlalchemy import Column, Date, Float, MetaData, String, Table
from sqlalchemy.dialects.postgresql import ARRAY

metadata = MetaData()

item_catalog_table = Table(
    "item_catalog",
    metadata,
    Column("item_id", String, primary_key=True),
    Column("category", String, nullable=False),
    Column("subcategory", String, nullable=False),
    Column("brand", String, nullable=False),
    Column("price", Float, nullable=False),
    Column("tags", ARRAY(String), nullable=False),
    Column("description", String, nullable=False),
    Column("image_embedding", ARRAY(Float), nullable=True),
    Column("text_embedding", ARRAY(Float), nullable=False),
)

users_table = Table(
    "users",
    metadata,
    Column("user_id", String, primary_key=True),
    Column("signup_date", Date, nullable=False),
)
