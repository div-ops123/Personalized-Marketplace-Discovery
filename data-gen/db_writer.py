"""Side-effecting writes of the item catalog, users, and raw event tables."""

import logging

import pandas as pd
from sqlalchemy.engine import Engine

from events_schema import click_events_table, events_metadata, impression_events_table, purchase_events_table
from schema import item_catalog_table, metadata, users_table

logger = logging.getLogger(__name__)

# Keeps a single multi-row INSERT under Postgres's ~65535 bound-parameter
# limit even for the widest event table (9 columns).
_WRITE_CHUNKSIZE = 1000


def create_reference_tables(engine: Engine) -> None:
    """Creates the item_catalog and users tables if they don't already exist.

    Args:
        engine: A SQLAlchemy engine for the target warehouse.
    """
    metadata.create_all(engine)
    logger.debug("Ensured item_catalog and users tables exist.")


def write_item_catalog(items_df: pd.DataFrame, engine: Engine) -> None:
    """Appends the generated Item Catalog rows to the warehouse.

    Args:
        items_df: Output of item_catalog.build_item_catalog, with a
            text_embedding column already added.
        engine: A SQLAlchemy engine for the target warehouse.
    """
    items_df.to_sql(
        item_catalog_table.name, engine, if_exists="append", index=False, method="multi"
    )
    logger.info("Wrote %d rows to %s.", len(items_df), item_catalog_table.name)


def write_users(users_df: pd.DataFrame, engine: Engine) -> None:
    """Appends the generated user population rows to the warehouse.

    Args:
        users_df: Output of user_population.build_users.
        engine: A SQLAlchemy engine for the target warehouse.
    """
    users_df.to_sql(users_table.name, engine, if_exists="append", index=False, method="multi")
    logger.info("Wrote %d rows to %s.", len(users_df), users_table.name)


def create_event_tables(engine: Engine) -> None:
    """Creates the impression/click/purchase event tables if they don't exist.

    Args:
        engine: A SQLAlchemy engine for the target warehouse.
    """
    events_metadata.create_all(engine)
    logger.debug("Ensured impression_events, click_events, and purchase_events tables exist.")


def write_impressions(impressions_df: pd.DataFrame, engine: Engine) -> None:
    """Appends Impression Event rows to the warehouse.

    Args:
        impressions_df: Rows matching impression_events_table's schema.
        engine: A SQLAlchemy engine for the target warehouse.
    """
    if impressions_df.empty:
        return
    impressions_df.to_sql(
        impression_events_table.name,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=_WRITE_CHUNKSIZE,
    )
    logger.debug("Wrote %d rows to %s.", len(impressions_df), impression_events_table.name)


def write_clicks(clicks_df: pd.DataFrame, engine: Engine) -> None:
    """Appends Click Event rows to the warehouse.

    Args:
        clicks_df: Rows matching click_events_table's schema.
        engine: A SQLAlchemy engine for the target warehouse.
    """
    if clicks_df.empty:
        return
    clicks_df.to_sql(
        click_events_table.name,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=_WRITE_CHUNKSIZE,
    )
    logger.debug("Wrote %d rows to %s.", len(clicks_df), click_events_table.name)


def write_purchases(purchases_df: pd.DataFrame, engine: Engine) -> None:
    """Appends Purchase Event rows to the warehouse.

    Args:
        purchases_df: Rows matching purchase_events_table's schema.
        engine: A SQLAlchemy engine for the target warehouse.
    """
    if purchases_df.empty:
        return
    purchases_df.to_sql(
        purchase_events_table.name,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=_WRITE_CHUNKSIZE,
    )
    logger.debug("Wrote %d rows to %s.", len(purchases_df), purchase_events_table.name)
