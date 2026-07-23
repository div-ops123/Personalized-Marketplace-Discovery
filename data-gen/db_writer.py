"""Side-effecting writes of the item catalog and users tables to the warehouse."""

import logging

import pandas as pd
from sqlalchemy.engine import Engine

from schema import item_catalog_table, metadata, users_table

logger = logging.getLogger(__name__)


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
