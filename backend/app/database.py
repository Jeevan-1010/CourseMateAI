import os
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from backend.app.models import Base


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured.")

    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


def check_database_connection() -> None:
    with get_engine().connect() as connection:
        connection.execute(text("SELECT 1"))


def initialize_database() -> None:
    Base.metadata.create_all(bind=get_engine())
