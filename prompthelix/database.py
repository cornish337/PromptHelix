import os
import logging # Added import
from prompthelix.utils.logging_utils import setup_logging
from prompthelix.config import settings

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from prompthelix.models.base import Base
# Import all models to ensure they are registered with Base.metadata
from prompthelix import models  # This imports prompthelix/models/__init__.py

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./prompthelix.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
print(f"DEBUG: database.py: SessionLocal defined at module level, bound to engine: {engine}")

logger = logging.getLogger(__name__)

__all__ = [
    "DATABASE_URL",
    "engine",
    "SessionLocal",
    "Base",
    "get_db",
    "init_db",
]

def get_db():
    print(f"DEBUG: prompthelix.database.get_db CALLED. Using SessionLocal bound to engine: {SessionLocal.kw['bind']}")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables if they do not exist."""

    # Logging is configured centrally by setup_logging() in main.py or cli.py
    logger = logging.getLogger(__name__)
    logger.info("Initializing database...")
    logger.info(f"Using database URL: {DATABASE_URL}")

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables checked/created.")
