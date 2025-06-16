import os
import logging # Added import
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from prompthelix.models.base import Base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./prompthelix.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

__all__ = [
    "DATABASE_URL",
    "engine",
    "SessionLocal",
    "Base",
    "get_db",
    "init_db",
]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    logging.basicConfig(level=logging.INFO)  # Added logging config
    logger = logging.getLogger(__name__)  # Added logger instance
    logger.info("Initializing database...")  # Added log message
    logger.info(f"Using database URL: {DATABASE_URL}") # Added log message
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables checked/created.") # Added log message
