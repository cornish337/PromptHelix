"""Shared SQLAlchemy declarative base used by all ORM models."""

from sqlalchemy.orm import declarative_base

# A single Base instance is used across the project so that models can be
# registered without creating import cycles.
Base = declarative_base()

__all__ = ["Base"]
