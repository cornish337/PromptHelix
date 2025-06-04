from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from .base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True) # Nullable for now, assuming password might not be set initially or for OAuth users
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

# TODO: Define a Session class for database interactions.
# This might involve creating a SQLAlchemy sessionmaker.
# from sqlalchemy.orm import sessionmaker
# from ..core.database import engine # Assuming engine is defined in core.database
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
class Session:
    # Placeholder for session-related methods or configurations
    pass
