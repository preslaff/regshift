"""
Database configuration and connection management.
"""

import databases
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Database URL
DATABASE_URL = settings.DATABASE_URL

# Create database instance
database = databases.Database(DATABASE_URL)

# SQLAlchemy engine
engine = sqlalchemy.create_engine(
    DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW
)

# Session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


async def get_database():
    """Dependency for getting database connection."""
    async with database.transaction():
        yield database


def get_db_session():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()