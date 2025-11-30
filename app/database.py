from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///bugflow.db")

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base model for ORM classes
Base = declarative_base()


def init_db():
    """
    Initialize database tables if they do not exist.
    This avoids 'table already exists' errors.
    """
    from app import models  # Import ORM models
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency that provides a DB session for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
