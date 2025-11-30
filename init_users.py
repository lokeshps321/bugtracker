#!/usr/bin/env python3
"""
Initialize database with default users using hashed passwords
"""
import os
import sys
from dotenv import load_dotenv

# Set environment to development for initializing
os.environ.setdefault("ENVIRONMENT", "development")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app import models, database
from app.auth import get_password_hash

# Load environment variables
load_dotenv()

# Check if SECRET_KEY is set
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    print("ERROR: SECRET_KEY environment variable must be set")
    print("Please set SECRET_KEY in your .env file")
    sys.exit(1)

def init_users():
    # Create database engine and session
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///bugflow.db")
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    models.Base.metadata.create_all(bind=engine)
    
    # Create session and add default users
    db = SessionLocal()
    try:
        # Check if users already exist
        existing_user_count = db.query(models.User).count()
        if existing_user_count > 0:
            print("Users already exist in database. Skipping initialization.")
            return
        
        # Create default users with hashed passwords
        default_users = [
            models.User(
                email="pm1@example.com",
                password_hash=get_password_hash("password"),
                role="project_manager"
            ),
            models.User(
                email="tester1@example.com",
                password_hash=get_password_hash("password"),
                role="tester"
            ),
            models.User(
                email="dev1@example.com",
                password_hash=get_password_hash("password"),
                role="developer"
            )
        ]
        
        for user in default_users:
            db.add(user)
        
        db.commit()
        print("Database initialized with default users successfully!")
        print("Default users created:")
        print("- pm1@example.com (password: password, role: project_manager)")
        print("- tester1@example.com (password: password, role: tester)")
        print("- dev1@example.com (password: password, role: developer)")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    init_users()