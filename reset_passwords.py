#!/usr/bin/env python3
"""
Reset passwords for existing users
"""
import os
import sys
from dotenv import load_dotenv

# Set environment to development
os.environ.setdefault("ENVIRONMENT", "development")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app import models
from passlib.context import CryptContext

# Load environment variables
load_dotenv()

# Initialize password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def reset_passwords():
    # Create database engine and session
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///bugflow.db")
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create session
    db = SessionLocal()
    try:
        # Get all users
        users = db.query(models.User).all()
        
        if not users:
            print("No users found in database!")
            return
        
        # Reset password for each user to "password"
        for user in users:
            new_hash = pwd_context.hash("password")
            user.password_hash = new_hash
            db.add(user)
            print(f"Reset password for {user.email} (role: {user.role})")
        
        db.commit()
        print("\nAll passwords reset to 'password' successfully!")
        print("\nYou can now login with:")
        for user in users:
            print(f"- {user.email} / password")
        
    except Exception as e:
        print(f"Error resetting passwords: {e}")
        db.rollback()
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    reset_passwords()
