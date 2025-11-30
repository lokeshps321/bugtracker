"""
Configuration module for BugFlow application
Handles loading environment variables and providing configuration values
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to manage environment variables"""
    
    # JWT Configuration
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///bugflow.db")
    
    # Email Configuration
    SMTP_SERVER: Optional[str] = os.getenv("SMTP_SERVER")
    SMTP_PORT: Optional[int] = int(os.getenv("SMTP_PORT", "587")) if os.getenv("SMTP_PORT") else 587
    SMTP_USER: Optional[str] = os.getenv("SMTP_USER")
    SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD")
    
    # CORS Configuration
    ALLOWED_ORIGINS: list = ["http://localhost:5173", "http://localhost:5174", "http://localhost:8501"]
    if os.getenv("ALLOWED_ORIGINS"):
        ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS").split(",")
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration values are set"""
        # Check and set SECRET_KEY
        cls.SECRET_KEY = os.getenv("SECRET_KEY", "")
        if not cls.SECRET_KEY:
            # For development purposes only - in production, always set a proper SECRET_KEY
            if os.getenv("TESTING") or os.getenv("ENVIRONMENT") == "development":
                cls.SECRET_KEY = "dev-secret-key-for-development-purposes-only"
                print("WARNING: Using default development SECRET_KEY. Set SECRET_KEY environment variable for production!")
            else:
                raise ValueError("SECRET_KEY environment variable must be set")

        # Check if email configuration is complete
        if not all([cls.SMTP_SERVER, cls.SMTP_USER, cls.SMTP_PASSWORD]):
            print("WARNING: Email configuration is incomplete. Email notifications will not work.")
            print("Please set SMTP_SERVER, SMTP_USER, and SMTP_PASSWORD in your .env file.")