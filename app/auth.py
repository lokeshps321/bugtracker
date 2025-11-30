import os
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from app import models, schemas, database
from app.config import Config
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from typing import Optional # <-- ADDED: Needed for Optional[timedelta]

# Load configuration
# For development/testing, we don't validate the config during import
if os.getenv("TESTING") or os.getenv("ENVIRONMENT") == "development":
    # For testing/development purposes
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-for-development-purposes-only")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
else:
    # For production, validate the config
    Config.validate_config()
    SECRET_KEY = Config.SECRET_KEY
    ALGORITHM = Config.ALGORITHM
    ACCESS_TOKEN_EXPIRE_MINUTES = Config.ACCESS_TOKEN_EXPIRE_MINUTES

# Initialize password hashing context with appropriate backend
# Use bcrypt as primary but with explicit backend specification to avoid detection issues during testing
pwd_context = CryptContext(
    schemes=["bcrypt", "pbkdf2_sha256"],
    deprecated="auto"
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password_and_maybe_migrate(db: Session, user: models.User, plain_password: str):
    """
    Verify the password and migrate to preferred scheme if needed.
    Returns True if verified, False otherwise.
    """
    hashed = user.password_hash
    try:
        verified = pwd_context.verify(plain_password, hashed)
    except Exception:
        # UnknownHashError or invalid format
        return False

    # Optional: auto-migrate to preferred scheme (bcrypt)
    if verified and pwd_context.needs_update(hashed):
        user.password_hash = pwd_context.hash(plain_password)
        db.add(user)
        db.commit()
        db.refresh(user)
    return verified


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def authenticate_user(db: Session, email: str, password: str):
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        return False
    if not verify_password_and_maybe_migrate(db, user, password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None): # <-- FIX: Added expires_delta
    to_encode = data.copy()
    
    # Calculate expiration time using the provided delta or the default minutes
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(database.get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        role: str = payload.get("role")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise credentials_exception
    # Add role to the user object for access in route handlers
    user.role = role
    return user


def require_role(required_role: str):
    """
    Dependency to check if user has a specific role
    """
    async def role_checker(current_user: models.User = Depends(get_current_user)):
        if not hasattr(current_user, 'role') or current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: {required_role} role required"
            )
        return current_user
    return role_checker


def require_any_role(*required_roles: str):
    """
    Dependency to check if user has any of the specified roles
    """
    async def role_checker(current_user: models.User = Depends(get_current_user)):
        if not hasattr(current_user, 'role') or current_user.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: requires one of roles: {', '.join(required_roles)}"
            )
        return current_user
    return role_checker
