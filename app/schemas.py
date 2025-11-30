from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import re

class UserBase(BaseModel):
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$', max_length=255)
    role: str = Field(..., min_length=1, max_length=50)

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128)

class User(UserBase):
    id: int
    class Config:
        orm_mode = True

class BugDescription(BaseModel):
    description: str

class Prediction(BaseModel):
    severity: str
    team: str

class BugCreate(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    description: str = Field(..., min_length=1, max_length=10000)
    project: str = Field(..., min_length=1, max_length=100)
    assigned_to_id: Optional[int] = Field(None, ge=1)

class Bug(BaseModel):
    id: int
    title: Optional[str]
    description: str
    severity: Optional[str]
    team: Optional[str]
    status: str
    reporter_id: int
    assigned_to_id: Optional[int]
    project: str
    created_at: datetime
    updated_at: datetime
    is_fake: bool
    class Config:
        orm_mode = True

class BugUpdate(BaseModel):
    bug_id: int = Field(..., ge=1)
    status: Optional[str] = Field(None, max_length=20)
    assigned_to_id: Optional[int] = Field(None, ge=1)
    severity: Optional[str] = Field(None, max_length=20)
    team: Optional[str] = Field(None, max_length=50)

class FeedbackCreate(BaseModel):
    bug_id: int
    correction_severity: Optional[str] = None
    correction_team: Optional[str] = None

class Feedback(FeedbackCreate):
    id: int
    created_at: datetime
    class Config:
        orm_mode = True

class Notification(BaseModel):
    id: int
    message: str
    recipient_team: Optional[str]
    sender_email: Optional[str]
    created_at: datetime
    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str