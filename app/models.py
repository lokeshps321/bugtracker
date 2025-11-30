from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from app.database import Base
import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)
    bugs_reported = relationship("Bug", back_populates="reporter", foreign_keys="Bug.reporter_id")
    bugs_assigned = relationship("Bug", back_populates="assigned_to", foreign_keys="Bug.assigned_to_id")

class Bug(Base):
    __tablename__ = "bugs"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)  # Bug title
    description = Column(Text, nullable=False)
    severity = Column(String)
    team = Column(String)
    status = Column(String, nullable=False)
    reporter_id = Column(Integer, ForeignKey("users.id"))
    assigned_to_id = Column(Integer, ForeignKey("users.id"))
    project = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_fake = Column(Boolean, default=False)
    reporter = relationship("User", back_populates="bugs_reported", foreign_keys=[reporter_id])
    assigned_to = relationship("User", back_populates="bugs_assigned", foreign_keys=[assigned_to_id])

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    bug_id = Column(Integer, ForeignKey("bugs.id"))
    correction_severity = Column(String)
    correction_team = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    message = Column(Text, nullable=False)
    recipient_team = Column(String)
    sender_email = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)