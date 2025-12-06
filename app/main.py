# app/main.py - FastAPI Backend (Final Fix for reporter_id)

import os
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session
from datetime import timedelta, datetime
from typing import Optional, List
import threading
import time

from app.config import Config

# Create FastAPI app
app = FastAPI(title="BugFlow AI Backend")

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    """Create database tables and preload ML models"""
    from app.database import Base, engine
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created/verified")
    
    # Preload ML models for faster predictions
    print("🔄 Preloading ML models...")
    try:
        import predict_bug
        predict_bug.load_model_and_vectorizer()
        print("✅ ML models preloaded - predictions will be fast!")
    except Exception as e:
        print(f"⚠️ ML model preload failed: {e}")

# --- DEPENDENCIES ---

from . import models
from . import schemas
from .database import SessionLocal, engine
from .auth import create_access_token, get_current_user, authenticate_user
from .ml_model import (
    predict as predict_bug_attributes,
    check_duplicate as check_for_duplicate
)
from .notifications import send_email_notification_sync

# --- MOCK/PLACEHOLDER UTILITIES ---
ACCESS_TOKEN_EXPIRE_MINUTES = 30
# --- END MOCK/PLACEHOLDER UTILITIES ---


# Global retraining state tracker
retraining_in_progress = False
retraining_lock = threading.Lock()

app = FastAPI(title="BugFlow AI Backend")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:8501", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting configuration
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize database
models.Base.metadata.create_all(bind=engine)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- NEW REQUIRED PYDANTIC SCHEMA ---
class BugUpdateWithFeedback(schemas.BugUpdate):
    bug_id: int
    status: Optional[str] = None
    correction_severity: Optional[str] = None
    correction_team: Optional[str] = None

# ----------------------------------------------------
# 1. AUTHENTICATION & USER ROUTES
# ----------------------------------------------------

@app.post("/token")
@limiter.limit("5/minute")  # Limit to 5 attempts per minute per IP to prevent brute force
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Use the authenticate_user function which properly hashes and verifies passwords
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                          detail="Incorrect username or password",
                          headers={"WWW-Authenticate": "Bearer"})

    # Determine role based on user email
    if user.email == "pm1@example.com":
        role = "project_manager"
    elif "tester" in user.email:
        role = "tester"
    elif "dev" in user.email:
        role = "developer"
    else:
        role = "user"  # default role

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "role": role},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/current_user", response_model=schemas.User)
async def read_users_me(current_user: schemas.User = Depends(get_current_user)):
    return current_user


# ----------------------------------------------------
# 2. BUG REPORTING & PREDICTION
# ----------------------------------------------------

@app.post("/predict")
async def predict_bug(data: schemas.BugCreate):
    global retraining_in_progress

    # Wait if retraining is in progress to ensure we get the latest model
    wait_count = 0
    while retraining_in_progress and wait_count < 10:  # Wait up to 5 seconds
        time.sleep(0.5)
        wait_count += 1

    # Combine title and description for better prediction
    full_text = f"{data.title}. {data.description}" if data.title else data.description
    severity, team = predict_bug_attributes(full_text)
    return {"severity": severity, "team": team}


@app.post("/report_bug", status_code=status.HTTP_201_CREATED)
async def report_bug(
    data: schemas.BugCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    # Only testers can report bugs, developers and project managers have different roles
    if hasattr(current_user, 'role') and current_user.role == 'developer':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Developers are not authorized to report bugs"
        )
    # Look up the user in the database to get their ID
    db_user = db.query(models.User).filter(models.User.email == current_user.email).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found in database")

    # Combine title and description for better prediction
    full_text = f"{data.title}. {data.description}" if data.title else data.description
    severity, team = predict_bug_attributes(full_text)

    # check_for_duplicate returns a Bug object or None
    duplicate_bug_object = check_for_duplicate(data.description, db)

    if duplicate_bug_object:
        # We need the ID from the object to return to the user
        duplicate_bug_id = duplicate_bug_object.id
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Possible duplicate bug ID: {duplicate_bug_id}. Please review before reporting."
        )

    new_bug = models.Bug(
        title=data.title if data.title else data.description[:50] + "...",  # Use title or first 50 chars
        description=data.description,
        project=data.project,
        severity=severity,
        team=team,
        reporter_id=db_user.id,  # Use the ID from the database user
        status="open"
    )
    db.add(new_bug)
    db.commit()
    db.refresh(new_bug)

    # Simple test email
    test_subject = "BugFlow Test Email"
    test_body = "This is a test email from BugFlow. If you receive this, email sending is working!"

    try:
        # Test email to your account (from .env)
        background_tasks.add_task(
            send_email_notification_sync,
            recipient_email=os.getenv("SMTP_USER"),
            subject=test_subject,
            body=test_body
        )
        print(f"Test email sent to {os.getenv('SMTP_USER')}")
    except Exception as e:
        print(f"Failed to send test email: {str(e)}")
        raise

    return new_bug

# ----------------------------------------------------
# 3. CRITICAL: BUG UPDATE & FEEDBACK ROUTE (MODIFIED)
# ----------------------------------------------------

import subprocess
import threading

def retrain_models():
    """
    Function to retrain the ML models after receiving feedback
    """
    global retraining_in_progress

    try:
        # Set retraining in progress flag
        with retraining_lock:
            retraining_in_progress = True

        print("Starting model retraining...")

        # Verify that the Python interpreter exists
        python_path = "/home/lokesh/try/bugflow/venv/bin/python"
        if not os.path.exists(python_path):
            print(f"Python interpreter not found at {python_path}")
            # Try to find the current Python interpreter
            import sys
            python_path = sys.executable
            print(f"Using current Python interpreter: {python_path}")

        # Call the fine-tune script to retrain models with new feedback
        result = subprocess.run(
            [python_path, "fine_tune_bert.py"],
            cwd="/home/lokesh/try/bugflow",
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout (increased for larger training sets)
        )
        if result.returncode == 0:
            print("Model retraining completed successfully")
            # After retraining, force model reload in memory to use new models
            try:
                from app import ml_model
                # Force reset the model state to ensure reload
                ml_model.models_loaded = False  # Reset the loaded flag so models will be reloaded
                ml_model.load_model()  # Reload the models to get the new versions
                print("New models loaded into memory after retraining")

                # Also force a reload of the predict_bug models
                import predict_bug
                predict_bug.models_loaded = False
                predict_bug.load_model_and_vectorizer()
                print("Predict bug models also reloaded")

            except Exception as e:
                print(f"Error reloading models after retraining: {str(e)}")
        else:
            print(f"Model retraining failed: {result.stderr}")
            print(f"Command output: {result.stdout}")
            # Even if retraining fails, we should still try to reload existing models
            # to ensure the latest available model is loaded
            try:
                from app import ml_model
                ml_model.models_loaded = False
                ml_model.load_model()
                import predict_bug
                predict_bug.models_loaded = False
                predict_bug.load_model_and_vectorizer()
                print("Existing models reloaded despite retraining failure")
            except Exception as e:
                print(f"Error reloading existing models: {str(e)}")
    except subprocess.TimeoutExpired:
        print("Model retraining timed out")
        # Handle timeout by trying to reload anyway
        try:
            from app import ml_model
            ml_model.models_loaded = False
            ml_model.load_model()
            import predict_bug
            predict_bug.models_loaded = False
            predict_bug.load_model_and_vectorizer()
            print("Models reloaded after timeout")
        except Exception as e:
            print(f"Error reloading models after timeout: {str(e)}")
    except Exception as e:
        print(f"Error during model retraining: {str(e)}")
        # Provide more detailed error information for debugging dependency issues
        if "No module named" in str(e):
            print("Dependency error detected - attempting to reload existing models...")
        # Try to reload existing models even when there's an error
        try:
            from app import ml_model
            ml_model.models_loaded = False
            ml_model.load_model()
            import predict_bug
            predict_bug.models_loaded = False
            predict_bug.load_model_and_vectorizer()
            print("Existing models reloaded despite error")
        except Exception as reload_error:
            print(f"Error reloading models after retraining error: {str(reload_error)}")
    finally:
        # Reset retraining in progress flag
        with retraining_lock:
            retraining_in_progress = False
        print("Model retraining process finished")
        # Ensure that any subsequent predictions wait for the retraining to complete
        # by briefly checking and ensuring the model load timestamp is updated
        try:
            import predict_bug
            predict_bug.check_and_reload_models()
            print("Model synchronization completed after retraining")
        except Exception as e:
            print(f"Error during model synchronization: {str(e)}")

@app.post("/update_bug")
async def update_bug_and_feedback(
    bug_data: BugUpdateWithFeedback,
    background_tasks: BackgroundTasks,
    user: schemas.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    bug = db.query(models.Bug).filter(models.Bug.id == bug_data.bug_id).first()
    if not bug:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bug not found")

    if bug_data.status:
        bug.status = bug_data.status

    feedback_recorded = False

    # Check for Severity Correction
    if bug_data.correction_severity and bug_data.correction_severity != "No Change":
        # FIX: The Feedback model only supports correction_severity and correction_team
        new_feedback = models.Feedback(
            bug_id=bug.id,
            correction_severity=bug_data.correction_severity, # Directly assign the corrected value
            # correction_team is left as None, as only severity is being corrected here
        )
        db.add(new_feedback)
        bug.severity = bug_data.correction_severity
        feedback_recorded = True

    # Check for Team Correction
    if bug_data.correction_team and bug_data.correction_team != "No Change":
        # FIX: Create a separate Feedback record for the Team correction
        # We need to ensure we don't overwrite severity correction in the same record
        # Note: In a real app, you might merge the records if both are corrected,
        # but creating a separate record is safer for this schema.
        new_feedback = models.Feedback(
            bug_id=bug.id,
            correction_team=bug_data.correction_team, # Directly assign the corrected value
            # correction_severity is left as None, as only team is being corrected here
        )
        db.add(new_feedback)
        bug.team = bug_data.correction_team
        feedback_recorded = True

    db.commit()
    db.refresh(bug)

    # NEW: Trigger retraining immediately if feedback was recorded
    if feedback_recorded:
        # Get total feedback count to see if we have enough to retrain
        total_feedback = db.query(models.Feedback).count()
        if total_feedback >= 1:  # Changed from original threshold to 1
            print(f"Feedback recorded (total: {total_feedback}), triggering immediate retrain...")
            # Run retraining synchronously for critical feedback to ensure model updates quickly
            import threading
            retrain_thread = threading.Thread(target=retrain_models)
            retrain_thread.start()

            # Wait briefly to allow retraining to begin, but don't block the response
            # This helps ensure subsequent predictions get updated models faster
            import time
            time.sleep(0.5)  # Brief wait to allow retraining to start

    notification_message = f"Bug #{bug.id} updated to status '{bug.status}' by {user.email}."
    if feedback_recorded:
        notification_message += " **(AI Feedback Recorded and Model Retraining Triggered)**"

    background_tasks.add_task(
        send_email_notification_sync,
        recipient_email="pm_alerts@example.com",
        subject=f"BugFlow Action: Bug #{bug.id}",
        body=notification_message
    )

    return {"message": notification_message}


@app.get("/bugs", response_model=List[schemas.Bug])
async def get_all_bugs(db: Session = Depends(get_db)):
    return db.query(models.Bug).all()


@app.delete("/bugs/{bug_id}")
async def delete_bug(
    bug_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """
    Delete a bug. Testers can delete their own bugs.
    Project managers can delete any bug.
    """
    bug = db.query(models.Bug).filter(models.Bug.id == bug_id).first()
    if not bug:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bug not found")
    
    # Get the database user to check reporter_id
    db_user = db.query(models.User).filter(models.User.email == current_user.email).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found in database")
    
    # Check permissions: testers can only delete their own bugs, PM can delete any
    user_role = getattr(current_user, 'role', 'user')
    is_pm = user_role == 'project_manager' or current_user.email == 'pm1@example.com'
    is_owner = bug.reporter_id == db_user.id
    
    if not (is_pm or is_owner):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete bugs you reported"
        )
    
    # Delete associated feedback first (foreign key constraint)
    db.query(models.Feedback).filter(models.Feedback.bug_id == bug_id).delete()
    
    # Delete the bug
    db.delete(bug)
    db.commit()
    
    return {"message": f"Bug #{bug_id} deleted successfully"}

@app.post("/trigger_retrain")
async def trigger_retrain(
    background_tasks: BackgroundTasks,
    current_user: schemas.User = Depends(get_current_user)
):
    """
    Manual endpoint to trigger model retraining
    """
    # Check if user has appropriate role (PM or Admin)
    if hasattr(current_user, 'role') and current_user.role not in ['project_manager']:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only project managers can trigger retraining")

    # Trigger retraining in background
    background_tasks.add_task(retrain_models)
    return {"message": "Model retraining triggered successfully"}

# ----------------------------------------------------
# 4. NEW: MLOPS DATA ENDPOINTS
# ----------------------------------------------------

@app.get("/feedback_count")
async def get_feedback_count(db: Session = Depends(get_db)):
    try:
        total_feedback = db.query(models.Feedback).count()
        return {"total_feedback": total_feedback}
    except Exception as e:
        print(f"Database error fetching feedback count: {e}")
        return {"total_feedback": 0}

@app.get("/feedback_history")
async def get_feedback_history(db: Session = Depends(get_db)):
    try:
        # Fetch all feedback and aggregate by date in Python for DB compatibility
        feedbacks = db.query(models.Feedback).all()
        from collections import defaultdict
        counts = defaultdict(int)
        
        for f in feedbacks:
            if f.created_at:
                date_str = f.created_at.strftime("%Y-%m-%d")
                counts[date_str] += 1
        
        # Convert to list of dicts and sort by date
        result = [{"date": k, "count": v} for k, v in sorted(counts.items())]
        return result
    except Exception as e:
        print(f"Database error fetching feedback history: {e}")
        return []
    except Exception as e:
        print(f"Database error fetching feedback history: {e}")
        return []

# ----------------------------------------------------
# 5. NOTIFICATION ROUTE
# ----------------------------------------------------

@app.get("/notifications")
async def get_notifications(db: Session = Depends(get_db)):
    """Get real notifications from database, plus recent activity"""
    notifications = []
    
    # Get notifications from database
    db_notifications = db.query(models.Notification).order_by(
        models.Notification.created_at.desc()
    ).limit(20).all()
    
    for n in db_notifications:
        notifications.append({
            "message": n.message,
            "timestamp": n.created_at.strftime("%Y-%m-%d %H:%M") if n.created_at else "N/A"
        })
    
    # Add recent bug activity as notifications
    recent_bugs = db.query(models.Bug).order_by(
        models.Bug.updated_at.desc()
    ).limit(5).all()
    
    for bug in recent_bugs:
        notifications.append({
            "message": f"Bug #{bug.id}: {bug.title or 'Untitled'} - Status: {bug.status}",
            "timestamp": bug.updated_at.strftime("%Y-%m-%d %H:%M") if bug.updated_at else "N/A"
        })
    
    # Add recent feedback as notifications
    feedback_count = db.query(models.Feedback).count()
    if feedback_count > 0:
        notifications.insert(0, {
            "message": f"MLOps: {feedback_count} expert corrections collected for model improvement",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
    
    # Sort by timestamp (recent first)
    return notifications if notifications else [{"message": "No notifications yet", "timestamp": ""}]

# ----------------------------------------------------
# 6. ROOT ROUTE
# ----------------------------------------------------

@app.get("/")
async def root():
    return {"message": "BugFlow Backend is running."}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring services"""
    return {"status": "healthy", "service": "bugflow-backend"}

@app.get("/admin/init-db")
async def initialize_database(db: Session = Depends(get_db)):
    """Initialize database with demo users - ONE TIME USE ONLY"""
    from passlib.context import CryptContext
    from app.models import User  # Import User model
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    # Check if users already exist
    existing = db.query(User).first()
    if existing:
        return {"message": "Database already initialized", "status": "skipped"}
    
    # Create demo users
    users_data = [
        {"email": "pm1@example.com", "password": "password", "role": "pm"},
        {"email": "tester1@example.com", "password": "password", "role": "tester"},
        {"email": "dev1@example.com", "password": "password", "role": "developer"},
    ]
    
    for user_data in users_data:
        hashed_password = pwd_context.hash(user_data["password"])
        user = User(
            email=user_data["email"],
            password_hash=hashed_password,
            role=user_data["role"]
        )
        db.add(user)
    
    db.commit()
    return {"message": "Database initialized with 3 demo users", "status": "success"}
