# 📁 BugFlow - File Guide

*A complete guide to every file in the project for learning and presentation*

---

## 📂 Project Structure Overview

```
bugflow/
├── 🚀 start.sh                    # Single startup script
├── 📖 README.md                   # Quick start guide
├── 📖 PROJECT_DOCUMENTATION.md    # Complete project documentation
├── 📖 FILE_GUIDE.md               # This file - explains all files
├── 📦 requirements.txt            # Python dependencies
├── 🔧 Procfile                    # Render deployment config
├── 🔧 render.yaml                 # Render blueprint
├── 🔧 runtime.txt                 # Python version for cloud
├── 🤖 predict_bug.py              # ML prediction logic
│
├── app/                           # Backend (FastAPI)
│   ├── main.py                    # API endpoints
│   ├── models.py                  # Database models
│   ├── schemas.py                 # Request/Response schemas
│   ├── auth.py                    # Authentication (JWT)
│   ├── database.py                # Database connection
│   ├── ml_model.py                # ML model integration
│   ├── notifications.py           # Email notifications
│   └── config.py                  # Configuration settings
│
├── frontend/                      # Frontend (Streamlit)
│   └── app.py                     # Complete UI application
│
├── severity_model_specialized/    # Fine-tuned severity model
└── team_model_specialized/        # Fine-tuned team model
```

---

## 🚀 Startup & Configuration Files

### `start.sh`
**Purpose**: One command to start everything

```bash
./start.sh  # Starts backend + frontend
```

**What it does**:
1. Activates Python virtual environment
2. Installs dependencies (first run only)
3. Starts FastAPI backend on port 8000
4. Starts Streamlit frontend on port 8501
5. Shows demo credentials

---

### `requirements.txt`
**Purpose**: Python package dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework for API |
| `uvicorn` | ASGI server |
| `sqlalchemy` | Database ORM |
| `transformers` | HuggingFace ML models |
| `torch` | PyTorch for ML |
| `sentence-transformers` | Deduplication model |
| `streamlit` | Frontend framework |
| `pydantic` | Data validation |

---

### `Procfile` & `render.yaml`
**Purpose**: Cloud deployment configuration (Render.com)

```
web: python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

---

## 🖥️ Backend Files (`app/` folder)

### `app/main.py` ⭐ MOST IMPORTANT
**Purpose**: All API endpoints (the heart of the backend)

**Key Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/token` | POST | User login, returns JWT |
| `/predict` | POST | ML prediction only |
| `/report_bug` | POST | Create new bug |
| `/bugs` | GET | List all bugs |
| `/update_bug` | POST | Update bug status/feedback |
| `/notifications` | GET | Get real-time notifications |
| `/feedback_count` | GET | MLOps feedback count |

**Key Features**:
- Model preloading on startup (fast predictions)
- PM correction triggers MLOps retraining
- Duplicate detection on bug creation

---

### `app/models.py`
**Purpose**: Database table definitions (SQLAlchemy ORM)

**Tables**:

```python
class User:        # id, email, password_hash, role
class Bug:         # id, title, description, severity, team, status
class Feedback:    # id, bug_id, correction_severity, correction_team
class Notification: # id, message, recipient_team, created_at
```

**Relationships**:
- User → Bugs (reporter_id, assigned_to_id)
- Bug → Feedback (corrections for MLOps)

---

### `app/schemas.py`
**Purpose**: Pydantic models for request/response validation

**Key Schemas**:

```python
class BugCreate:   # Input for creating bugs
class BugResponse: # Output bug data
class UserLogin:   # Login credentials
class Token:       # JWT token response
```

**Why Pydantic?**: Automatic validation, serialization, and OpenAPI docs

---

### `app/auth.py`
**Purpose**: JWT authentication and password hashing

**Functions**:

| Function | Purpose |
|----------|---------|
| `create_access_token()` | Generate JWT token |
| `get_current_user()` | Decode JWT, return user |
| `authenticate_user()` | Verify password |
| `hash_password()` | Bcrypt password hashing |

**Security**: Uses HS256 algorithm, 30-minute token expiry

---

### `app/database.py`
**Purpose**: Database connection setup

**Local**: SQLite (`bugflow.db`)
**Cloud**: PostgreSQL (via `DATABASE_URL` env var)

```python
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

---

### `app/ml_model.py`
**Purpose**: ML model integration for the backend

**Functions**:

| Function | Purpose |
|----------|---------|
| `load_prediction_model()` | Load severity/team models |
| `load_dedup_model()` | Load deduplication model |
| `predict()` | Make predictions |
| `check_for_duplicate()` | Find similar bugs |

---

### `app/notifications.py`
**Purpose**: Email notification system

**Trigger**: When bug status changes or PM makes correction

```python
def send_email_notification_sync(to_email, subject, body):
    # Uses Gmail SMTP
```

---

### `app/config.py`
**Purpose**: Configuration from environment variables

```python
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./bugflow.db")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
```

---

## 🎨 Frontend Files (`frontend/` folder)

### `frontend/app.py` ⭐ MOST IMPORTANT
**Purpose**: Complete Streamlit UI (1300+ lines)

**Structure**:

| Section | Lines | Purpose |
|---------|-------|---------|
| CSS Styling | 1-250 | Dark theme, glassmorphism |
| Helper Functions | 250-400 | API calls, cards, badges |
| Tester Pages | 470-620 | Report bugs, view status |
| Developer Pages | 620-670 | View assigned bugs, update status |
| PM Pages | 670-1050 | Dashboard, Kanban, Analytics, Corrections |
| Main Router | 1100-1332 | Login, role-based navigation |

**Key Features**:
- Role-based UI (Tester, Developer, PM)
- Real-time predictions with loading animation
- MLOps feedback for PM corrections
- Interactive charts (Plotly)

---

## 🤖 ML Files

### `predict_bug.py` ⭐ CORE ML LOGIC
**Purpose**: All ML prediction logic

**Models Used**:

| Model | Purpose | Accuracy |
|-------|---------|----------|
| `microsoft/codebert-base` | Severity classification | 86.35% |
| `microsoft/codebert-base` | Team classification | 83.40% |
| `all-MiniLM-L6-v2` | Deduplication | 85% threshold |

**Key Functions**:

```python
def load_model_and_vectorizer():
    """Load CodeBERT models for severity and team"""
    
def predict(text):
    """Returns (severity, team) for given bug description"""
    
def predict_with_confidence(text):
    """Returns predictions with confidence scores"""
```

**Flow**:
```
Bug Description → Tokenizer → CodeBERT → Softmax → Prediction
```

---

### `severity_model_specialized/` & `team_model_specialized/`
**Purpose**: Fine-tuned CodeBERT models

**Contents**:

| File | Purpose |
|------|---------|
| `config.json` | Model architecture config |
| `model.safetensors` | Trained weights (~500MB) |
| `tokenizer_config.json` | Tokenizer settings |
| `metrics.json` | Training accuracy results |

**Fine-tuning Details**:
- Base: `microsoft/codebert-base`
- Dataset: 9,820 bug descriptions
- Techniques: Label smoothing, class weighting, early stopping

---

## 📖 Documentation Files

### `README.md`
**Audience**: Quick start for new users
**Contents**: Features, installation, credentials

### `PROJECT_DOCUMENTATION.md`
**Audience**: Deep technical understanding
**Contents**: Architecture, ML approach, training details

### `FILE_GUIDE.md` (This file)
**Audience**: Learning & presentation
**Contents**: Every file explained

---

## 🔑 Key Concepts for Presentation

### 1. **Architecture**
> "We use a microservices approach with FastAPI backend and Streamlit frontend, connected via REST API."

### 2. **ML Pipeline**
> "Bug descriptions are tokenized by RobertaTokenizer, processed by fine-tuned CodeBERT models, and classified into severity levels and teams."

### 3. **Why CodeBERT?**
> "CodeBERT is pre-trained on 6.4 million code-documentation pairs, making it ideal for understanding technical bug descriptions."

### 4. **MLOps**
> "Project Managers can correct predictions. After 50 corrections, the system triggers automatic model retraining."

### 5. **Deduplication**
> "We use Sentence Transformers to create embeddings and cosine similarity to detect duplicates (threshold: 0.85)."

---

## 🎯 Quick Reference for Demo

```bash
# Start everything
./start.sh

# Access
Frontend: http://localhost:8501
Backend:  http://localhost:8000
API Docs: http://localhost:8000/docs

# Demo Login
Tester:    tester1 / test123
Developer: dev1 / dev123
PM:        pm1 / pm123
```

---

*Created for BugFlow Project Presentation*
