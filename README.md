#  BugFlow - AI-Powered Bug Tracking System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red?logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange?logo=pytorch)
![CodeBERT](https://img.shields.io/badge/Model-CodeBERT-purple)

**An intelligent bug tracking system with ML-powered severity classification, team assignment, and duplicate detection**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [ML Models](#-ml-models) • [API Documentation](#-api-documentation)

</div>

---

##  Features

### Core Functionality
| Feature | Description |
|---------|-------------|
| ** AI Severity Prediction** | Automatically classifies bugs as Low, Medium, High, or Critical (86.35% accuracy) |
| ** Smart Team Assignment** | Routes bugs to Backend, Frontend, Mobile, or DevOps teams (83.40% accuracy) |
| ** Duplicate Detection** | Identifies similar bugs using semantic similarity (85% threshold) |
| ** Email Notifications** | Sends alerts on ticket creation and status changes |
| ** MLOps Feedback Loop** | PM corrections trigger automatic model retraining |

### Role-Based Access
| Role | Capabilities |
|------|-------------|
| **Tester** | Report bugs, view predictions, track status |
| **Developer** | View assigned bugs, update status, add comments |
| **Project Manager** | Review all bugs, correct predictions, trigger retraining |

---

##  Quick Start

### Prerequisites
- Python 3.12+
- CUDA GPU (optional, for faster inference)

### One Command Startup
```bash
# Clone and run
git clone https://github.com/yourusername/bugflow.git
cd bugflow
./start.sh
```

### Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r frontend/requirements.txt

# Start backend (Terminal 1)
uvicorn app.main:app --reload --port 8000

# Start frontend (Terminal 2)
cd frontend && streamlit run app.py --server.port 8501
```

### Access Points
| Service | URL |
|---------|-----|
| **Frontend** | http://localhost:8501 |
| **Backend API** | http://localhost:8000 |
| **API Docs** | http://localhost:8000/docs |

### Demo Credentials
| Role | Username | Password |
|------|----------|----------|
| Tester | `tester1` | `test123` |
| Developer | `dev1` | `dev123` |
| Project Manager | `pm1` | `pm123` |

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BugFlow Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐        ┌──────────────┐        ┌──────────┐  │
│   │   Streamlit  │  HTTP  │   FastAPI    │  SQL   │  SQLite/ │  │
│   │   Frontend   │◄──────►│   Backend    │◄──────►│ PostgreSQL│  │
│   └──────────────┘        └──────────────┘        └──────────┘  │
│                                  │                               │
│                                  ▼                               │
│                     ┌────────────────────────┐                   │
│                     │     ML Pipeline        │                   │
│                     ├────────────────────────┤                   │
│                     │ ┌──────────────────┐   │                   │
│                     │ │   CodeBERT       │   │                   │
│                     │ │ Severity Model   │   │                   │
│                     │ │   (86.35%)       │   │                   │
│                     │ └──────────────────┘   │                   │
│                     │ ┌──────────────────┐   │                   │
│                     │ │   CodeBERT       │   │                   │
│                     │ │  Team Model      │   │                   │
│                     │ │   (83.40%)       │   │                   │
│                     │ └──────────────────┘   │                   │
│                     │ ┌──────────────────┐   │                   │
│                     │ │  MiniLM-L6-v2    │   │                   │
│                     │ │  Deduplication   │   │                   │
│                     │ └──────────────────┘   │                   │
│                     └────────────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure
```
bugflow/
├── app/                          # FastAPI Backend
│   ├── main.py                   # API endpoints
│   ├── models.py                 # SQLAlchemy models
│   ├── schemas.py                # Pydantic schemas
│   ├── auth.py                   # JWT authentication
│   ├── database.py               # Database connection
│   ├── ml_model.py               # ML integration
│   └── notifications.py          # Email service
├── frontend/
│   └── app.py                    # Streamlit UI
├── severity_model_specialized/   # Fine-tuned severity model
├── team_model_specialized/       # Fine-tuned team model
├── predict_bug.py                # ML prediction logic
├── start.sh                      # One-command startup
└── requirements.txt              # Dependencies
```

---

##  ML Models

### Why CodeBERT?

We evaluated several models for bug classification:

| Model | Pros | Cons | Decision |
|-------|------|------|----------|
| BERT | Well-established | Not optimized for code |  |
| GPT-3/4 | Powerful | Expensive, API dependency | |
| DistilBERT | Fast | Less accurate | |
| **CodeBERT** | Pre-trained on code + docs | Larger model size |  **Selected** |

**CodeBERT** (`microsoft/codebert-base`) is ideal because:
- Pre-trained on 6.4M+ code-documentation pairs
- Understands programming terminology and error messages
- 125M parameters, fine-tunes efficiently

### Training Approach

#### Techniques Used
| Technique | Purpose | Value |
|-----------|---------|-------|
| **Label Smoothing** | Prevent overconfidence | 0.1 |
| **Class Weighting** | Handle imbalanced data | balanced |
| **Early Stopping** | Prevent overfitting | patience=4 |
| **Mixed Precision** | Faster training | fp16 |

#### Training Configuration
```python
TrainingArguments(
    num_train_epochs=15,
    learning_rate=1.5e-5,
    warmup_ratio=0.15,
    weight_decay=0.02,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    fp16=True
)
```

### Model Performance

#### Severity Classification (86.35% accuracy)
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Low | 83.1% | 83.5% | 83.3% |
| Medium | 84.1% | 85.5% | 84.8% |
| High | 85.4% | 85.1% | 85.3% |
| **Critical** | **92.7%** | **91.2%** | **91.9%** |

#### Team Assignment (83.40% accuracy)
| Team | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Backend | 78.0% | 84.5% | 81.1% |
| Frontend | 82.8% | 84.3% | 83.5% |
| **Mobile** | **92.7%** | 83.8% | **88.1%** |
| DevOps | 86.5% | 79.0% | 82.6% |

### Prediction Examples

| Bug Description | Predicted Severity | Predicted Team |
|-----------------|-------------------|----------------|
| "Database timeout causing 500 errors" | HIGH (93%) | Backend (94%) |
| "iOS app crashes on launch" | CRITICAL (82%) | Mobile (96%) |
| "Docker container won't start in K8s" | CRITICAL (91%) | DevOps (96%) |
| "Button misaligned on mobile" | MEDIUM (92%) | Mobile (97%) |
| "Minor typo in footer" | LOW (82%) | Frontend (85%) |

---

##  API Documentation

### Authentication
```bash
# Login
POST /token
Content-Type: application/x-www-form-urlencoded
username=tester1&password=test123

# Response
{"access_token": "eyJ...", "token_type": "bearer"}
```

### Bug Operations

#### Create Bug
```bash
POST /bugs
Authorization: Bearer {token}
Content-Type: application/json

{
  "title": "Login page crash",
  "description": "App crashes when clicking login button",
  "project": "WebApp"
}

# Response includes ML predictions
{
  "id": 1,
  "predicted_severity": "high",
  "predicted_team": "Frontend",
  "severity_confidence": 0.89,
  "team_confidence": 0.92
}
```

#### Get Predictions
```bash
POST /predict
Content-Type: application/json

{"description": "Database connection timeout"}

# Response
{
  "severity": "high",
  "team": "Backend",
  "severity_confidence": 0.93,
  "team_confidence": 0.94
}
```

---

##  MLOps: Continuous Learning

BugFlow supports continuous model improvement through PM feedback:

```
1. Bug submitted → ML predicts severity/team
2. PM reviews prediction
3. If incorrect → PM provides correction
4. Corrections stored in feedback database
5. When feedback ≥ 50 → Automatic retraining triggered
6. New model deployed (hot reload)
```

---

## Cloud Deployment

### Render (Backend)
1. Connect GitHub repo
2. Set environment variables:
   - `DATABASE_URL`: PostgreSQL connection string
   - `SECRET_KEY`: Auto-generate
   - `USE_BASE_MODEL`: `true` (if models unavailable)

### Streamlit Cloud (Frontend)
1. Deploy from `frontend/app.py`
2. Set secret: `API_URL = https://your-backend.onrender.com`

---

##  Dataset

Training dataset: 9,820 bug descriptions
- Source: GitHub issues, Stack Overflow, synthetic data
- Split: 80% train, 10% validation, 10% test
- Balanced across severity levels and teams

---

##  Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | FastAPI, SQLAlchemy, Pydantic |
| **Frontend** | Streamlit, Plotly, Lottie |
| **ML/AI** | PyTorch, Transformers, CodeBERT |
| **Database** | SQLite (dev), PostgreSQL (prod) |
| **Auth** | JWT, bcrypt |
| **Dedup** | Sentence Transformers (MiniLM-L6-v2) |

---

##  License

MIT License - feel free to use for any purpose.

---

<div align="center">

**Built with ❤️ using FastAPI, Streamlit, and CodeBERT**

</div>
