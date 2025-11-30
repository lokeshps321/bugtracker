# BugFlow 🐛⚡

**Intelligent Bug Tracking System with AI/ML & MLOps**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)

> Enterprise-grade bug tracking with state-of-the-art AI that achieves **96.7% severity** and **95.3% team assignment** accuracy.

---

## 🎯 Overview

BugFlow is a production-ready bug tracking system enhanced with **fine-tuned DistilBERT models** that automatically:
- **Classifies bug severity** (low/medium/high/critical) - 96.7% accuracy
- **Assigns to teams** (Backend/Frontend/Mobile/DevOps) - 95.3% accuracy  
- **Detects duplicates** using semantic similarity
- **Learns from corrections** through MLOps feedback loop

### Key Features
- ✅ **Real-time AI predictions** (<100ms latency)
- ✅ **MLOps continuous learning** (PM corrections improve models)
- ✅ **Role-based access** (Tester/Developer/Project Manager)
- ✅ **Modern UI** with Streamlit + interactive Plotly charts
- ✅ **Production-ready** FastAPI backend with JWT auth

---

## 📊 Performance Metrics

| Model | Accuracy | Training Data | Training Time |
|-------|----------|---------------|---------------|
| **Severity Classification** | 96.7% | 10K samples | 15 min (CPU) |
| **Team Assignment** | 95.3% | 50K samples | 90 min (GPU) |
| **Deduplication** | TBD | 20K triplets | 5 min (GPU) |

**Per-Team Performance:**
- Backend: 97.1% F1-score
- Frontend: 94.8% F1-score
- Mobile: 93.5% F1-score
- DevOps: 88.3% F1-score

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     BugFlow Stack                        │
├─────────────────────────────────────────────────────────┤
│  Frontend: Streamlit 1.41.1                             │
│  Backend:  FastAPI 0.115.6 + SQLAlchemy 2.0.36          │
│  Database: SQLite (PostgreSQL-ready)                    │
│  ML:       PyTorch 2.5.1 + DistilBERT + Transformers    │
│  GPU:      NVIDIA CUDA 12.1 support                     │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- NVIDIA GPU (optional, for faster training)
- 8GB RAM minimum

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/lokeshps321/bugflow.git
cd bugflow

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize database
python init_users.py

# 5. Start backend (Terminal 1)
uvicorn app.main:app --reload --port 8000

# 6. Start frontend (Terminal 2)
streamlit run frontend/app.py
```

### Access
- **Frontend:** http://localhost:8501
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### Demo Accounts
- **Tester:** tester@bugflow.com / tester123
- **Developer:** dev@bugflow.com / dev123
- **PM:** pm@bugflow.com / pm123

---

## 📁 Project Structure

```
bugflow/
├── app/                        # Backend (FastAPI)
│   ├── main.py                # API routes
│   ├── models.py              # Database models
│   ├── auth.py                # JWT authentication
│   ├── ml_model.py            # ML prediction service
│   └── config.py              # Configuration
├── frontend/                   # Frontend (Streamlit)
│   └── app.py                 # Main UI
├── datasets/                   # Training data
│   └── preprocessed/          # Cleaned datasets
├── severity_model_new/         # Fine-tuned severity classifier
├── team_model_90plus/          # Fine-tuned team classifier
├── dedup_model/               # Deduplication embeddings
├── train_*.py                 # Training scripts
├── preprocess_dataset.py      # Data preprocessing
├── test_mlops.py              # MLOps testing
├── BUGFLOW_PRD.md             # Complete product specification
└── requirements.txt           # Python dependencies
```

---

## 🤖 ML Pipeline

### Dataset
- **Source:** 5.3 million real GitHub bug reports
- **Processed:** 99,757 cleaned and labeled samples
- **Split:** 70% train / 15% validation / 15% test

### Models

#### 1. Severity Classifier (DistilBERT)
```python
# Fine-tuned on 10K samples
# Classes: low, medium, high, critical
# Accuracy: 96.7%
python train_severity_fast.py
```

#### 2. Team Assignment (DistilBERT)
```python
# Fine-tuned on 50K samples with class weighting
# Classes: Backend, Frontend, Mobile, DevOps
# Accuracy: 95.3%
python train_team_90plus.py
```

#### 3. Deduplication (Sentence-Transformer)
```python
# Triplet loss training
# Similarity threshold: 0.85
python train_deduplication.py
```

---

## 🔄 MLOps Workflow

```
1. Tester reports bug → AI predicts severity + team
2. Bug saved to database
3. PM reviews prediction
4. PM corrects if wrong ✏️
5. Correction saved as feedback
6. System tracks correction count
7. At 50 corrections → Retrain models 🔄
8. Deploy improved model
9. Future predictions are better ✨
```

**Test MLOps:**
```bash
python test_mlops.py
```

---

## 🎨 Features

### For Testers
- Report bugs with free-text descriptions
- AI auto-classifies severity and assigns team
- Duplicate detection before submission
- Track bug status

### For Developers
- View assigned bugs
- Update status (open → in_progress → resolved)
- Quick status transitions

### For Project Managers
- **Dashboard:** Overall stats, severity distribution
- **Kanban Board:** Visual bug flow by status
- **Analytics:** Interactive Plotly charts
- **Correct AI:** One-click prediction corrections
- **MLOps:** Track model improvements

---

## 📈 API Endpoints

### Authentication
```http
POST /token
Body: { "username": "user@email.com", "password": "pass" }
Response: { "access_token": "jwt_token" }
```

### Prediction
```http
POST /predict
Headers: Authorization: Bearer {token}
Body: { "description": "Login button crashes", "project": "WebApp" }
Response: { "severity": "high", "team": "Frontend" }
```

### Report Bug
```http
POST /report_bug
Body: {
    "description": "...",
    "severity": "high",
    "team": "Backend"
}
```

### Update Bug (with MLOps)
```http
POST /update_bug
Body: {
    "bug_id": 123,
    "correction_severity": "critical",
    "correction_team": "DevOps"
}
```

---

## 🔧 Advanced Usage

### Retrain Models

```bash
# Severity model
python train_severity_model.py \
    --samples 50000 \
    --epochs 5 \
    --batch-size 32

# Team model
python train_team_90plus.py \
    --use-gpu \
    --class-weights

# Deduplication
python train_deduplication.py
```

### GPU Setup

```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 📚 Documentation

- **[BUGFLOW_PRD.md](BUGFLOW_PRD.md)** - Complete product specification with:
  - System architecture
  - ML pipeline details
  - Fine-tuning methodology
  - MLOps implementation
  - API specifications
  - Deployment guide

---

## 🛠️ Technology Stack

**Backend:**
- FastAPI 0.115.6
- SQLAlchemy 2.0.36
- PyTorch 2.5.1
- Transformers 4.47.1
- Sentence-Transformers

**Frontend:**
- Streamlit 1.41.1
- Plotly 5.24.1
- Pandas 2.2.3

**ML/AI:**
- DistilBERT (distilbert-base-uncased)
- Sentence-Transformers (all-MiniLM-L6-v2)
- CUDA 12.1

---

## 🚀 Deployment

### Docker (Recommended)

```dockerfile
# Backend
docker build -t bugflow-backend .
docker run -p 8000:8000 --gpus all bugflow-backend

# Frontend
docker build -t bugflow-frontend -f Dockerfile.frontend .
docker run -p 8501:8501 bugflow-frontend
```

### Production Checklist

- [ ] Migrate to PostgreSQL
- [ ] Set up HTTPS/SSL
- [ ] Configure CORS properly
- [ ] Enable rate limiting
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure automated backups
- [ ] Set up CI/CD pipeline
- [ ] Deploy ML models to cloud storage

---

## 🎯 Future Roadmap

**Short-term (3 months):**
- Automated model retraining (Airflow/Celery)
- A/B testing for model versions
- Email notifications
- Comprehensive unit tests

**Mid-term (6 months):**
- Multi-language support
- Slack/Teams integration
- Advanced analytics
- Mobile app (React Native)

**Long-term (1 year):**
- LLM integration (GPT-4)
- Automated bug fixing suggestions
- Multi-tenant SaaS
- Explainable AI

---

## 🤝 Contributing

This is a private repository. For collaboration requests, contact the maintainer.

---

## 📄 License

Private - All Rights Reserved

---

## 👤 Author

**Lokesh**
- GitHub: [@lokeshps321](https://github.com/lokeshps321)

---

## 🙏 Acknowledgments

- **Dataset:** GitHub Issues (5.3M bug reports)
- **Base Models:** HuggingFace Transformers
- **UI Framework:** Streamlit
- **Inspiration:** Modern MLOps practices

---

## 📞 Support

For issues or questions:
1. Check [BUGFLOW_PRD.md](BUGFLOW_PRD.md) for detailed documentation
2. Review API docs at http://localhost:8000/docs
3. Contact repository owner

---

**Built with ❤️ and fine-tuned AI**
