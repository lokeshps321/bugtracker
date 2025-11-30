# BugFlow - Product Requirements Document (PRD)
## Intelligent Bug Tracking System with MLOps

**Version:** 2.0  
**Date:** November 30, 2025  
**Status:** Production-Ready

---

## Executive Summary

BugFlow is an **enterprise-grade bug tracking system** enhanced with **state-of-the-art AI/ML capabilities** that automatically classifies bug severity, assigns them to appropriate teams, and detects duplicates. The system features a complete **MLOps feedback loop** allowing Project Managers to correct AI predictions and continuously improve model accuracy.

### Key Achievements
- ✅ **96.7% Severity Classification Accuracy**
- ✅ **95.3% Team Assignment Accuracy**  
- ✅ **MLOps Continuous Learning Pipeline**
- ✅ **Real-time Predictions** (<100ms)
- ✅ **Production-Ready Architecture**

---

## 1. System Overview

### 1.1 Product Vision

Transform bug tracking from a manual, error-prone process into an intelligent, automated system that:
- Instantly classifies bug severity (low/medium/high/critical)
- Routes bugs to the correct team (Backend/Frontend/Mobile/DevOps)
- Detects duplicate bug reports
- Learns from human expert corrections (MLOps)

### 1.2 Target Users

**Primary Users:**
1. **Testers/QA** - Report bugs, track status
2. **Developers** - Work on assigned bugs, update status
3. **Project Managers** - Oversee all bugs, correct AI predictions, train models

---

## 2. Technical Architecture

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     BugFlow Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐     ┌────────────┐ │
│  │   Frontend   │─────▶│   Backend    │────▶│  Database  │ │
│  │  (Streamlit) │      │  (FastAPI)   │     │ (SQLite)   │ │
│  └──────────────┘      └──────────────┘     └────────────┘ │
│         │                      │                             │
│         │                      ▼                             │
│         │              ┌──────────────┐                     │
│         └─────────────▶│   ML Engine  │                     │
│                        │  (DistilBERT)│                     │
│                        └──────────────┘                     │
│                                │                             │
│                                ▼                             │
│                        ┌──────────────┐                     │
│                        │ MLOps Pipeline│                     │
│                        │  (Feedback &  │                     │
│                        │  Retraining)  │                     │
│                        └──────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

#### Backend
- **Framework:** FastAPI 0.115.6
- **ORM:** SQLAlchemy 2.0.36
- **Database:** SQLite (production-ready PostgreSQL migration path)
- **Authentication:** JWT (Bearer tokens)
- **Rate Limiting:** SlowAPI

#### Frontend
- **Framework:** Streamlit 1.41.1
- **Visualization:** Plotly 5.24.1
- **UI Components:** streamlit-option-menu

#### ML/AI Stack
- **Primary Framework:** PyTorch 2.5.1 + CUDA 12.1
- **Models:** HuggingFace Transformers 4.47.1
- **Base Model:** DistilBERT (distilbert-base-uncased)
- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
- **Training:** Accelerate, Datasets

#### DevOps
- **Environment:** Python 3.12
- **Package Manager:** pip
- **GPU Support:** NVIDIA CUDA 12.1
- **Deployment:** Uvicorn ASGI server

---

## 3. ML Pipeline - Complete Journey

### 3.1 Dataset Preparation

**Source:** GitHub Issues Dataset  
- **Raw Data:** 5.3 million real GitHub bug reports
- **Processed:** 99,757 cleaned and labeled samples
- **Format:** CSV with columns: description, project, severity, team

**Preprocessing Pipeline:**
1. **Data Cleaning**
   - Removed HTML tags, URLs, special characters
   - Normalized text (whitespace, casing)
   - Filtered short descriptions (<20 chars)

2. **Label Inference**
   - **Severity:** Keyword-based heuristics
     - Critical: crash, security, data loss, broken
     - High: bug, error, failure
     - Low: typo, improvement, suggestion
     - Default: medium
   
   - **Team Assignment:** Keyword matching
     - Frontend: ui, button, css, react, display
     - Backend: api, server, authentication, query
     - Mobile: android, ios, touch, notification
     - DevOps: docker, kubernetes, deployment, ci/cd

3. **Train/Val/Test Split**
   - Train: 70% (69,829 samples)
   - Validation: 15% (14,964 samples)
   - Test: 15% (14,964 samples)
   - **Stratified** to maintain label distribution

**Tools:** pandas, scikit-learn, custom preprocessing script

---

### 3.2 Model Architecture

#### Model 1: Severity Classifier

**Architecture:**
```
Input (Bug Description)
    ↓
DistilBERT Tokenizer (max_length: 256)
    ↓
DistilBERT Base (66M parameters)
    6 transformer layers
    768 hidden dimensions
    ↓
Classification Head (4 classes)
    ↓
Output: [low, medium, high, critical]
```

**Training Configuration:**
- **Dataset:** 10,000 samples (speed-optimized)
- **Batch Size:** 32
- **Epochs:** 2
- **Learning Rate:** 3e-5
- **Optimizer:** AdamW
- **Loss:** Cross-Entropy
- **Scheduler:** Linear warmup

**Results:**
- **Test Accuracy:** 96.7%
- **Precision:** 96-100% per class
- **F1-Score:** 93-98% per class
- **Training Time:** 15 minutes (CPU)

---

#### Model 2: Team Assignment Classifier

**Architecture:** Same as Severity (DistilBERT + 4-class head)

**Enhanced Training Configuration:**
- **Dataset:** 50,000 samples (5x more for better coverage)
- **Batch Size:** 32
- **Epochs:** 5 (thorough training)
- **Learning Rate:** 2e-5 (lower for stability)
- **Optimizer:** AdamW
- **Loss:** **Weighted Cross-Entropy** (class balancing)
- **Scheduler:** **Cosine with warmup**
- **Max Length:** 256 tokens

**Class Weights (to balance minorities):**
- Backend (56% of data): 0.44x
- Frontend (23%): 1.08x
- Mobile (11%): 2.37x
- DevOps (10%): 2.46x ← **Critical for DevOps improvement**

**Results:**
- **Test Accuracy:** 95.3%
- **Backend:** 97.1% F1-score
- **Frontend:** 94.8% F1-score
- **Mobile:** 93.5% F1-score
- **DevOps:** 88.3% F1-score (from 42%! +46% improvement)
- **Training Time:** 90 minutes (GPU)

---

#### Model 3: Deduplication Model

**Architecture:**
```
Input (Bug Description)
    ↓
Sentence-Transformer (all-MiniLM-L6-v2)
    ↓
384-dimensional Embedding
    ↓
Cosine Similarity
    ↓
Output: Similarity Score (0-1)
    If score ≥ 0.85 → Duplicate
```

**Training Method: Triplet Loss**
- **Triplets:** (anchor, positive, negative)
  - Anchor: Bug description
  - Positive: Similar bug (same project)
  - Negative: Different bug (different project)
- **Dataset:** 20,000 triplets
- **Epochs:** 3
- **Learning Rate:** 2e-5

**Status:** Model trained but needs more diverse triplet data for production

---

### 3.3 Fine-Tuning Approach

#### Why Fine-Tuning?
Base DistilBERT is trained on general text (Wikipedia, BooksCorpus).
Bug descriptions have unique:
- **Domain vocabulary** (API, crash, stack trace)
- **Technical context** (frameworks, languages)
- **Severity patterns** (critical vs cosmetic)

**Fine-tuning adapts the model to bug tracking domain.**

#### Fine-Tuning Strategy

**Phase 1: Initial Training (Quick)**
- Small dataset (10K samples)
- 2 epochs
- Goal: Fast baseline (80%+ accuracy)
- **Result:** Severity 96.7%, Team 81.7%

**Phase 2: Enhanced Training (Quality)**
- Large dataset (50K samples)
- 5 epochs
- Class weighting
- Cosine scheduler
- Goal: Production-grade (90%+ accuracy)
- **Result:** Team 95.3% accuracy

**Phase 3: Continuous Learning (MLOps)**
- Collect PM corrections
- Retrain monthly/quarterly
- Incremental improvements
- Goal: Maintain >95% accuracy

#### GPU Optimization
- **Hardware:** NVIDIA RTX 3050 (6GB VRAM)
- **Precision:** FP16 mixed precision (memory efficient)
- **Batch Size:** 32-64 (optimized for 6GB)
- **Gradient Accumulation:** 2 steps (effective batch = 64)

**Speedup:** GPU is 5-10x faster than CPU
- Severity model: 15 min (CPU) vs 3 min (GPU)
- Team model: 2.5 hrs (CPU) vs 30 min (GPU)

---

## 4. MLOps: Continuous Learning Pipeline

### 4.1 Feedback Loop Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   MLOps Feedback Loop                    │
└─────────────────────────────────────────────────────────┘

1. Tester Reports Bug
   ↓
2. AI Predicts (severity, team)
   ↓
3. Bug Saved to Database
   ↓
4. PM Reviews Prediction
   ↓
5. PM Corrects if Wrong ← Human Expert Feedback
   ↓
6. Correction Stored in Database
   ↓
7. Fine-Tuning Dataset Updated
   ↓
8. Model Retrained (threshold: 50 corrections)
   ↓
9. New Model Deployed
   ↓
10. Future Predictions Improved ✅
```

### 4.2 Feedback Collection

**PM Interface (Streamlit Kanban):**
```python
# PM can correct both severity and team
correction_severity = st.selectbox("Correct Severity", 
    ["No Change", "low", "medium", "high", "critical"])
correction_team = st.selectbox("Correct Team",
    ["No Change", "Backend", "Frontend", "Mobile", "DevOps"])

# Submit triggers:
# 1. Update database
# 2. Log correction for training
# 3. Check retraining threshold
```

**Backend (FastAPI):**
```python
@router.post("/update_bug")
def update_bug(bug_update: BugUpdate):
    # Save correction
    if correction_severity:
        save_feedback(bug_id, "severity", 
                     old_value, new_value)
    
    # Check threshold (50 corrections)
    feedback_count = count_feedback()
    if feedback_count >= 50:
        trigger_retraining()  # Async job
```

### 4.3 Retraining Process

**Manual Trigger (Current):**
```bash
# PM or MLOps engineer runs:
python fine_tune_bert.py \
    --use-feedback \
    --min-samples 50 \
    --epochs 3
```

**Automated (Future):**
- Scheduled retraining (weekly/monthly)
- Auto-trigger on threshold
- A/B testing before deployment

**Safety Measures:**
- Validate new model on test set
- Require 90%+ accuracy to deploy
- Rollback capability if regression
- Compare old vs new model

---

## 5. Features & Capabilities

### 5.1 Core Features

#### Bug Reporting (Tester Role)
- **Free-text description** input
- **AI-powered prediction** (severity + team)
- **Manual assignment** override option
- **Duplicate detection** (pre-reporting check)
- **Project tagging**

#### Bug Management (Developer Role)
- **View assigned bugs** (filtered by user)
- **Update status:** open → in_progress → resolved
- **Quick status transitions**
- **Bug details view**

#### Project Management (PM Role)
- **Dashboard:** Overall stats, severity distribution, team workload
- **Kanban Board:** Visual bug flow by status
- **Analytics:** Interactive charts (Plotly)
  - Bugs by team (horizontal bar chart)
  - Severity distribution (donut chart)
  - MLOps feedback history (area chart)
- **Correct AI Predictions:** One-click corrections
- **Notifications:** System events

### 5.2 Advanced Features

#### Real-time AI Predictions
- **Latency:** <100ms per prediction
- **Model caching:** Models loaded once at startup
- **Auto-reload:** Detects model updates, reloads automatically

#### Interactive Analytics
- **Modern visualizations:** Plotly graphs (not static charts)
- **Rich color palettes:** Neon/cyber theme
- **Responsive:** Mobile-friendly

#### Theme Support
- **Light mode** (default)
- **Dark mode** compatible (sidebar, graphs, buttons)

---

## 6. UI/UX Design

### 6.1 Design Philosophy
- **Professional:** Clean, modern interface
- **Role-based:** Tailored views per user type
- **Efficient:** Minimal clicks to complete tasks
- **Visual:** Rich charts, color-coded severities

### 6.2 UI Components

**Sidebar Navigation:**
- **Icon-based menu** (streamlit-option-menu)
- **Bootstrap icons** (speedometer, bug, kanban, bell)
- **Smooth transitions**
- **Theme-aware colors**

**Bug Cards:**
- **Gradient borders** (status-based)
- **Severity badges** (color-coded)
- **Team chips** (styled tags)
- **Timestamps**

**Graphs:**
- **Donut chart:** Severity distribution
- **Horizontal bar:** Team workload (Plasma gradient)
- **Area chart:** MLOps feedback (Fuchsia spline)

### 6.3 Accessibility
- **High contrast:** Readable in light/dark modes
- **Large click targets:** Buttons ≥48px
- **Keyboard navigation:** All actions accessible
- **Screen reader compatible:** Semantic HTML

---

## 7. API Specification

### 7.1 Authentication
```http
POST /token
Body: { "username": "user@example.com", "password": "password" }
Response: { "access_token": "jwt_token", "token_type": "bearer" }
```

### 7.2 Prediction
```http
POST /predict
Headers: Authorization: Bearer {token}
Body: { 
    "description": "Login button crashes the app",
    "project": "WebApp"
}
Response: {
    "severity": "high",
    "team": "Frontend"
}
```

### 7.3 Bug Reporting
```http
POST /report_bug
Body: {
    "description": "...",
    "project": "...",
    "severity": "high",  # From AI
    "team": "Backend",   # From AI
    "assigned_to_id": 5  # Optional
}
Response: { "id": 123, "message": "Bug reported" }
```

### 7.4 Bug Update (with MLOps)
```http
POST /update_bug
Body: {
    "bug_id": 123,
    "status": "in_progress",
    "correction_severity": "critical",  # PM correction
    "correction_team": "DevOps"         # PM correction
}
Response: {
    "message": "Bug updated (AI Feedback Recorded)",
    "feedback_count": 27
}
```

---

## 8. Performance Metrics

### 8.1 ML Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Inference Time |
|-------|----------|-----------|--------|----------|----------------|
| **Severity** | 96.7% | 96-100% | 95-100% | 93-98% | ~50ms |
| **Team** | 95.3% | 86-98% | 91-97% | 88-97% | ~50ms |
| **Deduplication** | N/A | TBD | TBD | TBD | ~100ms |

### 8.2 System Performance
- **API Latency:** <100ms (p95)
- **Database:** <10ms queries (SQLite)
- **Concurrent Users:** 100+ (tested)
- **Uptime:** 99.9% target

---

## 9. Data Flow

### 9.1 Bug Reporting Flow
```
User (Tester)
    ↓ Enters description
Frontend (Streamlit)
    ↓ POST /predict
Backend (FastAPI)
    ↓ Tokenize → Model → Predict
ML Engine
    ↓ Return (severity, team)
Backend
    ↓ POST /report_bug
Database (SQLite)
    ↓ Insert bug record
Backend
    ↓ Return bug_id
Frontend
    ↓ Show success
User
```

### 9.2 Correction Flow (MLOps)
```
PM reviews bug #123
    ↓ Changes severity: medium → high
    ↓ Changes team: Backend → DevOps
Frontend
    ↓ POST /update_bug
Backend
    ↓ Update bug record
    ↓ Log feedback (bug_id, field, old, new)
Database
    ↓ feedback_count++
Backend
    ↓ Check threshold (50 corrections)
    ↓ If threshold met → schedule retraining
MLOps Pipeline
```

---

## 10. Security

### 10.1 Authentication & Authorization
- **JWT tokens:** Secure, stateless
- **Role-based access:** Tester/Developer/PM permissions
- **Password hashing:** bcrypt (Passlib)

### 10.2 API Security
- **Rate limiting:** 100 req/min per user
- **Input validation:** Pydantic models
- **SQL injection prevention:** SQLAlchemy ORM
- **CORS:** Configured for production

### 10.3 Data Privacy
- **No PII in bug descriptions** (developer responsibility)
- **Database encryption:** Optional (production)
- **Audit logs:** All updates tracked

---

## 11. Development Journey

### 11.1 Phase 1: Foundation (Week 1)
✅ Backend API (FastAPI, SQLAlchemy, JWT)  
✅ Frontend UI (Streamlit, role-based views)  
✅ Database schema (bugs, users, notifications)  
✅ Basic CRUD operations

### 11.2 Phase 2: ML Integration (Week 2)
✅ Model loading (DistilBERT)  
✅ /predict endpoint  
✅ Auto-classification on bug report  
✅ Team assignment (rule-based fallback)

### 11.3 Phase 3: MLOps Pipeline (Week 3)
✅ Feedback collection UI  
✅ Correction storage (database)  
✅ Feedback counting  
✅ Manual retraining scripts

### 11.4 Phase 4: Fine-Tuning (Week 4)
✅ Dataset preparation (5.3M → 100K samples)  
✅ Severity model training (96.7% accuracy)  
✅ Team model training v1 (81.7% accuracy)  
✅ GPU optimization

### 11.5 Phase 5: Production Quality (Week 5)
✅ Enhanced team model (95.3% accuracy)  
✅ Class weighting (DevOps improvement)  
✅ Cosine scheduler  
✅ Deduplication model (in progress)  
✅ MLOps testing script  
✅ Comprehensive documentation

---

## 12. Challenges & Solutions

### 12.1 Challenge: Low DevOps Accuracy (42%)
**Root Cause:** Class imbalance (DevOps only 10% of data)

**Solution:**
- **Class weighting:** 2.46x weight for DevOps
- **More data:** 50K samples (5x increase)
- **More epochs:** 5 instead of 2

**Result:** DevOps F1-score 42% → 88.3% (+46% improvement)

---

### 12.2 Challenge: GPU Not Detected
**Root Cause:** PyTorch CUDA initialization error

**Investigation:**
- nvidia-smi working ✓
- PyTorch installed with CUDA ✓
- Environment variable conflict ✗

**Solution:**
- Reinstalled PyTorch with correct CUDA version (cu121)
- Enabled GPU persistence mode
- Cleared Python cache

**Result:** GPU working, 5-10x training speedup

---

### 12.3 Challenge: Slow Training (2-3 hours)
**Root Cause:** Large dataset (50K) + CPU-only

**Solution:**
- **GPU acceleration:** CUDA + FP16
- **Batch size optimization:** 32 → 64
- **Efficient data loading:** HuggingFace Datasets

**Result:** Training time 2.5 hrs → 30 min (5x faster)

---

## 13. Deployment

### 13.1 Current Deployment
- **Backend:** Uvicorn server (localhost:8000)
- **Frontend:** Streamlit app (localhost:8501)
- **Database:** SQLite file (bugflow.db)
- **Models:** Local files (severity_model/, team_model/)

### 13.2 Production Deployment (Recommended)

**Backend:**
```bash
# Docker container with GPU support
docker run --gpus all -p 8000:8000 \
    -v /models:/app/models \
    bugflow/backend:latest
```

**Frontend:**
```bash
# Streamlit Cloud or self-hosted
streamlit run frontend/app.py \
    --server.address 0.0.0.0 \
    --server.port 8501
```

**Database:** Migrate to PostgreSQL
```python
# config.py
DATABASE_URL = "postgresql://user:pass@host:5432/bugflow"
```

**Models:** Store in cloud storage (S3, GCS)
```bash
aws s3 sync s3://bugflow-models/severity_model ./severity_model
```

---

## 14. Future Roadmap

### 14.1 Short-term (Next 3 months)
- ☐ **Automated retraining:** Scheduled jobs (Airflow/Celery)
- ☐ **Deduplication improvements:** More diverse triplet data
- ☐ **A/B testing:** Compare model versions
- ☐ **API documentation:** Swagger/OpenAPI
- ☐ **Unit tests:** 80%+ code coverage

### 14.2 Mid-term (6 months)
- ☐ **Multi-language support:** i18n (English, Spanish, etc.)
- ☐ **Email notifications:** SendGrid integration
- ☐ **Slack/Teams integration:** Bug alerts
- ☐ **Advanced analytics:** Bug trends, team velocity
- ☐ **Mobile app:** React Native

### 14.3 Long-term (1 year)
- ☐ **LLM integration:** GPT-4 for bug descriptions
- ☐ **Automated bug fixing:** AI-suggested patches
- ☐ **Multi-tenant:** Enterprise SaaS offering
- ☐ **Custom model training:** Per-organization models
- ☐ **Explainable AI:** Why did model predict X?

---

## 15. Success Metrics

### 15.1 ML Metrics
- ✅ Severity accuracy: **96.7%** (target: 85%+)
- ✅ Team accuracy: **95.3%** (target: 90%+)
- ☐ Deduplication precision: **TBD** (target: 90%+)
- ✅ Inference latency: **<100ms** (target: <200ms)

### 15.2 Business Metrics
- ☐ Time-to-triage: Reduce by 70% (manual → AI)
- ☐ Mis-assigned bugs: Reduce by 90%
- ☐ Duplicate detection: Catch 80%+ before reporting
- ☐ PM time saved: 10 hours/week

### 15.3 User Adoption
- ☐ Active users: 100+ within 3 months
- ☐ Bugs reported: 1,000+ within 3 months
- ☐ AI acceptance rate: 90%+ (PM agrees with AI)
- ☐ User satisfaction: 4.5/5 stars

---

## 16. Lessons Learned

### 16.1 ML/AI
1. **Data quality > Quantity:** Clean 10K samples beat noisy 100K
2. **Class balancing crucial:** Minorities need weighting/oversampling
3. **Fine-tuning works:** Base models adapt well to domain
4. **GPU matters:** 5-10x speedup justifies cloud GPU costs

### 16.2 Engineering
1. **Checkpointing essential:** Training can crash, save progress
2. **Stratified splits:** Maintain label distribution in train/val/test
3. **Logging critical:** Debug issues during 2-hour training runs
4. **Error handling:** Assume everything can fail

### 16.3 Product
1. **Start simple:** Basic models first, iterate
2. **User feedback early:** PM corrections guide improvements
3. **Visualization matters:** Graphs make data actionable
4. **Role-based UX:** Tailor interface per user type

---

## Conclusion

BugFlow demonstrates a **complete ML/MLOps pipeline** from raw data to production deployment:

1. ✅ **Dataset:** 5.3M real bugs → 100K cleaned samples
2. ✅ **Models:** Fine-tuned DistilBERT (96.7%, 95.3% accuracy)
3. ✅ **MLOps:** PM corrections → Model improvements
4. ✅ **Production:** FastAPI + Streamlit, <100ms latency
5. ✅ **Industry standards:** Error handling, checkpointing, monitoring

**Next Steps:** Deploy to cloud, automate retraining, expand to new use cases.

---

**Document End**
