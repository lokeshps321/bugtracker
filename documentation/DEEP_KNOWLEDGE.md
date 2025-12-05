# 🎓 BugFlow - Deep Knowledge Guide

*Complete technical knowledge to answer any question about the project*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why We Built This](#2-why-we-built-this)
3. [Technical Architecture](#3-technical-architecture)
4. [Machine Learning Deep Dive](#4-machine-learning-deep-dive)
5. [Natural Language Processing](#5-natural-language-processing)
6. [Model Training Process](#6-model-training-process)
7. [Backend Implementation](#7-backend-implementation)
8. [Frontend Implementation](#8-frontend-implementation)
9. [Database Design](#9-database-design)
10. [MLOps & Continuous Learning](#10-mlops--continuous-learning)
11. [Security Implementation](#11-security-implementation)
12. [API Design](#12-api-design)
13. [Deployment Architecture](#13-deployment-architecture)
14. [Common Interview Questions](#14-common-interview-questions)

---

## 1. Project Overview

### What is BugFlow?

BugFlow is an **AI-powered bug tracking system** that automatically triages software bugs using Natural Language Processing (NLP) and Deep Learning.

### Key Features

| Feature | Technology | Accuracy |
|---------|------------|----------|
| Severity Classification | CodeBERT (Fine-tuned) | 86.35% |
| Team Assignment | CodeBERT (Fine-tuned) | 83.40% |
| Duplicate Detection | Sentence Transformers | 85% threshold |
| Real-time Notifications | SMTP Email | N/A |
| Continuous Learning | MLOps Pipeline | Automatic |

### Why It Matters

Traditional bug tracking requires **manual triage**:
- Takes 5-10 minutes per bug
- Inconsistent across team members
- Creates bottlenecks
- Delays critical fixes

BugFlow **automates this** using AI, reducing triage time to **milliseconds**.

---

## 2. Why We Built This

### Problem Statement

Software teams waste significant time manually classifying bugs. A typical workflow:

```
Developer reports bug → Manager reads it → Manager decides severity → 
Manager assigns to team → Team picks it up → Work begins
```

This creates **delays of hours to days** for critical bugs.

### Our Solution

```
Developer reports bug → AI instantly predicts severity + team → 
Notification sent → Team immediately sees it → Work begins
```

Time savings: **90% reduction in triage time**

---

## 3. Technical Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     BugFlow Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐        ┌──────────────┐       ┌────────────┐  │
│  │   Frontend   │  HTTP  │   Backend    │  SQL  │  Database  │  │
│  │  (Streamlit) │◄──────►│  (FastAPI)   │◄─────►│  (SQLite/  │  │
│  │  Port 8501   │  REST  │  Port 8000   │  ORM  │  Postgres) │  │
│  └──────────────┘        └──────┬───────┘       └────────────┘  │
│                                 │                                │
│                                 ▼                                │
│                    ┌─────────────────────────┐                   │
│                    │     ML Pipeline         │                   │
│                    │  ┌─────────────────┐    │                   │
│                    │  │   Tokenizer     │    │                   │
│                    │  │(RobertaTokenizer)│   │                   │
│                    │  └────────┬────────┘    │                   │
│                    │           ▼             │                   │
│                    │  ┌─────────────────┐    │                   │
│                    │  │   CodeBERT      │    │                   │
│                    │  │ (125M params)   │    │                   │
│                    │  └────────┬────────┘    │                   │
│                    │           ▼             │                   │
│                    │  ┌─────────────────┐    │                   │
│                    │  │  Classification │    │                   │
│                    │  │  Head (4 cls)   │    │                   │
│                    │  └─────────────────┘    │                   │
│                    └─────────────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction

1. **User** → Enters bug description in Streamlit
2. **Frontend** → Sends HTTP POST to `/predict`
3. **Backend** → Tokenizes text, feeds to CodeBERT
4. **Model** → Returns probability distribution
5. **Backend** → Returns highest probability class
6. **Frontend** → Displays prediction to user

---

## 4. Machine Learning Deep Dive

### What is CodeBERT?

CodeBERT is a **bimodal pre-trained model** for programming language and natural language.

**Key Facts**:
- Developed by: Microsoft Research
- Parameters: 125 million
- Based on: RoBERTa architecture
- Pre-training data: 6.4 million code-documentation pairs
- Languages: 6 programming languages (Python, Java, JavaScript, PHP, Ruby, Go)

### Why CodeBERT (Not Regular BERT)?

| Aspect | BERT | CodeBERT | Why CodeBERT Wins |
|--------|------|----------|-------------------|
| Pre-training | Wikipedia + Books | Code + Docs | Understands code terminology |
| Vocabulary | General | Code-aware | Knows "API", "null pointer", "stack trace" |
| Context | Natural language | Code + NL | Better for technical descriptions |

### Model Architecture

```
Input: "Database timeout causing 500 error"
                    │
                    ▼
        ┌───────────────────────┐
        │    Tokenizer          │
        │  (RobertaTokenizer)   │
        └───────────┬───────────┘
                    │
                    ▼
        [CLS] Database timeout causing 500 error [SEP]
                    │
                    ▼
        ┌───────────────────────┐
        │   Embedding Layer     │
        │   (768 dimensions)    │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  12 Transformer       │
        │  Encoder Layers       │
        │  (Self-Attention)     │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Classification Head  │
        │  (Linear → Softmax)   │
        └───────────┬───────────┘
                    │
                    ▼
        [0.02, 0.08, 0.85, 0.05]  → "high" (85%)
         low   med  high crit
```

### Self-Attention Mechanism

Self-attention allows the model to understand relationships between words:

```
"Database timeout causing server crash"
     ↑           ↑              ↑
     └───────────┼──────────────┘
          Attention weights connect
          "Database" with "crash"
```

**Formula**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- Q = Query matrix
- K = Key matrix  
- V = Value matrix
- d_k = dimension of keys (for scaling)

---

## 5. Natural Language Processing

### Tokenization Process

**Input**: "iOS app crashes on launch"

**Step 1**: Subword Tokenization (BPE)
```
["iOS", "Ġapp", "Ġcrashes", "Ġon", "Ġlaunch"]
```
(Ġ represents a space before the word)

**Step 2**: Token IDs
```
[10706, 2771, 15719, 15, 3568]
```

**Step 3**: Add Special Tokens
```
[0, 10706, 2771, 15719, 15, 3568, 2]
     ↑                              ↑
   [CLS]                          [SEP]
```

### Why Subword Tokenization?

**Problem with word-level**:
- Unknown words (OOV): "NullPointerException" → [UNK]
- Vocabulary explosion

**BPE Solution**:
- "NullPointerException" → ["Null", "Pointer", "Exception"]
- Handles any word by breaking into known subwords

### Text Preprocessing

```python
def preprocess(text):
    # 1. Combine title + description
    full_text = f"{title}. {description}"
    
    # 2. Tokenize (max 128 tokens)
    tokens = tokenizer(
        full_text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    return tokens
```

---

## 6. Model Training Process

### Dataset Preparation

**Original Dataset**: GitHub Issues + Stack Overflow
**Size**: 9,820 bug descriptions
**Split**: 80% train, 10% validation, 10% test

**Distribution**:
```
Severity:              Team:
┌─────────┬──────┐     ┌──────────┬──────┐
│ Low     │ 25%  │     │ Backend  │ 28%  │
│ Medium  │ 25%  │     │ Frontend │ 38%  │
│ High    │ 25%  │     │ Mobile   │ 17%  │
│ Critical│ 25%  │     │ DevOps   │ 17%  │
└─────────┴──────┘     └──────────┴──────┘
```

### Training Configuration

```python
TrainingArguments(
    num_train_epochs=15,
    per_device_train_batch_size=16,
    learning_rate=1.5e-5,
    warmup_ratio=0.15,
    weight_decay=0.02,
    gradient_accumulation_steps=2,
    fp16=True,
    evaluation_strategy="epoch",
    load_best_model_at_end=True
)
```

### Key Training Techniques

#### 1. Label Smoothing (ε = 0.1)

**Problem**: Model becomes overconfident (99.9% predictions)

**Solution**: Soften labels
```
Hard label:  [0, 0, 1, 0]
Soft label:  [0.025, 0.025, 0.925, 0.025]
```

**Formula**:
```
y_smooth = (1 - ε) × y_true + ε / num_classes
```

#### 2. Class Weighting

**Problem**: Imbalanced dataset (Frontend has more samples)

**Solution**: Higher loss for minority classes
```python
class_weights = compute_class_weight('balanced', 
                                     classes=unique_labels, 
                                     y=train_labels)
# Result: [0.89, 0.65, 1.48, 1.51]
#         Backend Frontend Mobile DevOps
```

#### 3. Early Stopping

**Problem**: Model overfits after too many epochs

**Solution**: Stop when validation accuracy stops improving
```python
EarlyStoppingCallback(early_stopping_patience=4)
```

**What happened**:
- Severity: Best at epoch 8, stopped at epoch 12
- Team: Best at epoch 5, stopped at epoch 9

#### 4. Mixed Precision Training (FP16)

**Problem**: Training is slow and uses lots of memory

**Solution**: Use 16-bit floats instead of 32-bit
```python
fp16=True  # In TrainingArguments
```

**Benefits**:
- 2x faster training
- 50% less GPU memory
- No accuracy loss

---

## 7. Backend Implementation

### FastAPI Framework

**Why FastAPI?**

| Feature | Flask | Django | FastAPI |
|---------|-------|--------|---------|
| Performance | Medium | Low | **High** |
| Async Support | Add-on | Limited | **Native** |
| Type Hints | Optional | Optional | **Required** |
| Auto Docs | No | No | **Yes** |

### Key Endpoints

```python
# Authentication
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm):
    user = authenticate_user(db, form_data.username, form_data.password)
    token = create_access_token({"sub": user.email})
    return {"access_token": token}

# Prediction
@app.post("/predict")
async def predict_bug(data: BugCreate):
    full_text = f"{data.title}. {data.description}"
    severity, team = predict_bug_attributes(full_text)
    return {"severity": severity, "team": team}

# Bug Creation
@app.post("/report_bug")
async def report_bug(data: BugCreate, db: Session):
    # 1. Predict severity and team
    severity, team = predict_bug_attributes(full_text)
    
    # 2. Check for duplicates
    duplicate = check_for_duplicate(data.description, db)
    
    # 3. Save to database
    bug = Bug(title=data.title, description=data.description,
              severity=severity, team=team)
    db.add(bug)
    db.commit()
    
    return bug
```

### Dependency Injection

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/bugs")
async def get_bugs(db: Session = Depends(get_db)):
    return db.query(Bug).all()
```

---

## 8. Frontend Implementation

### Streamlit Framework

**Why Streamlit?**

- Pure Python (no JavaScript needed)
- Built-in components (buttons, forms, charts)
- Easy ML integration
- Fast prototyping

### Role-Based UI

```python
if user.role == "tester":
    show_tester_pages()  # Report bugs, view predictions
elif user.role == "developer":
    show_developer_pages()  # View assigned bugs, update status
elif user.role == "pm":
    show_pm_pages()  # Dashboard, analytics, corrections
```

### State Management

```python
# Session state persists across reruns
if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.token = None

# Update state after login
st.session_state.user = user_data
st.session_state.token = token
```

---

## 9. Database Design

### Entity Relationship Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│      User       │     │       Bug       │     │    Feedback     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ id (PK)         │     │ id (PK)         │     │ id (PK)         │
│ email           │────►│ reporter_id (FK)│     │ bug_id (FK)     │
│ password_hash   │────►│ assigned_to_id  │◄────│ correction_sev  │
│ role            │     │ title           │     │ correction_team │
└─────────────────┘     │ description     │     │ created_at      │
                        │ severity        │     └─────────────────┘
                        │ team            │
                        │ status          │
                        │ created_at      │
                        └─────────────────┘
```

### SQLAlchemy ORM

```python
class Bug(Base):
    __tablename__ = "bugs"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    description = Column(Text, nullable=False)
    severity = Column(String)  # low, medium, high, critical
    team = Column(String)      # Backend, Frontend, Mobile, DevOps
    status = Column(String)    # open, in_progress, resolved
    reporter_id = Column(Integer, ForeignKey("users.id"))
    assigned_to_id = Column(Integer, ForeignKey("users.id"))
```

---

## 10. MLOps & Continuous Learning

### Feedback Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLOps Feedback Loop                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Bug submitted ──► ML Prediction ──► User sees result        │
│                                              │                   │
│                                              ▼                   │
│  4. Model retrained ◄── 3. Threshold ◄── 2. PM Correction       │
│         │                   (50)             │                   │
│         ▼                                    ▼                   │
│  5. Hot reload ──► Better predictions    Stored in DB           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
@app.post("/update_bug")
async def update_bug(data: BugUpdate, db: Session):
    # Save PM correction as feedback
    if data.correction_severity or data.correction_team:
        feedback = Feedback(
            bug_id=data.bug_id,
            correction_severity=data.correction_severity,
            correction_team=data.correction_team
        )
        db.add(feedback)
        db.commit()
        
        # Check if threshold reached
        count = db.query(Feedback).count()
        if count >= 50:
            trigger_retraining()
```

---

## 11. Security Implementation

### JWT Authentication

**Token Generation**:
```python
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
```

**Token Structure**:
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.
eyJzdWIiOiJ0ZXN0ZXIxQGV4YW1wbGUuY29tIiwiZXhwIjoxNzAxNjk1NDAwfQ.
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Password Hashing

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"])

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)
```

---

## 12. API Design

### RESTful Principles

| Method | Endpoint | Action |
|--------|----------|--------|
| POST | /token | Login |
| GET | /bugs | List all bugs |
| POST | /bugs | Create bug |
| PATCH | /bugs/{id} | Update bug |
| POST | /predict | Get prediction only |

### Request/Response Examples

**Create Bug**:
```http
POST /report_bug
Content-Type: application/json
Authorization: Bearer eyJ...

{
  "title": "Login fails on Chrome",
  "description": "Users cannot login...",
  "project": "WebApp"
}

Response:
{
  "id": 42,
  "title": "Login fails on Chrome",
  "severity": "high",
  "team": "Frontend",
  "status": "open"
}
```

---

## 13. Deployment Architecture

### Cloud Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Cloud Deployment                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │  Streamlit   │     │   Render     │     │  PostgreSQL  │     │
│  │    Cloud     │────►│  (Backend)   │────►│   (Render)   │     │
│  │  (Frontend)  │     │   FastAPI    │     │              │     │
│  └──────────────┘     └──────────────┘     └──────────────┘     │
│                              │                                   │
│                              ▼                                   │
│                    ┌──────────────────┐                         │
│                    │  HuggingFace     │                         │
│                    │  Model Hub       │                         │
│                    │  (CodeBERT)      │                         │
│                    └──────────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 14. Common Interview Questions

### Q1: "Why did you choose CodeBERT over GPT?"

**Answer**: 
"CodeBERT is specifically designed for code-related tasks, pre-trained on 6.4 million code-documentation pairs. While GPT is powerful, it's overkill for classification tasks and requires expensive API calls. CodeBERT runs locally, is fine-tunable, and achieves 86% accuracy for our use case."

### Q2: "How does your model handle unseen words?"

**Answer**:
"We use BPE (Byte Pair Encoding) tokenization. It breaks unknown words into known subwords. For example, 'NullPointerException' becomes ['Null', 'Pointer', 'Exception']. This handles any technical terminology without out-of-vocabulary issues."

### Q3: "What is label smoothing and why use it?"

**Answer**:
"Label smoothing prevents overconfidence. Instead of training on hard labels [0,0,1,0], we use soft labels [0.025, 0.025, 0.925, 0.025]. This improves generalization because the model learns that predictions are probabilistic, not absolute."

### Q4: "How do you prevent overfitting?"

**Answer**:
"Three techniques: (1) Early stopping - we stop training when validation accuracy stops improving. (2) Weight decay (0.02) - L2 regularization penalizes large weights. (3) Dropout in transformer layers - randomly zeros activations during training."

### Q5: "Explain the MLOps pipeline."

**Answer**:
"Our MLOps enables continuous learning. When a PM corrects a prediction, it's stored as feedback. After 50 corrections, automatic retraining is triggered. The new model is hot-reloaded without restarting the server. This creates a feedback loop where the model improves over time."

### Q6: "How does duplicate detection work?"

**Answer**:
"We use Sentence Transformers (all-MiniLM-L6-v2) to convert bug descriptions into 384-dimensional embeddings. For a new bug, we compute cosine similarity with all existing bugs. If similarity > 0.85, we flag it as a potential duplicate. This catches semantically similar bugs even with different wording."

### Q7: "What's the difference between severity prediction and team assignment?"

**Answer**:
"They're separate classification tasks. Severity (low/medium/high/critical) assesses impact - 'database crash' is critical because it affects data. Team assignment (Backend/Frontend/Mobile/DevOps) determines routing - 'database crash' goes to Backend because it's server-side. We use two separate fine-tuned CodeBERT models."

### Q8: "How do you handle class imbalance?"

**Answer**:
"Class weighting. We compute weights inversely proportional to class frequency using sklearn's compute_class_weight('balanced'). Minority classes get higher weights, so mistakes on them cost more during training. This prevents the model from just predicting the majority class."

---

*Created for BugFlow Project Presentation | December 2025*
