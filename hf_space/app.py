"""
BugFlow ML Inference API - Hugging Face Space
This runs on Hugging Face Spaces with 16GB RAM (FREE!)
Enhanced with keyword-based boosting for DevOps and Mobile
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BugFlow ML Inference API")

# Model configuration
SEVERITY_MODEL = "loke007/bugflow-severity-classifier"
TEAM_MODEL = "loke007/bugflow-team-classifier"

# Global model variables
severity_model = None
team_model = None
dedup_model = None
tokenizer = None
severity_labels = ['low', 'medium', 'high', 'critical']
team_labels = ['Backend', 'Frontend', 'Mobile', 'DevOps']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Keyword-based detection for better DevOps and Mobile classification
DEVOPS_KEYWORDS = [
    "ci/cd", "cicd", "pipeline", "jenkins", "github actions", "gitlab ci", 
    "docker", "container", "kubernetes", "k8s", "helm", "pod", "deployment",
    "terraform", "ansible", "aws", "azure", "gcp", "cloud", "ec2", "s3",
    "load balancer", "nginx", "ssl", "certificate", "dns", "cdn",
    "prometheus", "grafana", "monitoring", "alerting", "logging", "elk",
    "infrastructure", "devops", "sre", "deploy", "rollback", "release",
    "nightly", "cron", "scheduled job"
]

# Backend keywords - these OVERRIDE DevOps when found together
BACKEND_KEYWORDS = [
    "jwt", "token", "auth", "oauth", "session", "login", "password",
    "api", "rest", "graphql", "endpoint", "database", "db", "sql",
    "redis", "cache", "queue", "webhook", "validation",
    "payment", "stripe", "transaction", "order", "checkout", "cart",
    "user", "account", "profile", "subscription", "billing"
]

MOBILE_KEYWORDS = [
    "ios", "android", "mobile", "iphone", "ipad", "samsung", "pixel",
    "react native", "flutter", "swift", "kotlin", "xcode", "android studio",
    "app store", "play store", "apk", "ipa", "provisioning", "certificate",
    "push notification", "fcm", "apns", "deep link", "biometric",
    "face id", "touch id", "fingerprint", "gps", "location", "camera",
    "battery", "offline", "tablet", "phone", "smartphone", "wearable"
]

SEVERITY_KEYWORDS = {
    "critical": [
        "crash", "crashes", "crashed", "crashing", "down", "outage", "data loss",
        "security vulnerability", "security breach", "breach", "hacked", "exploit",
        "vulnerability", "ddos", "attack", "penetration", "brute force",
        "production down", "system down", "server down", "site down", "app down",
        "cannot start", "won't start", "not starting", "complete failure",
        "all users affected", "everyone affected", "total outage", "catastrophic",
        "100% of users", "all users", "complete outage", "system outage",
        "double charged", "lost money", "money lost", "revenue lost", "lost revenue",
        "p0", "p0 incident", "critical incident", "p0-urgent", "urgent"
    ],
    "high": [
        "error", "errors", "fail", "fails", "failed", "failing", "failure",
        "not working", "doesn't work", "broken", "breaking", "breaks",
        "block", "blocked", "blocking", "blocker", "stuck", "freeze", "frozen",
        "cannot", "can't", "unable", "impossible", "prevents", "preventing",
        "authentication", "login fail", "logout", "session expir", "timeout",
        "data missing", "data lost", "corrupt", "unavailable", "unresponsive",
        "urgent", "asap", "major bug", "serious",
        "not triggering", "not responding", "not scaling",
        "affecting all", "p1", "incident"
    ],
    "medium": [
        "issue", "problem", "incorrect", "wrong", "unexpected",
        "slow", "delay", "delayed", "latency", "performance", "degraded", "degradation",
        "intermittent", "sometimes", "occasionally", "inconsistent", "flaky",
        "confusing", "unclear", "misleading", "usability", "ux issue",
        "improvement needed", "needs fix", "should be", "supposed to",
        "takes too long", "optimization", "optimize"
    ],
    "low": [
        "typo", "typos", "spelling", "grammar", "cosmetic", "visual", "aesthetic",
        "minor", "small", "trivial", "nice to have", "enhancement", "suggestion",
        "request", "feature request", "would be nice", "consider",
        "documentation", "docs", "readme", "comment", "tooltip", "label",
        "low priority", "not urgent", "when you have time", "polish"
    ]
}

def detect_team_by_keywords(text):
    """Detect team based on keyword matching"""
    text_lower = text.lower()
    
    devops_score = sum(1 for kw in DEVOPS_KEYWORDS if kw in text_lower)
    mobile_score = sum(1 for kw in MOBILE_KEYWORDS if kw in text_lower)
    backend_score = sum(1 for kw in BACKEND_KEYWORDS if kw in text_lower)
    
    # Backend keywords override DevOps (JWT, auth, API are backend, not DevOps)
    if backend_score >= 1 and devops_score >= 1:
        return "Backend", 0.90  # Backend takes priority for auth/API issues
    
    if devops_score >= 2:
        return "DevOps", 0.95
    if mobile_score >= 2:
        return "Mobile", 0.95
    if backend_score >= 2:
        return "Backend", 0.90
    if devops_score == 1:
        return "DevOps", 0.80
    if mobile_score == 1:
        return "Mobile", 0.80
    if backend_score == 1:
        return "Backend", 0.75
    
    return None, 0

def detect_severity_by_keywords(text):
    """Detect severity based on keyword matching with smart priority"""
    text_lower = text.lower()
    
    # Count matches for each severity level
    scores = {
        "critical": sum(1 for kw in SEVERITY_KEYWORDS["critical"] if kw in text_lower),
        "high": sum(1 for kw in SEVERITY_KEYWORDS["high"] if kw in text_lower),
        "medium": sum(1 for kw in SEVERITY_KEYWORDS["medium"] if kw in text_lower),
        "low": sum(1 for kw in SEVERITY_KEYWORDS["low"] if kw in text_lower)
    }
    
    # Strong LOW indicators that completely override high severity
    # (typo, spelling, cosmetic, documentation are ALWAYS low priority)
    strong_low = ["typo", "typos", "spelling", "cosmetic", "documentation", "docs", 
                  "low priority", "minor issue", "trivial", "polish", "readme",
                  "brand guide", "color palette", "font", "design review"]
    has_strong_low = any(kw in text_lower for kw in strong_low)
    
    # Medium indicators for intermittent/flaky issues
    medium_indicators = ["intermittent", "flaky", "sometimes", "occasionally", 
                         "workaround", "re-running", "retry", "not blocking",
                         "30%", "partial", "inconsistent"]
    has_medium_indicator = any(kw in text_lower for kw in medium_indicators)
    
    # Check for crashes/outages/security first (these are always critical)
    if scores["critical"] >= 1:
        return "critical", 0.95
    
    # If strong low indicators found, it's ALWAYS LOW (override high)
    if has_strong_low:
        return "low", 0.92
    
    # Intermittent/flaky issues are MEDIUM not HIGH
    if has_medium_indicator and scores["high"] >= 1:
        return "medium", 0.85
    
    # Normal priority: high > medium > low
    if scores["high"] >= 1:
        return "high", 0.90
    if scores["medium"] >= 1:
        return "medium", 0.80
    if scores["low"] >= 1:
        return "low", 0.75
    
    return None, 0

class PredictRequest(BaseModel):
    description: str
    title: str = ""

class PredictResponse(BaseModel):
    severity: str
    team: str
    severity_confidence: float = 0.0
    team_confidence: float = 0.0

def load_models():
    """Load models from Hugging Face Hub"""
    global severity_model, team_model, dedup_model, tokenizer, severity_labels, team_labels
    
    logger.info(f"🔄 Loading models on {device}...")
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    
    # Load severity model
    logger.info(f"Loading severity model: {SEVERITY_MODEL}")
    severity_model = RobertaForSequenceClassification.from_pretrained(SEVERITY_MODEL)
    severity_model.to(device)
    severity_model.eval()
    
    # Try to load severity labels
    try:
        from huggingface_hub import hf_hub_download
        import json
        label_path = hf_hub_download(repo_id=SEVERITY_MODEL, filename="severity_labels.json")
        with open(label_path, 'r') as f:
            label_map = json.load(f)
            severity_labels = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
        logger.info(f"Severity labels: {severity_labels}")
    except Exception as e:
        logger.warning(f"Using default severity labels: {e}")
    
    # Load team model
    logger.info(f"Loading team model: {TEAM_MODEL}")
    team_model = RobertaForSequenceClassification.from_pretrained(TEAM_MODEL)
    team_model.to(device)
    team_model.eval()
    
    # Try to load team labels
    try:
        from huggingface_hub import hf_hub_download
        import json
        label_path = hf_hub_download(repo_id=TEAM_MODEL, filename="team_labels.json")
        with open(label_path, 'r') as f:
            label_map = json.load(f)
            team_labels = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
        logger.info(f"Team labels: {team_labels}")
    except Exception as e:
        logger.warning(f"Using default team labels: {e}")
    
    # Load deduplication model (MiniLM-L6 for semantic similarity)
    logger.info("Loading deduplication model: all-MiniLM-L6-v2")
    dedup_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("✅ Deduplication model loaded!")
    
    logger.info("✅ All models loaded successfully!")

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()

@app.get("/")
async def root():
    return {"message": "BugFlow ML Inference API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": severity_model is not None}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict severity and team for a bug description"""
    global severity_model, team_model, tokenizer
    
    if severity_model is None or team_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Combine title and description
    text = f"{request.title}. {request.description}" if request.title else request.description
    
    try:
        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        # Predict severity with model
        with torch.no_grad():
            outputs = severity_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            sev_conf, sev_idx = torch.max(probs, dim=1)
            severity = severity_labels[sev_idx.item()]
            severity_confidence = sev_conf.item()
        
        # Boost severity for blocking/failing keywords
        kw_severity, kw_sev_conf = detect_severity_by_keywords(text)
        if kw_severity and severity in ["low", "medium"]:
            # Always boost if keywords indicate high/critical severity
            severity = kw_severity
            severity_confidence = max(severity_confidence, kw_sev_conf)
        
        # Predict team with model
        with torch.no_grad():
            outputs = team_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            team_conf, team_idx = torch.max(probs, dim=1)
            team = team_labels[team_idx.item()]
            team_confidence = team_conf.item()
        
        # Keyword-based boost for DevOps and Mobile
        kw_team, kw_conf = detect_team_by_keywords(text)
        if kw_team:
            # If keywords strongly indicate DevOps/Mobile and model is uncertain
            if kw_conf > 0.8 and team not in ["DevOps", "Mobile"]:
                # Override if model confidence is low or keyword match is strong
                if team_confidence < 0.8 or kw_conf >= 0.95:
                    logger.info(f"Keyword boost: {team} -> {kw_team} (kw={kw_conf:.2f}, model={team_confidence:.2f})")
                    team = kw_team
                    team_confidence = max(team_confidence, kw_conf)
        
        return PredictResponse(
            severity=severity,
            team=team,
            severity_confidence=severity_confidence,
            team_confidence=team_confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Deduplication models
class DuplicateCheckRequest(BaseModel):
    description: str
    existing_descriptions: list[str] = []

class DuplicateCheckResponse(BaseModel):
    is_duplicate: bool
    duplicate_index: int = -1
    similarity_score: float = 0.0

@app.post("/check_duplicate", response_model=DuplicateCheckResponse)
async def check_duplicate(request: DuplicateCheckRequest):
    """Check if a bug description is a duplicate of existing bugs"""
    global dedup_model
    
    if dedup_model is None:
        raise HTTPException(status_code=503, detail="Deduplication model not loaded")
    
    if not request.existing_descriptions:
        return DuplicateCheckResponse(is_duplicate=False, duplicate_index=-1, similarity_score=0.0)
    
    try:
        # Encode all descriptions
        all_descriptions = [request.description] + request.existing_descriptions
        embeddings = dedup_model.encode(all_descriptions)
        
        # Calculate similarity between new description and all existing ones
        new_embedding = embeddings[0].reshape(1, -1)
        existing_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(new_embedding, existing_embeddings)[0]
        
        max_similarity = float(np.max(similarities))
        max_index = int(np.argmax(similarities))
        
        # Threshold for duplicate detection (0.9 = 90% similar)
        is_duplicate = max_similarity > 0.9
        
        return DuplicateCheckResponse(
            is_duplicate=is_duplicate,
            duplicate_index=max_index if is_duplicate else -1,
            similarity_score=max_similarity
        )
        
    except Exception as e:
        logger.error(f"Deduplication error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

