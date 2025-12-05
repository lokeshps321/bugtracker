"""
BugFlow ML Inference API - Hugging Face Space
This runs on Hugging Face Spaces with 16GB RAM (FREE!)
Enhanced with keyword-based boosting for DevOps and Mobile
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
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
    "backup", "restore", "staging", "production", "environment",
    "infrastructure", "devops", "sre", "deploy", "rollback", "release",
    "nightly", "cron", "scheduled job", "sync", "migration script"
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
    "critical": ["crash", "crashes", "down", "outage", "data loss", "security", 
                 "vulnerability", "breach", "blocked", "broken", "fail", "failure"],
    "high": ["error", "not working", "bug", "issue", "problem", "cannot", "unable"],
}

def detect_team_by_keywords(text):
    """Detect team based on keyword matching"""
    text_lower = text.lower()
    
    devops_score = sum(1 for kw in DEVOPS_KEYWORDS if kw in text_lower)
    mobile_score = sum(1 for kw in MOBILE_KEYWORDS if kw in text_lower)
    
    if devops_score >= 2:
        return "DevOps", 0.95
    if mobile_score >= 2:
        return "Mobile", 0.95
    if devops_score == 1:
        return "DevOps", 0.80
    if mobile_score == 1:
        return "Mobile", 0.80
    
    return None, 0

def detect_severity_by_keywords(text):
    """Boost severity based on keywords"""
    text_lower = text.lower()
    
    for kw in SEVERITY_KEYWORDS["critical"]:
        if kw in text_lower:
            return "critical" if "down" in text_lower or "crash" in text_lower else None, 0.90
    
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
    global severity_model, team_model, tokenizer, severity_labels, team_labels
    
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
        
        # Boost severity for critical keywords
        kw_severity, kw_sev_conf = detect_severity_by_keywords(text)
        if kw_severity and severity in ["low", "medium"] and kw_sev_conf > severity_confidence:
            severity = "high"  # Boost to at least high if critical keywords found
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

