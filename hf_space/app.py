"""
BugFlow ML Inference API - Hugging Face Space
This runs on Hugging Face Spaces with 16GB RAM (FREE!)
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
        
        # Predict severity
        with torch.no_grad():
            outputs = severity_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            sev_conf, sev_idx = torch.max(probs, dim=1)
            severity = severity_labels[sev_idx.item()]
            severity_confidence = sev_conf.item()
        
        # Predict team
        with torch.no_grad():
            outputs = team_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            team_conf, team_idx = torch.max(probs, dim=1)
            team = team_labels[team_idx.item()]
            team_confidence = team_conf.item()
        
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
