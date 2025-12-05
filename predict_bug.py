"""
BugFlow ML Prediction Module

Supports:
1. Local fine-tuned models (development)
2. Hugging Face Hub models (cloud deployment)
3. Base model fallback (when nothing else available)
"""

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch
import logging
import os
import time
import json

logger = logging.getLogger(__name__)

# Global variables
severity_model = None
team_model = None
tokenizer = None
severity_labels = None
team_labels = None
dedup_model = None
models_loaded = False
last_model_load_time = 0
last_model_version = 0
model_load_timestamp = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hugging Face Hub configuration
HF_SEVERITY_MODEL = os.getenv('HF_SEVERITY_MODEL', '')  # e.g., "username/bugflow-severity-classifier"
HF_TEAM_MODEL = os.getenv('HF_TEAM_MODEL', '')  # e.g., "username/bugflow-team-classifier"


def load_model_from_hub_or_local(model_type):
    """
    Load model from Hugging Face Hub or local directory.
    Priority: Local fine-tuned > Hugging Face Hub > Base model
    """
    global device
    
    if model_type == "severity":
        hf_repo = HF_SEVERITY_MODEL
        local_paths = ['./severity_model_specialized', './severity_model_new', './severity_model']
        default_labels = ['low', 'medium', 'high', 'critical']
        label_file = 'severity_labels.json'
    else:  # team
        hf_repo = HF_TEAM_MODEL
        local_paths = ['./team_model_specialized', './team_model_new', './team_model']
        default_labels = ['Backend', 'Frontend', 'Mobile', 'DevOps']
        label_file = 'team_labels.json'
    
    model = None
    labels = default_labels
    use_base_model = os.getenv('USE_BASE_MODEL', 'false').lower() == 'true'
    
    # Option 1: Try local fine-tuned models first (for development)
    if not use_base_model:
        for path in local_paths:
            if os.path.exists(path):
                try:
                    model = RobertaForSequenceClassification.from_pretrained(path)
                    logger.info(f"✅ Loaded {model_type} model from local: {path}")
                    
                    # Load labels
                    label_path = os.path.join(path, label_file)
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            label_map = json.load(f)
                            labels = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
                    
                    model.to(device)
                    model.eval()
                    return model, labels
                except Exception as e:
                    logger.warning(f"Failed to load from {path}: {e}")
    
    # Option 2: Try Hugging Face Hub (for cloud deployment)
    if hf_repo and not use_base_model:
        try:
            logger.info(f"🔄 Loading {model_type} model from Hugging Face Hub: {hf_repo}")
            model = RobertaForSequenceClassification.from_pretrained(hf_repo)
            
            # Try to load labels from hub
            try:
                from huggingface_hub import hf_hub_download
                label_path = hf_hub_download(repo_id=hf_repo, filename=label_file)
                with open(label_path, 'r') as f:
                    label_map = json.load(f)
                    labels = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
            except Exception:
                logger.info(f"Using default labels for {model_type}")
            
            logger.info(f"✅ Loaded {model_type} model from Hugging Face Hub!")
            model.to(device)
            model.eval()
            return model, labels
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face Hub: {e}")
    
    # Option 3: Fallback to base CodeBERT model
    logger.info(f"🔄 Using base CodeBERT model for {model_type} (no fine-tuned model available)")
    model = RobertaForSequenceClassification.from_pretrained(
        'microsoft/codebert-base',
        num_labels=len(default_labels)
    )
    model.to(device)
    model.eval()
    return model, default_labels


def load_model_and_vectorizer(reload=False):
    """Load or reload all models and initialize global state."""
    global severity_model, team_model, tokenizer, severity_labels, team_labels, dedup_model
    global models_loaded, last_model_load_time, last_model_version, model_load_timestamp
    
    try:
        # Load tokenizer
        logger.info("🔄 Loading tokenizer...")
        tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        
        # Load severity model
        logger.info("🔄 Loading severity model...")
        severity_model, severity_labels = load_model_from_hub_or_local("severity")
        
        # Load team model
        logger.info("🔄 Loading team model...")
        team_model, team_labels = load_model_from_hub_or_local("team")
        
        # Load deduplication model
        try:
            logger.info("🔄 Loading deduplication model...")
            dedup_model = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))
            logger.info("✅ Deduplication model loaded")
        except Exception as e:
            logger.warning(f"Dedup model not available: {str(e)}")

        models_loaded = True
        last_model_load_time = time.time()
        last_model_version += 1
        model_load_timestamp = time.time()

        logger.info(f"✅ All models loaded successfully on {device}")
        return None, (severity_model, team_model, tokenizer, severity_labels, team_labels)
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def check_and_reload_models():
    """Check for model updates and reload if necessary (for MLOps)."""
    global last_model_load_time
    
    # Debounce: Don't check more than once every 5 seconds
    if time.time() - last_model_load_time < 5:
        return False

    try:
        # Check timestamps for local models
        updates = []
        for path in ['./severity_model_new/model.safetensors', './team_model_new/model.safetensors',
                      './severity_model_new/pytorch_model.bin', './team_model_new/pytorch_model.bin']:
            if os.path.exists(path):
                updates.append(os.path.getmtime(path))
        
        if not updates:
            return False
            
        most_recent = max(updates)
        if most_recent > last_model_load_time + 1:
            logger.info("🔄 Detecting model update, reloading...")
            load_model_and_vectorizer(reload=True)
            return True
            
        return False
    except Exception:
        return False


def predict(description):
    """Predict severity and team for a bug description."""
    global models_loaded, tokenizer, severity_model, team_model, severity_labels, team_labels, device

    if not models_loaded:
        load_model_and_vectorizer()
    else:
        check_and_reload_models()

    try:
        # Tokenize
        inputs = tokenizer(
            description,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        # Severity prediction
        severity = "medium"
        if severity_model:
            with torch.no_grad():
                outputs = severity_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                idx = torch.argmax(probs, dim=1).item()
                severity = severity_labels[idx] if severity_labels and idx < len(severity_labels) else "medium"

        # Team prediction
        team = "Backend"
        if team_model:
            with torch.no_grad():
                outputs = team_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                idx = torch.argmax(probs, dim=1).item()
                team = team_labels[idx] if team_labels and idx < len(team_labels) else "Backend"
        else:
            # Fallback heuristics if no model
            desc_lower = description.lower()
            if any(x in desc_lower for x in ["ui", "frontend", "css", "button", "display"]): 
                team = "Frontend"
            elif any(x in desc_lower for x in ["ios", "android", "mobile", "app"]): 
                team = "Mobile"
            elif any(x in desc_lower for x in ["deploy", "docker", "kubernetes", "ci/cd", "devops"]): 
                team = "DevOps"

        return severity, team

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "medium", "Backend"


def predict_with_confidence(description):
    """Predict with confidence scores."""
    global models_loaded, tokenizer, severity_model, team_model, severity_labels, team_labels, device

    if not models_loaded:
        load_model_and_vectorizer()

    result = {
        'severity': 'medium', 'team': 'Backend',
        'severity_confidence': 0.0, 'team_confidence': 0.0
    }

    try:
        inputs = tokenizer(
            description,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        if severity_model:
            with torch.no_grad():
                outputs = severity_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                conf, idx = torch.max(probs, dim=1)
                result['severity'] = severity_labels[idx.item()]
                result['severity_confidence'] = conf.item()

        if team_model:
            with torch.no_grad():
                outputs = team_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                conf, idx = torch.max(probs, dim=1)
                result['team'] = team_labels[idx.item()]
                result['team_confidence'] = conf.item()

        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return result
