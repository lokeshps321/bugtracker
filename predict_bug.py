from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
import logging
import os
import time

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

def load_model_and_vectorizer(reload=False):
    """
    Load all ML models, tokenizer, labels, and deduplication model.
    Moves models to GPU if available for faster inference.
    """
    global severity_model, team_model, tokenizer, severity_labels, team_labels, dedup_model, models_loaded, last_model_load_time, last_model_version, model_load_timestamp, device

    try:
        logger.info(f"Loading models (reload={reload}) on device: {device}...")
        
        # Initialize tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # Load severity model
        try:
            if os.path.exists('./severity_model_new'):
                severity_model = DistilBertForSequenceClassification.from_pretrained('./severity_model_new')
                logger.info("Loaded FINE-TUNED severity model from ./severity_model_new")
            else:
                severity_model = DistilBertForSequenceClassification.from_pretrained('./severity_model')
                logger.info("Loaded old severity model from ./severity_model")
            
            severity_model.to(device)
            severity_model.eval()
            
            # Load labels
            try:
                severity_labels = pd.read_json('severity_labels.json', typ='series').tolist()
            except Exception:
                severity_labels = ['low', 'medium', 'high', 'critical']
        except Exception as e:
            logger.error(f"Failed to load severity model: {str(e)}")
            raise

        # Load team model
        team_model = None
        team_labels = ['Backend', 'Frontend', 'Mobile', 'DevOps']
        try:
            if os.path.exists('./team_model_new'):
                team_model = DistilBertForSequenceClassification.from_pretrained('./team_model_new')
            elif os.path.exists('./team_model'):
                team_model = DistilBertForSequenceClassification.from_pretrained('./team_model')
            
            if team_model:
                team_model.to(device)
                team_model.eval()
                try:
                    team_labels = pd.read_json('team_labels.json', typ='series').tolist()
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Team model not found: {str(e)}")

        # Load deduplication model
        try:
            dedup_model = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))
            logger.info("Deduplication model loaded")
        except Exception as e:
            logger.warning(f"Dedup model not available: {str(e)}")

        models_loaded = True
        last_model_load_time = time.time()
        last_model_version += 1
        model_load_timestamp = time.time()

        logger.info(f"Models loaded successfully on {device}")
        return None, (severity_model, team_model, tokenizer, severity_labels, team_labels)
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def check_and_reload_models():
    """Check for model updates and reload if necessary."""
    global last_model_load_time
    
    # Debounce: Don't check more than once every 5 seconds
    if time.time() - last_model_load_time < 5:
        return False

    try:
        # Check timestamps (simplified for performance)
        updates = []
        for path in ['./severity_model_new/pytorch_model.bin', './team_model_new/pytorch_model.bin']:
            if os.path.exists(path):
                updates.append(os.path.getmtime(path))
        
        if not updates:
            return False
            
        most_recent = max(updates)
        if most_recent > last_model_load_time + 1:
            logger.info("Detecting model update, reloading...")
            load_model_and_vectorizer(reload=True)
            return True
            
        return False
    except Exception:
        return False

def predict(description):
    """Predict severity and team using GPU acceleration."""
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

        # Severity
        severity = "medium"
        if severity_model:
            with torch.no_grad():
                outputs = severity_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                idx = torch.argmax(probs, dim=1).item()
                severity = severity_labels[idx] if severity_labels and idx < len(severity_labels) else "medium"

        # Team
        team = "backend"
        if team_model:
            with torch.no_grad():
                outputs = team_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                idx = torch.argmax(probs, dim=1).item()
                team = team_labels[idx] if team_labels and idx < len(team_labels) else "backend"
        else:
            # Fallback heuristics
            desc_lower = description.lower()
            if any(x in desc_lower for x in ["ui", "frontend", "css", "button"]): team = "frontend"
            elif any(x in desc_lower for x in ["db", "database", "sql"]): team = "database"
            elif any(x in desc_lower for x in ["ios", "android", "mobile"]): team = "mobile"

        return severity, team

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "medium", "backend"

def predict_with_confidence(description):
    """Predict with confidence scores on GPU."""
    global models_loaded, tokenizer, severity_model, team_model, severity_labels, team_labels, device

    if not models_loaded:
        load_model_and_vectorizer()

    result = {
        'severity': 'medium', 'team': 'backend',
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
