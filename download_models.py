#!/usr/bin/env python3
"""
Download pre-trained ML models for BugFlow deployment.
Run this script on first deployment to download models.
"""
import os
import gdown  # pip install gdown

# Google Drive file IDs (you'll need to upload models to Google Drive)
# For now, we'll use the base models from HuggingFace

MODEL_URLS = {
    "severity": "distilbert-base-uncased",
    "team": "distilbert-base-uncased",
}

def download_base_models():
    """Download base DistilBERT models from HuggingFace"""
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    
    print("📥 Downloading base models from HuggingFace...")
    
    # Create model directories
    os.makedirs("severity_model", exist_ok=True)
    os.makedirs("team_model", exist_ok=True)
    
    # Download severity model (we'll use base model, fine-tune on first run)
    print("  ✓ Severity model...")
    severity_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4  # low, medium, high, critical
    )
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    severity_model.save_pretrained("severity_model")
    tokenizer.save_pretrained("severity_model")
    
    # Download team model
    print("  ✓ Team model...")
    team_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4  # Backend, Frontend, Mobile, DevOps
    )
    team_model.save_pretrained("team_model")
    tokenizer.save_pretrained("team_model")
    
    print("✅ Base models downloaded successfully!")
    print("\n⚠️  Note: These are base models.")
    print("   Fine-tune them with your data for better accuracy.")

if __name__ == "__main__":
    download_base_models()
