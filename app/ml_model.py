"""
BugFlow ML Model Module - Cloud Optimized

Memory-optimized for Render's 512MB free tier.
Deduplication is disabled on cloud to save memory.
"""

from sklearn.metrics.pairwise import cosine_similarity
from app import models
from sqlalchemy.orm import Session
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# Cloud mode detection - skip heavy models to fit in 512MB
CLOUD_MODE = os.getenv('HF_SPACE_URL') is not None or os.getenv('RENDER', '') == 'true'

# Log cloud mode status
if CLOUD_MODE:
    logger.info("☁️  Cloud mode enabled - deduplication disabled to save memory")

# Global variables
dedup_model = None
vectorizer = None
ml_model = None
models_loaded = False
dedup_models_loaded = False


def load_prediction_model():
    """
    Load only ML models for prediction (without deduplication model).
    On cloud, this is a no-op since predictions go to HF Space.
    """
    global vectorizer, ml_model, models_loaded
    
    if CLOUD_MODE:
        logger.info("Cloud mode: Predictions handled by HF Space")
        models_loaded = True
        return
    
    try:
        import predict_bug
        vectorizer, ml_model = predict_bug.load_model_and_vectorizer()
        models_loaded = True
        logger.info("Prediction models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading prediction models: {str(e)}")
        raise


def load_dedup_model():
    """
    Load deduplication model.
    SKIPPED on cloud to save memory (512MB limit on Render).
    """
    global dedup_model, dedup_models_loaded
    
    # Skip on cloud - dedup model is too large for 512MB Render
    if CLOUD_MODE:
        logger.info("Cloud mode: Skipping deduplication model to save memory")
        dedup_models_loaded = True
        return
    
    try:
        # Only import SentenceTransformer when NOT in cloud mode
        from sentence_transformers import SentenceTransformer
        dedup_model = SentenceTransformer('all-MiniLM-L6-v2')
        dedup_models_loaded = True
        logger.info("Deduplication model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading deduplication model: {str(e)}")
        raise


def load_model():
    """
    Load both prediction and deduplication models.
    On cloud, skips heavy models to save memory.
    """
    load_prediction_model()
    load_dedup_model()


def predict(description: str):
    """
    Predict severity and team for a bug description.
    On cloud, delegates to HF Space via predict_bug module.
    """
    import predict_bug
    return predict_bug.predict(description)


def check_duplicate(description: str, db: Session):
    """
    Check if a bug description is a duplicate.
    Returns the existing bug object if found, else None.
    On cloud, calls HF Space API for deduplication.
    """
    global dedup_model, dedup_models_loaded
    
    try:
        bugs = db.query(models.Bug).all()
        if not bugs:
            return None
        
        descriptions = [bug.description for bug in bugs]
        
        # On cloud, call HF Space API for deduplication
        if CLOUD_MODE:
            import requests
            HF_SPACE_URL = os.getenv('HF_SPACE_URL', 'https://loke007-bugflow-inference.hf.space')
            try:
                response = requests.post(
                    f"{HF_SPACE_URL}/check_duplicate",
                    json={
                        "description": description,
                        "existing_descriptions": descriptions
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    if result.get("is_duplicate", False):
                        duplicate_index = result.get("duplicate_index", -1)
                        if 0 <= duplicate_index < len(bugs):
                            logger.info(f"Duplicate found via HF Space (similarity: {result.get('similarity_score', 0):.2f})")
                            return bugs[duplicate_index]
                return None
            except Exception as e:
                logger.error(f"HF Space deduplication call failed: {str(e)}")
                return None
        
        # Local mode: use local model
        if not dedup_models_loaded:
            load_dedup_model()
        
        if dedup_model is None:
            return None
        
        embeddings = dedup_model.encode([description] + descriptions)
        similarities = cosine_similarity([embeddings[0]], embeddings[1:])[0]

        max_similarity = np.max(similarities)
        if max_similarity > 0.9:  # Threshold for duplicate
            index = np.argmax(similarities)
            return bugs[index]

        return None

    except Exception as e:
        logger.error(f"Deduplication error: {str(e)}")
        return None
