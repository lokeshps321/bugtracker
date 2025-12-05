"""
BugFlow ML Prediction Module - Cloud Version

Uses Hugging Face Space for inference (16GB RAM)
Falls back to rule-based predictions if Space unavailable
"""

import os
import logging
import requests
from typing import Tuple

logger = logging.getLogger(__name__)

# Hugging Face Space URL for ML inference
HF_SPACE_URL = os.getenv('HF_SPACE_URL', 'https://loke007-bugflow-inference.hf.space')

def predict(description: str) -> Tuple[str, str]:
    """
    Predict severity and team for a bug description.
    Uses Hugging Face Space for inference, falls back to rules if unavailable.
    """
    try:
        response = requests.post(
            f"{HF_SPACE_URL}/predict",
            json={"description": description, "title": ""},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Prediction from HF Space: {data}")
            return data.get('severity', 'medium'), data.get('team', 'Backend')
        else:
            logger.warning(f"HF Space returned {response.status_code}, using fallback")
            return _fallback_predict(description)
            
    except requests.exceptions.RequestException as e:
        logger.warning(f"HF Space unavailable: {e}, using fallback")
        return _fallback_predict(description)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return _fallback_predict(description)


def predict_with_confidence(description: str) -> dict:
    """Predict with confidence scores."""
    try:
        response = requests.post(
            f"{HF_SPACE_URL}/predict",
            json={"description": description, "title": ""},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            severity, team = _fallback_predict(description)
            return {
                'severity': severity,
                'team': team,
                'severity_confidence': 0.5,
                'team_confidence': 0.5
            }
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        severity, team = _fallback_predict(description)
        return {
            'severity': severity,
            'team': team,
            'severity_confidence': 0.5,
            'team_confidence': 0.5
        }


def _fallback_predict(description: str) -> Tuple[str, str]:
    """Rule-based fallback predictions when HF Space is unavailable."""
    desc_lower = description.lower()
    
    # Severity prediction
    if any(word in desc_lower for word in ['crash', 'critical', 'data loss', 'security', 'breach', 'down', 'outage']):
        severity = "critical"
    elif any(word in desc_lower for word in ['error', 'bug', 'broken', 'fails', 'not working', 'urgent']):
        severity = "high"
    elif any(word in desc_lower for word in ['slow', 'performance', 'improvement', 'enhance']):
        severity = "medium"
    else:
        severity = "low"
    
    # Team prediction
    if any(word in desc_lower for word in ['ui', 'frontend', 'button', 'display', 'css', 'layout', 'design', 'page']):
        team = "Frontend"
    elif any(word in desc_lower for word in ['ios', 'android', 'mobile', 'app store', 'phone']):
        team = "Mobile"
    elif any(word in desc_lower for word in ['deploy', 'devops', 'ci/cd', 'docker', 'kubernetes', 'infrastructure', 'server']):
        team = "DevOps"
    else:
        team = "Backend"
    
    logger.info(f"Using fallback prediction: severity={severity}, team={team}")
    return severity, team


def load_model_and_vectorizer(reload=False):
    """Compatibility function - no-op in cloud mode."""
    logger.info("Cloud mode: Using HF Space for predictions")
    return None, None


def check_and_reload_models():
    """Compatibility function - no-op in cloud mode."""
    return False
