"""
Simple fallback predictions when ML models are not available.
Used for cloud deployment where trained models haven't been uploaded yet.
"""

def simple_severity_prediction(description):
    """Rule-based severity prediction"""
    description_lower = description.lower()
    
    # Critical keywords
    if any(word in description_lower for word in ['crash', 'critical', 'data loss', 'security', 'breach', 'urgent', 'down', 'outage']):
        return "Critical"
    
    # High priority keywords
    if any(word in description_lower for word in ['error', 'bug', 'broken', 'fails', 'not working', 'urgent']):
        return "High"
    
    # Medium priority keywords  
    if any(word in description_lower for word in ['slow', 'performance', 'improvement', 'enhance']):
        return "Medium"
    
    # Default to low
    return "Low"

def simple_team_prediction(description):
    """Rule-based team assignment"""
    description_lower = description.lower()
    
    if any(word in description_lower for word in ['api', 'server', 'backend', 'database', 'query', 'sql']):
        return "Backend"
    
    if any(word in description_lower for word in ['ui', 'frontend', 'button', 'display', 'css', 'layout', 'design']):
        return "Frontend"
    
    if any(word in description_lower for word in ['database', 'db', 'query', 'data', 'sql', 'migration']):
        return "Database"
    
    if any(word in description_lower for word in ['deploy', 'devops', 'ci/cd', 'docker', 'kubernetes', 'infrastructure']):
        return "DevOps"
    
    # Default
    return "Backend"

def predict_with_fallback(description):
    """Try ML predictions, fallback to rules if models not available"""
    try:
        from predict_bug import predict
        severity, team = predict(description)
        return severity, team
    except Exception as e:
        print(f"ML models not available, using rule-based fallback: {e}")
        severity = simple_severity_prediction(description)
        team = simple_team_prediction(description)
        return severity, team
