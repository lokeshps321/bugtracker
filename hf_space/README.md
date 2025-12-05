---
title: BugFlow ML Inference
emoji: 🐛
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
app_port: 7860
---

# BugFlow ML Inference API

This Space hosts the ML inference API for BugFlow, a smart bug tracking system with AI-powered severity classification and team assignment.

## Models Used
- **Severity Classifier**: [loke007/bugflow-severity-classifier](https://huggingface.co/loke007/bugflow-severity-classifier)
- **Team Classifier**: [loke007/bugflow-team-classifier](https://huggingface.co/loke007/bugflow-team-classifier)

## API Endpoints

### POST /predict
Predict severity and team for a bug description.

**Request:**
```json
{
  "description": "App crashes when clicking login button",
  "title": "Login Crash Bug"
}
```

**Response:**
```json
{
  "severity": "high",
  "team": "Frontend",
  "severity_confidence": 0.92,
  "team_confidence": 0.87
}
```

### GET /health
Health check endpoint.
