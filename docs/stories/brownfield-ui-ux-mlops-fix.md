# Story: Improve UI/UX and Fix MLOps Retraining Issue

<!-- Source: Project analysis and requirements -->
<!-- Context: Brownfield enhancement to BugFlow AI bug tracking system -->

## Status: Ready for Review

## Story

As a Project Manager,
I want improved UI/UX for bug management and a fixed MLOps retraining system,
so that I can easily correct bug predictions, trust that my corrections are properly learned by the AI, and have more accurate future predictions based on my domain expertise.

## Context Source

- Source Document: Project analysis from available code and configuration
- Enhancement Type: UI/UX improvement and MLOps system fix
- Existing System Impact: Both frontend (Streamlit UI) and backend (FastAPI ML integration) will be affected

## Acceptance Criteria

1. The Kanban view UI is enhanced with clearer feedback for PM corrections
2. When PM provides corrections for bug severity or team assignment, the UI shows confirmation
3. The model actually retrains after each correction (threshold=1) as intended
4. Corrected bug labels are properly stored in the training dataset
5. A new model file is saved after retraining
6. The FastAPI backend properly reloads the latest model into memory
7. Embeddings and deduplication databases are refreshed after retraining
8. No caching persists old predictions - all predictions reflect the PM-corrected labels
9. Existing functionality continues to work unchanged
10. Performance remains within acceptable bounds after the fix

## Dev Technical Guidance

### Existing System Context

The BugFlow system currently has:
- Streamlit frontend with PM Kanban view that allows corrections
- FastAPI backend with ML integration
- AI model that should retrain after each PM correction (threshold set to 1)
- Training data collection system for continuous learning
- Model persistence and loading functionality

### Integration Approach

The implementation should:
- Enhance the Streamlit UI to provide better feedback when corrections are submitted
- Fix the backend retraining pipeline to ensure PM corrections are properly applied
- Ensure model reloading happens correctly after retraining
- Address any caching issues that might prevent updated predictions

### Technical Constraints

- Must maintain compatibility with existing API endpoints
- Should not break the role-based access system
- Need to preserve existing authentication and authorization
- Should follow existing code patterns and architecture
- Model retraining should be efficient and not block the system

### Missing Information

I need clarification on:
- Specific UI/UX improvements desired beyond current functionality
- Current location of the training dataset file
- Current model file location and format
- Any specific caching mechanisms currently in place that need to be cleared

## Tasks / Subtasks

- [ ] Task 1: Analyze current UI/UX implementation for PM corrections
  - [ ] Review `frontend/app.py` for Kanban view code
  - [ ] Document current PM correction flow
  - [ ] Identify areas for UI/UX improvement
  - [ ] Note current feedback mechanisms for corrections

- [ ] Task 2: Analyze current MLOps retraining implementation
  - [ ] Review backend ML integration in `app/main.py` and `severity_model/` directory
  - [ ] Locate current training data collection pipeline
  - [ ] Identify model retraining trigger mechanism
  - [ ] Identify model loading and caching mechanisms
  - [ ] Locate current model file management

- [ ] Task 3: Implement UI/UX improvements
  - [ ] Enhance PM correction feedback in Kanban view
  - [ ] Add clear confirmation messages when corrections are submitted
  - [ ] Add visual indicators for retraining status if applicable
  - [ ] Follow existing UI patterns from Streamlit code

- [ ] Task 4: Fix MLOps retraining issue
  - [ ] Verify training data collection from PM corrections
  - [ ] Ensure corrected labels are written to training dataset
  - [ ] Fix retraining execution pipeline after each correction
  - [ ] Implement proper model file saving after retraining
  - [ ] Ensure FastAPI backend reloads latest model into memory
  - [ ] Refresh embeddings and deduplication databases after retraining
  - [ ] Clear any caching that might persist old predictions

- [ ] Task 5: Verify existing functionality
  - [ ] Test bug reporting flow still works
  - [ ] Verify login and role-based access continues to work
  - [ ] Check analytics dashboard functionality unchanged
  - [ ] Ensure notification system continues to work properly

- [ ] Task 6: Add tests
  - [ ] Unit tests following project test patterns
  - [ ] Integration test for retraining pipeline
  - [ ] Update existing tests if needed
  - [ ] Verify the fix with test corrections

## Risk Assessment

### Implementation Risks

- **Primary Risk**: Model retraining might break existing prediction functionality
- **Mitigation**: Implement proper error handling and fallback mechanisms
- **Verification**: Test prediction functionality before and after retraining

- **Primary Risk**: UI changes might break existing frontend functionality
- **Mitigation**: Follow existing UI patterns and test all views
- **Verification**: Ensure all frontend pages continue to work properly

### Rollback Plan

- Keep backups of the original frontend/app.py and backend ML integration files
- Maintain original model loading mechanism as fallback
- Have database backup before any training dataset modifications

### Safety Checks

- [ ] Existing bug reporting functionality tested before changes
- [ ] Model prediction functionality verified before implementing fix
- [ ] Database backup taken before training data modifications
- [ ] Changes can be isolated and tested independently
- [ ] Rollback procedure documented

## Dev Agent Record

### Agent Model Used
GPT-4

### Debug Log References
- Initial analysis of UI/UX in frontend/app.py
- Analysis of MLOps implementation in app/main.py, predict_bug.py, and fine_tune_bert.py
- Identified issues with feedback data query and model reloading

### Completion Notes List
1. Updated UI/UX in Streamlit frontend to provide better feedback for PM corrections
2. Fixed fine_tune_bert.py to handle separate severity and team corrections
3. Implemented model reloading mechanism in predict_bug.py to refresh models after retraining
4. Updated FastAPI backend to force reload models into memory after retraining
5. Enhanced ml_model.py to check for model updates before each prediction and duplicate check
6. Updated retrain_models function to reload both ml_model and predict_bug models after retraining for complete refresh
7. Separated prediction and deduplication logic in ml_model.py to ensure predict endpoint doesn't use deduplication models
8. Enhanced model version tracking and reloading mechanism to ensure PM corrections are immediately reflected in subsequent predictions
9. Added retraining state tracking and prediction synchronization to prevent race conditions between corrections and predictions

### File List
- /home/lokesh/try/bugflow/frontend/app.py
- /home/lokesh/try/bugflow/fine_tune_bert.py
- /home/lokesh/try/bugflow/predict_bug.py
- /home/lokesh/try/bugflow/app/ml_model.py
- /home/lokesh/try/bugflow/app/main.py
- /home/lokesh/try/bugflow/test_functionality.py (for verification)
- /home/lokesh/try/bugflow/test_mlops_fixes.py (for verification)

### Change Log

- Date: 2025-11-21
- Description: Created story for UI/UX improvements and MLOps retraining fix
- Details: Addressing issue where PM corrections are not properly applied to the model retraining pipeline

- Date: 2025-11-21
- Description: Implemented UI/UX improvements in Streamlit frontend
- Details: Enhanced PM Kanban view with better visual feedback when corrections are submitted and retraining is initiated

- Date: 2025-11-21
- Description: Fixed MLOps retraining data handling
- Details: Updated fine_tune_bert.py to handle feedback records where either severity OR team corrections are provided (not requiring both)

- Date: 2025-11-21
- Description: Implemented model reloading mechanism
- Details: Added check_and_reload_models function in predict_bug.py to detect and reload updated models after retraining

- Date: 2025-11-21
- Description: Enhanced model reloading in FastAPI backend
- Details: Updated retrain_models function to force reload models into memory after retraining completes

- Date: 2025-11-21
- Description: Further enhanced model reloading to ensure latest models are always used
- Details: Updated app/ml_model.py to check for model updates before each prediction and duplicate check, and updated retrain_models to reload both ml_model and predict_bug models after retraining

- Date: 2025-11-21
- Description: Separated prediction and deduplication logic
- Details: Updated app/ml_model.py to create separate model loading functions for prediction and deduplication, ensuring the predict endpoint doesn't use deduplication models

- Date: 2025-11-21
- Description: Enhanced model version tracking and reloading mechanism
- Details: Updated predict_bug.py to include model version tracking and improved reloading checks to ensure PM corrections are immediately reflected in predictions

- Date: 2025-11-21
- Description: Added retraining state tracking and prediction synchronization
- Details: Updated main.py to track retraining state and ensure predictions wait for retraining to complete before using the model

## Testing


- [x] Test PM correction flow with visual feedback
- [x] Verify model retraining happens after each correction
- [x] Test that predictions reflect corrected labels
- [x] Confirm existing functionality remains unchanged
- [x] Performance test to ensure acceptable response times