# Deploy BugFlow with Your Trained Models (96.7% Accuracy)

## Problem
Your trained models are too large for GitHub (4.36 GB), but you want to deploy them to the cloud.

## Solution: Google Drive Storage

### Step 1: Upload Models to Google Drive

1. **Create a folder in Google Drive** called `bugflow-models`

2. **Upload these folders:**
   ```
   bugflow-models/
   ├── severity_model_new/     (your 96.7% accuracy model)
   ├── team_model_90plus/      (your 95.3% accuracy model)
   └── dedup_model/            (your deduplication model)
   ```

3. **Make the folder public:**
   - Right-click folder → Share
   - Change to "Anyone with the link"
   - Copy the sharing link

### Step 2: Extract Folder ID from Link

Your link looks like:
```
https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9
```

The ID is: `1a2b3c4d5e6f7g8h9`

### Step 3: Update download_models.py

Edit `/home/lokesh/try/bugflow/download_models.py`:

```python
#!/usr/bin/env python3
"""
Download trained BugFlow models from Google Drive
"""
import gdown
import os

# Your Google Drive folder ID (get from sharing link)
MODELS_FOLDER_ID = "YOUR_FOLDER_ID_HERE"

def download_trained_models():
    """Download pre-trained models from Google Drive"""
    print("📥 Downloading trained models from Google Drive...")
    
    # Download entire folder
    url = f"https://drive.google.com/drive/folders/{MODELS_FOLDER_ID}"
    gdown.download_folder(url, quiet=False, use_cookies=False)
    
    print("✅ Trained models downloaded successfully!")
    print("   - Severity: 96.7% accuracy")
    print("   - Team: 95.3% accuracy") 
    print("   - Deduplication: Ready")

if __name__ == "__main__":
    download_trained_models()
```

### Step 4: Update requirements.txt

Add to `requirements.txt`:
```
gdown>=4.7.1
```

### Step 5: Push Code to GitHub

```bash
cd /home/lokesh/try/bugflow

# Add only code files
git add download_models.py requirements.txt
git commit -m "Add trained model download from Google Drive"

# Push (should be fast now - no large files!)
git push -u origin main
```

### Step 6: Deploy to Render

1. Go to Render.com → Your Backend Service
2. Update Build Command:
   ```
   pip install -r requirements.txt && python download_models.py
   ```
3. This will download your trained models on first deployment!

---

## Alternative: Quick Command-Line Upload

If you don't want to use Google Drive UI:

```bash
# Install rclone
sudo apt install rclone

# Configure Google Drive
rclone config

# Upload models
cd /home/lokesh/try/bugflow
rclone copy severity_model_new gdrive:bugflow-models/severity_model_new
rclone copy team_model_90plus gdrive:bugflow-models/team_model_90plus  
rclone copy dedup_model gdrive:bugflow-models/dedup_model
```

---

## For Right Now: Push Code Only

```bash
# Current status: Code changes staged, models in .gitignore
git push -u origin main
```

This should work because:
- ✅ Models are in .gitignore (won't be tracked)
- ✅ Only code is staged (~20 MB)
- ✅ Models will be downloaded on deployment

---

## Summary

| What | Where | Size |
|------|-------|------|
| **Code** | GitHub | ~20 MB ✅ |
| **Models** | Google Drive | 4.36 GB ✅ |
| **Deployment** | Render downloads from Drive | Automatic |

**Your 96.7% accuracy models WILL be deployed** - just stored separately! 🎉
