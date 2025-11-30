# ML Models Storage Strategy for Cloud Deployment

## Problem
ML model files are too large for GitHub (4+ GB total):
- `severity_model_new/` - ~500 MB
- `team_model_90plus/` - ~500 MB  
- `dedup_model/` - ~400 MB
- `datasets/` - ~3 GB

## Solutions

### Option 1: Base Models Only (Fastest Deploy)
**Use untrained base models from HuggingFace**

**Pros:**
- ✅ Small size (~500 MB total)
- ✅ Fast deployment
- ✅ No storage costs

**Cons:**
- ❌ Lower accuracy (need to fine-tune on production data)

**Implementation:**
1. Push code without models
2. On Render deployment, run `download_models.py`
3. Models downloaded from HuggingFace on first run

### Option 2: Google Drive Storage (Recommended)
**Store trained models in Google Drive, download on deployment**

**Steps:**
1. Upload models to Google Drive (make public or shareable link)
2. Get shareable link for each model folder
3. Use `gdown` to download during deployment

**Example:**
```python
import gdown
gdown.download_folder(
    "https://drive.google.com/drive/folders/YOUR_FOLDER_ID",
    quiet=False
)
```

### Option 3: Cloud Storage (Production)
**Use AWS S3, Google Cloud Storage, or Azure Blob**

**Pros:**
- ✅ Professional solution
- ✅ Versioning
- ✅ Fast downloads
- ✅ Can serve models via CDN

**Cons:**
- ❌ Costs money (but very cheap ~$0.50/month)

**Free tier options:**
- AWS S3: 5GB free for 1 year
- Google Cloud: 5GB free forever
- Cloudflare R2: 10GB free forever

### Option 4: Git LFS (Large File Storage)
**GitHub's built-in large file solution**

**Pros:**
- ✅ Integrated with GitHub
- ✅ Version control

**Cons:**
- ❌ Limited free tier (1GB storage, 1GB bandwidth)
- ❌ Need to pay after free tier

**Setup:**
```bash
git lfs install
git lfs track "*.bin"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Track large files with LFS"
```

## Recommended Approach for Free Deployment

**Use Option 1 for now:**
1. Deploy with base models
2. Add fine-tuning script that runs on first deployment
3. Store corrected predictions in PostgreSQL
4. Retrain models periodically on production data

**Implementation:**
```bash
# 1. Remove models from git
git rm -r --cached *_model* datasets/

# 2. Update .gitignore to exclude models
# (already done)

# 3. Commit and push (small size now)
git add .
git commit -m "Remove large model files"
git push origin main

# 4. On Render deployment, add build command:
pip install -r requirements.txt && python download_models.py
```

## Current Setup (After This Fix)

**Git will contain:**
- ✅ Code (~10 MB)
- ✅ Training scripts
- ✅ Download script
- ❌ NO large model files
- ❌ NO datasets

**Total repo size:** ~15 MB (easily pushable to GitHub)

**Models will be:**
- Downloaded from HuggingFace on deployment
- OR uploaded to Google Drive and downloaded
- OR fine-tuned on production data
