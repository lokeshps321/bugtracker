# Quick Cloud Deployment Commands

## Step 1: Push to GitHub
```bash
cd /home/lokesh/try/bugflow

# Add all files
git add .
git commit -m "Ready for cloud deployment"

# Push to GitHub (create repo first at github.com/new)
git remote add origin https://github.com/lokeshps321/bugflow.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy Backend (Render.com)
1. Go to https://render.com/dashboard
2. Click "New +" → "Web Service"
3. Connect GitHub → Select `bugflow` repo
4. Configure:
   - **Name**: bugflow-backend
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
5. Add PostgreSQL:
   - Click "New +" → "PostgreSQL"
   - **Name**: bugflow-db
   - **Plan**: Free
6. Link database to backend (copy DATABASE_URL)

## Step 3: Deploy Frontend (Streamlit Cloud)
1. Go to https://share.streamlit.io
2. Click "New app"
3. Configure:
   - **Repository**: lokeshps321/bugflow
   - **Branch**: main
   - **Main file**: frontend/app.py
4. Add secret (Advanced settings → Secrets):
   ```
   API_URL = "https://your-backend.onrender.com"
   ```
5. Deploy!

## Step 4: Initialize Database
1. Go to Render → Backend → Shell
2. Run:
   ```bash
   python init_users.py
   python create_sample_data.py
   ```

## Done! 🎉
Your app: https://bugflow.streamlit.app
