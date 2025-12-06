# Deploy BugFlow to Hugging Face Spaces (5 minutes)

Your backend is ready - just needs to be pushed to Hugging Face!

## Step 1: Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Owner**: loke007 (your account)
   - **Space name**: bugflow-inference
   - **License**: Apache 2.0
   - **SDK**: Docker
   - **Hardware**: CPU basic (FREE)
3. Click **Create Space**

## Step 2: Push Your Code

```bash
# Clone the empty space
cd /tmp
git clone https://huggingface.co/spaces/loke007/bugflow-inference
cd bugflow-inference

# Copy your files
cp ~/try/bugflow/hf_space/* .

# Push to HF
git add .
git commit -m "Initial deployment"
git push
```

## Step 3: Wait for Build

- HF will build your Docker container (2-3 minutes)
- Once green, your API is live at: `https://loke007-bugflow-inference.hf.space`

## Step 4: Update Streamlit Secrets

Go to Streamlit Cloud → Your App → Settings → Secrets:

```toml
API_URL = "https://loke007-bugflow-inference.hf.space"
```

## Why HF Spaces is Better

| Feature | Render Free | HF Spaces |
|---------|-------------|-----------|
| Cold start | 30-60 seconds | ~5 seconds |
| Sleeps after | 15 min | Never (free tier) |
| RAM | 512MB | 16GB |
| GPU available | No | Yes (paid) |
