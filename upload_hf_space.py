"""
Upload app.py to HuggingFace Space using API
"""
from huggingface_hub import HfApi, login
import os

# Get token from .env file
token = None
try:
    with open(".env", "r") as f:
        for line in f:
            if "hf_" in line:
                # Extract token
                import re
                match = re.search(r'hf_[a-zA-Z0-9]+', line)
                if match:
                    token = match.group(0)
                    break
except:
    pass

if not token:
    print("❌ No token found in .env")
    exit(1)

print(f"Found token: {token[:10]}...")
login(token=token)

api = HfApi()

# Upload the app.py file
print("📤 Uploading app.py to HuggingFace Space...")
try:
    api.upload_file(
        path_or_fileobj="hf_space/app.py",
        path_in_repo="app.py",
        repo_id="loke007/bugflow-inference",
        repo_type="space",
        commit_message="Add keyword boost for DevOps/Mobile predictions"
    )
    print("✅ Successfully uploaded app.py to HuggingFace Space!")
    print("🚀 Space will rebuild with keyword boost for DevOps/Mobile")
except Exception as e:
    print(f"❌ Upload failed: {e}")
