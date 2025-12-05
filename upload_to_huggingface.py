#!/usr/bin/env python3
"""
Upload BugFlow fine-tuned models to Hugging Face Hub.

Usage:
    1. Create account at https://huggingface.co/join
    2. Get token at https://huggingface.co/settings/tokens (with write access)
    3. Run: python upload_to_huggingface.py --token YOUR_HF_TOKEN --username YOUR_USERNAME

This will create two public model repositories:
    - YOUR_USERNAME/bugflow-severity-classifier
    - YOUR_USERNAME/bugflow-team-classifier
"""

import os
import shutil
import argparse
from pathlib import Path

def create_clean_model_dir(source_dir, dest_dir):
    """Copy only essential model files (no training checkpoints)."""
    essential_files = [
        'config.json',
        'model.safetensors',
        'pytorch_model.bin',  # Alternative format
        'tokenizer_config.json',
        'vocab.json',
        'vocab.txt',
        'merges.txt',
        'special_tokens_map.json',
        'severity_labels.json',
        'team_labels.json',
        'label_map.json',
    ]
    
    os.makedirs(dest_dir, exist_ok=True)
    
    copied = []
    for f in essential_files:
        src = os.path.join(source_dir, f)
        if os.path.exists(src):
            dst = os.path.join(dest_dir, f)
            shutil.copy2(src, dst)
            copied.append(f)
            print(f"  ✓ Copied {f}")
    
    return copied

def create_model_card(model_type, model_dir):
    """Create README.md (model card) for Hugging Face."""
    if model_type == "severity":
        content = """---
language: en
tags:
- bug-classification
- severity-classification
- software-engineering
license: apache-2.0
datasets:
- custom-github-issues
pipeline_tag: text-classification
---

# BugFlow Severity Classifier

Fine-tuned CodeBERT model for classifying bug report severity levels.

## Labels
- **Low**: Minor issues, cosmetic changes
- **Medium**: Standard bugs affecting some functionality
- **High**: Important bugs affecting major functionality  
- **Critical**: System crashes, data loss, security issues

## Usage

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

model = RobertaForSequenceClassification.from_pretrained("YOUR_USERNAME/bugflow-severity-classifier")
tokenizer = RobertaTokenizer.from_pretrained("YOUR_USERNAME/bugflow-severity-classifier")

text = "Application crashes when clicking login button"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=1)
labels = ['low', 'medium', 'high', 'critical']
predicted = labels[torch.argmax(probs).item()]
print(f"Severity: {predicted}")
```

## Training
- Base model: microsoft/codebert-base
- Dataset: Custom GitHub issues dataset + domain-specific bugs
- Fine-tuned using Hugging Face Transformers
"""
    else:  # team
        content = """---
language: en
tags:
- bug-classification
- team-assignment
- software-engineering
license: apache-2.0
datasets:
- custom-github-issues
pipeline_tag: text-classification
---

# BugFlow Team Classifier

Fine-tuned CodeBERT model for assigning bugs to the appropriate development team.

## Labels
- **Frontend**: UI, CSS, layout, display issues
- **Backend**: API, server, database, logic issues
- **Mobile**: iOS, Android, mobile app issues
- **DevOps**: Deployment, CI/CD, infrastructure issues

## Usage

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

model = RobertaForSequenceClassification.from_pretrained("YOUR_USERNAME/bugflow-team-classifier")
tokenizer = RobertaTokenizer.from_pretrained("YOUR_USERNAME/bugflow-team-classifier")

text = "Button not responding on click"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=1)
labels = ['Backend', 'Frontend', 'Mobile', 'DevOps']
predicted = labels[torch.argmax(probs).item()]
print(f"Team: {predicted}")
```

## Training
- Base model: microsoft/codebert-base
- Dataset: Custom GitHub issues dataset + domain-specific bugs
- Fine-tuned using Hugging Face Transformers
"""
    
    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(content)
    print(f"  ✓ Created README.md")


def upload_to_hub(model_dir, repo_name, token):
    """Upload model directory to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
        
        api = HfApi()
        
        # Create repo (will fail silently if exists)
        try:
            create_repo(repo_name, token=token, exist_ok=True)
            print(f"  ✓ Repository created/verified: {repo_name}")
        except Exception as e:
            print(f"  ⚠ Repo might already exist: {e}")
        
        # Upload all files
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_name,
            token=token,
            commit_message="Upload BugFlow fine-tuned model"
        )
        print(f"  ✓ Uploaded to https://huggingface.co/{repo_name}")
        return True
        
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"ERROR uploading: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload BugFlow models to Hugging Face Hub")
    parser.add_argument("--token", required=True, help="Hugging Face API token")
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--skip-upload", action="store_true", help="Only prepare files, skip upload")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    temp_dir = base_dir / "models_for_deploy"
    
    print("\n" + "="*60)
    print("🚀 BugFlow Model Upload to Hugging Face Hub")
    print("="*60)
    
    # Prepare severity model
    print("\n📦 Preparing severity classifier...")
    severity_source = base_dir / "severity_model_new"
    severity_dest = temp_dir / "severity"
    
    if not severity_source.exists():
        print(f"  ❌ Source not found: {severity_source}")
        return
    
    create_clean_model_dir(severity_source, severity_dest)
    create_model_card("severity", severity_dest)
    
    # Prepare team model
    print("\n📦 Preparing team classifier...")
    team_source = base_dir / "team_model_new"
    team_dest = temp_dir / "team"
    
    if not team_source.exists():
        print(f"  ❌ Source not found: {team_source}")
        return
    
    create_clean_model_dir(team_source, team_dest)
    create_model_card("team", team_dest)
    
    if args.skip_upload:
        print("\n✅ Files prepared in models_for_deploy/")
        print("   Run without --skip-upload to upload to Hugging Face Hub")
        return
    
    # Upload to Hub
    print("\n☁️  Uploading severity classifier...")
    severity_repo = f"{args.username}/bugflow-severity-classifier"
    upload_to_hub(severity_dest, severity_repo, args.token)
    
    print("\n☁️  Uploading team classifier...")
    team_repo = f"{args.username}/bugflow-team-classifier"
    upload_to_hub(team_dest, team_repo, args.token)
    
    print("\n" + "="*60)
    print("✅ DONE! Your models are now on Hugging Face Hub!")
    print("="*60)
    print(f"\nSeverity: https://huggingface.co/{severity_repo}")
    print(f"Team:     https://huggingface.co/{team_repo}")
    print("\n📝 Next steps:")
    print("   1. Set these env vars on Render.com:")
    print(f"      HF_SEVERITY_MODEL={severity_repo}")
    print(f"      HF_TEAM_MODEL={team_repo}")
    print("   2. Redeploy your backend")

if __name__ == "__main__":
    main()
