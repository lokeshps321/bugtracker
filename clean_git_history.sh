#!/bin/bash
# Clean Git History and Start Fresh for Cloud Deployment

echo "🧹 Cleaning Git History - Removing Large Model Files"
echo "======================================================"

# WARNING: This will rewrite git history!
read -p "This will create a fresh repository. Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Step 1: Remove current git
echo "📦 Step 1: Backing up current code..."
cd ..
tar -czf bugflow-backup-$(date +%Y%m%d-%H%M%S).tar.gz bugflow/ --exclude='bugflow/.git' --exclude='bugflow/venv'
echo "✅ Backup created"

# Step 2: Remove .git folder and start fresh
echo "📦 Step 2: Removing old git history..."
cd bugflow
rm -rf .git

# Step 3: Initialize fresh git repository
echo "📦 Step 3: Initializing fresh repository..."
git init
git add .gitignore
git commit -m "Initial commit - gitignore"

# Step 4: Add all CODE files (models excluded by .gitignore)
echo "📦 Step 4: Adding code files..."
git add *.py *.md *.txt *.sh app/ frontend/ Procfile
git commit -m "Add BugFlow application code"

# Step 5: Show new repo size
echo ""
echo "📊 New Repository Size:"
du -sh .git
echo ""
echo "✅ Clean repository ready!"
echo ""
echo "Next steps:"
echo "1. Create new GitHub repository at https://github.com/new"
echo "2. Run: git remote add origin https://github.com/YOUR_USERNAME/bugflow.git"
echo "3. Run: git push -u origin main"
echo ""
echo "Expected push size: ~20-30 MB (should complete in seconds!)"
