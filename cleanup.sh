#!/bin/bash
# Cleanup script - Prepare repository for GitHub

echo "🧹 Cleaning up repository for GitHub..."
echo ""

# Remove data files
echo "→ Removing data files..."
rm -rf data/images/
mkdir -p data  # Keep directory

# Remove model files
echo "→ Removing model files..."
rm -rf models/
mkdir -p models  # Keep directory

# Remove logs
echo "→ Removing logs..."
rm -f *.log

# Remove Python cache
echo "→ Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

# Remove any existing venv
echo "→ Removing old virtual environments..."
rm -rf venv/ env/ .venv/ ENV/

# Remove temporary files
echo "→ Removing temporary files..."
rm -rf temp/ tmp/ backup/
rm -f *.tmp *.temp *.bak

# Remove system files
echo "→ Removing system files..."
find . -name ".DS_Store" -delete
find . -name "Thumbs.db" -delete

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📁 Current directory contents:"
ls -la
echo ""
echo "📊 Directory size:"
du -sh .
echo ""
echo "Next steps:"
echo "1. Create venv: python3.10 -m venv venv"
echo "2. Activate: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
echo "3. Install deps: pip install -r requirements.txt"
echo "4. Initialize git: git init"
echo "5. Add files: git add ."
echo "6. Commit: git commit -m 'Initial commit'"
