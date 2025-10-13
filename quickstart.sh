#!/bin/bash
# Quick start script for ML Product Pricing

echo "=========================================="
echo "ML Product Pricing - Quick Start"
echo "=========================================="

# Step 1: Create virtual environment
echo ""
echo "Step 1: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "Virtual environment created ✓"
else
    echo "Virtual environment already exists ✓"
fi

# Step 2: Activate virtual environment
echo ""
echo "Step 2: Activating virtual environment..."
source venv/Scripts/activate

# Step 3: Install dependencies
echo ""
echo "Step 3: Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Download spaCy model (optional for NLP)
echo ""
echo "Step 4: Downloading spaCy model (optional)..."
python -m spacy download en_core_web_sm || echo "Note: spaCy model download failed (optional)"

# Step 5: Run training (without optimization and images for quick start)
echo ""
echo "=========================================="
echo "Starting Training (Quick Mode)"
echo "=========================================="
python train.py --config config.yaml

# Step 6: Generate predictions
echo ""
echo "=========================================="
echo "Generating Predictions"
echo "=========================================="
python predict.py --config config.yaml

echo ""
echo "=========================================="
echo "Quick Start Completed!"
echo "=========================================="
echo "Check test_out.csv for predictions"
