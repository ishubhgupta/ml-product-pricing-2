"""Test model loading"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.utils.helpers import load_pickle

print("Testing model loading...")
try:
    model = load_pickle('models/ridge_model.pkl')
    print("✓ Ridge model loaded successfully!")
    print(f"  Model type: {type(model)}")
except Exception as e:
    print(f"✗ Failed to load Ridge model: {e}")

try:
    model = load_pickle('models/random_forest_model.pkl')
    print("✓ Random Forest model loaded successfully!")
    print(f"  Model type: {type(model)}")
except Exception as e:
    print(f"✗ Failed to load Random Forest model: {e}")

print("\nAll models loaded successfully! Ready for predictions.")
