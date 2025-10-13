"""
Test script to verify ML Product Pricing setup.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError as e:
        print(f"  ✗ pandas: {e}")
        return False
    
    try:
        import sklearn
        print("  ✓ scikit-learn")
    except ImportError as e:
        print(f"  ✗ scikit-learn: {e}")
        return False
    
    try:
        from PIL import Image
        print("  ✓ Pillow")
    except ImportError as e:
        print(f"  ✗ Pillow: {e}")
        return False
    
    # Optional dependencies
    try:
        import xgboost
        print("  ✓ xgboost (optional)")
    except ImportError:
        print("  ⚠ xgboost (optional - not installed)")
    
    try:
        import lightgbm
        print("  ✓ lightgbm (optional)")
    except ImportError:
        print("  ⚠ lightgbm (optional - not installed)")
    
    try:
        import optuna
        print("  ✓ optuna (optional)")
    except ImportError:
        print("  ⚠ optuna (optional - not installed)")
    
    try:
        import torch
        print("  ✓ torch (optional)")
    except ImportError:
        print("  ⚠ torch (optional - not installed for image features)")
    
    return True


def test_data_files():
    """Test that data files exist."""
    print("\nTesting data files...")
    
    files_to_check = [
        '../train1.csv',
        '../test1.csv'
    ]
    
    all_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - NOT FOUND")
            all_exist = False
    
    return all_exist


def test_config():
    """Test configuration file."""
    print("\nTesting configuration...")
    
    if not os.path.exists('config.yaml'):
        print("  ✗ config.yaml - NOT FOUND")
        return False
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("  ✓ config.yaml loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error loading config.yaml: {e}")
        return False


def test_modules():
    """Test custom modules."""
    print("\nTesting custom modules...")
    
    try:
        from src.data.data_loader import DataLoader
        print("  ✓ DataLoader")
    except ImportError as e:
        print(f"  ✗ DataLoader: {e}")
        return False
    
    try:
        from src.features.text_features import TextFeatureExtractor
        print("  ✓ TextFeatureExtractor")
    except ImportError as e:
        print(f"  ✗ TextFeatureExtractor: {e}")
        return False
    
    try:
        from src.features.image_features import ImageFeatureExtractor
        print("  ✓ ImageFeatureExtractor")
    except ImportError as e:
        print(f"  ✗ ImageFeatureExtractor: {e}")
        return False
    
    try:
        from src.models.base_models import RidgeModel, RandomForestModel
        print("  ✓ Model classes")
    except ImportError as e:
        print(f"  ✗ Model classes: {e}")
        return False
    
    try:
        from src.evaluation.metrics import calculate_smape, ModelEvaluator
        print("  ✓ Evaluation metrics")
    except ImportError as e:
        print(f"  ✗ Evaluation metrics: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality with sample data."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from src.evaluation.metrics import calculate_smape
        
        # Test SMAPE calculation
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        smape = calculate_smape(y_true, y_pred)
        
        print(f"  ✓ SMAPE calculation: {smape:.4f}")
        
        # Test data loader with sample data
        from src.data.data_loader import DataLoader
        import pandas as pd
        
        # Create sample dataframe
        sample_df = pd.DataFrame({
            'sample_id': [1, 2, 3],
            'catalog_content': ['test product 1', 'test product 2', 'test product 3'],
            'image_link': ['http://example.com/1.jpg', 'http://example.com/2.jpg', 'http://example.com/3.jpg'],
            'price': [10.0, 20.0, 30.0]
        })
        
        loader = DataLoader()
        loader.train_df = sample_df
        stats = loader.get_price_statistics()
        print(f"  ✓ Data loader: Mean price = {stats['mean']:.2f}")
        
        # Test text feature extraction
        from src.features.text_features import TextFeatureExtractor
        
        text_extractor = TextFeatureExtractor(max_features=10)
        texts = ['Product A with value: 10.0 unit: oz', 'Product B Pack of 5']
        
        ipq_features = text_extractor.extract_ipq(texts[1])
        print(f"  ✓ Text feature extraction: IPQ = {ipq_features['ipq']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error in functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("ML PRODUCT PRICING - SETUP VERIFICATION")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Data Files", test_data_files()))
    results.append(("Configuration", test_config()))
    results.append(("Custom Modules", test_modules()))
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run training: python train.py --config config.yaml")
        print("  2. Generate predictions: python predict.py --config config.yaml")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues before proceeding.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check data file paths in config.yaml")
        print("  - Ensure data files are in the correct location")
        return 1


if __name__ == "__main__":
    sys.exit(main())
