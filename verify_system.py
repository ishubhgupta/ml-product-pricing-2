"""
System verification script to ensure everything is working correctly.
Run this before starting optimized training.
"""

import os
import sys
import joblib
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def verify_dependencies():
    """Verify all required packages are installed."""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Verifying Dependencies")
    logger.info("="*60)
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'xgboost', 
        'lightgbm', 'optuna', 'pyyaml', 'joblib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✓ {package}")
        except ImportError:
            logger.error(f"✗ {package} - NOT FOUND")
            missing.append(package)
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False
    
    logger.info("✓ All dependencies installed")
    return True


def verify_data_files():
    """Verify required data files exist."""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Verifying Data Files")
    logger.info("="*60)
    
    required_files = [
        '../train1.csv',
        '../test1.csv',
        'config.yaml'
    ]
    
    missing = []
    for filepath in required_files:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            logger.info(f"✓ {filepath} ({size_mb:.2f} MB)")
        else:
            logger.error(f"✗ {filepath} - NOT FOUND")
            missing.append(filepath)
    
    if missing:
        logger.error(f"\nMissing files: {', '.join(missing)}")
        return False
    
    logger.info("✓ All data files present")
    return True


def verify_project_structure():
    """Verify project directory structure."""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Verifying Project Structure")
    logger.info("="*60)
    
    required_dirs = [
        'src/data',
        'src/features',
        'src/models',
        'src/evaluation',
        'src/utils',
        'models'
    ]
    
    required_files = [
        'src/data/data_loader.py',
        'src/features/text_features.py',
        'src/models/base_models.py',
        'src/evaluation/metrics.py',
        'src/utils/helpers.py',
        'train.py',
        'predict.py'
    ]
    
    all_good = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            logger.info(f"✓ {directory}/")
        else:
            logger.error(f"✗ {directory}/ - NOT FOUND")
            all_good = False
    
    for filepath in required_files:
        if os.path.exists(filepath):
            logger.info(f"✓ {filepath}")
        else:
            logger.error(f"✗ {filepath} - NOT FOUND")
            all_good = False
    
    if all_good:
        logger.info("✓ Project structure complete")
    return all_good


def verify_save_load_mechanism():
    """Verify model save/load works correctly."""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Verifying Save/Load Mechanism")
    logger.info("="*60)
    
    try:
        from src.models.base_models import RidgeModel
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        # Create a simple test model
        logger.info("Creating test model...")
        model = RidgeModel()
        X_test = np.random.rand(10, 5)
        y_test = np.random.rand(10)
        model.fit(X_test, y_test, scale=False)
        
        # Save model
        test_path = 'models/test_verification.pkl'
        logger.info(f"Saving test model to {test_path}...")
        model.save(test_path)
        
        # Verify file exists
        if not os.path.exists(test_path):
            logger.error(f"✗ Model file not created!")
            return False
        
        size_kb = os.path.getsize(test_path) / 1024
        logger.info(f"✓ Model saved ({size_kb:.2f} KB)")
        
        # Load model
        logger.info("Loading test model...")
        loaded_model = RidgeModel()
        loaded_model.load(test_path)
        
        # Verify loaded model works
        logger.info("Testing loaded model predictions...")
        pred1 = model.predict(X_test, scale=False)
        pred2 = loaded_model.predict(X_test, scale=False)
        
        if np.allclose(pred1, pred2):
            logger.info("✓ Loaded model predictions match original")
        else:
            logger.error("✗ Loaded model predictions differ!")
            return False
        
        # Clean up
        os.remove(test_path)
        logger.info("✓ Save/load mechanism working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Save/load verification failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def verify_helpers():
    """Verify helper functions work correctly."""
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Verifying Helper Functions")
    logger.info("="*60)
    
    try:
        from src.utils.helpers import save_pickle, load_pickle
        import numpy as np
        
        # Test data
        test_data = {
            'array': np.random.rand(10, 5),
            'list': [1, 2, 3, 4, 5],
            'dict': {'key': 'value'}
        }
        
        # Save
        test_path = 'models/test_helpers.pkl'
        logger.info("Testing save_pickle...")
        save_pickle(test_data, test_path)
        
        if not os.path.exists(test_path):
            logger.error("✗ Helper save failed!")
            return False
        
        logger.info("✓ save_pickle works")
        
        # Load
        logger.info("Testing load_pickle...")
        loaded_data = load_pickle(test_path)
        
        if np.allclose(test_data['array'], loaded_data['array']):
            logger.info("✓ load_pickle works")
        else:
            logger.error("✗ Loaded data differs!")
            return False
        
        # Clean up
        os.remove(test_path)
        logger.info("✓ Helper functions working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Helper verification failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    logger.info("\n" + "="*70)
    logger.info("ML PRODUCT PRICING - SYSTEM VERIFICATION")
    logger.info("="*70)
    
    checks = [
        ("Dependencies", verify_dependencies),
        ("Data Files", verify_data_files),
        ("Project Structure", verify_project_structure),
        ("Save/Load Mechanism", verify_save_load_mechanism),
        ("Helper Functions", verify_helpers)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            logger.error(f"\n✗ {check_name} check failed with exception: {str(e)}")
            results.append((check_name, False))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*70)
    
    all_passed = True
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status} - {check_name}")
        if not result:
            all_passed = False
    
    logger.info("="*70)
    
    if all_passed:
        logger.info("\n✓ ALL CHECKS PASSED - SYSTEM READY FOR TRAINING")
        logger.info("\nYou can now run:")
        logger.info("  Basic training:    python train.py --config config.yaml")
        logger.info("  Optimized training: python train.py --config config.yaml --optimize")
        return 0
    else:
        logger.error("\n✗ SOME CHECKS FAILED - PLEASE FIX ISSUES BEFORE TRAINING")
        return 1


if __name__ == "__main__":
    sys.exit(main())
