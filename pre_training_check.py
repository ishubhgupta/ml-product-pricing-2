#!/usr/bin/env python3
"""
Pre-training safety check - Run this before optimized training to ensure everything is ready.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def check_disk_space():
    """Check if enough disk space is available."""
    import shutil
    stats = shutil.disk_usage(os.getcwd())
    free_gb = stats.free / (1024**3)
    print(f"\n📊 Disk Space: {free_gb:.2f} GB free")
    
    if free_gb < 1:
        print("⚠️  WARNING: Less than 1GB free. Consider freeing up space.")
        return False
    return True

def check_models_directory():
    """Ensure models directory is ready."""
    if os.path.exists('models'):
        files = os.listdir('models')
        pkl_files = [f for f in files if f.endswith('.pkl')]
        if pkl_files:
            print(f"\n📁 Found {len(pkl_files)} existing model files:")
            for f in pkl_files:
                size_mb = os.path.getsize(f'models/{f}') / (1024**2)
                print(f"   - {f} ({size_mb:.2f} MB)")
            
            response = input("\n⚠️  Existing models will be overwritten. Continue? (yes/no): ")
            if response.lower() != 'yes':
                print("❌ Aborted by user")
                return False
    return True

def check_data_integrity():
    """Quick check of data files."""
    print("\n📋 Checking data files...")
    try:
        import pandas as pd
        
        # Check train data
        train_df = pd.read_csv('../train1.csv', nrows=5)
        required_cols = ['sample_id', 'catalog_content', 'price']
        if all(col in train_df.columns for col in required_cols):
            print("   ✓ Train data format correct")
        else:
            print("   ✗ Train data missing required columns")
            return False
        
        # Check test data
        test_df = pd.read_csv('../test1.csv', nrows=5)
        if 'sample_id' in test_df.columns and 'catalog_content' in test_df.columns:
            print("   ✓ Test data format correct")
        else:
            print("   ✗ Test data missing required columns")
            return False
            
    except Exception as e:
        print(f"   ✗ Error reading data: {str(e)}")
        return False
    
    return True

def estimate_training_time(optimize=True):
    """Estimate training time based on mode."""
    if optimize:
        print("\n⏱️  Estimated Training Time: 30-60 minutes")
        print("   - 4 models × 50 trials each")
        print("   - Hyperparameter optimization with Optuna")
    else:
        print("\n⏱️  Estimated Training Time: 10-15 minutes")
        print("   - 4 models with default parameters")
    print("   - Ensemble creation")
    print("   - Model validation and saving")

def main():
    print("="*70)
    print("🔍 PRE-TRAINING SAFETY CHECK")
    print("="*70)
    
    checks = [
        ("Disk Space", check_disk_space),
        ("Models Directory", check_models_directory),
        ("Data Integrity", check_data_integrity)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
                print(f"❌ {check_name} check failed")
        except Exception as e:
            print(f"❌ {check_name} check error: {str(e)}")
            all_passed = False
    
    if not all_passed:
        print("\n❌ SAFETY CHECKS FAILED")
        print("Please resolve issues before training.")
        return 1
    
    print("\n" + "="*70)
    print("✅ ALL SAFETY CHECKS PASSED")
    print("="*70)
    
    # Show training options
    print("\n🚀 Ready to start training!")
    print("\nAvailable commands:")
    print("\n1. Basic Training (Fast, 10-15 min):")
    print("   python train.py --config config.yaml")
    print("\n2. Optimized Training (Recommended, 30-60 min):")
    print("   python train.py --config config.yaml --optimize")
    
    # Ask which mode
    print("\n" + "-"*70)
    response = input("Which mode? (1=basic, 2=optimized, q=quit): ").strip()
    
    if response == '1':
        estimate_training_time(optimize=False)
        confirm = input("\nStart basic training? (yes/no): ")
        if confirm.lower() == 'yes':
            print("\n🚀 Starting basic training...")
            os.system("python train.py --config config.yaml")
    elif response == '2':
        estimate_training_time(optimize=True)
        confirm = input("\nStart optimized training? (yes/no): ")
        if confirm.lower() == 'yes':
            print("\n🚀 Starting optimized training...")
            os.system("python train.py --config config.yaml --optimize")
    else:
        print("\n👋 Exiting. Run training manually when ready.")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
