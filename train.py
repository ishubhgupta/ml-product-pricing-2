"""
Main training script for ML Product Pricing.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.helpers import setup_logging, load_config, save_pickle, log_transform, ensure_dir
from src.data.data_loader import DataLoader
from src.features.text_features import TextFeatureExtractor
from src.features.image_features import ImageFeatureExtractor
from src.models.base_models import (
    RidgeModel, RandomForestModel, XGBoostModel, LightGBMModel, EnsembleModel
)
from src.models.optimization import HyperparameterOptimizer
from src.evaluation.metrics import ModelEvaluator, calculate_smape


logger = logging.getLogger(__name__)


def main(config_path: str, use_optimization: bool = True, use_images: bool = False):
    """
    Main training pipeline.
    
    Args:
        config_path: Path to configuration file
        use_optimization: Whether to use hyperparameter optimization
        use_images: Whether to use image features
    """
    # Setup
    setup_logging()
    logger.info("="*60)
    logger.info("ML PRODUCT PRICING - TRAINING PIPELINE")
    logger.info("="*60)
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Create output directories
    ensure_dir(config['output']['model_dir'])
    ensure_dir(config['data']['features_dir'])
    
    # =====================================================================
    # STEP 1: Load Data
    # =====================================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Loading Data")
    logger.info("="*60)
    
    data_loader = DataLoader(
        train_path=config['data']['train_csv'],
        test_path=config['data']['test_csv']
    )
    
    train_df = data_loader.load_training_data()
    data_loader.validate_data_integrity(train_df, is_train=True)
    
    # Get price statistics
    price_stats = data_loader.get_price_statistics()
    logger.info(f"Price statistics: {price_stats}")
    
    # =====================================================================
    # STEP 2: Feature Engineering - Text
    # =====================================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Extracting Text Features")
    logger.info("="*60)
    
    text_extractor = TextFeatureExtractor(
        max_features=config['features']['text']['tfidf_max_features'],
        ngram_range=tuple(config['features']['text']['tfidf_ngram_range']),
        min_df=config['features']['text']['min_df'],
        max_df=config['features']['text']['max_df']
    )
    
    # Extract features
    X_text = text_extractor.extract_all_features(
        train_df['catalog_content'].tolist(),
        fit_tfidf=True
    )
    
    # Save text extractor
    save_pickle(text_extractor, 'models/text_extractor.pkl')
    logger.info(f"Text feature shape: {X_text.shape}")
    
    # =====================================================================
    # STEP 3: Feature Engineering - Images (Optional)
    # =====================================================================
    if use_images:
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Extracting Image Features")
        logger.info("="*60)
        
        # Use train_image_dir if available, otherwise fall back to image_dir
        image_dir = config['data'].get('train_image_dir', config['data'].get('image_dir', 'data/images'))
        
        image_extractor = ImageFeatureExtractor(
            image_dir=image_dir,
            image_size=tuple(config['features']['image']['image_size']),
            model_name=config['features']['image']['model_name'],
            use_cache=config['features']['image']['use_cache']
        )
        
        logger.info(f"Downloading images to: {image_dir}")
        
        # Extract features
        X_image = image_extractor.extract_all_features(
            train_df['image_link'].tolist(),
            train_df['sample_id'].tolist()
        )
        
        # Save image extractor
        save_pickle(image_extractor, 'models/image_extractor.pkl')
        logger.info(f"Image feature shape: {X_image.shape}")
        
        # Combine features
        X = np.hstack([X_text, X_image])
        logger.info(f"Combined feature shape: {X.shape}")
    else:
        X = X_text
        logger.info("Skipping image features (use --use-images to enable)")
    
    # =====================================================================
    # STEP 4: Prepare Target Variable
    # =====================================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Preparing Target Variable")
    logger.info("="*60)
    
    y = train_df['price'].values
    
    # Apply log transformation
    if config['training']['target_transform'] == 'log':
        y_transformed = log_transform(y)
        logger.info(f"Applied log transformation to prices")
        logger.info(f"Original price range: [{y.min():.2f}, {y.max():.2f}]")
        logger.info(f"Transformed range: [{y_transformed.min():.2f}, {y_transformed.max():.2f}]")
    else:
        y_transformed = y
    
    # =====================================================================
    # STEP 5: Train/Validation Split
    # =====================================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Train/Validation Split")
    logger.info("="*60)
    
    from sklearn.model_selection import train_test_split
    
    # Create price bins for stratification
    price_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_transformed, 
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state'],
        stratify=price_bins
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    
    # =====================================================================
    # STEP 6: Hyperparameter Optimization (Optional)
    # =====================================================================
    if use_optimization:
        logger.info("\n" + "="*60)
        logger.info("STEP 6: Hyperparameter Optimization")
        logger.info("="*60)
        
        # Get optimization config with defaults
        opt_config = config.get('optimization', {})
        optimizer = HyperparameterOptimizer(
            n_trials=opt_config.get('n_trials', 20),
            cv_folds=opt_config.get('cv_folds', 3),
            random_state=config['training']['random_state'],
            timeout=opt_config.get('timeout', 600),
            early_stopping_rounds=opt_config.get('early_stopping_rounds', 5)
        )
        
        logger.info(f"Optimization settings: {optimizer.n_trials} trials, {optimizer.timeout}s timeout, {optimizer.cv_folds}-fold CV")
        
        # Optimize each model type
        ridge_params = optimizer.optimize_ridge(X_train, y_train)
        rf_params = optimizer.optimize_random_forest(X_train, y_train)
        xgb_params = optimizer.optimize_xgboost(X_train, y_train)
        lgb_params = optimizer.optimize_lightgbm(X_train, y_train)
    else:
        # Use default parameters
        ridge_params = {'alpha': 10.0}
        rf_params = {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 2}
        xgb_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8}
        lgb_params = {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.05, 'num_leaves': 31}
    
    # =====================================================================
    # STEP 7: Train Models
    # =====================================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 7: Training Models")
    logger.info("="*60)
    
    models = {}
    results = []
    
    # Ridge
    logger.info("\nTraining Ridge Regression...")
    ridge = RidgeModel(**ridge_params)
    ridge.fit(X_train, y_train)
    models['ridge'] = ridge
    
    # Random Forest
    logger.info("\nTraining Random Forest...")
    rf = RandomForestModel(**rf_params)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    
    # XGBoost
    logger.info("\nTraining XGBoost...")
    xgb = XGBoostModel(**xgb_params)
    xgb.fit(X_train, y_train)
    models['xgboost'] = xgb
    
    # LightGBM
    logger.info("\nTraining LightGBM...")
    lgb = LightGBMModel(**lgb_params)
    lgb.fit(X_train, y_train)
    models['lightgbm'] = lgb
    
    # =====================================================================
    # STEP 8: Evaluate Models
    # =====================================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 8: Evaluating Models")
    logger.info("="*60)
    
    evaluator = ModelEvaluator(
        cv_folds=config['training']['cv_folds'],
        random_state=config['training']['random_state']
    )
    
    from src.utils.helpers import reverse_log_transform
    
    for name, model in models.items():
        # Predictions on validation set
        y_val_pred_transformed = model.predict(X_val)
        
        # Reverse transformation
        if config['training']['target_transform'] == 'log':
            y_val_pred = reverse_log_transform(y_val_pred_transformed)
            y_val_original = reverse_log_transform(y_val)
        else:
            y_val_pred = y_val_pred_transformed
            y_val_original = y_val
        
        # Calculate SMAPE
        result = evaluator.evaluate_predictions(y_val_original, y_val_pred, name)
        results.append(result)
    
    # Compare models
    evaluator.compare_models(results)
    
    # Select best model
    best_result = min(results, key=lambda x: x['smape'])
    best_model_name = best_result['model_name']
    best_model = models[best_model_name]
    
    logger.info(f"\nBest Model: {best_model_name} (SMAPE: {best_result['smape']:.4f})")
    
    # =====================================================================
    # STEP 9: Create Ensemble (Optional)
    # =====================================================================
    if config['ensemble']['use_stacking']:
        logger.info("\n" + "="*60)
        logger.info("STEP 9: Creating Ensemble Model")
        logger.info("="*60)
        
        # Simple weighted average based on validation performance
        weights = []
        model_list = []
        for result in results:
            model_name = result['model_name']
            # Inverse SMAPE as weight (lower SMAPE = higher weight)
            weight = 1.0 / (result['smape'] + 1e-6)
            weights.append(weight)
            model_list.append(models[model_name])
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        ensemble = EnsembleModel(model_list, weights)
        
        # Evaluate ensemble
        y_val_pred_transformed = ensemble.predict(X_val)
        if config['training']['target_transform'] == 'log':
            y_val_pred = reverse_log_transform(y_val_pred_transformed)
            y_val_original = reverse_log_transform(y_val)
        else:
            y_val_pred = y_val_pred_transformed
            y_val_original = y_val
        
        ensemble_result = evaluator.evaluate_predictions(y_val_original, y_val_pred, "Ensemble")
        
        if ensemble_result['smape'] < best_result['smape']:
            logger.info(f"Ensemble improves SMAPE: {ensemble_result['smape']:.4f} vs {best_result['smape']:.4f}")
            best_model = ensemble
            best_model_name = "Ensemble"
        else:
            logger.info(f"Ensemble does not improve performance, using {best_model_name}")
    
    # =====================================================================
    # STEP 10: Save Best Model
    # =====================================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 10: Saving Models")
    logger.info("="*60)
    
    # Save all models with validation
    for name, model in models.items():
        model_path = f"models/{name}_model.pkl"
        try:
            model.save(model_path)
            # Verify the saved model can be loaded
            import joblib
            test_load = joblib.load(model_path)
            logger.info(f"✓ {name} model saved and verified: {model_path}")
        except Exception as e:
            logger.error(f"✗ Failed to save/verify {name} model: {str(e)}")
            raise
    
    # Save best model with validation
    best_model_path = "models/best_model.pkl"
    try:
        if best_model_name == "Ensemble":
            ensemble.save(best_model_path)
        else:
            best_model.save(best_model_path)
        
        # Verify the best model can be loaded
        import joblib
        test_load = joblib.load(best_model_path)
        logger.info(f"✓ Best model ({best_model_name}) saved and verified: {best_model_path}")
    except Exception as e:
        logger.error(f"✗ Failed to save/verify best model: {str(e)}")
        raise
    
    # Save configuration for prediction
    import json
    config_path = 'models/config.json'
    try:
        with open(config_path, 'w') as f:
            json.dump({
                'best_model': best_model_name,
                'use_images': use_images,
                'target_transform': config['training']['target_transform']
            }, f, indent=2)
        logger.info(f"✓ Configuration saved: {config_path}")
    except Exception as e:
        logger.error(f"✗ Failed to save configuration: {str(e)}")
        raise
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"All models saved in models/ directory")
    logger.info(f"Ready for prediction with: python predict.py --config config.yaml")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML Product Pricing models")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--optimize', action='store_true', help='Enable hyperparameter optimization')
    parser.add_argument('--use-images', action='store_true', help='Enable image feature extraction')
    
    args = parser.parse_args()
    
    main(args.config, use_optimization=args.optimize, use_images=args.use_images)
