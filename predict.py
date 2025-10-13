"""
Prediction script for generating test_out.csv
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

from src.utils.helpers import setup_logging, load_pickle, reverse_log_transform, clip_predictions
from src.data.data_loader import DataLoader


logger = logging.getLogger(__name__)


def main(config_path: str, model_path: str = None):
    """
    Generate predictions for test data.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to trained model (default: models/best_model.pkl)
    """
    # Setup
    setup_logging('prediction.log')
    logger.info("="*60)
    logger.info("ML PRODUCT PRICING - PREDICTION PIPELINE")
    logger.info("="*60)
    
    # Load configuration
    from src.utils.helpers import load_config
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Load model configuration
    import json
    with open('models/config.json', 'r') as f:
        model_config = json.load(f)
    
    logger.info(f"Best model: {model_config['best_model']}")
    logger.info(f"Use images: {model_config['use_images']}")
    logger.info(f"Target transform: {model_config['target_transform']}")
    
    # =====================================================================
    # STEP 1: Load Test Data
    # =====================================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Loading Test Data")
    logger.info("="*60)
    
    data_loader = DataLoader(test_path=config['data']['test_csv'])
    test_df = data_loader.load_test_data()
    data_loader.validate_data_integrity(test_df, is_train=False)
    
    # =====================================================================
    # STEP 2: Load Feature Extractors
    # =====================================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Loading Feature Extractors")
    logger.info("="*60)
    
    # Load text extractor
    text_extractor = load_pickle('models/text_extractor.pkl')
    logger.info("Text extractor loaded")
    
    # Load image extractor if needed
    if model_config['use_images']:
        image_extractor = load_pickle('models/image_extractor.pkl')
        logger.info("Image extractor loaded")
    
    # =====================================================================
    # STEP 3: Extract Features from Test Data
    # =====================================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Extracting Features from Test Data")
    logger.info("="*60)
    
    # Extract text features
    logger.info("Extracting text features...")
    X_text = text_extractor.extract_all_features(
        test_df['catalog_content'].tolist(),
        fit_tfidf=False  # Use already fitted vectorizer
    )
    logger.info(f"Text feature shape: {X_text.shape}")
    
    # Extract image features if needed
    if model_config['use_images']:
        logger.info("Extracting image features...")
        X_image = image_extractor.extract_all_features(
            test_df['image_link'].tolist(),
            test_df['sample_id'].tolist()
        )
        logger.info(f"Image feature shape: {X_image.shape}")
        
        # Combine features
        X_test = np.hstack([X_text, X_image])
        logger.info(f"Combined feature shape: {X_test.shape}")
    else:
        X_test = X_text
    
    # =====================================================================
    # STEP 4: Load Model and Generate Predictions
    # =====================================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Loading Model and Generating Predictions")
    logger.info("="*60)
    
    # Determine model path
    if model_path is None:
        # Use the specific model file instead of best_model.pkl
        best_model_name = model_config['best_model']
        if best_model_name == 'Ensemble':
            model_path = 'models/best_model.pkl'
        else:
            model_path = f'models/{best_model_name}_model.pkl'
    
    logger.info(f"Loading model: {model_config['best_model']}")
    
    # Load model based on type with error handling
    try:
        if model_config['best_model'] == 'Ensemble':
            from src.models.base_models import EnsembleModel
            model = EnsembleModel.load(model_path)
            logger.info(f"✓ Ensemble model loaded successfully from {model_path}")
        else:
            # Load individual model - need to instantiate the right model class
            from src.models.base_models import (
                RidgeModel, RandomForestModel, XGBoostModel, LightGBMModel
            )
            
            model_name = model_config['best_model']
            if model_name == 'ridge':
                model = RidgeModel()
            elif model_name == 'random_forest':
                model = RandomForestModel()
            elif model_name == 'xgboost':
                model = XGBoostModel()
            elif model_name == 'lightgbm':
                model = LightGBMModel()
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            model.load(model_path)
            logger.info(f"✓ {model_name} model loaded successfully from {model_path}")
            
        # Verify model is fitted and ready
        if hasattr(model, 'is_fitted'):
            if not model.is_fitted:
                raise ValueError("Loaded model is not fitted!")
        
        # Verify model has predict method
        if not hasattr(model, 'predict'):
            raise AttributeError("Loaded model does not have predict method!")
            
    except Exception as e:
        logger.error(f"✗ Failed to load model from {model_path}")
        logger.error(f"Error: {str(e)}")
        logger.error("Please ensure the model was trained and saved correctly.")
        raise
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions_transformed = model.predict(X_test)
    
    # Reverse transformation if needed
    if model_config['target_transform'] == 'log':
        predictions = reverse_log_transform(predictions_transformed)
        logger.info("Reversed log transformation")
    else:
        predictions = predictions_transformed
    
    # Clip predictions to reasonable range
    predictions = clip_predictions(predictions, min_val=0.01, max_val=10000.0)
    
    logger.info(f"Predictions generated: {len(predictions)}")
    logger.info(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    logger.info(f"Prediction mean: {predictions.mean():.2f}")
    logger.info(f"Prediction median: {np.median(predictions):.2f}")
    
    # =====================================================================
    # STEP 5: Create Output CSV
    # =====================================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Creating Output CSV")
    logger.info("="*60)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Validate output
    logger.info(f"Output shape: {output_df.shape}")
    logger.info(f"Sample IDs: {len(output_df['sample_id'].unique())}")
    
    # Check for missing values
    if output_df.isnull().any().any():
        logger.error("Output contains missing values!")
        logger.error(output_df.isnull().sum())
        raise ValueError("Output validation failed: missing values detected")
    
    # Check for duplicate sample IDs
    if output_df['sample_id'].duplicated().any():
        logger.error("Output contains duplicate sample IDs!")
        raise ValueError("Output validation failed: duplicate sample IDs")
    
    # Check price values
    if (output_df['price'] <= 0).any():
        logger.error("Output contains non-positive prices!")
        raise ValueError("Output validation failed: non-positive prices")
    
    logger.info("Output validation passed ✓")
    
    # =====================================================================
    # STEP 6: Save Output
    # =====================================================================
    output_path = config['data']['output_csv']
    output_df.to_csv(output_path, index=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Predictions saved to: {output_path}")
    logger.info(f"{'='*60}")
    logger.info("First 10 predictions:")
    logger.info(output_df.head(10).to_string(index=False))
    
    logger.info("\n" + "="*60)
    logger.info("PREDICTION COMPLETED SUCCESSFULLY!")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions for test data")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    
    args = parser.parse_args()
    
    main(args.config, args.model)
