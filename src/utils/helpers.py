"""
Utility functions for the ML Product Pricing system.
"""

import os
import logging
import yaml
import pickle
import numpy as np
from typing import Any, Dict


def setup_logging(log_file: str = 'ml_pricing.log', level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object to pickle file using joblib for consistency.
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    import joblib
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath, compress=3)


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file using joblib for consistency.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    import joblib
    return joblib.load(filepath)


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)


def clip_predictions(predictions: np.ndarray, min_val: float = 0.01, max_val: float = 10000.0) -> np.ndarray:
    """
    Clip predictions to reasonable price range.
    
    Args:
        predictions: Array of predictions
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clipped predictions
    """
    return np.clip(predictions, min_val, max_val)


def reverse_log_transform(log_prices: np.ndarray) -> np.ndarray:
    """
    Reverse log transformation for prices.
    
    Args:
        log_prices: Log-transformed prices
        
    Returns:
        Original scale prices
    """
    return np.exp(log_prices)


def log_transform(prices: np.ndarray) -> np.ndarray:
    """
    Apply log transformation to prices.
    
    Args:
        prices: Original prices
        
    Returns:
        Log-transformed prices
    """
    return np.log(prices + 1)  # Add 1 to avoid log(0)
