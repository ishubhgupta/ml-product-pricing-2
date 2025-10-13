"""
SMAPE metric and evaluation utilities.
"""

import numpy as np
import logging
from typing import List, Dict, Any
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer


logger = logging.getLogger(__name__)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    SMAPE = (100/n) * Σ(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        SMAPE score (0-200, lower is better)
    """
    # Ensure arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    # Calculate SMAPE
    smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
    
    return float(smape)


def smape_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    SMAPE scorer for sklearn (negative because sklearn maximizes scores).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Negative SMAPE (for maximization)
    """
    return -calculate_smape(y_true, y_pred)


class ModelEvaluator:
    """
    Evaluates model performance using SMAPE and cross-validation.
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize evaluator.
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random seed
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scorer = make_scorer(smape_scorer, greater_is_better=False)
    
    def evaluate_single_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a single model using cross-validation.
        
        Args:
            model: Sklearn-compatible model
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model: {model.__class__.__name__}")
        
        # Perform cross-validation
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.scorer)
        
        # Convert back to positive SMAPE
        cv_scores = -cv_scores
        
        results = {
            'model_name': model.__class__.__name__,
            'mean_smape': np.mean(cv_scores),
            'std_smape': np.std(cv_scores),
            'cv_scores': cv_scores.tolist(),
            'min_smape': np.min(cv_scores),
            'max_smape': np.max(cv_scores)
        }
        
        logger.info(f"Mean SMAPE: {results['mean_smape']:.4f} ± {results['std_smape']:.4f}")
        
        return results
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate predictions on a single dataset.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        smape = calculate_smape(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE (for reference)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        results = {
            'model_name': model_name,
            'smape': smape,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        logger.info(f"{model_name} - SMAPE: {smape:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        return results
    
    def compare_models(self, results: List[Dict[str, Any]]) -> None:
        """
        Compare multiple model results.
        
        Args:
            results: List of result dictionaries
        """
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        
        # Sort by mean SMAPE
        sorted_results = sorted(results, key=lambda x: x.get('mean_smape', x.get('smape', float('inf'))))
        
        for i, result in enumerate(sorted_results, 1):
            model_name = result['model_name']
            if 'mean_smape' in result:
                score = f"{result['mean_smape']:.4f} ± {result['std_smape']:.4f}"
            else:
                score = f"{result['smape']:.4f}"
            
            logger.info(f"{i}. {model_name}: {score}")
        
        logger.info("="*60)
        logger.info(f"Best Model: {sorted_results[0]['model_name']}")
        logger.info("="*60 + "\n")


def calculate_confidence_interval(scores: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Calculate confidence interval for scores.
    
    Args:
        scores: Array of scores
        confidence: Confidence level
        
    Returns:
        (lower_bound, upper_bound)
    """
    from scipy import stats
    
    mean = np.mean(scores)
    std_err = stats.sem(scores)
    interval = std_err * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
    
    return (mean - interval, mean + interval)
