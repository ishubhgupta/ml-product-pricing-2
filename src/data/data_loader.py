"""
Data loading and validation module.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict
from pathlib import Path


logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and validation of training and test data.
    """
    
    def __init__(self, train_path: Optional[str] = None, test_path: Optional[str] = None):
        """
        Initialize DataLoader.
        
        Args:
            train_path: Path to training CSV file
            test_path: Path to test CSV file
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        
    def load_training_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load training data from CSV.
        
        Args:
            filepath: Path to training CSV (overrides init path)
            
        Returns:
            Training dataframe
        """
        path = filepath or self.train_path
        if path is None:
            raise ValueError("Training data path not provided")
            
        logger.info(f"Loading training data from {path}")
        df = pd.read_csv(path)
        
        # Validate required columns
        required_cols = ['sample_id', 'catalog_content', 'image_link', 'price']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Basic validation
        logger.info(f"Loaded {len(df)} training samples")
        logger.info(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        logger.info(f"Missing values: {df.isnull().sum().to_dict()}")
        
        self.train_df = df
        return df
    
    def load_test_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load test data from CSV.
        
        Args:
            filepath: Path to test CSV (overrides init path)
            
        Returns:
            Test dataframe
        """
        path = filepath or self.test_path
        if path is None:
            raise ValueError("Test data path not provided")
            
        logger.info(f"Loading test data from {path}")
        df = pd.read_csv(path)
        
        # Validate required columns
        required_cols = ['sample_id', 'catalog_content', 'image_link']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Loaded {len(df)} test samples")
        
        self.test_df = df
        return df
    
    def validate_data_integrity(self, df: pd.DataFrame, is_train: bool = True) -> bool:
        """
        Validate data integrity and quality.
        
        Args:
            df: Dataframe to validate
            is_train: Whether this is training data
            
        Returns:
            True if validation passes
        """
        logger.info("Validating data integrity...")
        
        # Check for duplicate sample IDs
        if df['sample_id'].duplicated().any():
            logger.warning(f"Found {df['sample_id'].duplicated().sum()} duplicate sample IDs")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        
        # Validate prices (for training data)
        if is_train and 'price' in df.columns:
            if (df['price'] <= 0).any():
                logger.error("Found non-positive prices in training data")
                return False
            
            # Check for extreme outliers
            q1 = df['price'].quantile(0.25)
            q3 = df['price'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            outliers = ((df['price'] < lower_bound) | (df['price'] > upper_bound)).sum()
            logger.info(f"Found {outliers} extreme price outliers ({outliers/len(df)*100:.2f}%)")
        
        # Validate image links
        if 'image_link' in df.columns:
            invalid_links = df[df['image_link'].isnull() | (df['image_link'] == '')].shape[0]
            if invalid_links > 0:
                logger.warning(f"Found {invalid_links} invalid image links")
        
        # Validate catalog content
        if 'catalog_content' in df.columns:
            empty_content = df[df['catalog_content'].isnull() | (df['catalog_content'].str.strip() == '')].shape[0]
            if empty_content > 0:
                logger.warning(f"Found {empty_content} empty catalog content entries")
        
        logger.info("Data validation completed")
        return True
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split training data into train and validation sets.
        
        Args:
            test_size: Proportion of data for validation
            random_state: Random seed
            
        Returns:
            Train and validation dataframes
        """
        if self.train_df is None:
            raise ValueError("Training data not loaded. Call load_training_data() first.")
        
        from sklearn.model_selection import train_test_split
        
        # Stratified split based on price bins
        price_bins = pd.qcut(self.train_df['price'], q=10, labels=False, duplicates='drop')
        
        train_df, val_df = train_test_split(
            self.train_df,
            test_size=test_size,
            random_state=random_state,
            stratify=price_bins
        )
        
        logger.info(f"Split data: {len(train_df)} train, {len(val_df)} validation")
        return train_df, val_df
    
    def get_price_statistics(self) -> Dict[str, float]:
        """
        Get statistics about price distribution.
        
        Returns:
            Dictionary of price statistics
        """
        if self.train_df is None or 'price' not in self.train_df.columns:
            raise ValueError("Training data with prices not loaded")
        
        prices = self.train_df['price']
        return {
            'mean': prices.mean(),
            'median': prices.median(),
            'std': prices.std(),
            'min': prices.min(),
            'max': prices.max(),
            'q25': prices.quantile(0.25),
            'q75': prices.quantile(0.75),
            'skewness': prices.skew(),
            'kurtosis': prices.kurtosis()
        }
