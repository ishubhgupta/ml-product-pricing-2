"""
Text feature extraction module.
"""

import re
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """
    Extracts features from catalog content text.
    """
    
    def __init__(self, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 3, max_df: float = 0.8):
        """
        Initialize text feature extractor.
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: Range for n-grams
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.tfidf_vectorizer = None
        
    def extract_ipq(self, text: str) -> Dict[str, float]:
        """
        Extract Item Pack Quantity (IPQ) from text.
        
        Args:
            text: Catalog content text
            
        Returns:
            Dictionary with IPQ features
        """
        # Common patterns for pack quantities
        patterns = [
            r'pack of (\d+)',
            r'\(pack of (\d+)\)',
            r'(\d+)\s*pack',
            r'(\d+)\s*count',
            r'(\d+)\s*ct',
            r'(\d+)[-\s]*piece',
            r'set of (\d+)',
        ]
        
        quantities = []
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            quantities.extend([int(m) for m in matches])
        
        # Also look for "Value: X.0 Unit: Count" pattern
        value_count_pattern = r'value:\s*(\d+\.?\d*)\s*unit:\s*count'
        value_matches = re.findall(value_count_pattern, text_lower)
        if value_matches:
            quantities.extend([float(m) for m in value_matches])
        
        if quantities:
            ipq = max(quantities)  # Use maximum found quantity
        else:
            ipq = 1.0  # Default to 1 if not found
        
        # Cap extreme outliers
        ipq = min(ipq, 1000)
        
        return {
            'ipq': ipq,
            'ipq_log': np.log1p(ipq),
            'is_multipack': 1.0 if ipq > 1 else 0.0
        }
    
    def extract_product_value(self, text: str) -> Dict[str, float]:
        """
        Extract product value and unit from text.
        
        Args:
            text: Catalog content text
            
        Returns:
            Dictionary with value features
        """
        # Pattern: "Value: 288.0 Unit: Fl Oz"
        value_pattern = r'value:\s*(\d+\.?\d*)\s*unit:\s*([a-z\s]+)'
        matches = re.findall(value_pattern, text.lower())
        
        # Initialize all features with default values
        features = {
            'product_value': 0.0,
            'product_value_log': 0.0,
            'has_value': 0.0,
            'unit_is_volume': 0.0,
            'unit_is_weight': 0.0,
            'unit_is_count': 0.0
        }
        
        if matches:
            value, unit = matches[0]
            value = float(value)
            features['product_value'] = value
            features['product_value_log'] = np.log1p(value)
            features['has_value'] = 1.0
            
            # Add unit type features
            unit = unit.strip()
            features['unit_is_volume'] = 1.0 if any(u in unit for u in ['oz', 'ml', 'liter', 'gallon']) else 0.0
            features['unit_is_weight'] = 1.0 if any(u in unit for u in ['lb', 'kg', 'gram', 'ounce']) else 0.0
            features['unit_is_count'] = 1.0 if 'count' in unit else 0.0
        
        return features
    
    def extract_text_statistics(self, text: str) -> Dict[str, float]:
        """
        Extract statistical features from text.
        
        Args:
            text: Catalog content text
            
        Returns:
            Dictionary with text statistics
        """
        if not text or pd.isna(text):
            return {
                'text_length': 0,
                'word_count': 0,
                'avg_word_length': 0,
                'sentence_count': 0,
                'has_bullets': 0.0,
                'bullet_count': 0
            }
        
        # Basic statistics
        text_length = len(text)
        words = text.split()
        word_count = len(words)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Count sentences (approximate)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Count bullet points
        bullet_count = text.count('Bullet Point')
        has_bullets = 1.0 if bullet_count > 0 else 0.0
        
        return {
            'text_length': text_length,
            'text_length_log': np.log1p(text_length),
            'word_count': word_count,
            'word_count_log': np.log1p(word_count),
            'avg_word_length': avg_word_length,
            'sentence_count': sentence_count,
            'has_bullets': has_bullets,
            'bullet_count': bullet_count
        }
    
    def extract_product_categories(self, text: str) -> Dict[str, float]:
        """
        Extract product category features based on keywords.
        
        Args:
            text: Catalog content text
            
        Returns:
            Dictionary with category features
        """
        text_lower = text.lower() if text else ""
        
        categories = {
            'cat_food_beverage': ['tea', 'coffee', 'juice', 'drink', 'beverage', 'food', 'snack'],
            'cat_candy_sweets': ['candy', 'chocolate', 'sweet', 'gummy', 'caramel'],
            'cat_health_wellness': ['organic', 'natural', 'gluten free', 'vegan', 'healthy'],
            'cat_spices_seasoning': ['spice', 'seasoning', 'herb', 'flavor', 'salt', 'pepper'],
            'cat_cooking': ['cooking', 'baking', 'kitchen', 'recipe'],
        }
        
        features = {}
        for cat_name, keywords in categories.items():
            features[cat_name] = 1.0 if any(kw in text_lower for kw in keywords) else 0.0
        
        return features
    
    def extract_brand_features(self, text: str) -> Dict[str, float]:
        """
        Extract brand-related features.
        
        Args:
            text: Catalog content text
            
        Returns:
            Dictionary with brand features
        """
        features = {
            'has_brand_name': 0.0,
            'is_premium': 0.0
        }
        
        if not text:
            return features
        
        text_lower = text.lower()
        
        # Check if "Item Name:" is present (indicates structured product name)
        if 'item name:' in text_lower:
            features['has_brand_name'] = 1.0
        
        # Premium indicators
        premium_keywords = ['premium', 'gourmet', 'organic', 'natural', 'artisan', 'craft']
        features['is_premium'] = 1.0 if any(kw in text_lower for kw in premium_keywords) else 0.0
        
        return features
    
    def fit_tfidf(self, texts: List[str]) -> None:
        """
        Fit TF-IDF vectorizer on texts.
        
        Args:
            texts: List of text documents
        """
        logger.info(f"Fitting TF-IDF with max_features={self.max_features}, ngram_range={self.ngram_range}")
        
        # Clean texts
        texts = [str(t) if t else "" for t in texts]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )
        
        self.tfidf_vectorizer.fit(texts)
        logger.info(f"TF-IDF fitted with {len(self.tfidf_vectorizer.vocabulary_)} features")
    
    def extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract TF-IDF features from texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            TF-IDF feature matrix
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf() first.")
        
        texts = [str(t) if t else "" for t in texts]
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def extract_all_features(self, texts: List[str], fit_tfidf: bool = False) -> np.ndarray:
        """
        Extract all text features.
        
        Args:
            texts: List of text documents
            fit_tfidf: Whether to fit TF-IDF vectorizer
            
        Returns:
            Combined feature matrix
        """
        logger.info(f"Extracting text features from {len(texts)} documents")
        
        # Fit TF-IDF if requested
        if fit_tfidf:
            self.fit_tfidf(texts)
        
        # Extract TF-IDF features
        tfidf_features = self.extract_tfidf_features(texts)
        
        # Extract other features
        other_features = []
        for text in texts:
            features = {}
            features.update(self.extract_ipq(text))
            features.update(self.extract_product_value(text))
            features.update(self.extract_text_statistics(text))
            features.update(self.extract_product_categories(text))
            features.update(self.extract_brand_features(text))
            other_features.append(list(features.values()))
        
        other_features = np.array(other_features)
        
        # Combine all features
        combined_features = np.hstack([tfidf_features, other_features])
        
        logger.info(f"Extracted {combined_features.shape[1]} total text features")
        logger.info(f"  - TF-IDF features: {tfidf_features.shape[1]}")
        logger.info(f"  - Other features: {other_features.shape[1]}")
        
        return combined_features
