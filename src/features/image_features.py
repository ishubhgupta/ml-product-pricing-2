"""
Image feature extraction module.
"""

import os
import numpy as np
import logging
from typing import List, Dict, Optional
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm


logger = logging.getLogger(__name__)


class ImageFeatureExtractor:
    """
    Extracts features from product images.
    """
    
    def __init__(self, image_dir: str, image_size: tuple = (224, 224), 
                 model_name: str = "simple", use_cache: bool = True):
        """
        Initialize image feature extractor.
        
        Args:
            image_dir: Directory to store downloaded images
            image_size: Target image size
            model_name: Model to use ('simple' or 'efficientnet')
            use_cache: Whether to cache downloaded images
        """
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.model_name = model_name
        self.use_cache = use_cache
        self.model = None
        
        if model_name == "efficientnet":
            self._init_efficientnet()
    
    def _init_efficientnet(self):
        """Initialize EfficientNet model for feature extraction."""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            logger.info("Loading EfficientNet-B0 model...")
            self.model = models.efficientnet_b0(pretrained=True)
            self.model.eval()
            
            # Remove classifier to get features
            self.model.classifier = torch.nn.Identity()
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("EfficientNet model loaded successfully")
        except ImportError:
            logger.warning("PyTorch not available, falling back to simple features")
            self.model_name = "simple"
    
    def download_image(self, image_url: str, sample_id: int) -> Optional[str]:
        """
        Download image from URL.
        
        Args:
            image_url: URL of the image
            sample_id: Sample ID for naming
            
        Returns:
            Path to downloaded image or None if failed
        """
        image_path = self.image_dir / f"{sample_id}.jpg"
        
        # Check cache
        if self.use_cache and image_path.exists():
            return str(image_path)
        
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Verify it's an image
            img = Image.open(BytesIO(response.content))
            img.verify()
            
            # Save image
            img = Image.open(BytesIO(response.content))
            img.save(image_path)
            
            return str(image_path)
        except Exception as e:
            logger.debug(f"Failed to download image {sample_id}: {str(e)}")
            return None
    
    def download_images_batch(self, image_links: List[str], sample_ids: List[int], 
                             max_workers: int = 10) -> Dict[int, Optional[str]]:
        """
        Download multiple images in parallel.
        
        Args:
            image_links: List of image URLs
            sample_ids: List of sample IDs
            max_workers: Number of parallel workers
            
        Returns:
            Dictionary mapping sample_id to image path
        """
        logger.info(f"Downloading {len(image_links)} images...")
        
        results = {}
        successful = 0
        
        for sample_id, image_url in tqdm(zip(sample_ids, image_links), total=len(image_links), desc="Downloading images"):
            path = self.download_image(image_url, sample_id)
            results[sample_id] = path
            if path is not None:
                successful += 1
        
        logger.info(f"Successfully downloaded {successful}/{len(image_links)} images ({successful/len(image_links)*100:.1f}%)")
        
        return results
    
    def extract_simple_features(self, image_path: str) -> np.ndarray:
        """
        Extract simple features from image (color, size, etc.).
        
        Args:
            image_path: Path to image file
            
        Returns:
            Feature vector
        """
        try:
            img = Image.open(image_path)
            
            # Image dimensions
            width, height = img.size
            aspect_ratio = width / height if height > 0 else 1.0
            area = width * height
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize for consistent processing
            img = img.resize((64, 64))
            img_array = np.array(img)
            
            # Color histogram (simplified)
            r_hist = np.histogram(img_array[:,:,0], bins=8, range=(0, 256))[0]
            g_hist = np.histogram(img_array[:,:,1], bins=8, range=(0, 256))[0]
            b_hist = np.histogram(img_array[:,:,2], bins=8, range=(0, 256))[0]
            
            # Normalize histograms
            r_hist = r_hist / r_hist.sum()
            g_hist = g_hist / g_hist.sum()
            b_hist = b_hist / b_hist.sum()
            
            # Mean colors
            mean_r = img_array[:,:,0].mean() / 255.0
            mean_g = img_array[:,:,1].mean() / 255.0
            mean_b = img_array[:,:,2].mean() / 255.0
            
            # Color variance
            var_r = img_array[:,:,0].std() / 255.0
            var_g = img_array[:,:,1].std() / 255.0
            var_b = img_array[:,:,2].std() / 255.0
            
            # Brightness
            brightness = (mean_r + mean_g + mean_b) / 3.0
            
            # Combine features
            features = np.concatenate([
                [width, height, aspect_ratio, np.log1p(area)],
                r_hist, g_hist, b_hist,
                [mean_r, mean_g, mean_b],
                [var_r, var_g, var_b],
                [brightness]
            ])
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting features from {image_path}: {str(e)}")
            # Return zero features if error
            return np.zeros(44)  # 4 + 8*3 + 3 + 3 + 1 = 44 features
    
    def extract_cnn_features(self, image_path: str) -> np.ndarray:
        """
        Extract CNN features using pre-trained model.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Feature vector
        """
        if self.model is None:
            return self.extract_simple_features(image_path)
        
        try:
            import torch
            
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            
            with torch.no_grad():
                features = self.model(img_tensor)
            
            return features.squeeze().numpy()
            
        except Exception as e:
            logger.debug(f"Error extracting CNN features from {image_path}: {str(e)}")
            return np.zeros(1280)  # EfficientNet-B0 feature size
    
    def extract_features_batch(self, image_paths: List[Optional[str]]) -> np.ndarray:
        """
        Extract features from multiple images.
        
        Args:
            image_paths: List of image paths (can contain None)
            
        Returns:
            Feature matrix
        """
        logger.info(f"Extracting features from {len(image_paths)} images using {self.model_name} model")
        
        features_list = []
        valid_count = 0
        
        for path in tqdm(image_paths, desc="Extracting image features"):
            if path is None or not os.path.exists(path):
                # Use zero features for missing images
                if self.model_name == "efficientnet":
                    features = np.zeros(1280)
                else:
                    features = np.zeros(44)
            else:
                if self.model_name == "efficientnet":
                    features = self.extract_cnn_features(path)
                else:
                    features = self.extract_simple_features(path)
                valid_count += 1
            
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        logger.info(f"Extracted features from {valid_count}/{len(image_paths)} valid images")
        logger.info(f"Feature matrix shape: {features_array.shape}")
        
        return features_array
    
    def extract_all_features(self, image_links: List[str], sample_ids: List[int]) -> np.ndarray:
        """
        Download images and extract features.
        
        Args:
            image_links: List of image URLs
            sample_ids: List of sample IDs
            
        Returns:
            Feature matrix
        """
        # Download images
        image_paths_dict = self.download_images_batch(image_links, sample_ids)
        
        # Create ordered list of paths
        image_paths = [image_paths_dict.get(sid) for sid in sample_ids]
        
        # Extract features
        features = self.extract_features_batch(image_paths)
        
        return features
