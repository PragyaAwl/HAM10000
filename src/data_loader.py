"""
HAM10000 Dataset Loader

This module implements the HAM10000 dataset loader with metadata parsing,
image loading, and label encoding for 7 skin lesion classes.

Requirements: 1.1, 1.5
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HAM10000DataLoader:
    """
    HAM10000 dataset loader with metadata parsing and image loading capabilities.
    
    Supports loading metadata CSV, parsing lesion classifications, and loading images
    with proper error handling for corrupted files.
    """
    
    # 7 skin lesion classes as specified in requirements
    LESION_CLASSES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
    
    def __init__(self, 
                 metadata_path: str = None,
                 images_part1_path: str = None,
                 images_part2_path: str = None):
        """
        Initialize the HAM10000 data loader.
        
        Args:
            metadata_path: Path to HAM10000_metadata.csv (optional)
            images_part1_path: Path to HAM10000_images_part_1 directory (optional)
            images_part2_path: Path to HAM10000_images_part_2 directory (optional)
        """
        if metadata_path:
            self.metadata_path = Path(metadata_path)
            # Validate paths exist
            if not self.metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        else:
            self.metadata_path = None
            
        if images_part1_path:
            self.images_part1_path = Path(images_part1_path)
            if not self.images_part1_path.exists():
                raise FileNotFoundError(f"Images part 1 directory not found: {images_part1_path}")
        else:
            self.images_part1_path = None
            
        if images_part2_path:
            self.images_part2_path = Path(images_part2_path)
            if not self.images_part2_path.exists():
                raise FileNotFoundError(f"Images part 2 directory not found: {images_part2_path}")
        else:
            self.images_part2_path = None
        
        # Initialize label encoder
        self.label_to_idx = {label: idx for idx, label in enumerate(self.LESION_CLASSES)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Cache for loaded metadata
        self._metadata = None
        
    def load_single_image(self, image_id: str, image_dir: str = None) -> Optional[np.ndarray]:
        """
        Load a single image by ID from a specific directory or auto-detect.
        
        Args:
            image_id: Image ID (e.g., 'ISIC_0027419')
            image_dir: Specific directory to search (optional)
            
        Returns:
            Image as numpy array (H, W, C) or None if loading failed
        """
        if image_dir:
            # Load from specific directory
            filename = f"{image_id}.jpg"
            image_path = Path(image_dir) / filename
            if not image_path.exists():
                return None
                
            try:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    return np.array(img)
            except Exception as e:
                logger.warning(f"Error loading image {image_id} from {image_dir}: {e}")
                return None
        else:
            # Use existing load_image method
            return self.load_image(image_id)
    def load_metadata(self, metadata_path: str = None) -> pd.DataFrame:
        """
        Load and parse HAM10000 metadata CSV.
        
        Args:
            metadata_path: Optional path to metadata file (uses instance path if None)
        
        Returns:
            DataFrame with columns: lesion_id, image_id, dx, dx_type, age, sex, localization
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
            pd.errors.EmptyDataError: If metadata file is empty
            ValueError: If required columns are missing
        """
        if metadata_path:
            metadata_file = Path(metadata_path)
        else:
            metadata_file = self.metadata_path
            
        try:
            logger.info(f"Loading metadata from {metadata_file}")
            metadata = pd.read_csv(metadata_file)
            
            # Validate required columns
            required_columns = ['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization']
            missing_columns = [col for col in required_columns if col not in metadata.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in metadata: {missing_columns}")
            
            # Add label encoding
            metadata['label'] = metadata['dx'].map(self.label_to_idx)
            
            # Validate diagnosis classes
            unique_dx = set(metadata['dx'].unique())
            expected_dx = set(self.LESION_CLASSES)
            if not unique_dx.issubset(expected_dx):
                unexpected = unique_dx - expected_dx
                logger.warning(f"Found unexpected diagnosis classes: {unexpected}")
            
            logger.info(f"Loaded metadata with {len(metadata)} records")
            logger.info(f"Class distribution:\n{metadata['dx'].value_counts()}")
            
            self._metadata = metadata
            return metadata
            
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"Metadata file is empty: {metadata_file}")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
    
    def get_image_path(self, image_id: str) -> Optional[Path]:
        """
        Get the full path to an image file.
        
        Args:
            image_id: Image ID (e.g., 'ISIC_0027419')
            
        Returns:
            Path to image file if found, None otherwise
        """
        filename = f"{image_id}.jpg"
        
        # Check part 1 directory first
        if self.images_part1_path:
            path1 = self.images_part1_path / filename
            if path1.exists():
                return path1
            
        # Check part 2 directory
        if self.images_part2_path:
            path2 = self.images_part2_path / filename
            if path2.exists():
                return path2
            
        return None
    
    def load_image(self, image_id: str) -> Optional[np.ndarray]:
        """
        Load a single image by ID with error handling for corrupted files.
        
        Args:
            image_id: Image ID (e.g., 'ISIC_0027419')
            
        Returns:
            Image as numpy array (H, W, C) or None if loading failed
        """
        image_path = self.get_image_path(image_id)
        if image_path is None:
            logger.warning(f"Image not found: {image_id}")
            return None
            
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.array(img)
                
        except (IOError, OSError) as e:
            logger.warning(f"Failed to load corrupted image {image_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading image {image_id}: {e}")
            return None
    
    def load_images(self, image_ids: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Load multiple images by ID with error handling.
        
        Args:
            image_ids: List of image IDs
            
        Returns:
            Dictionary mapping image_id to image array (or None if failed)
        """
        images = {}
        failed_count = 0
        
        logger.info(f"Loading {len(image_ids)} images...")
        
        for i, image_id in enumerate(image_ids):
            if i % 1000 == 0:
                logger.info(f"Loaded {i}/{len(image_ids)} images")
                
            image = self.load_image(image_id)
            images[image_id] = image
            
            if image is None:
                failed_count += 1
        
        logger.info(f"Loaded {len(image_ids) - failed_count}/{len(image_ids)} images successfully")
        if failed_count > 0:
            logger.warning(f"Failed to load {failed_count} images")
            
        return images
    
    def encode_labels(self, diagnoses: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Encode diagnosis labels to numerical values.
        
        Args:
            diagnoses: List or Series of diagnosis codes (dx)
            
        Returns:
            Numpy array of encoded labels
            
        Raises:
            ValueError: If unknown diagnosis codes are found
        """
        if isinstance(diagnoses, pd.Series):
            diagnoses = diagnoses.tolist()
            
        encoded = []
        unknown_labels = set()
        
        for dx in diagnoses:
            if dx in self.label_to_idx:
                encoded.append(self.label_to_idx[dx])
            else:
                unknown_labels.add(dx)
                # Use -1 for unknown labels (will be filtered out later)
                encoded.append(-1)
        
        if unknown_labels:
            logger.warning(f"Found unknown diagnosis codes: {unknown_labels}")
            
        return np.array(encoded)
    
    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """
        Decode numerical labels back to diagnosis codes.
        
        Args:
            encoded_labels: Array of encoded labels
            
        Returns:
            List of diagnosis codes
        """
        decoded = []
        for label in encoded_labels:
            if label in self.idx_to_label:
                decoded.append(self.idx_to_label[label])
            else:
                decoded.append('unknown')
        return decoded
    
    def get_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced dataset.
        
        Args:
            labels: Array of encoded labels
            
        Returns:
            Dictionary mapping class index to weight
        """
        from collections import Counter
        
        # Count valid labels (exclude -1)
        valid_labels = labels[labels >= 0]
        class_counts = Counter(valid_labels)
        
        # Calculate inverse frequency weights
        total_samples = len(valid_labels)
        num_classes = len(self.LESION_CLASSES)
        
        weights = {}
        for class_idx in range(num_classes):
            count = class_counts.get(class_idx, 1)  # Avoid division by zero
            weights[class_idx] = total_samples / (num_classes * count)
            
        return weights
    
    def get_metadata(self) -> pd.DataFrame:
        """
        Get cached metadata or load if not already loaded.
        
        Returns:
            Metadata DataFrame
        """
        if self._metadata is None:
            return self.load_metadata()
        return self._metadata
    
    def get_dataset_info(self) -> Dict:
        """
        Get comprehensive dataset information.
        
        Returns:
            Dictionary with dataset statistics and information
        """
        metadata = self.get_metadata()
        
        info = {
            'total_samples': len(metadata),
            'unique_lesions': metadata['lesion_id'].nunique(),
            'unique_images': metadata['image_id'].nunique(),
            'class_distribution': metadata['dx'].value_counts().to_dict(),
            'age_stats': {
                'mean': metadata['age'].mean(),
                'std': metadata['age'].std(),
                'min': metadata['age'].min(),
                'max': metadata['age'].max(),
                'missing': metadata['age'].isna().sum()
            },
            'sex_distribution': metadata['sex'].value_counts().to_dict(),
            'localization_distribution': metadata['localization'].value_counts().to_dict(),
            'dx_type_distribution': metadata['dx_type'].value_counts().to_dict()
        }
        
        return info


def create_ham10000_loader(base_path: str = None) -> HAM10000DataLoader:
    """
    Convenience function to create HAM10000DataLoader with default paths.
    
    Args:
        base_path: Base path to HAM10000 data. If None, uses default kaggle cache path.
        
    Returns:
        Configured HAM10000DataLoader instance
    """
    if base_path is None:
        # Default path for kaggle dataset
        base_path = "c:/Users/agarw/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2"
    
    base_path = Path(base_path)
    
    metadata_path = base_path / "HAM10000_metadata.csv"
    images_part1_path = base_path / "HAM10000_images_part_1"
    images_part2_path = base_path / "HAM10000_images_part_2"
    
    return HAM10000DataLoader(
        metadata_path=str(metadata_path),
        images_part1_path=str(images_part1_path),
        images_part2_path=str(images_part2_path)
    )


if __name__ == "__main__":
    # Example usage
    loader = create_ham10000_loader()
    
    # Load metadata
    metadata = loader.load_metadata()
    print(f"Loaded {len(metadata)} metadata records")
    
    # Get dataset info
    info = loader.get_dataset_info()
    print(f"Dataset info: {info}")
    
    # Test image loading
    sample_image_ids = metadata['image_id'].head(5).tolist()
    images = loader.load_images(sample_image_ids)
    print(f"Loaded {sum(1 for img in images.values() if img is not None)} out of {len(sample_image_ids)} sample images")
    
    # Test label encoding
    sample_labels = metadata['dx'].head(10)
    encoded = loader.encode_labels(sample_labels)
    decoded = loader.decode_labels(encoded)
    print(f"Label encoding test: {list(zip(sample_labels.tolist(), encoded, decoded))}")