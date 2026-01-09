"""
Image Preprocessing Pipeline for HAM10000 Dataset

This module implements image preprocessing for EfficientNet-B0 input including
resizing to 224x224 and ImageNet normalization.

Requirements: 1.2
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Optional, Dict, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HAM10000ImagePreprocessor:
    """
    Image preprocessing pipeline for HAM10000 dataset.
    
    Handles resizing images to 224x224 for EfficientNet-B0 input and applies
    ImageNet normalization.
    """
    
    # ImageNet normalization parameters
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Target image size for EfficientNet-B0
    TARGET_SIZE = (224, 224)
    
    def __init__(self, 
                 target_size: Tuple[int, int] = TARGET_SIZE,
                 normalize_mean: List[float] = None,
                 normalize_std: List[float] = None,
                 interpolation: str = 'bilinear'):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            normalize_mean: Mean values for normalization (defaults to ImageNet)
            normalize_std: Std values for normalization (defaults to ImageNet)
            interpolation: Interpolation method for resizing ('bilinear', 'nearest', 'bicubic')
        """
        self.target_size = target_size
        self.normalize_mean = normalize_mean or self.IMAGENET_MEAN
        self.normalize_std = normalize_std or self.IMAGENET_STD
        
        # Validate normalization parameters
        if len(self.normalize_mean) != 3 or len(self.normalize_std) != 3:
            raise ValueError("Normalization mean and std must have 3 values for RGB channels")
        
        # Set interpolation mode
        interpolation_modes = {
            'bilinear': transforms.InterpolationMode.BILINEAR,
            'nearest': transforms.InterpolationMode.NEAREST,
            'bicubic': transforms.InterpolationMode.BICUBIC
        }
        
        if interpolation not in interpolation_modes:
            raise ValueError(f"Unsupported interpolation mode: {interpolation}")
        
        self.interpolation = interpolation_modes[interpolation]
        
        # Create preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=self.interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
        
        # Create transform without normalization for validation
        self.transform_no_norm = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=self.interpolation),
            transforms.ToTensor()
        ])
        
        logger.info(f"Initialized preprocessor with target size {self.target_size}")
        logger.info(f"Normalization - Mean: {self.normalize_mean}, Std: {self.normalize_std}")
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image], normalize: bool = True) -> torch.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image: Input image as numpy array (H, W, C) or PIL Image
            normalize: Whether to apply normalization
            
        Returns:
            Preprocessed image tensor (C, H, W)
            
        Raises:
            ValueError: If image format is invalid
            RuntimeError: If preprocessing fails
        """
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                if image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
                
                # Ensure uint8 format
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        # Assume normalized to [0, 1]
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                image = Image.fromarray(image)
            
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Expected numpy array or PIL Image, got {type(image)}")
            
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply preprocessing
            if normalize:
                processed = self.transform(image)
            else:
                processed = self.transform_no_norm(image)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise RuntimeError(f"Image preprocessing failed: {e}")
    
    def preprocess_batch(self, images: List[Union[np.ndarray, Image.Image]], normalize: bool = True) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of input images
            normalize: Whether to apply normalization
            
        Returns:
            Batch tensor (N, C, H, W)
        """
        processed_images = []
        failed_count = 0
        
        for i, image in enumerate(images):
            try:
                processed = self.preprocess_image(image, normalize=normalize)
                processed_images.append(processed)
            except Exception as e:
                logger.warning(f"Failed to preprocess image {i}: {e}")
                failed_count += 1
                # Create a zero tensor as placeholder
                if normalize:
                    placeholder = torch.zeros(3, *self.target_size)
                else:
                    placeholder = torch.zeros(3, *self.target_size)
                processed_images.append(placeholder)
        
        if failed_count > 0:
            logger.warning(f"Failed to preprocess {failed_count}/{len(images)} images")
        
        # Stack into batch tensor
        batch_tensor = torch.stack(processed_images)
        return batch_tensor
    
    def validate_preprocessing(self, processed_tensor: torch.Tensor) -> Dict[str, bool]:
        """
        Validate that preprocessing was applied correctly.
        
        Args:
            processed_tensor: Preprocessed image tensor (C, H, W) or batch (N, C, H, W)
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        # Handle both single image and batch
        if processed_tensor.ndim == 3:
            # Single image (C, H, W)
            batch_tensor = processed_tensor.unsqueeze(0)
        elif processed_tensor.ndim == 4:
            # Batch (N, C, H, W)
            batch_tensor = processed_tensor
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {processed_tensor.ndim}D")
        
        # Check dimensions
        n, c, h, w = batch_tensor.shape
        validation_results['correct_channels'] = (c == 3)
        validation_results['correct_height'] = (h == self.target_size[0])
        validation_results['correct_width'] = (w == self.target_size[1])
        validation_results['correct_shape'] = all([
            validation_results['correct_channels'],
            validation_results['correct_height'],
            validation_results['correct_width']
        ])
        
        # Check data type
        validation_results['correct_dtype'] = (batch_tensor.dtype == torch.float32)
        
        # Check value ranges (for normalized images, values should be roughly in [-3, 3] range)
        min_val = batch_tensor.min().item()
        max_val = batch_tensor.max().item()
        validation_results['reasonable_range'] = (-5.0 <= min_val <= 5.0 and -5.0 <= max_val <= 5.0)
        
        # Check for NaN or infinite values
        validation_results['no_nan'] = not torch.isnan(batch_tensor).any().item()
        validation_results['no_inf'] = not torch.isinf(batch_tensor).any().item()
        
        # Overall validation
        validation_results['valid'] = all([
            validation_results['correct_shape'],
            validation_results['correct_dtype'],
            validation_results['reasonable_range'],
            validation_results['no_nan'],
            validation_results['no_inf']
        ])
        
        return validation_results
    
    def denormalize(self, normalized_tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor back to [0, 1] range.
        
        Args:
            normalized_tensor: Normalized tensor (C, H, W) or (N, C, H, W)
            
        Returns:
            Denormalized tensor in [0, 1] range
        """
        # Convert mean and std to tensors
        mean = torch.tensor(self.normalize_mean).view(-1, 1, 1)
        std = torch.tensor(self.normalize_std).view(-1, 1, 1)
        
        # Handle batch dimension
        if normalized_tensor.ndim == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        
        # Denormalize: x = (x_norm * std) + mean
        denormalized = normalized_tensor * std + mean
        
        # Clamp to [0, 1] range
        denormalized = torch.clamp(denormalized, 0.0, 1.0)
        
        return denormalized
    
    def create_dataset(self, 
                      dataframe,
                      image_dirs: List[str],
                      transform: Optional[transforms.Compose] = None,
                      augment_minority_classes: bool = False,
                      augmentation_factor: int = 2) -> Dataset:
        """
        Create dataset from dataframe with advanced augmentation options.
        
        Args:
            dataframe: DataFrame with image metadata
            image_dirs: List of directories containing images
            transform: Transform to apply to images
            augment_minority_classes: Whether to augment minority classes
            augmentation_factor: Factor by which to augment minority classes
            
        Returns:
            HAM10000Dataset instance
        """
        from data_loader import HAM10000DataLoader
        
        # Load images
        data_loader = HAM10000DataLoader()
        images = {}
        labels = {}
        
        for _, row in dataframe.iterrows():
            image_id = row['image_id']
            
            # Try to load image from any of the directories
            image_loaded = False
            for img_dir in image_dirs:
                try:
                    image = data_loader.load_single_image(image_id, img_dir)
                    if image is not None:
                        images[image_id] = image
                        labels[image_id] = row['label']
                        image_loaded = True
                        break
                except:
                    continue
            
            if not image_loaded:
                logger.warning(f"Could not load image {image_id}")
        
        # Handle minority class augmentation
        image_ids = list(images.keys())
        
        if augment_minority_classes and augmentation_factor > 1:
            # Calculate class frequencies
            class_counts = {}
            for img_id in image_ids:
                label = labels[img_id]
                class_counts[label] = class_counts.get(label, 0) + 1
            
            # Find minority classes (less than median frequency)
            frequencies = list(class_counts.values())
            median_freq = np.median(frequencies)
            
            # Augment minority classes
            augmented_ids = image_ids.copy()
            for img_id in image_ids:
                label = labels[img_id]
                if class_counts[label] < median_freq:
                    # Add multiple copies for augmentation
                    for _ in range(augmentation_factor - 1):
                        augmented_ids.append(img_id)
            
            image_ids = augmented_ids
            logger.info(f"Augmented dataset from {len(images)} to {len(image_ids)} samples")
        
        return HAM10000Dataset(images, labels, image_ids, self, transform)
        
    
    def get_preprocessing_stats(self) -> Dict[str, Union[List[float], Tuple[int, int]]]:
        """
        Get preprocessing configuration statistics.
        
        Returns:
            Dictionary with preprocessing parameters
        """
        return {
            'target_size': self.target_size,
            'normalize_mean': self.normalize_mean,
            'normalize_std': self.normalize_std,
            'interpolation': str(self.interpolation)
        }


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000 with preprocessing.
    """
    
    def __init__(self, 
                 images: Dict[str, np.ndarray],
                 labels: Dict[str, int],
                 image_ids: List[str],
                 preprocessor: HAM10000ImagePreprocessor,
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize HAM10000 dataset.
        
        Args:
            images: Dictionary mapping image_id to image array
            labels: Dictionary mapping image_id to label
            image_ids: List of image IDs to include in dataset
            preprocessor: Image preprocessor instance
            transform: Additional transforms to apply
        """
        self.images = images
        self.labels = labels
        self.image_ids = [img_id for img_id in image_ids if img_id in images and img_id in labels]
        self.preprocessor = preprocessor
        self.transform = transform
        
        # Filter out missing images/labels
        missing_images = [img_id for img_id in image_ids if img_id not in images]
        missing_labels = [img_id for img_id in image_ids if img_id not in labels]
        
        if missing_images:
            logger.warning(f"Missing images for {len(missing_images)} image IDs")
        if missing_labels:
            logger.warning(f"Missing labels for {len(missing_labels)} image IDs")
        
        logger.info(f"Dataset initialized with {len(self.image_ids)} samples")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (preprocessed_image, label, image_id)
        """
        image_id = self.image_ids[idx]
        image = self.images[image_id]
        label = self.labels[image_id]
        
        # Preprocess image
        try:
            # If additional transforms are provided, apply them to the PIL image first
            # before the preprocessor's transforms
            if self.transform:
                # Convert numpy array to PIL Image if needed
                if isinstance(image, np.ndarray):
                    if image.dtype != np.uint8:
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)
                    pil_image = Image.fromarray(image)
                else:
                    pil_image = image
                
                # Ensure RGB mode
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Apply additional transforms to PIL image
                transformed_image = self.transform(pil_image)
                
                # If transform returns a tensor, we're done
                if isinstance(transformed_image, torch.Tensor):
                    return transformed_image, label, image_id
                else:
                    # If transform returns PIL image, apply preprocessor
                    processed_image = self.preprocessor.preprocess_image(transformed_image, normalize=True)
            else:
                # No additional transforms, just preprocess
                processed_image = self.preprocessor.preprocess_image(image, normalize=True)
            
            return processed_image, label, image_id
            
        except Exception as e:
            logger.error(f"Error processing sample {idx} (image_id: {image_id}): {e}")
            # Return zero tensor as fallback
            zero_tensor = torch.zeros(3, *self.preprocessor.target_size)
            return zero_tensor, label, image_id


def create_data_loaders(images: Dict[str, np.ndarray],
                       labels: Dict[str, int],
                       train_image_ids: List[str],
                       val_image_ids: List[str],
                       test_image_ids: List[str],
                       batch_size: int = 32,
                       num_workers: int = 0,
                       preprocessor: HAM10000ImagePreprocessor = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch data loaders for train, validation, and test sets.
    
    Args:
        images: Dictionary mapping image_id to image array
        labels: Dictionary mapping image_id to label
        train_image_ids: List of training image IDs
        val_image_ids: List of validation image IDs
        test_image_ids: List of test image IDs
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        preprocessor: Image preprocessor (creates default if None)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if preprocessor is None:
        preprocessor = HAM10000ImagePreprocessor()
    
    # Create datasets
    train_dataset = HAM10000Dataset(images, labels, train_image_ids, preprocessor)
    val_dataset = HAM10000Dataset(images, labels, val_image_ids, preprocessor)
    test_dataset = HAM10000Dataset(images, labels, test_image_ids, preprocessor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Created data loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    preprocessor = HAM10000ImagePreprocessor()
    
    # Test with a dummy image
    dummy_image = np.random.randint(0, 255, (450, 600, 3), dtype=np.uint8)
    
    # Preprocess single image
    processed = preprocessor.preprocess_image(dummy_image)
    print(f"Processed image shape: {processed.shape}")
    
    # Validate preprocessing
    validation = preprocessor.validate_preprocessing(processed)
    print(f"Validation results: {validation}")
    
    # Test batch processing
    batch_images = [dummy_image] * 4
    batch_processed = preprocessor.preprocess_batch(batch_images)
    print(f"Batch processed shape: {batch_processed.shape}")
    
    # Test denormalization
    denormalized = preprocessor.denormalize(processed)
    print(f"Denormalized range: [{denormalized.min():.3f}, {denormalized.max():.3f}]")