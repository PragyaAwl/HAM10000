#!/usr/bin/env python3
"""
Quick test to verify transform pipeline is working correctly.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import HAM10000DataLoader
from image_preprocessing import HAM10000ImagePreprocessor
from models.advanced_trainer import AdvancedAugmentation

def test_transform_pipeline():
    """Test that the transform pipeline works without errors."""
    
    print("🧪 Testing transform pipeline...")
    
    # Load a small sample of data
    data_loader = HAM10000DataLoader()
    metadata_path = "c:/Users/agarw/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/HAM10000_metadata.csv"
    metadata = data_loader.load_metadata(metadata_path)
    
    # Take just first 5 samples
    sample_metadata = metadata.head(5)
    print(f"Testing with {len(sample_metadata)} samples")
    
    # Load images
    images_dir1 = "c:/Users/agarw/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_1"
    images_dir2 = "c:/Users/agarw/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_2"
    
    images = {}
    labels = {}
    
    for _, row in sample_metadata.iterrows():
        image_id = row['image_id']
        
        # Try to load image
        image = data_loader.load_single_image(image_id, images_dir1)
        if image is None:
            image = data_loader.load_single_image(image_id, images_dir2)
        
        if image is not None:
            images[image_id] = image
            labels[image_id] = row['label']
            print(f"✓ Loaded {image_id}: {image.shape}")
        else:
            print(f"✗ Failed to load {image_id}")
    
    if not images:
        print("❌ No images loaded, cannot test transforms")
        return
    
    # Create preprocessor and dataset
    preprocessor = HAM10000ImagePreprocessor()
    image_ids = list(images.keys())
    
    # Test with PIL-only transforms
    print("\n🔄 Testing with PIL-only transforms...")
    train_transform = AdvancedAugmentation.get_train_transforms_pil_only()
    
    from image_preprocessing import HAM10000Dataset
    dataset = HAM10000Dataset(images, labels, image_ids, preprocessor, train_transform)
    
    # Test loading samples
    for i in range(min(3, len(dataset))):
        try:
            sample, label, img_id = dataset[i]
            print(f"✓ Sample {i}: {sample.shape}, label: {label}, id: {img_id}")
        except Exception as e:
            print(f"✗ Sample {i} failed: {e}")
            return
    
    print("\n🎉 Transform pipeline test PASSED!")
    return True

if __name__ == "__main__":
    test_transform_pipeline()