"""
Integration test for HAM10000 data loading and preprocessing pipeline.

Tests the complete pipeline from data loading to preprocessing.
"""

import pytest
import numpy as np
from src.data_loader import create_ham10000_loader
from src.image_preprocessing import HAM10000ImagePreprocessor
from src.dataset_splits import create_ham10000_splits


def test_complete_pipeline_integration():
    """Test the complete data loading and preprocessing pipeline."""
    try:
        # Test 1: Load HAM10000 data
        loader = create_ham10000_loader()
        metadata = loader.load_metadata()
        
        print(f"Loaded {len(metadata)} metadata records")
        assert len(metadata) > 0, "Should load metadata successfully"
        
        # Test 2: Load a small sample of images
        sample_image_ids = metadata['image_id'].head(10).tolist()
        images = loader.load_images(sample_image_ids)
        
        successful_images = {img_id: img for img_id, img in images.items() if img is not None}
        print(f"Successfully loaded {len(successful_images)} out of {len(sample_image_ids)} images")
        
        # Test 3: Create dataset splits
        train_metadata, val_metadata, test_metadata = create_ham10000_splits(
            metadata, test_size=0.2, val_size=0.1, clean_data=True
        )
        
        print(f"Dataset splits - Train: {len(train_metadata)}, Val: {len(val_metadata)}, Test: {len(test_metadata)}")
        assert len(train_metadata) > 0, "Should have training data"
        assert len(test_metadata) > 0, "Should have test data"
        
        # Test 4: Image preprocessing
        if successful_images:
            preprocessor = HAM10000ImagePreprocessor()
            
            # Test single image preprocessing
            sample_image = list(successful_images.values())[0]
            processed = preprocessor.preprocess_image(sample_image)
            
            print(f"Processed image shape: {processed.shape}")
            assert processed.shape == (3, 224, 224), "Should resize to 224x224"
            
            # Test batch preprocessing
            sample_images = list(successful_images.values())[:3]
            batch_processed = preprocessor.preprocess_batch(sample_images)
            
            print(f"Batch processed shape: {batch_processed.shape}")
            assert batch_processed.shape[0] == len(sample_images), "Batch size should match"
            assert batch_processed.shape[1:] == (3, 224, 224), "Each image should be 3x224x224"
            
            # Test validation
            validation = preprocessor.validate_preprocessing(batch_processed)
            assert validation['valid'], f"Preprocessing validation should pass: {validation}"
        
        print("✓ Complete pipeline integration test passed!")
        
    except FileNotFoundError:
        pytest.skip("HAM10000 dataset not available for integration testing")
    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")


def test_label_encoding_integration():
    """Test label encoding with real HAM10000 data."""
    try:
        loader = create_ham10000_loader()
        metadata = loader.load_metadata()
        
        # Test label encoding
        sample_labels = metadata['dx'].head(100)
        encoded = loader.encode_labels(sample_labels)
        decoded = loader.decode_labels(encoded)
        
        # Check that encoding/decoding is consistent
        for orig, dec in zip(sample_labels, decoded):
            assert orig == dec, f"Label encoding should be reversible: {orig} != {dec}"
        
        # Check class weights calculation
        weights = loader.get_class_weights(encoded)
        assert len(weights) == len(loader.LESION_CLASSES), "Should have weights for all classes"
        
        print("✓ Label encoding integration test passed!")
        
    except FileNotFoundError:
        pytest.skip("HAM10000 dataset not available for integration testing")


if __name__ == "__main__":
    test_complete_pipeline_integration()
    test_label_encoding_integration()
    print("All integration tests passed!")