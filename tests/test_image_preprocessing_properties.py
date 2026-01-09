"""
Property-based tests for image preprocessing pipeline.

Tests Property 2: Image preprocessing consistency
Validates: Requirements 1.2
"""

import pytest
import numpy as np
import torch
from PIL import Image
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import List, Tuple

from src.image_preprocessing import HAM10000ImagePreprocessor


class TestImagePreprocessingProperties:
    """Property-based tests for image preprocessing."""
    
    @given(
        height=st.integers(min_value=50, max_value=1000),
        width=st.integers(min_value=50, max_value=1000),
        batch_size=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_2_image_preprocessing_consistency(self, height, width, batch_size):
        """
        Property 2: Image preprocessing consistency
        
        For any set of input images, preprocessing should produce images with 
        identical dimensions and pixel values in the normalized range.
        
        Validates: Requirements 1.2
        """
        # Create preprocessor
        preprocessor = HAM10000ImagePreprocessor()
        
        # Generate random images with different sizes
        images = []
        for _ in range(batch_size):
            # Create random RGB image
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            images.append(image)
        
        # Preprocess images individually
        processed_individual = []
        for image in images:
            processed = preprocessor.preprocess_image(image, normalize=True)
            processed_individual.append(processed)
        
        # Preprocess images as batch
        processed_batch = preprocessor.preprocess_batch(images, normalize=True)
        
        # Property 1: All processed images should have identical dimensions
        target_shape = (3, 224, 224)  # (C, H, W)
        for i, processed in enumerate(processed_individual):
            assert processed.shape == target_shape, f"Image {i} should have shape {target_shape}, got {processed.shape}"
        
        assert processed_batch.shape == (batch_size, 3, 224, 224), f"Batch should have shape ({batch_size}, 3, 224, 224), got {processed_batch.shape}"
        
        # Property 2: Individual and batch processing should produce identical results
        for i in range(batch_size):
            torch.testing.assert_close(
                processed_individual[i], 
                processed_batch[i], 
                msg=f"Individual and batch processing should be identical for image {i}"
            )
        
        # Property 3: Pixel values should be in reasonable normalized range
        # After ImageNet normalization, values typically range from about -2.5 to 2.5
        for i, processed in enumerate(processed_individual):
            min_val = processed.min().item()
            max_val = processed.max().item()
            assert -5.0 <= min_val <= 5.0, f"Image {i} min value {min_val} outside reasonable range [-5, 5]"
            assert -5.0 <= max_val <= 5.0, f"Image {i} max value {max_val} outside reasonable range [-5, 5]"
        
        # Property 4: No NaN or infinite values
        for i, processed in enumerate(processed_individual):
            assert not torch.isnan(processed).any(), f"Image {i} contains NaN values"
            assert not torch.isinf(processed).any(), f"Image {i} contains infinite values"
        
        # Property 5: Preprocessing validation should pass
        for i, processed in enumerate(processed_individual):
            validation = preprocessor.validate_preprocessing(processed)
            assert validation['valid'], f"Image {i} failed preprocessing validation: {validation}"
    
    @given(
        original_height=st.integers(min_value=50, max_value=1000),
        original_width=st.integers(min_value=50, max_value=1000),
        target_height=st.integers(min_value=32, max_value=512),
        target_width=st.integers(min_value=32, max_value=512)
    )
    @settings(max_examples=50)
    def test_property_2_resize_consistency(self, original_height, original_width, target_height, target_width):
        """
        Property 2 extension: Resize consistency
        
        For any input image size and target size, resizing should produce 
        consistent output dimensions.
        
        Validates: Requirements 1.2
        """
        # Create preprocessor with custom target size
        preprocessor = HAM10000ImagePreprocessor(target_size=(target_height, target_width))
        
        # Create random image
        image = np.random.randint(0, 255, (original_height, original_width, 3), dtype=np.uint8)
        
        # Preprocess image
        processed = preprocessor.preprocess_image(image, normalize=True)
        
        # Property: Output should have exact target dimensions
        expected_shape = (3, target_height, target_width)
        assert processed.shape == expected_shape, f"Expected shape {expected_shape}, got {processed.shape}"
        
        # Property: Validation should confirm correct dimensions
        validation = preprocessor.validate_preprocessing(processed)
        assert validation['correct_height'], "Height validation should pass"
        assert validation['correct_width'], "Width validation should pass"
        assert validation['correct_shape'], "Shape validation should pass"
    
    @given(
        mean_values=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=3, max_size=3),
        std_values=st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=3, max_size=3)
    )
    @settings(max_examples=50)
    def test_property_2_normalization_consistency(self, mean_values, std_values):
        """
        Property 2 extension: Normalization consistency
        
        For any normalization parameters, the preprocessing should apply 
        normalization consistently across all images.
        
        Validates: Requirements 1.2
        """
        # Create preprocessor with custom normalization
        preprocessor = HAM10000ImagePreprocessor(
            normalize_mean=mean_values,
            normalize_std=std_values
        )
        
        # Create two identical images
        image1 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        image2 = image1.copy()
        
        # Preprocess both images
        processed1 = preprocessor.preprocess_image(image1, normalize=True)
        processed2 = preprocessor.preprocess_image(image2, normalize=True)
        
        # Property: Identical inputs should produce identical outputs
        torch.testing.assert_close(processed1, processed2, msg="Identical images should produce identical preprocessed results")
        
        # Property: Denormalization should be consistent
        denorm1 = preprocessor.denormalize(processed1)
        denorm2 = preprocessor.denormalize(processed2)
        
        torch.testing.assert_close(denorm1, denorm2, msg="Denormalization should be consistent for identical inputs")
        
        # Property: Denormalized values should be in [0, 1] range
        assert denorm1.min() >= 0.0, "Denormalized values should be >= 0"
        assert denorm1.max() <= 1.0, "Denormalized values should be <= 1"
    
    @given(
        num_images=st.integers(min_value=1, max_value=20),
        image_height=st.integers(min_value=100, max_value=500),
        image_width=st.integers(min_value=100, max_value=500)
    )
    @settings(max_examples=30)
    def test_property_2_batch_processing_consistency(self, num_images, image_height, image_width):
        """
        Property 2 extension: Batch processing consistency
        
        For any batch of images, batch processing should produce the same 
        results as individual processing.
        
        Validates: Requirements 1.2
        """
        preprocessor = HAM10000ImagePreprocessor()
        
        # Generate batch of random images
        images = []
        for _ in range(num_images):
            image = np.random.randint(0, 255, (image_height, image_width, 3), dtype=np.uint8)
            images.append(image)
        
        # Process individually
        individual_results = []
        for image in images:
            processed = preprocessor.preprocess_image(image, normalize=True)
            individual_results.append(processed)
        
        # Process as batch
        batch_result = preprocessor.preprocess_batch(images, normalize=True)
        
        # Property: Batch processing should match individual processing
        assert batch_result.shape[0] == num_images, f"Batch should have {num_images} images"
        
        for i in range(num_images):
            torch.testing.assert_close(
                individual_results[i], 
                batch_result[i], 
                msg=f"Batch processing should match individual processing for image {i}"
            )
        
        # Property: All images in batch should have consistent dimensions
        for i in range(num_images):
            assert batch_result[i].shape == (3, 224, 224), f"Image {i} in batch should have shape (3, 224, 224)"
    
    @given(
        pixel_values=st.lists(
            st.integers(min_value=0, max_value=255), 
            min_size=3*32*32,  # Reduced size for faster generation
            max_size=3*32*32
        )
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow])
    def test_property_2_pixel_value_handling(self, pixel_values):
        """
        Property 2 extension: Pixel value handling consistency
        
        For any valid pixel values, preprocessing should handle them consistently
        without producing invalid outputs.
        
        Validates: Requirements 1.2
        """
        # Reshape pixel values into image
        image_array = np.array(pixel_values, dtype=np.uint8).reshape(32, 32, 3)
        
        preprocessor = HAM10000ImagePreprocessor()
        
        # Preprocess image
        processed = preprocessor.preprocess_image(image_array, normalize=True)
        
        # Property: Output should be valid tensor
        assert isinstance(processed, torch.Tensor), "Output should be a tensor"
        assert processed.dtype == torch.float32, "Output should be float32"
        assert processed.shape == (3, 224, 224), "Output should have correct shape"
        
        # Property: No invalid values
        assert not torch.isnan(processed).any(), "Output should not contain NaN"
        assert not torch.isinf(processed).any(), "Output should not contain infinite values"
        
        # Property: Validation should pass
        validation = preprocessor.validate_preprocessing(processed)
        assert validation['valid'], f"Preprocessing validation should pass: {validation}"
        
        # Property: Denormalization should work
        denormalized = preprocessor.denormalize(processed)
        assert denormalized.min() >= 0.0, "Denormalized values should be >= 0"
        assert denormalized.max() <= 1.0, "Denormalized values should be <= 1"
    
    @given(
        interpolation_mode=st.sampled_from(['bilinear', 'nearest', 'bicubic'])
    )
    @settings(max_examples=10)
    def test_property_2_interpolation_consistency(self, interpolation_mode):
        """
        Property 2 extension: Interpolation consistency
        
        For any interpolation mode, preprocessing should produce valid results
        with consistent dimensions.
        
        Validates: Requirements 1.2
        """
        preprocessor = HAM10000ImagePreprocessor(interpolation=interpolation_mode)
        
        # Create test image
        image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        # Preprocess image
        processed = preprocessor.preprocess_image(image, normalize=True)
        
        # Property: Output dimensions should be consistent regardless of interpolation
        assert processed.shape == (3, 224, 224), f"Shape should be (3, 224, 224) for {interpolation_mode} interpolation"
        
        # Property: Output should be valid
        validation = preprocessor.validate_preprocessing(processed)
        assert validation['valid'], f"Preprocessing should be valid for {interpolation_mode} interpolation"
        
        # Property: Values should be in reasonable range
        assert -5.0 <= processed.min() <= 5.0, f"Min value should be reasonable for {interpolation_mode}"
        assert -5.0 <= processed.max() <= 5.0, f"Max value should be reasonable for {interpolation_mode}"


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v"])