"""
Property-based tests for EfficientNet model adaptation.

Tests Property 6: Pre-trained model loading
**Validates: Requirements 2.1**
"""

import pytest
import torch
import torch.nn as nn
from hypothesis import given, strategies as st, settings
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.efficientnet_adapter import EfficientNetAdapter


class TestModelAdaptationProperties:
    """Property-based tests for model adaptation functionality."""
    
    @given(
        num_classes=st.integers(min_value=2, max_value=1000),
        batch_size=st.integers(min_value=1, max_value=8)
    )
    @settings(max_examples=100, deadline=60000)  # 60 second timeout for model loading
    def test_pretrained_model_loading_property(self, num_classes: int, batch_size: int):
        """
        Property 6: Pre-trained model loading
        
        For any valid number of classes and batch size, loading a pre-trained model
        should successfully restore all weights and model structure.
        
        **Feature: ham10000-qspice-pipeline, Property 6: Pre-trained model loading**
        **Validates: Requirements 2.1**
        """
        # Create adapter with random number of classes
        adapter = EfficientNetAdapter(num_classes=num_classes)
        
        # Load and adapt pre-trained model
        model = adapter.create_adapted_model("efficientnet_b0")
        
        # Validate model structure
        validation_results = adapter.validate_model_structure(model)
        
        # Property assertions
        assert validation_results['has_correct_output_classes'], \
            f"Model should have {num_classes} output classes, got {validation_results['output_classes']}"
        
        assert validation_results['output_classes'] == num_classes, \
            f"Output classes mismatch: expected {num_classes}, got {validation_results['output_classes']}"
        
        assert validation_results['model_parameters'] > 0, \
            "Model should have parameters loaded"
        
        # Test forward pass with random batch size
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                model = model.cuda()
            
            output = model(dummy_input)
            
            # Verify output shape
            assert output.shape == (batch_size, num_classes), \
                f"Output shape should be ({batch_size}, {num_classes}), got {output.shape}"
            
            # Verify output is valid (no NaN or Inf)
            assert torch.isfinite(output).all(), \
                "Model output should not contain NaN or Inf values"
    
    @given(
        model_names=st.sampled_from(["efficientnet_b0"])  # Can expand to other models later
    )
    @settings(max_examples=10, deadline=60000)  # Fewer examples due to model loading time
    def test_model_architecture_consistency_property(self, model_names: str):
        """
        Property: Model architecture consistency
        
        For any supported model architecture, the loading process should
        consistently produce models with the same structure and parameter count.
        
        **Feature: ham10000-qspice-pipeline, Property 6: Pre-trained model loading**
        **Validates: Requirements 2.1**
        """
        adapter = EfficientNetAdapter(num_classes=7)
        
        # Load the same model twice
        model1 = adapter.create_adapted_model(model_names)
        model2 = adapter.create_adapted_model(model_names)
        
        # Both models should have identical structure
        validation1 = adapter.validate_model_structure(model1)
        validation2 = adapter.validate_model_structure(model2)
        
        assert validation1['output_classes'] == validation2['output_classes'], \
            "Models of same architecture should have identical output classes"
        
        assert validation1['model_parameters'] == validation2['model_parameters'], \
            "Models of same architecture should have identical parameter count"
        
        # Both models should have the same architecture (but different classifier weights)
        # We test this by checking that the feature extractor weights are identical
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                model1 = model1.cuda()
                model2 = model2.cuda()
            
            # Extract features before the classifier (should be identical)
            # For EfficientNet, we can access the feature extractor
            if hasattr(model1, 'features') and hasattr(model2, 'features'):
                features1 = model1.features(dummy_input)
                features2 = model2.features(dummy_input)
                
                # Features should be identical (same pre-trained weights)
                assert torch.allclose(features1, features2, atol=1e-6), \
                    "Feature extractors should produce identical outputs with same pre-trained weights"
            
            # The final outputs will be different due to random classifier initialization
            # This is expected and correct behavior
    
    def test_model_device_handling_property(self):
        """
        Property: Model device handling
        
        Models should be properly moved to the correct device and handle
        device-specific operations correctly.
        
        **Feature: ham10000-qspice-pipeline, Property 6: Pre-trained model loading**
        **Validates: Requirements 2.1**
        """
        adapter = EfficientNetAdapter(num_classes=7)
        model = adapter.create_adapted_model("efficientnet_b0")
        
        # Check that model is on expected device
        expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_device = next(model.parameters()).device
        
        assert str(model_device).startswith(str(expected_device).split(':')[0]), \
            f"Model should be on {expected_device}, but is on {model_device}"
        
        # Test that model can handle inputs on the same device
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(expected_device)
            output = model(dummy_input)
            
            assert output.device == expected_device, \
                f"Output should be on {expected_device}, but is on {output.device}"


if __name__ == "__main__":
    # Run a simple test to verify the property tests work
    test_instance = TestModelAdaptationProperties()
    
    print("Running property test for model adaptation...")
    try:
        # Test with specific values
        test_instance.test_pretrained_model_loading_property(num_classes=7, batch_size=2)
        test_instance.test_model_architecture_consistency_property(model_names="efficientnet_b0")
        test_instance.test_model_device_handling_property()
        print("✓ Property tests passed!")
    except Exception as e:
        print(f"✗ Property tests failed: {str(e)}")