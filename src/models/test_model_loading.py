"""
Simple test script to validate EfficientNet model loading and adaptation.
"""

import torch
from efficientnet_adapter import EfficientNetAdapter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test basic model loading and adaptation functionality."""
    try:
        # Create adapter
        adapter = EfficientNetAdapter(num_classes=7)
        
        # Create adapted model
        model = adapter.create_adapted_model("efficientnet_b0")
        
        # Validate model structure
        validation_results = adapter.validate_model_structure(model)
        
        print("Model Loading Test Results:")
        print(f"- Correct output classes: {validation_results['has_correct_output_classes']}")
        print(f"- Output classes: {validation_results['output_classes']}")
        print(f"- Model parameters: {validation_results['model_parameters']:,}")
        print(f"- Device: {validation_results['device']}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 224, 224)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
            output = model(dummy_input)
            print(f"- Forward pass successful: {output.shape}")
        
        return validation_results['has_correct_output_classes']
        
    except Exception as e:
        logger.error(f"Model loading test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("✓ Model loading and adaptation test passed!")
    else:
        print("✗ Model loading and adaptation test failed!")