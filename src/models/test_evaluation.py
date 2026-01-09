"""
Test script for model evaluation functionality.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_adapter import EfficientNetAdapter
from models.model_evaluator import ModelEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data(num_samples: int = 100, num_classes: int = 7) -> DataLoader:
    """Create test data with some predictable patterns."""
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Add some class-specific patterns to make evaluation meaningful
    for i in range(num_samples):
        class_id = labels[i].item()
        # Add class-specific pattern
        pattern_value = (class_id + 1) * 0.1
        images[i, 0, :5, :5] = pattern_value
    
    dataset = TensorDataset(images, labels)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    return test_loader


def test_evaluation_functionality():
    """Test the comprehensive evaluation functionality."""
    try:
        logger.info("Creating test data...")
        test_loader = create_test_data(num_samples=70)
        
        logger.info("Creating model...")
        adapter = EfficientNetAdapter(num_classes=7)
        model = adapter.create_adapted_model("efficientnet_b0")
        
        logger.info("Setting up evaluator...")
        evaluator = ModelEvaluator()
        
        logger.info("Running comprehensive evaluation...")
        results = evaluator.evaluate_model_comprehensive(
            model=model,
            test_loader=test_loader,
            save_path="results/test_evaluation.json"
        )
        
        # Print summary
        evaluator.print_evaluation_summary(results)
        
        # Test visualization functions
        logger.info("Testing visualization functions...")
        conf_matrix = np.array(results['confusion_matrix'])
        
        evaluator.generate_confusion_matrix_plot(
            conf_matrix,
            save_path="results/test_confusion_matrix.png",
            normalize=True
        )
        
        evaluator.generate_per_class_performance_plot(
            results['per_class_metrics'],
            save_path="results/test_per_class_performance.png"
        )
        
        # Test model saving with evaluation
        logger.info("Testing model saving with evaluation...")
        evaluator.save_model_with_evaluation(
            model=model,
            evaluation_results=results,
            model_path="models/test_evaluated_model.pth",
            results_path="results/test_detailed_results.json"
        )
        
        print("\nEvaluation Test Results:")
        print(f"- Overall accuracy: {results['overall_accuracy']:.2f}%")
        print(f"- Macro F1-score: {results['f1_macro']:.3f}")
        print(f"- Total samples: {results['total_samples']}")
        print(f"- Number of classes: {len(results['class_names'])}")
        print(f"- Confusion matrix shape: {np.array(results['confusion_matrix']).shape}")
        
        # Validate key metrics exist
        required_keys = [
            'overall_accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'per_class_metrics', 'confusion_matrix', 'class_names'
        ]
        
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            print(f"✗ Missing required keys: {missing_keys}")
            return False
        
        # Validate per-class metrics
        for class_name, metrics in results['per_class_metrics'].items():
            required_class_keys = ['precision', 'recall', 'f1_score', 'accuracy', 'support']
            missing_class_keys = [key for key in required_class_keys if key not in metrics]
            if missing_class_keys:
                print(f"✗ Missing per-class keys for {class_name}: {missing_class_keys}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Evaluation test failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_evaluation_functionality()
    if success:
        print("✓ Evaluation functionality test passed!")
    else:
        print("✗ Evaluation functionality test failed!")