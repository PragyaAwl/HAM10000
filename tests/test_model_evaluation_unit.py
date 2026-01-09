"""
Unit tests for model evaluation functionality.

Tests metric calculations with known inputs and validates confusion matrix generation.
**Validates: Requirements 2.5**
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path
import tempfile
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.model_evaluator import ModelEvaluator
from models.efficientnet_adapter import EfficientNetAdapter


class TestModelEvaluationUnit:
    """Unit tests for model evaluation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator()
        self.class_names = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
    
    def test_metric_calculation_perfect_predictions(self):
        """Test metric calculations with perfect predictions."""
        # Create perfect predictions (all correct)
        y_true = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])  # Perfect predictions
        y_prob = np.eye(7)[y_pred]  # One-hot probabilities
        avg_loss = 0.1
        
        results = self.evaluator._calculate_comprehensive_metrics(
            y_true, y_pred, y_prob, avg_loss
        )
        
        # Perfect predictions should give 100% accuracy
        assert results['overall_accuracy'] == 100.0
        assert results['precision_macro'] == 1.0
        assert results['recall_macro'] == 1.0
        assert results['f1_macro'] == 1.0
        
        # All per-class metrics should be 1.0 for classes that appear
        for class_name, metrics in results['per_class_metrics'].items():
            if metrics['support'] > 0:  # Only check classes that appear in data
                assert metrics['precision'] == 1.0
                assert metrics['recall'] == 1.0
                assert metrics['f1_score'] == 1.0
                assert metrics['accuracy'] == 1.0
    
    def test_metric_calculation_random_predictions(self):
        """Test metric calculations with completely random predictions."""
        np.random.seed(42)  # For reproducible results
        
        # Create random predictions
        y_true = np.array([0, 1, 2, 3, 4, 5, 6] * 10)  # 70 samples, balanced
        y_pred = np.random.randint(0, 7, size=70)  # Random predictions
        y_prob = np.random.rand(70, 7)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize to probabilities
        avg_loss = 2.0
        
        results = self.evaluator._calculate_comprehensive_metrics(
            y_true, y_pred, y_prob, avg_loss
        )
        
        # Random predictions should give low accuracy (around 1/7 ≈ 14.3%)
        assert 0.0 <= results['overall_accuracy'] <= 100.0
        assert 0.0 <= results['precision_macro'] <= 1.0
        assert 0.0 <= results['recall_macro'] <= 1.0
        assert 0.0 <= results['f1_macro'] <= 1.0
        
        # Check that all required keys exist
        required_keys = [
            'overall_accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'per_class_metrics', 'confusion_matrix', 'class_names'
        ]
        for key in required_keys:
            assert key in results
    
    def test_confusion_matrix_generation(self):
        """Test confusion matrix generation with known inputs."""
        # Create known predictions for easy verification
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])  # Some correct, some incorrect
        y_prob = np.eye(7)[y_pred][:, :7]  # Convert to probabilities
        avg_loss = 1.5
        
        results = self.evaluator._calculate_comprehensive_metrics(
            y_true, y_pred, y_prob, avg_loss
        )
        
        confusion_matrix = np.array(results['confusion_matrix'])
        
        # Expected confusion matrix:
        # True\Pred  0  1  2  3  4  5  6
        #     0     [1  1  0  0  0  0  0]  # 2 samples of class 0: 1 correct, 1 predicted as 1
        #     1     [0  2  0  0  0  0  0]  # 2 samples of class 1: both predicted as 1
        #     2     [1  0  1  0  0  0  0]  # 2 samples of class 2: 1 correct, 1 predicted as 0
        #   3-6     [0  0  0  0  0  0  0]  # No samples for classes 3-6
        
        expected_matrix = np.array([
            [1, 1, 0, 0, 0, 0, 0],  # Class 0
            [0, 2, 0, 0, 0, 0, 0],  # Class 1
            [1, 0, 1, 0, 0, 0, 0],  # Class 2
            [0, 0, 0, 0, 0, 0, 0],  # Class 3
            [0, 0, 0, 0, 0, 0, 0],  # Class 4
            [0, 0, 0, 0, 0, 0, 0],  # Class 5
            [0, 0, 0, 0, 0, 0, 0]   # Class 6
        ])
        
        assert np.array_equal(confusion_matrix, expected_matrix)
        
        # Check accuracy calculation from confusion matrix
        correct_predictions = np.trace(confusion_matrix)
        total_predictions = np.sum(confusion_matrix)
        expected_accuracy = (correct_predictions / total_predictions) * 100
        
        assert abs(results['overall_accuracy'] - expected_accuracy) < 1e-6
    
    def test_per_class_metrics_calculation(self):
        """Test per-class metrics calculation with known inputs."""
        # Create simple case: 3 classes, clear patterns
        y_true = np.array([0, 0, 0, 1, 1, 2])  # 3 of class 0, 2 of class 1, 1 of class 2
        y_pred = np.array([0, 0, 1, 1, 1, 2])  # 2/3 correct for class 0, 2/2 for class 1, 1/1 for class 2
        y_prob = np.eye(7)[y_pred][:, :7]
        avg_loss = 1.0
        
        results = self.evaluator._calculate_comprehensive_metrics(
            y_true, y_pred, y_prob, avg_loss
        )
        
        per_class = results['per_class_metrics']
        
        # Class 0: 2 correct out of 3 true, 2 predicted as class 0 (all correct)
        # Precision = 2/2 = 1.0, Recall = 2/3 ≈ 0.667
        assert abs(per_class['mel']['precision'] - 1.0) < 1e-6
        assert abs(per_class['mel']['recall'] - 2/3) < 1e-6
        assert per_class['mel']['support'] == 3
        
        # Class 1: 2 correct out of 2 true, 3 predicted as class 1 (2 correct)
        # Precision = 2/3 ≈ 0.667, Recall = 2/2 = 1.0
        assert abs(per_class['nv']['precision'] - 2/3) < 1e-6
        assert abs(per_class['nv']['recall'] - 1.0) < 1e-6
        assert per_class['nv']['support'] == 2
        
        # Class 2: 1 correct out of 1 true, 1 predicted as class 2 (all correct)
        # Precision = 1/1 = 1.0, Recall = 1/1 = 1.0
        assert abs(per_class['bcc']['precision'] - 1.0) < 1e-6
        assert abs(per_class['bcc']['recall'] - 1.0) < 1e-6
        assert per_class['bcc']['support'] == 1
    
    def test_evaluation_with_dummy_model(self):
        """Test full evaluation pipeline with a dummy model."""
        # Create simple test data
        images = torch.randn(20, 3, 224, 224)
        labels = torch.randint(0, 7, (20,))
        dataset = TensorDataset(images, labels)
        test_loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Create a simple dummy model
        adapter = EfficientNetAdapter(num_classes=7)
        model = adapter.create_adapted_model("efficientnet_b0")
        
        # Run evaluation
        results = self.evaluator.evaluate_model_comprehensive(
            model=model,
            test_loader=test_loader
        )
        
        # Validate structure of results
        assert isinstance(results, dict)
        assert 'overall_accuracy' in results
        assert 'confusion_matrix' in results
        assert 'per_class_metrics' in results
        assert len(results['per_class_metrics']) == 7
        
        # Validate ranges
        assert 0.0 <= results['overall_accuracy'] <= 100.0
        assert results['total_samples'] == 20
        assert 0 <= results['correct_predictions'] <= 20
    
    def test_save_and_load_evaluation_results(self):
        """Test saving and loading evaluation results."""
        # Create dummy results
        results = {
            'overall_accuracy': 85.5,
            'precision_macro': 0.82,
            'recall_macro': 0.81,
            'f1_macro': 0.815,
            'confusion_matrix': [[10, 1], [2, 8]],
            'class_names': ['class_0', 'class_1']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save results
            self.evaluator._save_evaluation_results(results, temp_path)
            
            # Load and verify
            with open(temp_path, 'r') as f:
                loaded_results = json.load(f)
            
            assert loaded_results['overall_accuracy'] == 85.5
            assert loaded_results['precision_macro'] == 0.82
            assert loaded_results['confusion_matrix'] == [[10, 1], [2, 8]]
            
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
    
    def test_class_distribution_calculation(self):
        """Test class distribution calculation in metrics."""
        # Create unbalanced dataset
        y_true = np.array([0, 0, 0, 0, 1, 1, 2])  # 4 of class 0, 2 of class 1, 1 of class 2
        y_pred = np.array([0, 0, 0, 1, 1, 1, 2])  # Some predictions
        y_prob = np.eye(7)[y_pred][:, :7]
        avg_loss = 1.0
        
        results = self.evaluator._calculate_comprehensive_metrics(
            y_true, y_pred, y_prob, avg_loss
        )
        
        per_class = results['per_class_metrics']
        
        # Check support counts
        assert per_class['mel']['support'] == 4  # Class 0
        assert per_class['nv']['support'] == 2   # Class 1
        assert per_class['bcc']['support'] == 1  # Class 2
        assert per_class['akiec']['support'] == 0  # Class 3
        
        # Check support percentages
        total_samples = 7
        assert abs(per_class['mel']['support_percent'] - (4/7)*100) < 1e-6
        assert abs(per_class['nv']['support_percent'] - (2/7)*100) < 1e-6
        assert abs(per_class['bcc']['support_percent'] - (1/7)*100) < 1e-6
        assert per_class['akiec']['support_percent'] == 0.0
    
    def test_edge_case_empty_classes(self):
        """Test handling of classes with no samples."""
        # Create data with only 3 out of 7 classes
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 1, 2])
        y_prob = np.eye(7)[y_pred][:, :7]
        avg_loss = 1.0
        
        results = self.evaluator._calculate_comprehensive_metrics(
            y_true, y_pred, y_prob, avg_loss
        )
        
        per_class = results['per_class_metrics']
        
        # Classes 3-6 should have zero support and zero metrics
        for class_idx in [3, 4, 5, 6]:
            class_name = self.class_names[class_idx]
            assert per_class[class_name]['support'] == 0
            assert per_class[class_name]['support_percent'] == 0.0
            # Precision, recall, f1 should be 0.0 for classes with no samples
            assert per_class[class_name]['precision'] == 0.0
            assert per_class[class_name]['recall'] == 0.0
            assert per_class[class_name]['f1_score'] == 0.0


if __name__ == "__main__":
    # Run tests manually for verification
    test_instance = TestModelEvaluationUnit()
    test_instance.setup_method()
    
    print("Running unit tests for model evaluation...")
    
    try:
        test_instance.test_metric_calculation_perfect_predictions()
        print("✓ Perfect predictions test passed")
        
        test_instance.test_metric_calculation_random_predictions()
        print("✓ Random predictions test passed")
        
        test_instance.test_confusion_matrix_generation()
        print("✓ Confusion matrix test passed")
        
        test_instance.test_per_class_metrics_calculation()
        print("✓ Per-class metrics test passed")
        
        test_instance.test_evaluation_with_dummy_model()
        print("✓ Dummy model evaluation test passed")
        
        test_instance.test_save_and_load_evaluation_results()
        print("✓ Save/load results test passed")
        
        test_instance.test_class_distribution_calculation()
        print("✓ Class distribution test passed")
        
        test_instance.test_edge_case_empty_classes()
        print("✓ Empty classes test passed")
        
        print("✓ All unit tests passed!")
        
    except Exception as e:
        print(f"✗ Unit tests failed: {str(e)}")
        raise