"""
EfficientNet-B0 model adapter for HAM10000 skin lesion classification.

This module provides functionality to load pre-trained EfficientNet-B0 models,
adapt them for 7-class skin lesion classification, and validate performance.
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EfficientNetAdapter:
    """Adapter for EfficientNet-B0 model for HAM10000 classification."""
    
    def __init__(self, num_classes: int = 7):
        """
        Initialize the EfficientNet adapter.
        
        Args:
            num_classes: Number of output classes (7 for HAM10000)
        """
        self.num_classes = num_classes
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_pretrained_model(self, model_name: str = "efficientnet_b0") -> nn.Module:
        """
        Load pre-trained EfficientNet-B0 with ImageNet weights.
        
        Args:
            model_name: Name of the model architecture
            
        Returns:
            Pre-trained model
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading pre-trained {model_name} model...")
            model = timm.create_model(model_name, pretrained=True)
            logger.info(f"Successfully loaded {model_name} with ImageNet weights")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load pre-trained model {model_name}: {str(e)}")
    
    def adapt_for_ham10000(self, model: nn.Module) -> nn.Module:
        """
        Adapt pre-trained model for HAM10000 7-class classification.
        
        Args:
            model: Pre-trained model to adapt
            
        Returns:
            Adapted model with correct output dimensions
        """
        logger.info(f"Adapting model for {self.num_classes}-class HAM10000 classification...")
        
        # Get the number of features in the classifier
        if hasattr(model, 'classifier'):
            num_features = model.classifier.in_features
            # Replace the classifier head
            model.classifier = nn.Linear(num_features, self.num_classes)
        elif hasattr(model, 'head'):
            num_features = model.head.in_features
            # Replace the head
            model.head = nn.Linear(num_features, self.num_classes)
        else:
            raise ValueError("Model does not have a recognized classifier layer")
        
        # Move model to appropriate device
        model = model.to(self.device)
        
        logger.info(f"Model adapted successfully. Output classes: {self.num_classes}")
        return model
    
    def create_adapted_model(self, model_name: str = "efficientnet_b0") -> nn.Module:
        """
        Create a complete adapted model by loading and adapting in one step.
        
        Args:
            model_name: Name of the model architecture
            
        Returns:
            Adapted model ready for HAM10000 classification
        """
        pretrained_model = self.load_pretrained_model(model_name)
        adapted_model = self.adapt_for_ham10000(pretrained_model)
        self.model = adapted_model
        return adapted_model
    
    def validate_model_structure(self, model: nn.Module) -> Dict[str, any]:
        """
        Validate that the model has the correct structure for HAM10000.
        
        Args:
            model: Model to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'has_correct_output_classes': False,
            'output_classes': None,
            'model_parameters': 0,
            'device': str(next(model.parameters()).device)
        }
        
        # Check output dimensions with a dummy input
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            try:
                output = model(dummy_input)
                validation_results['output_classes'] = output.shape[1]
                validation_results['has_correct_output_classes'] = (output.shape[1] == self.num_classes)
            except Exception as e:
                logger.error(f"Model validation failed: {str(e)}")
                return validation_results
        
        # Count parameters
        validation_results['model_parameters'] = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Model validation: {validation_results}")
        return validation_results
    
    def evaluate_on_test_set(self, model: nn.Module, test_loader, criterion=None) -> Dict[str, float]:
        """
        Evaluate model performance on HAM10000 test set.
        
        Args:
            model: Model to evaluate
            test_loader: DataLoader for test set
            criterion: Loss function (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct_predictions': correct,
            'total_samples': total
        }
        
        logger.info(f"Test set evaluation: Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        return results
    
    def save_model(self, model: nn.Module, filepath: str) -> None:
        """
        Save the adapted model weights.
        
        Args:
            model: Model to save
            filepath: Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_classes': self.num_classes,
            'model_type': 'efficientnet_b0_ham10000'
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, model_name: str = "efficientnet_b0") -> nn.Module:
        """
        Load a saved adapted model.
        
        Args:
            filepath: Path to the saved model
            model_name: Name of the model architecture
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Create model architecture
        model = self.create_adapted_model(model_name)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
        return model