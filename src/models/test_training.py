"""
Test script for model training functionality with dummy data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_adapter import EfficientNetAdapter
from models.model_trainer import ModelTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_data(num_samples: int = 100, num_classes: int = 7) -> Tuple[DataLoader, DataLoader]:
    """Create dummy data loaders for testing."""
    # Create dummy images (224x224x3) and labels
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Create dataset
    dataset = TensorDataset(images, labels)
    
    # Split into train and validation
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    return train_loader, val_loader


def test_training_functionality():
    """Test the training functionality with dummy data."""
    try:
        logger.info("Creating dummy data...")
        train_loader, val_loader = create_dummy_data(num_samples=50)
        
        logger.info("Creating adapted model...")
        adapter = EfficientNetAdapter(num_classes=7)
        model = adapter.create_adapted_model("efficientnet_b0")
        
        logger.info("Setting up trainer...")
        trainer = ModelTrainer()
        
        logger.info("Starting fine-tuning (2 epochs for testing)...")
        trained_model, history = trainer.fine_tune_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,  # Short training for testing
            learning_rate=1e-3,
            patience=10,  # High patience for short test
            save_path="models/test_model.pth"
        )
        
        logger.info("Testing evaluation...")
        results = trainer.evaluate_model(trained_model, val_loader)
        
        print("\nTraining Test Results:")
        print(f"- Training completed successfully")
        print(f"- Final validation accuracy: {history['val_acc'][-1]:.2f}%")
        print(f"- Final validation loss: {history['val_loss'][-1]:.4f}")
        print(f"- Test accuracy: {results['accuracy']:.2f}%")
        print(f"- Test loss: {results['loss']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training test failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_training_functionality()
    if success:
        print("✓ Training functionality test passed!")
    else:
        print("✗ Training functionality test failed!")