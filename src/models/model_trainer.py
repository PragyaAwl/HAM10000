"""
Model training and fine-tuning functionality for HAM10000 dataset.

This module provides functionality to fine-tune EfficientNet models on HAM10000
with early stopping and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path
import time
import copy

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_score: Current validation score (higher is better)
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
            self.best_weights = copy.deepcopy(model.state_dict())
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    logger.info("Restored best weights from early stopping")
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
            
        return False


class ModelTrainer:
    """Trainer for fine-tuning models on HAM10000."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the trainer.
        
        Args:
            device: Device to use for training
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def fine_tune_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 5,
        save_path: Optional[str] = None
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        Fine-tune model on HAM10000 dataset.
        
        Args:
            model: Model to fine-tune
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            save_path: Path to save best model
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        model = model.to(self.device)
        
        # Setup optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # Setup early stopping
        early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        logger.info(f"Starting fine-tuning for {epochs} epochs...")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - start_time
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Early stopping check
            if early_stopping(val_acc, model):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save best model if path provided
        if save_path:
            self._save_model(model, save_path, history)
        
        logger.info("Fine-tuning completed")
        return model, history
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def _save_model(self, model: nn.Module, save_path: str, history: Dict[str, List[float]]) -> None:
        """Save model and training history."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_history': history,
            'model_type': 'efficientnet_b0_ham10000_finetuned'
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            criterion: Loss function
            
        Returns:
            Dictionary with evaluation metrics
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        model = model.to(self.device)
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
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
        
        logger.info(f"Test evaluation: Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        return results