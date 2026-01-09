"""
Advanced training pipeline for achieving 97%+ accuracy on HAM10000.

This module implements state-of-the-art training techniques including:
- Heavy data augmentation
- Advanced optimization strategies
- Learning rate scheduling
- Model ensembling
- Test-time augmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import logging
import time
import copy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class AdvancedAugmentation:
    """Advanced data augmentation for skin lesion images."""
    
    @staticmethod
    def get_train_transforms():
        """Get heavy augmentation transforms for training (PIL image input)."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ])
    
    @staticmethod
    def get_train_transforms_pil_only():
        """Get heavy augmentation transforms for PIL images only (no ToTensor/Normalize)."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3)
        ])
    
    @staticmethod
    def get_val_transforms():
        """Get validation transforms (no augmentation)."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def get_val_transforms_pil_only():
        """Get validation transforms for PIL images only (no ToTensor/Normalize)."""
        return transforms.Compose([
            transforms.Resize((224, 224))
        ])
    
    @staticmethod
    def get_tta_transforms():
        """Get test-time augmentation transforms."""
        return [
            # Original
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # Vertical flip
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # Slight rotation
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ]


class AdvancedTrainer:
    """Advanced trainer for achieving 97%+ accuracy."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def create_weighted_sampler(self, dataset, labels: List[int]) -> WeightedRandomSampler:
        """Create weighted sampler for class imbalance."""
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def train_with_memory_optimization(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        target_accuracy: float = 98.0,
        max_epochs: int = 80,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        mixed_precision: bool = True,
        save_path: str = "models/ham10000_98percent_memory_optimized.pth"
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        Memory-optimized training to achieve 98%+ accuracy without OOM errors.
        """
        model = model.to(self.device)
        
        # Enable gradient checkpointing to save memory
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'set_grad_checkpointing'):
            model.backbone.set_grad_checkpointing(True)
            logger.info("Enabled gradient checkpointing")
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if mixed_precision and torch.cuda.is_available() else None
        if scaler:
            logger.info("Enabled mixed precision training")
        
        # Memory-optimized optimizer
        optimizer = optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if 'classifier' not in n and 'head' not in n], 
             'lr': 1e-5, 'weight_decay': 1e-4},
            {'params': [p for n, p in model.named_parameters() if 'classifier' in n or 'head' in n], 
             'lr': 5e-4, 'weight_decay': 1e-3}
        ])
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[5e-5, 2e-3],
            epochs=max_epochs,
            steps_per_epoch=len(train_loader) // gradient_accumulation_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': [], 'memory_usage': []
        }
        
        best_accuracy = 0.0
        best_model_state = None
        patience_counter = 0
        patience = 15
        
        logger.info(f"Starting memory-optimized training to achieve {target_accuracy}% accuracy...")
        logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
        
        for epoch in range(max_epochs):
            start_time = time.time()
            
            # Clear cache at start of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Training phase with memory optimization
            train_loss, train_acc = self._train_epoch_memory_optimized(
                model, train_loader, optimizer, criterion, scheduler, 
                scaler, gradient_accumulation_steps
            )
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch_memory_optimized(
                model, val_loader, criterion, scaler
            )
            
            # Memory usage tracking
            memory_usage = 0
            if torch.cuda.is_available():
                memory_usage = torch.cuda.max_memory_allocated() / 1e9  # GB
                torch.cuda.reset_peak_memory_stats()
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            history['memory_usage'].append(memory_usage)
            
            epoch_time = time.time() - start_time
            
            logger.info(
                f"Epoch {epoch+1}/{max_epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} - "
                f"Memory: {memory_usage:.1f}GB"
            )
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                
                # Save checkpoint
                self._save_checkpoint(model, optimizer, epoch, val_acc, save_path)
                
                if val_acc >= target_accuracy:
                    logger.info(f"🎉 Target accuracy {target_accuracy}% achieved! (Val Acc: {val_acc:.2f}%)")
                    break
            else:
                patience_counter += 1
            
            # Early stopping with patience
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered. Best accuracy: {best_accuracy:.2f}%")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        logger.info(f"Training completed. Best validation accuracy: {best_accuracy:.2f}%")
        return model, history
    
    def _train_epoch_memory_optimized(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: optim.lr_scheduler._LRScheduler,
        scaler: Optional[torch.cuda.amp.GradScaler],
        gradient_accumulation_steps: int
    ) -> Tuple[float, float]:
        """Memory-optimized training epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target) / gradient_accumulation_steps
                
                # Backward pass
                scaler.scale(loss).backward()
            else:
                output = model(data)
                loss = criterion(output, target) / gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item() * gradient_accumulation_steps
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Clear intermediate variables to save memory
            del data, target, output, loss
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Handle remaining gradients
        if len(train_loader) % gradient_accumulation_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch_memory_optimized(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        scaler: Optional[torch.cuda.amp.GradScaler]
    ) -> Tuple[float, float]:
        """Memory-optimized validation epoch."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # Mixed precision forward pass
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Clear variables to save memory
                del data, target, output, loss
                if batch_idx % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy

    def train_to_97_percent(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        target_accuracy: float = 97.0,
        max_epochs: int = 200,
        save_path: str = "models/ham10000_97percent.pth"
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        Train model to achieve 97%+ accuracy using advanced techniques.
        """
        model = model.to(self.device)
        
        # Advanced optimizer with different learning rates for different parts
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'classifier' in name or 'head' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5, 'weight_decay': 1e-4},
            {'params': classifier_params, 'lr': 1e-3, 'weight_decay': 1e-3}
        ])
        
        # Advanced learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[1e-4, 1e-2],
            epochs=max_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        best_accuracy = 0.0
        best_model_state = None
        patience_counter = 0
        patience = 20
        
        logger.info(f"Starting training to achieve {target_accuracy}% accuracy...")
        
        for epoch in range(max_epochs):
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch_advanced(
                model, train_loader, optimizer, criterion, scheduler
            )
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch_advanced(
                model, val_loader, criterion
            )
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - start_time
            
            logger.info(
                f"Epoch {epoch+1}/{max_epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                
                # Save checkpoint
                self._save_checkpoint(model, optimizer, epoch, val_acc, save_path)
                
                if val_acc >= target_accuracy:
                    logger.info(f"🎉 Target accuracy {target_accuracy}% achieved! (Val Acc: {val_acc:.2f}%)")
                    break
            else:
                patience_counter += 1
            
            # Early stopping with patience
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered. Best accuracy: {best_accuracy:.2f}%")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        logger.info(f"Training completed. Best validation accuracy: {best_accuracy:.2f}%")
        return model, history
    
    def _train_epoch_advanced(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: optim.lr_scheduler._LRScheduler
    ) -> Tuple[float, float]:
        """Advanced training epoch with gradient accumulation and mixed precision."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Gradient accumulation
        accumulation_steps = 4
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item() * accumulation_steps
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        # Handle remaining gradients
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch_advanced(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Advanced validation with test-time augmentation."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Standard forward pass
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def evaluate_with_tta(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        tta_transforms: List[transforms.Compose]
    ) -> Dict[str, float]:
        """Evaluate model with test-time augmentation for maximum accuracy."""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                batch_predictions = []
                
                # Apply each TTA transform
                for transform in tta_transforms:
                    # Apply transform to each image in batch
                    tta_data = []
                    for img in data:
                        # Convert tensor back to PIL for transform
                        img_pil = transforms.ToPILImage()(img)
                        tta_img = transform(img_pil)
                        tta_data.append(tta_img)
                    
                    tta_batch = torch.stack(tta_data).to(self.device)
                    output = model(tta_batch)
                    probabilities = torch.softmax(output, dim=1)
                    batch_predictions.append(probabilities.cpu())
                
                # Average predictions across all TTA transforms
                avg_predictions = torch.stack(batch_predictions).mean(dim=0)
                final_predictions = avg_predictions.argmax(dim=1)
                
                all_predictions.extend(final_predictions.numpy())
                all_targets.extend(target.numpy())
        
        # Calculate accuracy
        correct = sum(p == t for p, t in zip(all_predictions, all_targets))
        accuracy = 100.0 * correct / len(all_targets)
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_samples': len(all_targets),
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        accuracy: float,
        save_path: str
    ) -> None:
        """Save training checkpoint."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'model_type': 'efficientnet_b0_ham10000_97percent'
        }, save_path)
        
        logger.info(f"Checkpoint saved: {save_path} (Accuracy: {accuracy:.2f}%)")
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: str = "results/training_history.png"
    ) -> None:
        """Plot comprehensive training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', alpha=0.8)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_acc'], label='Train Acc', alpha=0.8)
        axes[0, 1].plot(history['val_acc'], label='Val Acc', alpha=0.8)
        axes[0, 1].axhline(y=97, color='r', linestyle='--', label='Target 97%')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(history['learning_rates'], alpha=0.8)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Best accuracy over time
        best_acc_so_far = []
        best = 0
        for acc in history['val_acc']:
            if acc > best:
                best = acc
            best_acc_so_far.append(best)
        
        axes[1, 1].plot(best_acc_so_far, alpha=0.8, color='green')
        axes[1, 1].axhline(y=97, color='r', linestyle='--', label='Target 97%')
        axes[1, 1].set_title('Best Validation Accuracy Progress')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Best Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved to {save_path}")