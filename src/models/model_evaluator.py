"""
Comprehensive model evaluation and metrics calculation for HAM10000 classification.

This module provides detailed evaluation metrics including per-class performance,
confusion matrices, and comprehensive analysis for skin lesion classification.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for HAM10000 classification."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize the evaluator.
        
        Args:
            class_names: List of class names for HAM10000 (7 classes)
        """
        self.class_names = class_names or [
            'mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc'
        ]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def evaluate_model_comprehensive(
        self,
        model: nn.Module,
        test_loader,
        save_path: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Perform comprehensive model evaluation with detailed metrics.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            save_path: Optional path to save evaluation results
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        model = model.to(self.device)
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        logger.info("Starting comprehensive model evaluation...")
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Calculate comprehensive metrics
        results = self._calculate_comprehensive_metrics(
            y_true, y_pred, y_prob, total_loss / len(test_loader)
        )
        
        # Save results if path provided
        if save_path:
            self._save_evaluation_results(results, save_path)
        
        logger.info(f"Evaluation completed. Overall accuracy: {results['overall_accuracy']:.2f}%")
        return results
    
    def _calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        avg_loss: float
    ) -> Dict[str, any]:
        """Calculate comprehensive evaluation metrics."""
        
        # Overall metrics
        overall_accuracy = accuracy_score(y_true, y_pred) * 100
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Macro and weighted averages
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))
        
        # Per-class accuracy from confusion matrix
        per_class_accuracy = np.zeros(len(self.class_names))
        for i in range(len(self.class_names)):
            if conf_matrix[i].sum() > 0:  # If class has samples
                per_class_accuracy[i] = conf_matrix[i, i] / conf_matrix[i].sum()
        
        # Per-class metrics (only for classes that appear in the data)
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        max_class = max(unique_classes) if len(unique_classes) > 0 else 0
        
        # Pad arrays to match expected number of classes
        precision_per_class_full = np.zeros(len(self.class_names))
        recall_per_class_full = np.zeros(len(self.class_names))
        f1_per_class_full = np.zeros(len(self.class_names))
        
        # Fill in values for classes that appear in predictions
        if len(precision_per_class) > 0:
            precision_per_class_full[:len(precision_per_class)] = precision_per_class
            recall_per_class_full[:len(recall_per_class)] = recall_per_class
            f1_per_class_full[:len(f1_per_class)] = f1_per_class
        
        # Class distribution
        class_distribution = np.bincount(y_true, minlength=len(self.class_names))
        class_distribution_percent = class_distribution / len(y_true) * 100
        
        # Compile results
        results = {
            'overall_accuracy': overall_accuracy,
            'average_loss': avg_loss,
            'total_samples': len(y_true),
            'correct_predictions': int(np.sum(y_true == y_pred)),
            
            # Macro averages
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            
            # Weighted averages
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            
            # Per-class metrics
            'per_class_metrics': {
                self.class_names[i]: {
                    'precision': float(precision_per_class_full[i]),
                    'recall': float(recall_per_class_full[i]),
                    'f1_score': float(f1_per_class_full[i]),
                    'accuracy': float(per_class_accuracy[i]),
                    'support': int(class_distribution[i]),
                    'support_percent': float(class_distribution_percent[i])
                }
                for i in range(len(self.class_names))
            },
            
            # Confusion matrix
            'confusion_matrix': conf_matrix.tolist(),
            'confusion_matrix_normalized': np.divide(
                conf_matrix, 
                conf_matrix.sum(axis=1, keepdims=True), 
                out=np.zeros_like(conf_matrix, dtype=float), 
                where=conf_matrix.sum(axis=1, keepdims=True)!=0
            ).tolist(),
            
            # Class names for reference
            'class_names': self.class_names,
            
            # Raw predictions for further analysis
            'predictions': y_pred.tolist(),
            'true_labels': y_true.tolist(),
            'probabilities': y_prob.tolist()
        }
        
        return results
    
    def generate_confusion_matrix_plot(
        self,
        confusion_matrix: np.ndarray,
        save_path: Optional[str] = None,
        normalize: bool = True
    ) -> None:
        """
        Generate and save confusion matrix visualization.
        
        Args:
            confusion_matrix: Confusion matrix to plot
            save_path: Path to save the plot
            normalize: Whether to normalize the confusion matrix
        """
        plt.figure(figsize=(10, 8))
        
        if normalize:
            cm = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            cm = confusion_matrix
            fmt = 'd'
            title = 'Confusion Matrix'
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.close()
    
    def generate_per_class_performance_plot(
        self,
        per_class_metrics: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Generate per-class performance visualization.
        
        Args:
            per_class_metrics: Per-class metrics dictionary
            save_path: Path to save the plot
        """
        classes = list(per_class_metrics.keys())
        metrics = ['precision', 'recall', 'f1_score', 'accuracy']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [per_class_metrics[cls][metric] for cls in classes]
            
            bars = axes[i].bar(classes, values, alpha=0.7)
            axes[i].set_title(f'Per-Class {metric.replace("_", " ").title()}')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Per-class performance plot saved to {save_path}")
        
        plt.close()
    
    def _save_evaluation_results(self, results: Dict[str, any], save_path: str) -> None:
        """Save evaluation results to JSON file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a copy without numpy arrays for JSON serialization
        json_results = {
            key: value for key, value in results.items()
            if key not in ['predictions', 'true_labels', 'probabilities']
        }
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {save_path}")
    
    def print_evaluation_summary(self, results: Dict[str, any]) -> None:
        """Print a formatted summary of evaluation results."""
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
        print(f"Average Loss: {results['average_loss']:.4f}")
        print(f"Total Samples: {results['total_samples']}")
        print(f"Correct Predictions: {results['correct_predictions']}")
        
        print(f"\nMacro Averages:")
        print(f"  Precision: {results['precision_macro']:.3f}")
        print(f"  Recall: {results['recall_macro']:.3f}")
        print(f"  F1-Score: {results['f1_macro']:.3f}")
        
        print(f"\nWeighted Averages:")
        print(f"  Precision: {results['precision_weighted']:.3f}")
        print(f"  Recall: {results['recall_weighted']:.3f}")
        print(f"  F1-Score: {results['f1_weighted']:.3f}")
        
        print(f"\nPer-Class Performance:")
        print(f"{'Class':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'Accuracy':<9} {'Support':<8}")
        print("-" * 60)
        
        for class_name, metrics in results['per_class_metrics'].items():
            print(f"{class_name:<8} {metrics['precision']:<10.3f} {metrics['recall']:<8.3f} "
                  f"{metrics['f1_score']:<9.3f} {metrics['accuracy']:<9.3f} {metrics['support']:<8}")
        
        print("="*60)
    
    def save_model_with_evaluation(
        self,
        model: nn.Module,
        evaluation_results: Dict[str, any],
        model_path: str,
        results_path: str
    ) -> None:
        """
        Save model along with its evaluation results.
        
        Args:
            model: Model to save
            evaluation_results: Evaluation results
            model_path: Path to save model
            results_path: Path to save evaluation results
        """
        # Save model
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'evaluation_results': evaluation_results,
            'class_names': self.class_names,
            'model_type': 'efficientnet_b0_ham10000_evaluated'
        }, model_path)
        
        # Save detailed results
        self._save_evaluation_results(evaluation_results, results_path)
        
        logger.info(f"Model and evaluation results saved to {model_path} and {results_path}")