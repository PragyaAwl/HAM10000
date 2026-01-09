"""
Comprehensive training script to achieve 97%+ accuracy on HAM10000.

This script implements state-of-the-art training techniques to achieve
the highest possible accuracy on the HAM10000 skin lesion dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import argparse
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_loader import HAM10000DataLoader
from image_preprocessing import HAM10000ImagePreprocessor
from dataset_splits import HAM10000DatasetSplitter
from models.efficientnet_adapter import EfficientNetAdapter
from models.advanced_trainer import AdvancedTrainer, AdvancedAugmentation
from models.model_evaluator import ModelEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_97_percent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_data_loaders(
    metadata_path: str,
    images_dir1: str,
    images_dir2: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> tuple:
    """Set up data loaders with heavy augmentation."""
    
    logger.info("Loading HAM10000 dataset...")
    
    # Load data
    data_loader = HAM10000DataLoader()
    metadata = data_loader.load_metadata(metadata_path)
    
    logger.info(f"Loaded {len(metadata)} samples")
    logger.info(f"Class distribution:\n{metadata['dx'].value_counts()}")
    
    # Create dataset splits
    splitter = HAM10000DatasetSplitter(random_state=42)
    train_df, val_df, test_df = splitter.create_stratified_split(
        metadata, 
        val_size=val_split, 
        test_size=test_split
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Set up image preprocessing with advanced augmentation
    preprocessor = HAM10000ImagePreprocessor()
    
    # Create datasets with heavy augmentation for training
    train_dataset = preprocessor.create_dataset(
        train_df,
        [images_dir1, images_dir2],
        transform=AdvancedAugmentation.get_train_transforms_pil_only(),
        augment_minority_classes=True,  # Extra augmentation for rare classes
        augmentation_factor=3  # 3x more samples for minority classes
    )
    
    val_dataset = preprocessor.create_dataset(
        val_df,
        [images_dir1, images_dir2],
        transform=AdvancedAugmentation.get_val_transforms_pil_only()
    )
    
    test_dataset = preprocessor.create_dataset(
        test_df,
        [images_dir1, images_dir2],
        transform=AdvancedAugmentation.get_val_transforms_pil_only()
    )
    
    # Create weighted sampler for class imbalance
    trainer = AdvancedTrainer()
    train_labels = [sample[1] for sample in train_dataset]
    weighted_sampler = trainer.create_weighted_sampler(train_dataset, train_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=weighted_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_97_percent_model(
    metadata_path: str = "c:/Users/agarw/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/HAM10000_metadata.csv",
    images_dir1: str = "c:/Users/agarw/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_1",
    images_dir2: str = "c:/Users/agarw/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_2",
    target_accuracy: float = 97.0,
    max_epochs: int = 200,
    batch_size: int = 32
) -> None:
    """Main training function to achieve 97%+ accuracy."""
    
    logger.info("🚀 Starting HAM10000 training to achieve 97%+ accuracy")
    logger.info(f"Target accuracy: {target_accuracy}%")
    logger.info(f"Max epochs: {max_epochs}")
    logger.info(f"Batch size: {batch_size}")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set up data loaders
    train_loader, val_loader, test_loader = setup_data_loaders(
        metadata_path, images_dir1, images_dir2, batch_size
    )
    
    # Create model
    logger.info("Creating EfficientNet-B0 model...")
    adapter = EfficientNetAdapter(num_classes=7)
    model = adapter.create_adapted_model("efficientnet_b0")
    
    # Validate model structure
    validation_results = adapter.validate_model_structure(model)
    logger.info(f"Model validation: {validation_results}")
    
    # Create advanced trainer
    trainer = AdvancedTrainer(device=device)
    
    # Train the model
    logger.info("🔥 Starting intensive training...")
    trained_model, history = trainer.train_to_97_percent(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        target_accuracy=target_accuracy,
        max_epochs=max_epochs,
        save_path="models/ham10000_97percent_best.pth"
    )
    
    # Plot training history
    trainer.plot_training_history(history, "results/training_97_percent_history.png")
    
    # Final evaluation on test set
    logger.info("📊 Evaluating final model on test set...")
    evaluator = ModelEvaluator()
    
    # Standard evaluation
    test_results = evaluator.evaluate_model_comprehensive(
        model=trained_model,
        test_loader=test_loader,
        save_path="results/final_evaluation_97_percent.json"
    )
    
    # Print comprehensive results
    evaluator.print_evaluation_summary(test_results)
    
    # Test-time augmentation evaluation for maximum accuracy
    logger.info("🔬 Performing test-time augmentation evaluation...")
    tta_transforms = AdvancedAugmentation.get_tta_transforms()
    tta_results = trainer.evaluate_with_tta(
        model=trained_model,
        test_loader=test_loader,
        tta_transforms=tta_transforms
    )
    
    logger.info(f"📈 Final Results:")
    logger.info(f"Standard Test Accuracy: {test_results['overall_accuracy']:.2f}%")
    logger.info(f"TTA Test Accuracy: {tta_results['accuracy']:.2f}%")
    
    # Generate visualizations
    evaluator.generate_confusion_matrix_plot(
        np.array(test_results['confusion_matrix']),
        save_path="results/final_confusion_matrix_97_percent.png",
        normalize=True
    )
    
    evaluator.generate_per_class_performance_plot(
        test_results['per_class_metrics'],
        save_path="results/final_per_class_performance_97_percent.png"
    )
    
    # Save final model with evaluation
    evaluator.save_model_with_evaluation(
        model=trained_model,
        evaluation_results=test_results,
        model_path="models/ham10000_97percent_final.pth",
        results_path="results/ham10000_97percent_final_results.json"
    )
    
    # Success message
    final_accuracy = max(test_results['overall_accuracy'], tta_results['accuracy'])
    if final_accuracy >= target_accuracy:
        logger.info(f"🎉 SUCCESS! Achieved {final_accuracy:.2f}% accuracy (Target: {target_accuracy}%)")
    else:
        logger.info(f"⚠️  Achieved {final_accuracy:.2f}% accuracy (Target: {target_accuracy}%)")
        logger.info("Consider running with more epochs or different hyperparameters")
    
    return trained_model, test_results, tta_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HAM10000 model to 97%+ accuracy")
    parser.add_argument("--target-accuracy", type=float, default=97.0, help="Target accuracy percentage")
    parser.add_argument("--max-epochs", type=int, default=200, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    try:
        train_97_percent_model(
            target_accuracy=args.target_accuracy,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise