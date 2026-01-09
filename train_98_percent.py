#!/usr/bin/env python3
"""
98% Accuracy Training Script - Inspired by Detectron2 Success

This script implements techniques that achieved 98% accuracy on a computer vision task
with 250-300 images, adapted for HAM10000 skin lesion classification.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.enhanced_efficientnet import create_98_percent_model, create_memory_optimized_model, HighAccuracyTrainer
from models.model_evaluator import ModelEvaluator
from data_loader import HAM10000DataLoader
from image_preprocessing import HAM10000Dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_98_percent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_data_loaders(metadata_path: str, images_dir1: str, images_dir2: str, 
                      batch_size: int = 8, val_split: float = 0.15, test_split: float = 0.15):
    """
    Set up memory-optimized data loaders for HAM10000 dataset.
    
    Args:
        metadata_path: Path to metadata CSV
        images_dir1: Path to first images directory
        images_dir2: Path to second images directory
        batch_size: Batch size for training
        val_split: Validation split ratio
        test_split: Test split ratio
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("Setting up memory-optimized data loaders...")
    
    # Create data loader
    ham_loader = HAM10000DataLoader(
        metadata_path=metadata_path,
        images_part1_path=images_dir1,
        images_part2_path=images_dir2
    )
    
    # Load metadata
    metadata = ham_loader.load_metadata()
    
    # Memory-optimized transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),  # Reduced from 45
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Reduced
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Split data
    train_val_df, test_df = train_test_split(
        metadata, test_size=test_split, stratify=metadata['dx'], random_state=42
    )
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_split/(1-test_split), stratify=train_val_df['dx'], random_state=42
    )
    
    logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Load images for each split
    def load_images_for_split(df):
        images = {}
        labels = {}
        for _, row in df.iterrows():
            image_id = row['image_id']
            image = ham_loader.load_image(image_id)
            if image is not None:
                images[image_id] = image
                labels[image_id] = row['label']
        return images, labels
    
    # Load images for each split
    logger.info("Loading training images...")
    train_images, train_labels = load_images_for_split(train_df)
    
    logger.info("Loading validation images...")
    val_images, val_labels = load_images_for_split(val_df)
    
    logger.info("Loading test images...")
    test_images, test_labels = load_images_for_split(test_df)
    
    # Create preprocessor
    preprocessor = HAM10000ImagePreprocessor()
    
    # Create datasets
    train_dataset = HAM10000Dataset(
        images=train_images,
        labels=train_labels,
        image_ids=list(train_images.keys()),
        preprocessor=preprocessor,
        transform=train_transform
    )
    
    val_dataset = HAM10000Dataset(
        images=val_images,
        labels=val_labels,
        image_ids=list(val_images.keys()),
        preprocessor=preprocessor,
        transform=val_transform
    )
    
    test_dataset = HAM10000Dataset(
        images=test_images,
        labels=test_labels,
        image_ids=list(test_images.keys()),
        preprocessor=preprocessor,
        transform=val_transform
    )
    
    # Memory-optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Reduced for memory
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_98_percent_model():
    """Train HAM10000 model to achieve 98% accuracy using proven techniques with memory optimization."""
    
    print("🎯 HAM10000 Training for 98% Accuracy (Memory Optimized)")
    print("=" * 60)
    print("Techniques inspired by successful Detectron2 implementation:")
    print("✓ Memory-optimized EfficientNet-B1 (balanced accuracy/memory)")
    print("✓ Enhanced classifier head with attention")
    print("✓ Gradient accumulation for effective larger batch size")
    print("✓ Mixed precision training")
    print("✓ Advanced augmentation and regularization")
    print("✓ Test-time augmentation")
    print("=" * 60)
    
    # Check device and memory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        logger.info("Cleared GPU cache")
    
    # Create memory-optimized model (B1 instead of B4)
    logger.info("Creating memory-optimized EfficientNet-B1 model...")
    model = create_memory_optimized_model()
    
    # Get memory-optimized configuration
    trainer = HighAccuracyTrainer(device=device)
    config = trainer.get_memory_optimized_config()
    
    logger.info("Memory-optimized configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Set up data loaders with memory-optimized batch size
    logger.info("Setting up memory-optimized data loaders...")
    train_loader, val_loader, test_loader = setup_data_loaders(
        metadata_path="c:/Users/agarw/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/HAM10000_metadata.csv",
        images_dir1="c:/Users/agarw/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_1",
        images_dir2="c:/Users/agarw/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_2",
        batch_size=config['batch_size'],
        val_split=0.15,
        test_split=0.15
    )
    
    # Memory-optimized training with proven techniques
    logger.info("🚀 Starting memory-optimized enhanced training...")
    
    # Use advanced trainer with memory optimizations
    from models.advanced_trainer import AdvancedTrainer
    advanced_trainer = AdvancedTrainer(device=device)
    
    # Train with memory-optimized settings
    trained_model, history = advanced_trainer.train_with_memory_optimization(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        target_accuracy=98.0,  # Target 98% like your Detectron2 result
        max_epochs=config['epochs'],
        batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        mixed_precision=config['mixed_precision'],
        save_path="models/ham10000_98percent_memory_optimized.pth"
    )
    
    # Enhanced evaluation
    logger.info("📊 Performing comprehensive evaluation...")
    evaluator = ModelEvaluator()
    
    # Standard evaluation
    test_results = evaluator.evaluate_model_comprehensive(
        model=trained_model,
        test_loader=test_loader,
        save_path="results/enhanced_98_percent_evaluation.json"
    )
    
    # Test-time augmentation (like your preview generation with multiple views)
    logger.info("🔬 Test-time augmentation evaluation...")
    from models.advanced_trainer import AdvancedAugmentation
    tta_transforms = AdvancedAugmentation.get_tta_transforms()
    tta_results = advanced_trainer.evaluate_with_tta(
        model=trained_model,
        test_loader=test_loader,
        tta_transforms=tta_transforms
    )
    
    # Results summary
    print("\n" + "=" * 60)
    print("🎯 FINAL RESULTS - 98% ACCURACY TARGET")
    print("=" * 60)
    
    evaluator.print_evaluation_summary(test_results)
    
    print(f"\n📈 Performance Summary:")
    print(f"Standard Test Accuracy: {test_results['overall_accuracy']:.2f}%")
    print(f"TTA Test Accuracy: {tta_results['accuracy']:.2f}%")
    print(f"Target Accuracy: 98.00%")
    
    final_accuracy = max(test_results['overall_accuracy'], tta_results['accuracy'])
    
    if final_accuracy >= 98.0:
        print(f"\n🎉 SUCCESS! Achieved {final_accuracy:.2f}% accuracy!")
        print("✓ Target 98% accuracy reached!")
    elif final_accuracy >= 97.0:
        print(f"\n🔥 EXCELLENT! Achieved {final_accuracy:.2f}% accuracy!")
        print("✓ Very close to 98% target!")
    else:
        print(f"\n📊 Achieved {final_accuracy:.2f}% accuracy")
        print("Consider additional training or hyperparameter tuning")
    
    # Generate comprehensive visualizations
    logger.info("Creating visualizations...")
    advanced_trainer.plot_training_history(
        history, 
        "results/enhanced_98_percent_training_history.png"
    )
    
    evaluator.generate_confusion_matrix_plot(
        np.array(test_results['confusion_matrix']),
        save_path="results/enhanced_98_percent_confusion_matrix.png",
        normalize=True
    )
    
    evaluator.generate_per_class_performance_plot(
        test_results['per_class_metrics'],
        save_path="results/enhanced_98_percent_per_class_performance.png"
    )
    
    # Save final model
    evaluator.save_model_with_evaluation(
        model=trained_model,
        evaluation_results=test_results,
        model_path="models/ham10000_98percent_final_enhanced.pth",
        results_path="results/ham10000_98percent_final_enhanced_results.json"
    )
    
    print(f"\n💾 Model and results saved!")
    print(f"📁 Check 'models/' and 'results/' directories for outputs")
    
    return trained_model, test_results, tta_results


if __name__ == "__main__":
    try:
        train_98_percent_model()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise