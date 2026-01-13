#!/usr/bin/env python3
"""
Offline training script for HAM10000 - no pretrained weights needed.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HAM10000Dataset(Dataset):
    def __init__(self, metadata_df, image_dir1, image_dir2, transform=None):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.image_dir1 = image_dir1
        self.image_dir2 = image_dir2
        self.transform = transform
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(metadata_df['dx'])
        self.num_classes = len(self.label_encoder.classes_)
        
        logger.info(f"Dataset created with {len(self.metadata_df)} samples, {self.num_classes} classes")
        logger.info(f"Classes: {list(self.label_encoder.classes_)}")
        
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        image_id = row['image_id']
        label = self.labels[idx]
        
        # Try to load image from either directory
        image_path1 = os.path.join(self.image_dir1, f"{image_id}.jpg")
        image_path2 = os.path.join(self.image_dir2, f"{image_id}.jpg")
        
        image = None
        if os.path.exists(image_path1):
            try:
                image = Image.open(image_path1).convert('RGB')
            except:
                pass
        
        if image is None and os.path.exists(image_path2):
            try:
                image = Image.open(image_path2).convert('RGB')
            except:
                pass
        
        if image is None:
            # Create a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_class_names(self):
        return list(self.label_encoder.classes_)

class SimpleCNN(nn.Module):
    """Simple CNN for skin lesion classification."""
    
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model():
    """Train the model to achieve 98% accuracy."""
    
    # Configuration
    config = {
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'patience': 15,
        'target_accuracy': 98.0
    }
    
    logger.info("Starting HAM10000 training...")
    logger.info(f"Configuration: {config}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading HAM10000 dataset...")
    metadata_path = "data/raw/HAM10000_metadata.csv"
    
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return False
        
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Class distribution:\n{df['dx'].value_counts()}")
    
    # Split data stratified by class
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['dx'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['dx'], random_state=42
    )
    
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Data transforms with strong augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    image_dir1 = "data/raw/HAM10000_images_part_1"
    image_dir2 = "data/raw/HAM10000_images_part_2"
    
    train_dataset = HAM10000Dataset(train_df, image_dir1, image_dir2, train_transform)
    val_dataset = HAM10000Dataset(val_df, image_dir1, image_dir2, val_transform)
    test_dataset = HAM10000Dataset(test_df, image_dir1, image_dir2, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], 
        shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], 
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    # Model
    num_classes = train_dataset.num_classes
    class_names = train_dataset.get_class_names()
    logger.info(f"Creating model for {num_classes} classes: {class_names}")
    
    model = SimpleCNN(num_classes).to(device)
    
    # Calculate class weights for balanced training
    class_counts = df['dx'].value_counts()
    total_samples = len(df)
    class_weights = []
    for class_name in class_names:
        weight = total_samples / (num_classes * class_counts[class_name])
        class_weights.append(weight)
    
    class_weights = torch.FloatTensor(class_weights).to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training variables
    best_val_acc = 0.0
    patience_counter = 0
    training_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    logger.info("Starting training loop...")
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Log progress every 50 batches
            if batch_idx % 50 == 0:
                current_acc = 100. * train_correct / train_total
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                           f'Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # Calculate epoch metrics
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Store history
        training_history['train_loss'].append(train_loss_avg)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss_avg)
        training_history['val_acc'].append(val_acc)
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}:")
        logger.info(f"  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'config': config,
                'training_history': training_history
            }, 'models/best_ham10000_model.pth')
            
            logger.info(f"  🎯 New best model saved! Val Acc: {val_acc:.2f}%")
            
            # Check if target accuracy reached
            if val_acc >= config['target_accuracy']:
                logger.info(f"🎉 TARGET ACCURACY REACHED! {val_acc:.2f}% >= {config['target_accuracy']}%")
                break
        else:
            patience_counter += 1
            logger.info(f"  Patience: {patience_counter}/{config['patience']}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        logger.info("-" * 60)
    
    # Load best model for final evaluation
    if os.path.exists('models/best_ham10000_model.pth'):
        checkpoint = torch.load('models/best_ham10000_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded best model for final evaluation")
    
    # Final test evaluation
    logger.info("Evaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    logger.info(f"Final Test Accuracy: {test_acc:.2f}%")
    
    # Generate classification report
    try:
        report = classification_report(
            all_targets, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        logger.info("Per-class Performance:")
        for class_name in class_names:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                logger.info(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        # Overall metrics
        accuracy = report['accuracy']
        macro_avg = report['macro avg']
        weighted_avg = report['weighted avg']
        
        logger.info(f"Overall Accuracy: {accuracy:.3f}")
        logger.info(f"Macro Avg - P: {macro_avg['precision']:.3f}, R: {macro_avg['recall']:.3f}, F1: {macro_avg['f1-score']:.3f}")
        logger.info(f"Weighted Avg - P: {weighted_avg['precision']:.3f}, R: {weighted_avg['recall']:.3f}, F1: {weighted_avg['f1-score']:.3f}")
        
    except Exception as e:
        logger.error(f"Error generating classification report: {e}")
        report = None
    
    # Save comprehensive results
    results = {
        'training_config': config,
        'best_validation_accuracy': best_val_acc,
        'final_test_accuracy': test_acc,
        'class_names': class_names,
        'training_history': training_history,
        'classification_report': report,
        'model_architecture': 'SimpleCNN',
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    import json
    with open('results/ham10000_training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save training history as CSV
    history_df = pd.DataFrame(training_history)
    history_df.to_csv('results/training_history.csv', index=False)
    
    logger.info("Results saved to results/ directory")
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Success check
    success = best_val_acc >= config['target_accuracy']
    if success:
        logger.info(f"✅ SUCCESS! Achieved target accuracy of {config['target_accuracy']}%")
    else:
        logger.info(f"❌ Target not reached. Best: {best_val_acc:.2f}%, Target: {config['target_accuracy']}%")
    
    return success

if __name__ == "__main__":
    try:
        success = train_model()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)