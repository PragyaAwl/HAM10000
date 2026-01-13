#!/usr/bin/env python3
"""
Quick training script for HAM10000 - optimized for speed and 98% accuracy target.
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
        
        logger.info(f"Dataset: {len(self.metadata_df)} samples, {self.num_classes} classes")
        
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
            image = Image.new('RGB', (128, 128), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_class_names(self):
        return list(self.label_encoder.classes_)

class CompactCNN(nn.Module):
    """Compact CNN optimized for speed and accuracy."""
    
    def __init__(self, num_classes):
        super(CompactCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1 - 128x128 -> 64x64
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 2 - 64x64 -> 32x32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 3 - 32x32 -> 16x16
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            # Block 4 - 16x16 -> 8x8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
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
    """Train the model quickly to achieve 98% accuracy."""
    
    # Optimized configuration for speed
    config = {
        'batch_size': 64,  # Larger batch size for efficiency
        'num_epochs': 50,  # Fewer epochs
        'learning_rate': 0.003,  # Higher learning rate
        'weight_decay': 1e-4,
        'patience': 10,  # Less patience
        'target_accuracy': 98.0,
        'image_size': 128  # Smaller images for speed
    }
    
    logger.info("🚀 Starting FAST HAM10000 training for 98% accuracy...")
    logger.info(f"Config: {config}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load data
    metadata_path = "data/raw/HAM10000_metadata.csv"
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata not found: {metadata_path}")
        return False
        
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Quick data split
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['dx'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['dx'], random_state=42)
    
    logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Fast transforms
    train_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    image_dir1 = "data/raw/HAM10000_images_part_1"
    image_dir2 = "data/raw/HAM10000_images_part_2"
    
    train_dataset = HAM10000Dataset(train_df, image_dir1, image_dir2, train_transform)
    val_dataset = HAM10000Dataset(val_df, image_dir1, image_dir2, val_transform)
    test_dataset = HAM10000Dataset(test_df, image_dir1, image_dir2, val_transform)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Model
    num_classes = train_dataset.num_classes
    class_names = train_dataset.get_class_names()
    logger.info(f"Classes ({num_classes}): {class_names}")
    
    model = CompactCNN(num_classes).to(device)
    
    # Calculate class weights
    class_counts = df['dx'].value_counts()
    total_samples = len(df)
    class_weights = []
    for class_name in class_names:
        weight = total_samples / (num_classes * class_counts[class_name])
        class_weights.append(weight)
    
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training variables
    best_val_acc = 0.0
    patience_counter = 0
    
    logger.info("🎯 Starting training loop...")
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 30 == 0:
                current_acc = 100. * train_correct / train_total
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%')
        
        # Validation
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
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}:")
        logger.info(f"  Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.2f}%")
        logger.info(f"  Val: Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.2f}%")
        logger.info(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'config': config
            }, 'models/best_quick_model.pth')
            
            logger.info(f"  🎯 NEW BEST! Val Acc: {val_acc:.2f}%")
            
            # Check target
            if val_acc >= config['target_accuracy']:
                logger.info(f"🎉 TARGET ACHIEVED! {val_acc:.2f}% >= {config['target_accuracy']}%")
                break
        else:
            patience_counter += 1
            logger.info(f"  Patience: {patience_counter}/{config['patience']}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
        
        logger.info("-" * 50)
    
    # Final test evaluation
    logger.info("🧪 Final test evaluation...")
    
    if os.path.exists('models/best_quick_model.pth'):
        checkpoint = torch.load('models/best_quick_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
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
    logger.info(f"📊 Final Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    try:
        report = classification_report(all_targets, all_predictions, target_names=class_names, output_dict=True)
        
        logger.info("📈 Per-class Performance:")
        for class_name in class_names:
            if class_name in report:
                p, r, f1 = report[class_name]['precision'], report[class_name]['recall'], report[class_name]['f1-score']
                logger.info(f"  {class_name}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
        
        logger.info(f"Overall Accuracy: {report['accuracy']:.3f}")
        
    except Exception as e:
        logger.error(f"Error in classification report: {e}")
        report = None
    
    # Save results
    results = {
        'config': config,
        'best_validation_accuracy': best_val_acc,
        'final_test_accuracy': test_acc,
        'class_names': class_names,
        'classification_report': report
    }
    
    os.makedirs('results', exist_ok=True)
    import json
    with open('results/quick_training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("💾 Results saved to results/quick_training_results.json")
    logger.info(f"🏆 FINAL RESULT: Best validation accuracy = {best_val_acc:.2f}%")
    
    success = best_val_acc >= config['target_accuracy']
    if success:
        logger.info(f"✅ SUCCESS! Achieved {config['target_accuracy']}% target accuracy!")
    else:
        logger.info(f"❌ Target not reached. Best: {best_val_acc:.2f}%, Target: {config['target_accuracy']}%")
    
    return success

if __name__ == "__main__":
    try:
        success = train_model()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)