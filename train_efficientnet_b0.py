#!/usr/bin/env python3
"""
Train EfficientNet-B0 on HAM10000 dataset to achieve 98% accuracy.
This is the CORRECT implementation as specified in requirements.
"""

import os
import sys
import logging
import time
from datetime import datetime
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_class_names(self):
        return list(self.label_encoder.classes_)

class EfficientNetB0(nn.Module):
    """EfficientNet-B0 for HAM10000 skin lesion classification."""
    
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetB0, self).__init__()
        
        try:
            # Try to use torchvision EfficientNet
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=pretrained)
            
            # Replace classifier for our number of classes
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, num_classes)
            )
            
            logger.info(f"✅ Using torchvision EfficientNet-B0 (pretrained={pretrained})")
            
        except ImportError:
            # Fallback: Create a simplified EfficientNet-like architecture
            logger.warning("⚠️ torchvision EfficientNet not available, using simplified version")
            
            self.backbone = nn.Sequential(
                # Stem
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                
                # MBConv blocks (simplified)
                self._make_mbconv_block(32, 16, 1, 1),
                self._make_mbconv_block(16, 24, 2, 6),
                self._make_mbconv_block(24, 40, 2, 6),
                self._make_mbconv_block(40, 80, 3, 6),
                self._make_mbconv_block(80, 112, 3, 6),
                self._make_mbconv_block(112, 192, 4, 6),
                self._make_mbconv_block(192, 320, 1, 6),
                
                # Head
                nn.Conv2d(320, 1280, 1, bias=False),
                nn.BatchNorm2d(1280),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(1280, num_classes)
            )
    
    def _make_mbconv_block(self, in_channels, out_channels, num_layers, expand_ratio):
        """Create MBConv block (simplified)."""
        layers = []
        for i in range(num_layers):
            stride = 2 if i == 0 and in_channels != out_channels else 1
            layers.append(self._mbconv_layer(in_channels if i == 0 else out_channels, 
                                           out_channels, stride, expand_ratio))
        return nn.Sequential(*layers)
    
    def _mbconv_layer(self, in_channels, out_channels, stride, expand_ratio):
        """Single MBConv layer (simplified)."""
        expanded_channels = in_channels * expand_ratio
        
        layers = []
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride=stride, 
                     padding=1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        ])
        
        # Project
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.backbone(x)

def train_efficientnet_b0():
    """Train EfficientNet-B0 on HAM10000 to achieve 98% accuracy."""
    
    logger.info("🚀 Training EfficientNet-B0 on HAM10000 for 98% Accuracy")
    logger.info("=" * 60)
    
    # Configuration for EfficientNet-B0
    config = {
        'model_name': 'EfficientNet-B0',
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'patience': 15,
        'target_accuracy': 98.0,
        'image_size': 224,
        'pretrained': True
    }
    
    logger.info("📋 EfficientNet-B0 Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load HAM10000 dataset
    logger.info("📊 Loading HAM10000 dataset...")
    metadata_path = "data/raw/HAM10000_metadata.csv"
    
    if not os.path.exists(metadata_path):
        logger.error(f"❌ HAM10000 metadata not found: {metadata_path}")
        return False
        
    df = pd.read_csv(metadata_path)
    logger.info(f"✅ Loaded {len(df)} HAM10000 samples")
    logger.info(f"Class distribution:\\n{df['dx'].value_counts()}")
    
    # Stratified data split
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['dx'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['dx'], random_state=42
    )
    
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # EfficientNet-B0 optimized transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create HAM10000 datasets
    image_dir1 = "data/raw/HAM10000_images_part_1"
    image_dir2 = "data/raw/HAM10000_images_part_2"
    
    train_dataset = HAM10000Dataset(train_df, image_dir1, image_dir2, train_transform)
    val_dataset = HAM10000Dataset(val_df, image_dir1, image_dir2, val_transform)
    test_dataset = HAM10000Dataset(test_df, image_dir1, image_dir2, val_transform)
    
    # Data loaders
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
    
    # Create EfficientNet-B0 model
    num_classes = train_dataset.num_classes
    class_names = train_dataset.get_class_names()
    
    logger.info(f"🏗️ Creating EfficientNet-B0 for {num_classes} classes: {class_names}")
    model = EfficientNetB0(num_classes, pretrained=config['pretrained']).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ EfficientNet-B0 created: {total_params:,} total params, {trainable_params:,} trainable")
    
    # Class balancing for HAM10000
    class_counts = df['dx'].value_counts()
    total_samples = len(df)
    class_weights = []
    for class_name in class_names:
        weight = total_samples / (num_classes * class_counts[class_name])
        class_weights.append(weight)
    
    class_weights = torch.FloatTensor(class_weights).to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training variables
    best_val_acc = 0.0
    patience_counter = 0
    training_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rates': []
    }
    
    logger.info("🎯 Starting EfficientNet-B0 training for 98% accuracy...")
    
    # Training loop
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()
        
        # Training phase
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                current_acc = 100. * train_correct / train_total
                logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
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
        
        # Calculate metrics
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
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Log epoch results
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch}/{config['num_epochs']} - {epoch_time:.1f}s")
        logger.info(f"  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Check for best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best EfficientNet-B0 model
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'config': config,
                'training_history': training_history,
                'model_name': 'EfficientNet-B0'
            }, 'models/best_efficientnet_b0_ham10000.pth')
            
            logger.info(f"  🎯 NEW BEST EfficientNet-B0! Val Acc: {val_acc:.2f}%")
            
            # Check if 98% target reached
            if val_acc >= config['target_accuracy']:
                logger.info(f"🎉 98% TARGET ACHIEVED! EfficientNet-B0 Val Acc: {val_acc:.2f}%")
                break
        else:
            patience_counter += 1
            logger.info(f"  Patience: {patience_counter}/{config['patience']}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
        
        logger.info("-" * 60)
    
    # Final test evaluation
    logger.info("🧪 Final EfficientNet-B0 evaluation on HAM10000 test set...")
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
    logger.info(f"📊 Final EfficientNet-B0 Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    try:
        report = classification_report(
            all_targets, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        logger.info("📈 Per-class Performance:")
        for class_name in class_names:
            if class_name in report:
                p, r, f1 = report[class_name]['precision'], report[class_name]['recall'], report[class_name]['f1-score']
                logger.info(f"  {class_name}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
        
    except Exception as e:
        logger.error(f"Error generating classification report: {e}")
        report = None
    
    # Save final results
    results = {
        'model_name': 'EfficientNet-B0',
        'dataset': 'HAM10000',
        'config': config,
        'best_validation_accuracy': best_val_acc,
        'final_test_accuracy': test_acc,
        'class_names': class_names,
        'training_history': training_history,
        'classification_report': report,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'training_completed': datetime.now().isoformat()
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/efficientnet_b0_ham10000_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("💾 EfficientNet-B0 results saved to results/efficientnet_b0_ham10000_results.json")
    logger.info(f"🏆 TRAINING COMPLETED! Best EfficientNet-B0 validation accuracy: {best_val_acc:.2f}%")
    
    success = best_val_acc >= config['target_accuracy']
    if success:
        logger.info(f"✅ SUCCESS! EfficientNet-B0 achieved {config['target_accuracy']}% target on HAM10000!")
    else:
        logger.info(f"⚠️ Target not reached. Best: {best_val_acc:.2f}%, Target: {config['target_accuracy']}%")
    
    return success

if __name__ == "__main__":
    try:
        success = train_efficientnet_b0()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"EfficientNet-B0 training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)