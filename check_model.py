#!/usr/bin/env python3
"""
Check the current model checkpoint for training progress.
"""

import torch
import os
from datetime import datetime

def check_model_checkpoint():
    print("🔍 HAM10000 Training Progress Update")
    print("=" * 45)
    
    model_path = "models/best_ham10000_model.pth"
    
    if os.path.exists(model_path):
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            print("📊 Current Best Model:")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
            
            # File info
            size = os.path.getsize(model_path)
            mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
            print(f"  File Size: {size:,} bytes")
            print(f"  Last Updated: {mtime}")
            
            # Check if target reached
            val_acc = checkpoint.get('val_acc', 0)
            target = 98.0
            
            print(f"\n🎯 Progress to Target:")
            print(f"  Current: {val_acc:.2f}%")
            print(f"  Target: {target}%")
            
            if val_acc >= target:
                print("  ✅ TARGET ACHIEVED!")
            else:
                remaining = target - val_acc
                print(f"  🔄 Need {remaining:.2f}% more")
            
            # Training history if available
            if 'training_history' in checkpoint:
                history = checkpoint['training_history']
                if 'val_acc' in history and history['val_acc']:
                    print(f"\n📈 Training Progress:")
                    recent_accs = history['val_acc'][-5:]  # Last 5 epochs
                    for i, acc in enumerate(recent_accs):
                        epoch_num = len(history['val_acc']) - len(recent_accs) + i + 1
                        print(f"  Epoch {epoch_num}: {acc:.2f}%")
            
        except Exception as e:
            print(f"❌ Error reading checkpoint: {e}")
    else:
        print("❌ No model checkpoint found yet")
        print("🔄 Training still in progress...")

if __name__ == "__main__":
    check_model_checkpoint()