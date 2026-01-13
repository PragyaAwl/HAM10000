#!/usr/bin/env python3
"""
Check EfficientNet-B0 training progress.
"""

import os
from datetime import datetime

def check_efficientnet_training():
    print("🔍 EfficientNet-B0 Training Progress")
    print("=" * 40)
    
    # Check for EfficientNet-B0 model
    efficientnet_model = "models/best_efficientnet_b0_ham10000.pth"
    
    if os.path.exists(efficientnet_model):
        size = os.path.getsize(efficientnet_model)
        mtime = datetime.fromtimestamp(os.path.getmtime(efficientnet_model))
        print(f"✅ EfficientNet-B0 Model: {size:,} bytes")
        print(f"   Last Updated: {mtime}")
        
        # Try to load and check accuracy
        try:
            import torch
            checkpoint = torch.load(efficientnet_model, map_location='cpu')
            
            epoch = checkpoint.get('epoch', 'N/A')
            val_acc = checkpoint.get('val_acc', 'N/A')
            model_name = checkpoint.get('model_name', 'Unknown')
            
            print(f"   Model: {model_name}")
            print(f"   Epoch: {epoch}")
            print(f"   Validation Accuracy: {val_acc:.2f}%")
            
            if val_acc >= 98.0:
                print("   🎉 TARGET ACHIEVED!")
            else:
                print(f"   🎯 Progress: {val_acc:.2f}% / 98.0%")
                
        except Exception as e:
            print(f"   ❌ Error reading model: {e}")
    else:
        print("❌ No EfficientNet-B0 model saved yet")
        print("🔄 Training in progress...")
    
    # Check results
    results_file = "results/efficientnet_b0_ham10000_results.json"
    if os.path.exists(results_file):
        print("✅ Training results file exists")
    else:
        print("❌ No results file yet")
    
    print(f"\n⏰ Current time: {datetime.now()}")

if __name__ == "__main__":
    check_efficientnet_training()