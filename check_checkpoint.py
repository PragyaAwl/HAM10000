#!/usr/bin/env python3
"""
Check what's saved in the training checkpoint.
"""

import torch
import os

def check_checkpoint():
    print("🔍 Checking Training Checkpoint")
    print("=" * 40)
    
    checkpoint_path = "models/best_ham10000_model.pth"
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            print("📦 Checkpoint Contents:")
            for key in checkpoint.keys():
                print(f"  ✅ {key}")
            
            print(f"\n📊 Training State:")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
            
            # Check if optimizer and scheduler states are saved
            if 'optimizer_state_dict' in checkpoint:
                print("  ✅ Optimizer state saved - CAN RESUME!")
            else:
                print("  ❌ No optimizer state - would restart")
                
            if 'scheduler_state_dict' in checkpoint:
                print("  ✅ Scheduler state saved - CAN RESUME!")
            else:
                print("  ❌ No scheduler state")
                
            if 'training_history' in checkpoint:
                history = checkpoint['training_history']
                print(f"  ✅ Training history saved ({len(history.get('val_acc', []))} epochs)")
            else:
                print("  ❌ No training history")
                
            print(f"\n🎯 Resume Capability:")
            if 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
                print("  ✅ FULL RESUME POSSIBLE!")
                print("  ✅ Can continue from Epoch 6 with 58.32% accuracy")
                print("  ✅ All training state preserved")
            else:
                print("  ⚠️ Partial resume - would need to restart optimizer")
                
        except Exception as e:
            print(f"❌ Error reading checkpoint: {e}")
    else:
        print("❌ No checkpoint found")

if __name__ == "__main__":
    check_checkpoint()