#!/usr/bin/env python3
"""
Quick launcher for 97% accuracy training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from train_to_97_percent import train_97_percent_model

if __name__ == "__main__":
    print("🚀 Starting HAM10000 training to achieve 97%+ accuracy...")
    print("This will use advanced techniques including:")
    print("- Heavy data augmentation")
    print("- Advanced optimization strategies") 
    print("- Learning rate scheduling")
    print("- Test-time augmentation")
    print("- Class balancing")
    print()
    
    # Start training
    train_97_percent_model(
        target_accuracy=97.0,
        max_epochs=200,
        batch_size=32
    )