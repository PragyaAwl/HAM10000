#!/usr/bin/env python3
"""
Quick status check for resumed training.
"""

import os
from datetime import datetime

def quick_status():
    print("🔍 Resume Training Status")
    print("=" * 30)
    
    # Check checkpoint
    checkpoint_path = "models/best_ham10000_model.pth"
    if os.path.exists(checkpoint_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
        size = os.path.getsize(checkpoint_path)
        print(f"✅ Checkpoint: {size:,} bytes, modified {mtime}")
        
        # Check if recently modified (within last 5 minutes)
        import time
        if time.time() - os.path.getmtime(checkpoint_path) < 300:
            print("  🔄 Recently updated - training likely active!")
        else:
            print("  ⏰ Not recently updated")
    else:
        print("❌ No checkpoint found")
    
    # Check results
    results_path = "results/resumed_training_results.json"
    if os.path.exists(results_path):
        print("✅ Resume results file exists")
    else:
        print("❌ No resume results yet")
    
    print(f"\n⏰ Current time: {datetime.now()}")

if __name__ == "__main__":
    quick_status()