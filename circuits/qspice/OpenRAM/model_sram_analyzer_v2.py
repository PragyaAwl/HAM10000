#!/usr/bin/env python3
"""
Neural Network to SRAM Mapping & Inference Simulator
Load EfficientNet-B0 weights and simulate inference on custom SRAM circuit
"""

import torch
from pathlib import Path
import json
from datetime import datetime
import sys

# Set UTF-8 encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*80)
print("EFFICIENTNET-B0 -> SRAM INFERENCE SYSTEM")
print("="*80)

# =====================================================
# STEP 1: LOAD MODEL WEIGHTS
# =====================================================
print("\nSTEP 1: Loading Model Weights")
print("-"*80)

model_path = Path(r"C:\Users\agarw\OneDrive\Desktop\HAM10000\models\best_efficientnet_b0_ham10000.pth")

if not model_path.exists():
    print(f"ERROR: Model file not found: {model_path}")
    sys.exit(1)

print(f"OK: Model file found: {model_path}")
file_size_mb = model_path.stat().st_size / 1024 / 1024
print(f"OK: File size: {file_size_mb:.2f} MB")

# Load the model
try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"OK: Model loaded successfully")
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)

# =====================================================
# STEP 2: ANALYZE MODEL STRUCTURE
# =====================================================
print("\nSTEP 2: Analyzing Model Architecture")
print("-"*80)

# Extract state dict - handle different checkpoint formats
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"OK: Found 'model_state_dict' key in checkpoint")
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print(f"OK: Found 'state_dict' key in checkpoint")
elif isinstance(checkpoint, dict) and 'model' in checkpoint:
    state_dict = checkpoint['model']
    print(f"OK: Found 'model' key in checkpoint")
elif isinstance(checkpoint, dict):
    # Filter only tensors
    state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
    print(f"OK: Using checkpoint dict (filtered for tensors)")
else:
    state_dict = checkpoint
    print(f"OK: Using checkpoint as state_dict")

# Count parameters
total_params = 0
total_params_mb = 0
layer_info = {}

print("\nLayer-by-layer breakdown:")
print(f"{'Layer Name':<50} {'Shape':<20} {'Parameters':<15} {'Size (KB)'}")
print("-"*105)

for name, param in list(state_dict.items())[:50]:  # Show first 50 layers
    # Skip non-tensor entries
    if not isinstance(param, torch.Tensor):
        continue
    
    num_params = param.numel()
    param_size_kb = (num_params * param.element_size()) / 1024
    total_params += num_params
    total_params_mb += param_size_kb / 1024
    
    layer_info[name] = {
        'shape': list(param.shape),
        'params': num_params,
        'size_kb': param_size_kb,
        'dtype': str(param.dtype)
    }
    
    if num_params > 100000:  # Only show large layers
        shape_str = str(param.shape)
        print(f"{name:<50} {shape_str:<20} {num_params:<15,d} {param_size_kb:<15.2f}")

print("-"*105)
print(f"{'TOTAL':<50} {'':<20} {total_params:<15,d} {total_params_mb:.2f} MB")

# Get total from all layers (skip non-tensor entries)
total_params_all = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
total_params_mb_all = sum((p.numel() * p.element_size()) / 1024 / 1024 for p in state_dict.values() if isinstance(p, torch.Tensor))

print(f"\nTotal across ALL {len(state_dict)} layers:")
print(f"  Parameters: {total_params_all:,}")
print(f"  Size (float32): {total_params_mb_all:.2f} MB")

# =====================================================
# STEP 3: SRAM CAPACITY ANALYSIS
# =====================================================
print("\nSTEP 3: SRAM Capacity Analysis")
print("-"*80)

SRAM_SIZE_BITS = 2048  # 16x128 SRAM
SRAM_SIZE_BYTES = SRAM_SIZE_BITS / 8
SRAM_SIZE_KB = SRAM_SIZE_BYTES / 1024
SRAM_SIZE_MB = SRAM_SIZE_KB / 1024

print(f"SRAM Configuration (design1):")
print(f"  Capacity: {SRAM_SIZE_BITS:,} bits ({SRAM_SIZE_BYTES:.0f} bytes)")
print(f"  Size: {SRAM_SIZE_KB:.3f} KB = {SRAM_SIZE_MB:.6f} MB")
print(f"  Organization: 16-bit words * 128 rows")

print(f"\nModel vs SRAM Size:")
print(f"  Model size: {total_params_mb_all:.2f} MB (float32)")
print(f"  SRAM size: {SRAM_SIZE_MB:.6f} MB")
print(f"  Ratio: {total_params_mb_all / SRAM_SIZE_MB:.0f}:1")
print(f"  (Model is {total_params_mb_all / SRAM_SIZE_MB:.0f}x LARGER than SRAM)")

# =====================================================
# STEP 4: QUANTIZATION ANALYSIS
# =====================================================
print("\nSTEP 4: Quantization Requirements for SRAM Storage")
print("-"*80)

print(f"\nTo fit model in {SRAM_SIZE_MB:.6f} MB SRAM:")

compressions = [
    ("float32", 1),
    ("float16", 2),
    ("int8", 4),
    ("int4", 8),
    ("binary", 32),
]

print(f"\n{'Format':<15} {'Compression':<15} {'Size':<15} {'Fits?'}")
print("-"*50)

for format_name, ratio in compressions:
    compressed_size_mb = total_params_mb_all / ratio
    fits = "YES" if compressed_size_mb <= SRAM_SIZE_MB else "NO"
    symbol = "[OK]" if fits == "YES" else "[X]"
    print(f"{format_name:<15} {ratio}:1{'':<10} {compressed_size_mb:.4f} MB  {symbol}")

# =====================================================
# STEP 5: MEMORY ACCESS & TIMING
# =====================================================
print("\nSTEP 5: Memory Access Timing Analysis")
print("-"*80)

SRAM_ACCESS_TIME_NS = 1.2   # From datasheet
SRAM_CYCLE_TIME_NS = 10.0   # At 100 MHz
SRAM_CLOCK_MHZ = 100
WORD_SIZE_BITS = 16
WORD_SIZE_BYTES = 2
SRAM_NUM_WORDS = 128

print(f"\nSRAM Characteristics (design1):")
print(f"  Access time: {SRAM_ACCESS_TIME_NS} ns")
print(f"  Cycle time: {SRAM_CYCLE_TIME_NS} ns ({SRAM_CLOCK_MHZ} MHz)")
print(f"  Word size: {WORD_SIZE_BITS} bits ({WORD_SIZE_BYTES} bytes)")
print(f"  Total words: {SRAM_NUM_WORDS}")
print(f"  Total capacity: {SRAM_NUM_WORDS * WORD_SIZE_BYTES} bytes")

# Calculate reads needed
words_to_read = total_params / (WORD_SIZE_BITS / 32)  # 32-bit params per 16-bit word
reads_needed = words_to_read
total_read_time_ns = reads_needed * SRAM_CYCLE_TIME_NS
total_read_time_ms = total_read_time_ns / 1e6

print(f"\nMemory Access Estimation:")
print(f"  32-bit parameters: {total_params:,}")
print(f"  As 16-bit words: {int(words_to_read):,}")
print(f"  Read cycles needed: {int(reads_needed):,}")
print(f"  Time (at 100 MHz): {total_read_time_ms:.2f} ms")

# =====================================================
# STEP 6: POWER CONSUMPTION
# =====================================================
print("\nSTEP 6: Power Consumption Analysis")
print("-"*80)

SRAM_POWER_STANDBY_MW = 2.3
SRAM_POWER_READ_MW = 12.3
SRAM_POWER_WRITE_MW = 15.8

print(f"SRAM Power (from design1 datasheet):")
print(f"  Standby: {SRAM_POWER_STANDBY_MW} mW")
print(f"  Read @ 100MHz: {SRAM_POWER_READ_MW} mW")
print(f"  Write @ 100MHz: {SRAM_POWER_WRITE_MW} mW")

total_energy_mj = (SRAM_POWER_READ_MW * total_read_time_ms) / 1000

print(f"\nInference Power Budget:")
print(f"  Execution time: {total_read_time_ms:.2f} ms")
print(f"  Energy (SRAM): {total_energy_mj:.4f} mJ")
print(f"  Average power: {SRAM_POWER_READ_MW:.1f} mW")

# =====================================================
# STEP 7: SUMMARY
# =====================================================
print("\n" + "="*80)
print("SUMMARY & IMPLEMENTATION STRATEGY")
print("="*80)

summary = {
    "timestamp": datetime.now().isoformat(),
    "model": {
        "file": str(model_path),
        "total_parameters": total_params_all,
        "total_size_float32_mb": total_params_mb_all,
        "num_layers": len([p for p in state_dict.values() if isinstance(p, torch.Tensor)])
    },
    "sram": {
        "capacity_bits": SRAM_SIZE_BITS,
        "capacity_bytes": int(SRAM_SIZE_BYTES),
        "capacity_mb": SRAM_SIZE_MB,
        "word_size_bits": WORD_SIZE_BITS,
        "access_time_ns": SRAM_ACCESS_TIME_NS,
        "clock_mhz": SRAM_CLOCK_MHZ
    },
    "analysis": {
        "size_ratio": total_params_mb_all / SRAM_SIZE_MB,
        "can_fit_float32": (total_params_mb_all) <= SRAM_SIZE_MB,
        "can_fit_float16": (total_params_mb_all / 2) <= SRAM_SIZE_MB,
        "can_fit_int8": (total_params_mb_all / 4) <= SRAM_SIZE_MB,
        "can_fit_int4": (total_params_mb_all / 8) <= SRAM_SIZE_MB,
        "estimated_read_time_ms": total_read_time_ms,
        "estimated_power_mw": SRAM_POWER_READ_MW,
        "estimated_energy_mj": total_energy_mj
    }
}

print(f"\nKey Findings:")
print(f"  1. Model is {summary['analysis']['size_ratio']:.0f}x LARGER than single SRAM")
print(f"  2. Need weight compression/swapping strategy")
print(f"  3. INT8 quantization: {summary['analysis']['can_fit_int8']}")
print(f"  4. INT4 quantization: {summary['analysis']['can_fit_int4']}")
print(f"  5. Estimated read latency: {total_read_time_ms:.2f} ms")
print(f"  6. Power: {SRAM_POWER_READ_MW:.1f} mW")

print(f"\nRecommended Approach:")
print(f"  1. Use WEIGHT SWAPPING strategy:")
print(f"     - Store full model in main memory")
print(f"     - Load layer weights into SRAM during execution")
print(f"     - Process layer by layer")
print(f"  ")
print(f"  2. Quantization:")
print(f"     - Convert to INT8 (reduce size 4x to {total_params_mb_all/4:.2f} MB)")
print(f"     - Or INT4 (reduce 8x to {total_params_mb_all/8:.2f} MB)")
print(f"  ")
print(f"  3. For full inference calculation:")
print(f"     - Add processor clock cycles for computations")
print(f"     - Factor in weight loading time")
print(f"     - Estimate total end-to-end latency")

# Save analysis
output_path = Path(r"C:\Users\agarw\OneDrive\Desktop\HAM10000\circuits\qspice\OpenRAM\output\design1\model_sram_analysis.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nAnalysis saved: {output_path}")

print("\n" + "="*80)
