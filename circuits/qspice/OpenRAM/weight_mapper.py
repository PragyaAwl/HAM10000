#!/usr/bin/env python3
"""
Weight Mapper & SRAM Storage Simulator
Maps quantized model weights to SRAM memory layout
Calculates layer-by-layer storage and access patterns
"""

import torch
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

print("="*80)
print("WEIGHT MAPPER & SRAM STORAGE SIMULATOR")
print("="*80)

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = Path(r"C:\Users\agarw\OneDrive\Desktop\HAM10000\models\best_efficientnet_b0_ham10000.pth")
OUTPUT_DIR = Path(r"C:\Users\agarw\OneDrive\Desktop\HAM10000\circuits\qspice\OpenRAM\output\design1")

# SRAM Configuration
SRAM_WORD_SIZE_BYTES = 2  # 16 bits
SRAM_NUM_WORDS = 128
SRAM_TOTAL_BYTES = SRAM_WORD_SIZE_BYTES * SRAM_NUM_WORDS
SRAM_ACCESS_TIME_NS = 1.2
SRAM_CYCLE_TIME_NS = 10.0  # 100 MHz

# =====================================================
# STEP 1: LOAD AND ANALYZE LAYERS
# =====================================================
print("\nSTEP 1: Loading Model and Analyzing Layers")
print("-"*80)

checkpoint = torch.load(MODEL_PATH, map_location='cpu')
state_dict = checkpoint['model_state_dict']

# Group parameters by layer
layers = defaultdict(lambda: {'params': [], 'total_params': 0, 'total_bytes': 0})

for name, param in state_dict.items():
    if not isinstance(param, torch.Tensor):
        continue
    
    # Extract layer name (e.g., "features.0.0.weight" -> "features.0")
    parts = name.split('.')
    if len(parts) > 2:
        layer_key = '.'.join(parts[:-1])
    else:
        layer_key = name
    
    num_params = param.numel()
    size_bytes = num_params * 4  # float32
    
    layers[layer_key]['params'].append({
        'name': name,
        'shape': list(param.shape),
        'num_params': num_params,
        'size_bytes': size_bytes,
        'dtype': 'float32'
    })
    layers[layer_key]['total_params'] += num_params
    layers[layer_key]['total_bytes'] += size_bytes

print(f"Found {len(layers)} unique layers")
print(f"Total layers with parameters: {sum(1 for l in layers.values() if l['total_params'] > 0)}")

# =====================================================
# STEP 2: QUANTIZATION & SIZE REDUCTION
# =====================================================
print("\nSTEP 2: Quantization Analysis")
print("-"*80)

# For each layer, analyze quantization options
layer_strategies = {}

print(f"\n{'Layer':<40} {'float32':<12} {'int8':<12} {'int4':<12} {'strategy'}")
print("-"*85)

total_original = 0
total_int8 = 0
total_int4 = 0

for layer_name in sorted(layers.keys())[:15]:  # Show first 15
    layer = layers[layer_name]
    if layer['total_params'] == 0:
        continue
    
    original_kb = layer['total_bytes'] / 1024
    int8_kb = (layer['total_bytes'] / 4) / 1024
    int4_kb = (layer['total_bytes'] / 8) / 1024
    
    total_original += layer['total_bytes']
    total_int8 += layer['total_bytes'] / 4
    total_int4 += layer['total_bytes'] / 8
    
    # Determine best strategy
    if int4_kb < 0.5:
        strategy = "INT4 fits"
    elif int8_kb < 1.0:
        strategy = "INT8 fits"
    else:
        strategy = "Swap needed"
    
    print(f"{layer_name:<40} {original_kb:<12.2f} {int8_kb:<12.2f} {int4_kb:<12.2f} {strategy}")

total_original_all = sum(l['total_bytes'] for l in layers.values())
total_int8_all = total_original_all / 4
total_int4_all = total_original_all / 8

print("-"*85)
print(f"TOTAL:                                       {total_original_all/1024/1024:<12.2f} {total_int8_all/1024/1024:<12.2f} {total_int4_all/1024/1024:<12.2f}")

# =====================================================
# STEP 3: LAYER-BY-LAYER SRAM MAPPING
# =====================================================
print("\nSTEP 3: Layer-by-Layer SRAM Mapping (INT8 Quantization)")
print("-"*80)

print(f"\nSRAM Capacity: {SRAM_TOTAL_BYTES} bytes ({SRAM_NUM_WORDS} words)")
print(f"Mapping strategy: Load one layer at a time, process, then load next\n")

layer_mapping = []
current_sram_addr = 0
total_layers = len([l for l in layers.values() if l['total_params'] > 0])

print(f"{'Layer':<40} {'Params':<12} {'INT8 (KB)':<12} {'SRAM Addr':<12} {'Status'}")
print("-"*85)

for idx, layer_name in enumerate(sorted(layers.keys())):
    layer = layers[layer_name]
    if layer['total_params'] == 0:
        continue
    
    # Quantize to INT8
    int8_bytes = int(layer['total_bytes'] / 4)
    
    # Check if fits in SRAM
    fits_in_sram = int8_bytes <= SRAM_TOTAL_BYTES
    status = "FITS" if fits_in_sram else "SWAP"
    
    mapping = {
        'layer': layer_name,
        'original_params': layer['total_params'],
        'original_bytes': layer['total_bytes'],
        'int8_bytes': int8_bytes,
        'sram_addr_offset': current_sram_addr,
        'fits_in_sram': fits_in_sram,
        'num_sram_words': int(np.ceil(int8_bytes / SRAM_WORD_SIZE_BYTES))
    }
    layer_mapping.append(mapping)
    
    current_sram_addr = (current_sram_addr + mapping['num_sram_words']) % SRAM_NUM_WORDS
    
    # Print first 20 layers
    if idx < 20:
        print(f"{layer_name:<40} {layer['total_params']:<12,d} {int8_bytes/1024:<12.2f} {mapping['sram_addr_offset']:<12d} {status}")

print(f"\n... and {total_layers - 20} more layers")

# =====================================================
# STEP 4: MEMORY ACCESS PATTERNS
# =====================================================
print("\nSTEP 4: Inference Latency Calculation")
print("-"*80)

# Calculate total accesses needed
total_params = sum(l['total_params'] for l in layers.values())
total_int8_bytes = total_params

# Worst case: sequential read of all weights
read_accesses_needed = int(np.ceil(total_int8_bytes / SRAM_WORD_SIZE_BYTES))
read_time_ns = read_accesses_needed * SRAM_CYCLE_TIME_NS
read_time_ms = read_time_ns / 1e6

# Add computation time (assume MAC operations)
# EfficientNet-B0: ~390M FLOPs typical
# At 1 GHz: 390ms for pure computation
# Assuming 100MHz estimate for compute
compute_ops = 390e6  # 390M FLOPs
compute_clock_mhz = 100
compute_time_ms = (compute_ops / compute_clock_mhz) / 1e6

# Add overhead for layer swaps
num_layer_swaps = len([l for l in layer_mapping if not l['fits_in_sram']])
swap_overhead_ms = num_layer_swaps * 0.1  # 0.1ms per swap

total_inference_time_ms = read_time_ms + compute_time_ms + swap_overhead_ms

print(f"\nInference Timing Breakdown:")
print(f"  Weight reads: {read_accesses_needed:,} accesses")
print(f"  - Read time (SRAM): {read_time_ms:.3f} ms")
print(f"  - Compute time (estimated): {compute_time_ms:.1f} ms")
print(f"  - Layer swap overhead: {swap_overhead_ms:.3f} ms")
print(f"  - TOTAL INFERENCE: {total_inference_time_ms:.3f} ms")

# =====================================================
# STEP 5: POWER CALCULATION
# =====================================================
print("\nSTEP 5: Power & Energy Analysis")
print("-"*80)

SRAM_POWER_STANDBY_MW = 2.3
SRAM_POWER_READ_MW = 12.3
SRAM_POWER_WRITE_MW = 15.8

# Estimate duty cycle
read_fraction = read_time_ms / total_inference_time_ms if total_inference_time_ms > 0 else 0
compute_fraction = compute_time_ms / total_inference_time_ms if total_inference_time_ms > 0 else 0

sram_avg_power_mw = (read_fraction * SRAM_POWER_READ_MW + 
                     compute_fraction * SRAM_POWER_STANDBY_MW)

total_energy_mj = (sram_avg_power_mw * total_inference_time_ms) / 1000

print(f"\nPower Profile:")
print(f"  SRAM read power: {SRAM_POWER_READ_MW} mW")
print(f"  Read duty cycle: {read_fraction*100:.1f}%")
print(f"  Average power: {sram_avg_power_mw:.1f} mW")
print(f"  Total energy per inference: {total_energy_mj:.4f} mJ")

# =====================================================
# STEP 6: SUMMARY & RECOMMENDATIONS
# =====================================================
print("\n" + "="*80)
print("SUMMARY: MAPPING EfficientNet-B0 -> 2Kb SRAM")
print("="*80)

summary = {
    "model_analysis": {
        "total_parameters": int(total_params),
        "total_size_float32_mb": total_original_all / 1024 / 1024,
        "total_size_int8_mb": total_int8_all / 1024 / 1024,
        "total_size_int4_mb": total_int4_all / 1024 / 1024,
        "num_layers": total_layers,
    },
    "sram_configuration": {
        "word_size_bits": SRAM_WORD_SIZE_BYTES * 8,
        "num_words": SRAM_NUM_WORDS,
        "total_bytes": SRAM_TOTAL_BYTES,
        "access_time_ns": SRAM_ACCESS_TIME_NS,
        "cycle_time_ns": SRAM_CYCLE_TIME_NS,
        "clock_mhz": 1000 / SRAM_CYCLE_TIME_NS,
    },
    "mapping_strategy": {
        "quantization": "INT8 (4x compression)",
        "layers_fit_in_sram": sum(1 for l in layer_mapping if l['fits_in_sram']),
        "layers_requiring_swap": num_layer_swaps,
        "total_layers": total_layers,
    },
    "inference_performance": {
        "weight_read_time_ms": read_time_ms,
        "compute_time_ms": compute_time_ms,
        "swap_overhead_ms": swap_overhead_ms,
        "total_inference_time_ms": total_inference_time_ms,
        "throughput_inferences_per_sec": 1000 / total_inference_time_ms if total_inference_time_ms > 0 else 0,
    },
    "power_analysis": {
        "avg_power_mw": sram_avg_power_mw,
        "energy_per_inference_mj": total_energy_mj,
        "standby_power_mw": SRAM_POWER_STANDBY_MW,
        "read_power_mw": SRAM_POWER_READ_MW,
    }
}

print(f"\nModel Size Comparison:")
print(f"  Float32: {summary['model_analysis']['total_size_float32_mb']:.2f} MB")
print(f"  INT8:    {summary['model_analysis']['total_size_int8_mb']:.2f} MB (4x smaller)")
print(f"  INT4:    {summary['model_analysis']['total_size_int4_mb']:.2f} MB (8x smaller)")
print(f"  SRAM:    {SRAM_TOTAL_BYTES / 1024:.3f} KB")

print(f"\nMapping Strategy:")
print(f"  Approach: Layer-by-layer weight swapping")
print(f"  Layers that fit in SRAM: {summary['mapping_strategy']['layers_fit_in_sram']}/{total_layers}")
print(f"  Layers requiring swap: {num_layer_swaps}/{total_layers}")

print(f"\nInference Performance:")
print(f"  Latency: {summary['inference_performance']['total_inference_time_ms']:.3f} ms per inference")
print(f"  Throughput: {summary['inference_performance']['throughput_inferences_per_sec']:.1f} inferences/sec")

print(f"\nPower & Energy:")
print(f"  Average power: {summary['power_analysis']['avg_power_mw']:.1f} mW")
print(f"  Energy per inference: {summary['power_analysis']['energy_per_inference_mj']:.4f} mJ")

# =====================================================
# SAVE RESULTS
# =====================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save mapping
with open(OUTPUT_DIR / 'layer_mapping.json', 'w') as f:
    json.dump(layer_mapping[:50], f, indent=2)  # Save first 50 layers
print(f"\nLayer mapping saved: {OUTPUT_DIR / 'layer_mapping.json'}")

# Save summary
with open(OUTPUT_DIR / 'weight_mapping_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {OUTPUT_DIR / 'weight_mapping_summary.json'}")

print("\n" + "="*80)
