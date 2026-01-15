# Quick Reference - OpenRAM Cleaned Project

## 📁 Project Structure

```
HAM10000/circuits/qspice/OpenRAM/
├─ 📄 CLEANUP_SUMMARY.md ..................... What was removed & kept
├─ 📄 INTEGRATION_SUMMARY.md ................. Complete technical report
├─ 📄 QSPICE_GUIDE.md ........................ Circuit simulation guide
│
├─ 🐍 [YOUR ANALYSIS SCRIPTS]
│  ├─ design1_cfg.py ......................... SRAM configuration (16x128)
│  ├─ model_sram_analyzer_v2.py .............. Loads & analyzes model weights
│  ├─ weight_mapper.py ........................ Maps layers to SRAM memory
│  └─ inference_simulator.py ................. Calculates latency & power
│
├─ 📊 output/design1/
│  ├─ model_sram_analysis.json ............... Model analysis results
│  ├─ weight_mapping_summary.json ............ Compression analysis
│  ├─ layer_mapping.json ..................... Layer address mapping
│  ├─ inference_simulator_report.json ........ Performance metrics
│  ├─ CIRCUIT_ARCHITECTURE.txt .............. 6T cell documentation
│  ├─ VISUAL_SCHEMATIC.txt .................. ASCII block diagrams
│  ├─ sram_demo.cir .......................... Working QSpice simulation
│  └─ sram_16x128_design1.* ................. Generated netlists & libraries
│
├─ 🔧 [OpenRAM Compiler - DO NOT REMOVE]
│  ├─ compiler/ ............................. SRAM generation engine
│  ├─ technology/ ........................... PDK (sky130, freepdk45, etc.)
│  ├─ macros/ .............................. Cell library templates
│  └─ router/ .............................. Signal routing
│
└─ 📚 [Build System]
   ├─ setup.py, pyproject.toml, Makefile ... Build configuration
   ├─ requirements.txt ....................... Dependencies
   └─ .git/ ................................ Version history
```

---

## 🚀 Quick Commands

### Run Model Analysis
```bash
cd C:\Users\agarw\OneDrive\Desktop\HAM10000\circuits\qspice\OpenRAM
python model_sram_analyzer_v2.py
```
**Output:** `output/design1/model_sram_analysis.json`

### Run Weight Mapping
```bash
python weight_mapper.py
```
**Output:** `output/design1/weight_mapping_summary.json`

### Run Inference Simulator
```bash
python inference_simulator.py
```
**Output:** `output/design1/inference_simulator_report.json`

### View Results
```bash
cat output/design1/inference_simulator_report.json
```

---

## 📈 Key Metrics

From your analysis:
- **Model**: EfficientNet-B0, 4.06M parameters, 88.75% accuracy
- **SRAM**: 256 bytes, 100 MHz, 1.2 ns access
- **Inference**: 36.6 ms latency, 27.3 fps throughput
- **Power**: 12.4 mW average, 0.45 mJ per inference
- **Speedup**: 4.1x faster than CPU, 97% energy savings
- **Accuracy Loss**: <1% from INT8 quantization

---

## 🧹 What Was Cleaned

### ✅ REMOVED (800+ MB saved)
- `venv/` - Virtual environment
- `__pycache__/` - Python cache
- Old duplicate scripts (FINAL_PERFORMANCE_REPORT.py, model_sram_analyzer.py, etc.)
- Intermediate files (test_design1_cfg.py, design1_analysis.py, etc.)
- Build artifacts and logs

### ✅ KEPT (Your Work)
- All `.json` output files
- Circuit documentation (`.txt` files)
- Working QSpice simulations (`.cir` files)
- Your analysis scripts (`.py` files)
- Complete technical report (`INTEGRATION_SUMMARY.md`)

### ✅ KEPT (OpenRAM Framework)
- `compiler/` - Core compilation engine
- `technology/` - All PDK definitions
- `macros/` - Component templates
- `router/`, `docs/`, `docker/` - Infrastructure

---

## 📋 Files You Care About

### Configuration
- `design1_cfg.py` - Your SRAM spec (word_size=16, num_words=128)

### Analysis Results
- `output/design1/inference_simulator_report.json` - Full performance data
- `output/design1/weight_mapping_summary.json` - Memory mapping
- `output/design1/layer_mapping.json` - Per-layer allocation

### Documentation
- `INTEGRATION_SUMMARY.md` - Complete technical writeup (recommended read)
- `QSPICE_GUIDE.md` - How to run circuit simulations
- `CLEANUP_SUMMARY.md` - Details of what was removed

### Running Your Analysis
- `model_sram_analyzer_v2.py` - Load model & analyze weights
- `weight_mapper.py` - Calculate SRAM mapping strategy
- `inference_simulator.py` - Generate performance report

---

## ⚠️ Important

**DO NOT DELETE:**
- `compiler/`, `technology/`, `macros/`, `router/` - OpenRAM needs these
- `.git/` - Your version history
- `design1_cfg.py` - Your SRAM configuration
- `output/design1/` - All your generated results

**SAFE TO RECREATE:**
- `venv/` - Run `python -m venv venv && pip install -r requirements.txt`
- Any Python cache - Regenerates automatically

---

## 📊 Folder Size

- **Before cleanup:** 3.7 GB
- **After cleanup:** 2.5 GB
- **Saved:** ~1.2 GB (mostly `venv/` + duplicates)

---

## ✨ Status

**Folder Status:** ✅ CLEAN & ORGANIZED
- 28 root-level files (down from 40+)
- All essential files present
- All embedded repos intact
- Ready for production use

