# OpenRAM Design 1 Configuration
# Optimized for QSpice simulation & Python processing

# 1. Capacity & Word Size
word_size = 16          # 16-bit words (Good balance for quantized weights)
num_words = 128         # 128 rows (Total 2Kb)
num_banks = 1

# 2. Technology Selection
# Ensure you have the sky130 PDK or freepdk45 in your $OPENRAM_TECH path
tech_name = "sky130" 

# 3. Output Specs
output_name = "sram_16x128_design1"
output_path = "output/design1"

# 4. Simulation-Friendly Settings
netlist_only = True     # Skips layout to give you the .sp file quickly
analytical_delay = True # Speeds up characterization