# QSpice Simulation Guide for SRAM Design1

## Quick Start (3 Steps)

### Step 1: Download QSpice
- Go to: https://www.qorvo.com/design-hub/design-tools/interactive/qspice
- Download QSpice for Windows
- Install it (free, no registration required for basic use)

### Step 2: Open the Simulation File
1. Launch QSpice
2. File → Open
3. Navigate to: `output/design1/sram_simulation.cir`
4. Click Open

### Step 3: Run Simulation
1. Press **F5** or click Simulate → Run
2. Results display automatically
3. Use zoom and cursor tools to analyze signals

---

## Detailed Instructions

### Option A: Using QSpice GUI (Easiest)

**Installation:**
```
1. Download from https://www.qorvo.com/design-hub/design-tools/interactive/qspice
2. Run installer
3. Accept license and install
```

**Opening Your Simulation:**
```
1. Open QSpice
2. File → Open Circuit
3. Browse to: C:\Users\agarw\OneDrive\Desktop\HAM10000\circuits\qspice\OpenRAM\output\design1\
4. Select: sram_simulation.cir
5. Click Open
```

**Running the Simulation:**
```
1. Click the "Run" button or press F5
2. Wait for simulation to complete (~5-10 seconds)
3. Waveform window will open automatically
```

**Viewing Results:**
```
1. In the Waveform window:
   - Left side shows signal list
   - Bottom shows voltage vs time plot
   - Top toolbar has zoom/measure tools

2. To zoom:
   - Scroll wheel to zoom in/out
   - Click and drag to select region

3. To measure:
   - Use Cursor tool (shows times and values)
   - Measure button for automated measurements
```

---

### Option B: Using Command Line (Advanced)

**In PowerShell:**
```powershell
cd "C:\Users\agarw\OneDrive\Desktop\HAM10000\circuits\qspice\OpenRAM\output\design1"
qspice64.exe sram_simulation.cir
```

**Or with batch mode (runs in background):**
```powershell
qspice64.exe -b sram_simulation.cir
```

---

## What the Simulation Does

### Input Signals:
- **VDD**: 1.8V power supply
- **CK**: 100 MHz clock (10ns period)
- **WE**: Write enable (set to 0 = read mode)
- **RE**: Read enable (set to 1 = reading enabled)
- **CS**: Chip select (set to 1 = chip active)
- **A[0:6]**: Address bus (currently just A0 toggles)
- **DIN[0:15]**: Data input (test patterns on DIN0, DIN1)

### Expected Behavior:
```
Time (ns)   | Clock | Address | DIN  | Output (DOUT)
0-50        | 0Hz   | 0x00    | Low  | High-Z (disabled)
50-100      | 100M  | 0x01    | Data | Reflects memory content
```

### Output Signals to Observe:
- **DOUT[0:15]**: 16-bit data output bus
- **CK**: Clock signal synchronization
- **WE/RE**: Control signal status

---

## File Descriptions

### Files in `output/design1/`:

| File | Purpose |
|------|---------|
| `sram_simulation.cir` | ← **Start HERE** - Main QSpice netlist |
| `sram_16x128_design1.sp` | SRAM circuit description |
| `sram_16x128_design1.lib` | Timing library (.lib format) |
| `sram_16x128_design1.lef` | Physical layout info |
| `sram_16x128_design1_datasheet.txt` | Memory specifications |
| `compilation_summary.json` | Design metadata |

---

## Memory Interface (Quick Reference)

### Signal Definitions:
```
Pin Name    | Direction | Function
------------|-----------|------------------
CK          | Input     | Clock (rising edge)
WE          | Input     | Write Enable (1=write, 0=read)
RE          | Input     | Read Enable (1=read enabled)
CS          | Input     | Chip Select (1=active)
A[0:6]      | Input     | Address (128 locations)
DIN[0:15]   | Input     | Data Input (16-bit)
DOUT[0:15]  | Output    | Data Output (16-bit)
VDD         | Power     | 1.8V supply
VSS         | Ground    | 0V reference
```

### Memory Map:
```
Address  | Contents
---------|----------
0x00     | (empty/your data)
0x01     | (empty/your data)
...
0x7F     | (empty/your data)
(Total: 128 locations × 16 bits = 2 Kilobits)
```

---

## Simulation Parameters

**Transient Analysis:**
- Start time: 0 ns
- Stop time: 100 ns
- Max step: 0.01 ns (very fine resolution)

**Clock:**
- Frequency: 100 MHz
- Period: 10 ns
- Rise/Fall time: 0.5 ns

---

## Tips & Tricks

### 1. Zoom to See Details
- Use mouse wheel to zoom in/out
- Shift+Click to zoom to region
- Right-click for zoom options

### 2. Measure Timing
```
1. Click "Measure" button
2. Click two points on waveform
3. Time difference appears in status bar
```

### 3. Add More Signals
Edit `sram_simulation.cir` and change voltage sources:
```
VWE WE 0 DC 1    ← Change to 1 to enable writes
VRE RE 0 DC 0    ← Change to 0 to disable reads
```

### 4. Change Simulation Time
```
.tran 0 100n 0 0.01n
        ↑  ↑
      start stop
      Change 100n to 500n for longer simulation
```

### 5. Run Multiple Tests
Create multiple .cir files:
- `sram_read_test.cir`
- `sram_write_test.cir`
- `sram_stress_test.cir`

---

## Common Issues & Solutions

### Issue 1: "File not found" error
**Solution:** Use full path in QSpice:
```
C:\Users\agarw\OneDrive\Desktop\HAM10000\circuits\qspice\OpenRAM\output\design1\sram_simulation.cir
```

### Issue 2: Simulation runs but no output
**Solution:** Check if all voltage sources are defined
- Click View → SPICE Error Log
- Fix any missing netlist components

### Issue 3: Waveforms look weird/flat
**Solution:** Adjust simulation parameters
- Decrease max step: `.tran 0 100n 0 0.001n` (more resolution)
- Increase simulation time: `.tran 0 500n 0 0.01n`

### Issue 4: QSpice crashes
**Solution:** Reduce simulation complexity
- Decrease circuit size
- Increase max step size
- Reduce simulation time

---

## Next Steps After Viewing

### 1. Modify the Simulation
Edit `sram_simulation.cir` to test different scenarios:
```
* Test write operation
VWE WE 0 PULSE(0 1.8 20n 0.1n 0.1n 4.5n 10n)

* Test with different addresses
VA1 A1 0 PULSE(0 1.8 30n 0.1n 0.1n 4.5n 10n)
VA2 A2 0 PULSE(0 1.8 40n 0.1n 0.1n 4.5n 10n)
```

### 2. Create Custom Test Patterns
Write Python script to generate test stimuli:
```python
# Generate address sequence for all 128 locations
for addr in range(128):
    print(f"VA{i} A{i} 0 PULSE(...) at address {addr}")
```

### 3. Extract Timing Characteristics
Use `.meas` statements to automatically calculate:
- Access time
- Setup/hold times
- Clock-to-output delay
- Power consumption

### 4. Compare with Datasheet
- Measure actual timing from simulation
- Compare with `sram_16x128_design1_datasheet.txt`
- Verify specifications met

---

## Resources

**QSpice Official:**
- Website: https://www.qorvo.com/design-hub/design-tools/interactive/qspice
- Documentation: https://www.qorvo.com/design-hub/design-tools/interactive/qspice-user-guide
- Forums: https://www.qorvo.com/design-hub

**SPICE Basics:**
- LTspice Tutorial: https://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html
- SPICE Commands: https://www.analog.com/media/en/technical-documentation/user-guides/ltspice_help.pdf

**OpenRAM Documentation:**
- GitHub: https://github.com/VLSIDA/OpenRAM
- Wiki: https://github.com/VLSIDA/OpenRAM/wiki

---

## Quick Summary

```
1. Download QSpice from Qorvo website
2. Install it
3. Open: output/design1/sram_simulation.cir
4. Press F5 to run
5. View waveforms and analyze results
```

**Estimated time:** ~5 minutes total

---

Created: January 15, 2026
Design: 16×128 bit SRAM (2 Kb) in sky130 130nm technology
