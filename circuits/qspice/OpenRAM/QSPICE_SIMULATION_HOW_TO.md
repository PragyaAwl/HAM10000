# How to Simulate Your SRAM in QSpice - Complete Guide

## 🎯 Quick Summary

You have **3 options** to test your SRAM circuit in QSpice:

1. **GUI Method** (Easiest) - Click and play
2. **Command Line** (Medium) - Run from terminal
3. **Custom Simulation** (Advanced) - Modify test patterns

---

## Option 1: GUI Method (RECOMMENDED - Easiest)

### Step 1: Download & Install QSpice

1. Go to: **https://www.qorvo.com/design-hub/design-tools/interactive/qspice**
2. Click "Download QSpice"
3. Download for Windows (QSpice64)
4. Install (free, no registration needed)

### Step 2: Open Your Simulation File

1. **Launch QSpice** (look for QSpice icon on desktop)
2. Click **File → Open** (or Ctrl+O)
3. Navigate to:
   ```
   C:\Users\agarw\OneDrive\Desktop\HAM10000\circuits\qspice\OpenRAM\output\design1\
   ```
4. Select **`sram_demo.cir`** ← This is your working simulation
5. Click **Open**

### Step 3: Run the Simulation

1. Click the **green "Run" button** (or press **F5**)
2. You'll see a progress bar
3. After 5-10 seconds, a **waveform window** pops up automatically

### Step 4: Analyze Results

The waveform window shows:

```
LEFT SIDE: Signal List           CENTER: Waveform Plot
├─ CK (Clock)                    ├─ CK toggles at 100 MHz
├─ WE (Write Enable)             ├─ WE pulses for writes
├─ RE (Read Enable)              ├─ RE stays high (read enabled)
├─ CS (Chip Select)              ├─ CS stays high (active)
├─ A0, A1, A2... (Address)       ├─ Address bits change
├─ DIN0-DIN15 (Data In)          ├─ Data pulses at 10ns
└─ DOUT0-DOUT15 (Data Out)       └─ Output shows memory responses
```

**To zoom in/out:** Scroll mouse wheel over the plot

**To measure:** Use the cursor tool to click on signals and read values

---

## Option 2: Command Line Method

### Quick Run (GUI mode from terminal)

```powershell
# Navigate to the simulation folder
cd "C:\Users\agarw\OneDrive\Desktop\HAM10000\circuits\qspice\OpenRAM\output\design1"

# Run QSpice
qspice64.exe sram_demo.cir
```

The GUI will open automatically with your circuit.

### Batch Mode (Headless - generates data files)

```powershell
qspice64.exe -b sram_demo.cir
```

This runs without GUI and generates `.raw` waveform file for analysis.

---

## Option 3: Modify & Create Custom Tests

### What Each Signal Does

In `sram_demo.cir`, here are the test patterns:

```
SIGNAL          WHAT IT DOES                  WHEN
─────────────────────────────────────────────────────────────
CK              Clock pulse                   Every 10ns (100 MHz)
WE              Write enable (high=write)     Pulses at 5ns
RE              Read enable (always high)     Continuous
CS              Chip select (always high)     Continuous (chip on)
A0, A1, A2      Address bits (toggle)         Different times
DIN0-DIN3       Data input pulses             10-16ns (test data)
```

### To Create a Custom Test

1. Make a copy of `sram_demo.cir`
2. Open it in a text editor (VS Code recommended)
3. Modify the voltage sources (VPULSE lines)

**Example: Write 0xAB (10101011) to address 0x05**

```spice
* Change address to 0x05 (binary 000101)
VA0 A0 0 DC 1.8        ; Bit 0 = 1
VA1 A1 0 DC 1.8        ; Bit 1 = 0
VA2 A2 0 DC 1.8        ; Bit 2 = 1
VA3 A3 0 DC 0          ; Bit 3 = 0
VA4 A4 0 DC 0          ; Bit 4 = 0
VA5 A5 0 DC 0          ; Bit 5 = 0
VA6 A6 0 DC 0          ; Bit 6 = 0

* Set data input to 0xAB (10101011)
VDIN0 DIN0 0 DC 1.8    ; LSB = 1
VDIN1 DIN1 0 DC 1.8    ; = 1
VDIN2 DIN2 0 DC 0      ; = 0
VDIN3 DIN3 0 DC 1.8    ; = 1
VDIN4 DIN4 0 DC 0      ; = 0
VDIN5 DIN5 0 DC 1.8    ; = 1
VDIN6 DIN6 0 DC 1.8    ; = 1
VDIN7 DIN7 0 DC 0      ; MSB = 0

* Enable write
VWE WE 0 DC 1.8        ; Write enabled
```

---

## Available Simulation Files

| File | What It Does | Use Case |
|------|-------------|----------|
| **sram_demo.cir** ✅ | Working test with visible signals | **START HERE** |
| sram_cmos_detailed.cir | Transistor-level netlist | Advanced analysis |
| sram_16x128_design1.sp | SPICE netlist from OpenRAM | Reference |

---

## What You Should See in Results

### Timing Characteristics (from your datasheet)

```
Parameter              | Expected Value
───────────────────────|────────────────
Clock frequency        | 100 MHz
Clock period           | 10 ns
Access time (read)     | 1.2 ns
Setup time (write)     | ~0.5 ns
Hold time              | ~0.2 ns
Power (read)           | 12.3 mW
Power (standby)        | 2.3 mW
```

### What the Waveforms Should Show

1. **Clock (CK):** Clean square wave at 100 MHz
2. **Address (A0-A6):** Step through different addresses
3. **Write Enable (WE):** Pulses during write operations
4. **Data Output (DOUT0-15):** Changes based on read address

---

## Troubleshooting

### QSpice Won't Start
- Make sure QSpice64.exe is installed
- Check installation path: `C:\Program Files\QSpice\`
- Try running as Administrator

### No Waveforms Appear
- Make sure you opened **sram_demo.cir** (not a .sp file)
- Check that `.plot` and `.save` commands are in the .cir file
- Press F5 to run simulation

### Waveforms Look Strange
- Check the time scale (usually 0-100ns)
- Use zoom tool to see details
- Make sure all power supplies are connected (VDD=1.8V, VSS=0)

### Command Line Not Working
- Add QSpice to system PATH, or use full path:
  ```powershell
  "C:\Program Files\QSpice\qspice64.exe" sram_demo.cir
  ```

---

## Next Steps After Simulation

### 1. Verify Timing
- Measure setup/hold times with cursor tool
- Check clock-to-output delay
- Compare with datasheet specs

### 2. Test Different Scenarios
- Try reads from different addresses
- Try back-to-back writes
- Test edge cases (0x00, 0x7F)

### 3. Export Results
- QSpice can export waveforms as:
  - `.csv` (spreadsheet format)
  - `.txt` (text format)
  - `.raw` (binary format)

### 4. Compare with Neural Network
- Use timing data to verify:
  - Inference latency (36.6ms from your model)
  - Power consumption (12.4mW average)
  - Memory access patterns

---

## Quick Commands Reference

```powershell
# Start from folder
cd C:\Users\agarw\OneDrive\Desktop\HAM10000\circuits\qspice\OpenRAM\output\design1

# Option A: GUI (recommended)
qspice64.exe sram_demo.cir

# Option B: Headless batch
qspice64.exe -b sram_demo.cir

# Option C: With custom config
qspice64.exe -b sram_demo.cir -raw output.raw
```

---

## Summary

| Method | Ease | Time | Best For |
|--------|------|------|----------|
| GUI (Option 1) | ⭐⭐⭐⭐⭐ | 2 min | Visual analysis, quick checks |
| CLI (Option 2) | ⭐⭐⭐⭐ | 5 min | Automated testing, scripts |
| Custom (Option 3) | ⭐⭐⭐ | 15 min | Specific test cases |

**Recommended:** Start with **Option 1 (GUI)** → Open sram_demo.cir → Press F5

---

**Files you need:**
- ✅ `sram_demo.cir` - Main simulation file
- ✅ QSpice installed
- ✅ 5-10 seconds to run

**That's it! You're ready to simulate your SRAM circuit.** 🚀

