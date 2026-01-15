; OpenRAM SRAM Design1 - 16x128 (2Kb)
; Technology: sky130 (130nm)
; Date: Auto-generated

.title sky130 SRAM Compiler

* Memory array parameters
* Rows: 128
* Columns: 16
* Word size: 16 bits
* Address width: 7 bits

* Subcircuit definitions
.include "sky130_models.lib"

* Memory array instance
Xmem array[0:127] DATA[0:15] ADDR[0:6] WE RE CS CK VSS VDD sky130_sram_array

* Peripheral circuits
Xcontrol ctrl_logic WE RE CS CK CLKB DECODER_EN VSS VDD

* I/O Buffers
Xin_buf DATA_IN[0:15] DATA_INT[0:15] VSS VDD sky130_input_buffer
Xout_buf DATA_INT[0:15] DATA_OUT[0:15] VSS VDD sky130_output_buffer

.end
