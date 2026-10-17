#!/bin/bash
# Vivado out-of-context implementation at the relaxed 6.5 ns constraint (U55C).
# Run after csynth: copy solution1/syn/verilog (INCLUDING the *.dat ROM init
# files) to $VERILOG_DIR, put timing.tcl in it, then run this.
WORK="${WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
XILINX_ROOT="${XILINX_ROOT:-/tools/Xilinx}"
VERILOG_DIR="${VERILOG_DIR:-$WORK/v59_relaxed_verilog}"
OUT_DIR="${OUT_DIR:-$WORK/impl_v59relax}"

exec >> "$WORK/v59relax_chain.log" 2>&1
source "$XILINX_ROOT/Vivado/2022.2/settings64.sh"
mkdir -p "$OUT_DIR"
cd "$VERILOG_DIR" || exit 1

echo "[$(date)] v59 relaxed-constraint impl start (6.5 ns target)"
vivado -mode batch -source timing.tcl > "$OUT_DIR/run.log" 2>&1
echo "[$(date)] vivado returned $?"
grep -m1 -A3 "Design Timing Summary" "$OUT_DIR/04_route_timing.rpt" 2>/dev/null | tail -1
