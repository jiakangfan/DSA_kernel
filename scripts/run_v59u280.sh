#!/bin/bash
# U280 port of the reported v59relax design: identical verilog, identical
# timing.tcl directives and 6.5 ns constraint; only the part
# (xcu280-fsvh2892-2L-e) and the output directory differ.
WORK="${WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
XILINX_ROOT="${XILINX_ROOT:-/tools/Xilinx}"
VERILOG_DIR="${VERILOG_DIR:-$WORK/v59_u280_verilog}"
OUT_DIR="${OUT_DIR:-$WORK/impl_v59u280}"

exec >> "$WORK/v59u280_chain.log" 2>&1

# Guard: never run two of our own implementations at once.  Match the exact
# process name and only our own uid -- a pattern match on the full command line
# also matches this script and any ssh carrying it.
if ps -u "$(id -un)" -o comm= | grep -qx vivado; then
    echo "[$(date)] ABORT: one of our vivado runs is already active"; exit 1
fi

source "$XILINX_ROOT/Vivado/2022.2/settings64.sh"
mkdir -p "$OUT_DIR"
cd "$VERILOG_DIR" || exit 1

echo "[$(date)] v59 -> U280 impl start (xcu280-fsvh2892-2L-e, 6.5 ns, maxThreads 8)"
vivado -mode batch -source timing.tcl > "$OUT_DIR/run.log" 2>&1
echo "[$(date)] vivado returned $?"
grep -m1 -A3 "Design Timing Summary" "$OUT_DIR/04_route_timing.rpt" 2>/dev/null | tail -1
grep -m1 "Total On-Chip Power" "$OUT_DIR/04_route_power.rpt" 2>/dev/null
