#!/bin/bash
# U280 port of the reported v59relax design: identical verilog, identical timing.tcl
# directives and 6.5 ns constraint; only the part (xcu280-fsvh2892-2L-e) and output dir differ.
exec >> /home/jiakang/fpga_dsa_official_full/hls/v59u280_chain.log 2>&1
# guard: never run two of MY implementations at once (exact comm match, own uid only)
if ps -u jiakang -o comm= | grep -qx vivado; then echo "[$(date)] ABORT: a jiakang vivado is already running"; exit 1; fi
source /tools/Xilinx/Vivado/2022.2/settings64.sh
cd /home/jiakang/fpga_dsa_official_full/hls/v59_u280_verilog || exit 1
echo "[$(date)] v59 -> U280 impl start (xcu280-fsvh2892-2L-e, 6.5 ns target, maxThreads 8)"
vivado -mode batch -source timing.tcl > /home/jiakang/fpga_dsa_official_full/hls/impl_v59u280/run.log 2>&1
echo "[$(date)] vivado returned $?"
grep -m1 -A3 "Design Timing Summary" /home/jiakang/fpga_dsa_official_full/hls/impl_v59u280/04_route_timing.rpt 2>/dev/null | tail -1
grep -m1 "Total On-Chip Power" /home/jiakang/fpga_dsa_official_full/hls/impl_v59u280/04_route_power.rpt 2>/dev/null
