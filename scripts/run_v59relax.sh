#!/bin/bash
exec >> /home/jiakang/fpga_dsa_official_full/hls/v59relax_chain.log 2>&1
source /tools/Xilinx/Vivado/2022.2/settings64.sh
cd /home/jiakang/fpga_dsa_official_full/hls/v59_relaxed_verilog || exit 1
echo "[$(date)] v59 relaxed-constraint impl start (6.5 ns target)"
vivado -mode batch -source timing.tcl > /home/jiakang/fpga_dsa_official_full/hls/impl_v59relax/run.log 2>&1
echo "[$(date)] vivado returned $?"
grep -m1 -A3 "Design Timing Summary" /home/jiakang/fpga_dsa_official_full/hls/impl_v59relax/04_route_timing.rpt 2>/dev/null | tail -1
