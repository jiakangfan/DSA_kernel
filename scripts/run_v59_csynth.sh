#!/bin/bash
# C synthesis for v59.  Set WORK to the directory holding the sources and the
# tcl; it defaults to the directory this script lives in.
WORK="${WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
XILINX_ROOT="${XILINX_ROOT:-/tools/Xilinx}"

exec >> "$WORK/v59_chain.log" 2>&1
trap "echo \"[\$(date)] chain EXIT status=\$?\"" EXIT
cd "$WORK" || exit 1
source "$XILINX_ROOT/Vitis_HLS/2022.2/settings64.sh"

echo "[$(date)] v59 csynth start"
vitis_hls -f run_csynth_official_v59.tcl > csynth_official_v59.log 2>&1
echo "[$(date)] csynth returned $?"

R=official_dsa_decode_v59_csynth/solution1/syn/report/csynth.rpt
grep -m1 "^ *|+ official_dsa_decode " "$R"
echo "--- II violations ---"
grep -c "II Violation" csynth_official_v59.log
grep "II Violation" csynth_official_v59.log | sed "s/.*module .//;s/. (loop.*//" | sort -u
