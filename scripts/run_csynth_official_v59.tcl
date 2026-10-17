# C synthesis for the reported v59 design.
#   usage:  vitis_hls -f scripts/run_csynth_official_v59.tcl
# Run from the repository root, or set SRC_DIR / SYN_PART / SYN_PERIOD:
#   SYN_PART=xcu280-fsvh2892-2L-e vitis_hls -f scripts/run_csynth_official_v59.tcl

set src_dir [expr {[info exists ::env(SRC_DIR)]    ? $::env(SRC_DIR)    : "src"}]
set part    [expr {[info exists ::env(SYN_PART)]   ? $::env(SYN_PART)   : "xcu55c-fsvh2892-2L-e"}]
set period  [expr {[info exists ::env(SYN_PERIOD)] ? $::env(SYN_PERIOD) : "4.0"}]

open_project -reset official_dsa_decode_v59_csynth
set_top official_dsa_decode

add_files [file join $src_dir official_dsa_decode_v59.cpp] -cflags "-std=c++11 -I$src_dir"

open_solution -reset solution1 -flow_target vitis
set_part $part
create_clock -period $period -name default

config_interface -m_axi_addr64
config_interface -m_axi_alignment_byte_size 64
config_interface -m_axi_max_widen_bitwidth 512

csynth_design
exit
