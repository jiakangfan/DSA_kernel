open_project -reset official_dsa_decode_v59_csynth
set_top official_dsa_decode

add_files \
  /home/jiakang/fpga_dsa_official_full/hls/official_dsa_decode_v59.cpp \
  -cflags "-std=c++11"

open_solution -reset solution1 -flow_target vitis
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 4.0 -name default

config_interface -m_axi_addr64
config_interface -m_axi_alignment_byte_size 64
config_interface -m_axi_max_widen_bitwidth 512

csynth_design
exit
