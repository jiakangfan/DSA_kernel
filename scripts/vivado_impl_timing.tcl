set_param general.maxThreads 8
set part xcu55c-fsvh2892-2L-e
set out /home/jiakang/fpga_dsa_official_full/hls/impl_v59relax
file mkdir $out
file mkdir $out/ip
create_project -in_memory -part $part
set_property ip_output_repo $out/ip [current_project]
set_property target_language Verilog [current_project]
foreach t [glob -nocomplain *_ip.tcl] { puts "IP: $t"; source $t }
foreach ip [get_ips] { generate_target synthesis $ip }
read_verilog [glob *.v]
set fp [open clk.xdc w]
puts $fp "create_clock -period 6.500 -name ap_clk \[get_ports ap_clk\]"
close $fp
read_xdc clk.xdc

# The critical path is a pipeline enable fanning out to 5019 DSPs across all
# three SLRs: 95% wire, one logic level.  Every directive below targets that
# rather than area or runtime.
synth_design -top official_dsa_decode -part $part -mode out_of_context \
    -fanout_limit 200 -retiming
report_utilization -file $out/01_synth_util.rpt
report_timing_summary -file $out/01_synth_timing.rpt
write_checkpoint -force $out/synth.dcp
puts "===== SYNTH DONE ====="

opt_design -directive ExploreWithRemap
place_design -directive SSI_SpreadSLLs
report_timing_summary -file $out/02_place_timing.rpt
write_checkpoint -force $out/placed.dcp
puts "===== PLACE DONE ====="

phys_opt_design -directive AggressiveFanoutOpt
phys_opt_design -directive AggressiveExplore
report_timing_summary -file $out/03_physopt_timing.rpt
write_checkpoint -force $out/physopt.dcp
puts "===== PHYSOPT DONE ====="

route_design -directive AggressiveExplore
report_utilization    -file $out/04_route_util.rpt
report_timing_summary -file $out/04_route_timing.rpt
report_power          -file $out/04_route_power.rpt
write_checkpoint -force $out/routed.dcp
puts "===== ROUTE DONE ====="
