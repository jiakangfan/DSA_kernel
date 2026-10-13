# DSA_kernel

Streaming FPGA accelerator for the complete DeepSeek Sparse Attention (DSA) decode
operator, implemented in Vitis HLS for the AMD Alveo U55C.

This is the implementation reported in:

> J. Fan, R. Jayanth, and V. K. Prasanna, "Energy-Efficient Streaming FPGA Accelerator
> for DeepSeek Sparse Attention Decode," *International Workshop on Heterogeneous
> High-performance Reconfigurable Computing (H2RC)*, held with SC26, Chicago, IL,
> Nov. 2026.

The kernel fuses index scoring, exact top-2048 selection, K/V gather, sparse attention,
and the output projection into a single kernel, keeping all intermediates on chip.

## Files

| Path | What it is |
|---|---|
| `src/official_dsa_decode_v59.cpp` | The kernel. 2022 lines, 247 HLS pragmas. This is what becomes hardware. |
| `src/official_dsa_decode_v57.hpp` | Shared constants and types (head counts, dimensions, top-k, lane counts). Unchanged since v57. |
| `src/odsa_exp_lut.hpp` | The 256-entry exponential lookup table used by the integer softmax. |
| `src/test_official_dsa_decode_v59.cpp` | C-simulation testbench. Host-side only, zero HLS pragmas. |
| `scripts/run_csynth_official_v59.tcl` | Vitis HLS synthesis: target `xcu55c-fsvh2892-2L-e`, 4.0 ns clock, 512-bit AXI widening. |
| `scripts/run_v59_csynth.sh` | Driver for the above; greps the report for II violations. |
| `scripts/vivado_impl_timing.tcl` | Vivado out-of-context implementation at a relaxed 6.5 ns constraint. |
| `scripts/run_v59relax.sh` | Driver for the U55C implementation. |
| `scripts/run_v59u280.sh` | Same netlist retargeted to `xcu280-fsvh2892-2L-e`. |

## Configuration

| Parameter | Value |
|---|---|
| Indexer | 64 heads x 128 dims, signed INT4 |
| Selection | exact top-k, k = 2048, deterministic tie breaking |
| Attention | 128 heads x 576 dims (512 latent + 64 positional), signed INT8 |
| Physical head datapaths | 16, time-multiplexed over the 128 heads in 8 groups |
| Output projection | 512 -> 128 per head |
| Accumulation | INT32 |
| Context lengths | 4K, 8K, 16K, 32K, 64K, 128K, batch 1 |

## Results (U55C, post-route)

| Resource | Used | Available | % |
|---|---|---|---|
| LUT | 872,713 | 1,303,680 | 66.94 |
| FF | 439,196 | 2,607,360 | 16.84 |
| DSP | 4,784 | 9,024 | 53.01 |
| BRAM tile | 460 | 2,016 | 22.82 |
| URAM | 288 | 960 | 30.00 |

Achieved clock 140.9 MHz; total on-chip power 11.324 W.
Cycle count follows `C ~= 0.625 L + 27,805` across 4K-128K.

## Build

Both tool environments must be sourced or the tools fail silently:

```bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh
source /tools/Xilinx/Vitis_HLS/2022.2/settings64.sh
```

1. **C simulation** - needs `OFFICIAL_FIXTURE_DIR` pointing at a directory of
   reference tensors dumped from the model (`index_q.bin`, `index_k.bin`,
   `index_weights.bin`, `q_latent.bin`, `q_rope.bin`, `latent_cache.bin`,
   `pe_cache.bin`, `value_projection.bin`, `selected.bin`,
   `output_scaled_quant_f32.bin`). Fixtures are not in this repository.
   Pass criteria: zero top-k mismatches, max absolute output error <= 2.0,
   output cosine similarity >= 0.999999.
2. **Synthesis**: `vitis_hls -f scripts/run_csynth_official_v59.tcl`
3. **Implementation**: copy the generated `syn/verilog` directory (including the
   `*.dat` ROM initialisation files), then run Vivado on
   `scripts/vivado_impl_timing.tcl` with the working directory inside it.

## Reading the kernel

`src/official_dsa_decode_v59.cpp` carries earlier iterations alongside the live
design. Lines 7-456 and 1129-1289 are superseded (the binary heap and the
insertion-array selector are kept because they are the measured baselines for the
2.6x LUT comparison). The live call chain is:

```
official_dsa_decode              1923   top level, AXI interfaces
|-- exact_topk_v4                1087   phase 1
|   |-- score_scan_stage         1052   DATAFLOW: scoring and counting overlap
|   |   |-- index_score_stage     457   INT4 scoring, 2 tokens/cycle
|   |   \-- store_stage           671   buffer write + first histogram pass
|   \-- radix_select_stage        724   two further counting passes + emission
\-- compute_attention_all_heads_v3 1474 phase 2
    \-- odsa_gather_and_score    1453   DATAFLOW: gather and scoring overlap
```

Lines worth knowing:

| Line | Why |
|---|---|
| 1940-1953 | The fourteen AXI master bundles. `index_k0..k3` are four 512-bit ports; `latent_cache0/1` are two. |
| 543-556 | Fully unrolled 64 heads x 128 lanes, giving 16,384 signed-INT4 products per cycle. |
| 553 | `BIND_OP ... impl=fabric` - index multipliers bound to LUT fabric rather than DSP columns, which doubles the post-route clock. |
| 1433 | `BIND_OP ... impl=dsp` - attention multipliers bound to DSPs. |
| 1062-1067 | `DATAFLOW` plus two depth-512 streams: the first counting pass costs no extra cycles. |
| 747 | `DEPENDENCE ... dependent=false` on the per-lane histograms. |
| 1023 | The only pipeline in the file with II != 1: the tie-break merge carries a loop dependency on the lane pointers. Its trip count is bounded by the actual deficit rather than the full 2048. |
