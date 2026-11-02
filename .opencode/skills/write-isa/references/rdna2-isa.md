# RDNA2 ISA reference for gfx1031

This is the working reference for writing and tuning gfx1031 (RX 6700, Navi 22) code at the
instruction level. The complete opcode list, generated from AMD's machine-readable RDNA2
specification, is in `opcodes.md`. This file covers the execution model, the encoding formats,
the instructions that matter for compute kernels, the hazard rules, and the measured costs on this
card.

Sources:

- AMD machine-readable ISA specification, `amdgpu_isa_rdna2.xml` (MIT, Copyright (c) 2026 AMD).
- AMD RDNA2 ISA Reference Guide and the RDNA Performance Guide.
- Measured on the project card with `tools/IsaBench` and RGP.

## Execution model

- A WGP contains two CUs. `hipDeviceProp.multiProcessorCount` reports **18 on this card**, which is
  the WGP count; the hardware has 36 CUs. Account for this when converting a grid to waves per CU.
- A CU has two SIMD32 units and 64 stream processors; a WGP has four SIMD32. A wave32 VALU
  instruction issues in one cycle on its SIMD32 in the ideal case (GCN's four SIMD16 with four-cycle
  issue is the older model). The measured single-wave interval is ~1.75 cycles (see below), so one
  wave does not fill its SIMD.
- RDNA2 is wave32-native and also supports wave64. Kernels in this repository are wave32.
- LDS is 64 KB per CU (128 KB per WGP). It is 32 banks of 32 bits. The `ds_read_b32` microbenchmark
  showed no read bank penalty at any stride, while writes serialize on same-bank addresses (see the
  measured table below). AMD's guides describe read conflicts too, so treat the read result as
  specific to the b32 test and re-measure before relying on it for wide reads.
- VGPRs are allocated per wave in granularity. The compiler reports the real count as
  `.amdhsa_next_free_vgpr` in the assembly and `vgpr_count` in the code-object metadata. Use those,
  not a guessed budget. SGPRs are separate and hold uniform values, descriptors and lane selects.
- The scalar unit executes one instruction per wave per cycle and is independent of the vector
  path. Address arithmetic that is uniform across a wave belongs in SGPRs; the compiler usually
  does this automatically.

## Encoding formats

| format | width | typical use |
| --- | ---: | --- |
| `VOP1` | 32 | one vector source: `v_mov`, `v_cvt`, `v_exp`, `v_rcp`, `v_sqrt` |
| `VOP2` | 32 | two vector sources: `v_add_f32`, `v_mul_f32`, `v_and_b32`, `v_lshlrev_b32` |
| `VOP3` | 64 | three sources, modifiers, or an SGPR destination (compares, `v_readlane_b32`) |
| `VOP3P` | 64 | packed fp16 and three-operand dots: `v_pk_fma_f16`, `v_dot2_f32_f16` |
| `VOPC` | 32 | vector compare producing VCC |
| `SOP1` / `SOP2` | 32 | scalar ALU and moves |
| `SOPC` | 32 | scalar compare |
| `SOPK` | 32 | scalar with a 16-bit immediate |
| `SOPP` | 32 | control: branch, waitcnt, barrier, endpgm |
| `SMEM` | 64 | scalar memory: `s_load_dword`, `s_buffer_load_dword` |
| `MUBUF` / `MTBUF` | 64 | buffer load/store through a resource descriptor |
| `FLAT` / `GLOBAL` / `SCRATCH` | 64 | flat, global and private memory; HIP global pointers compile to `global_*` |
| `DS` | 64 | LDS: `ds_read_b32/b64/b128`, `ds_write`, `ds_bpermute`, atomics |
| `EXP` | 64 | export to the graphics pipeline (not used in compute here) |
| `MIMG` | 64 | image sample/load (not used in compute here) |

`v_readlane_b32` and `v_writelane_b32` are VOP3 and their destination is an **SGPR** on RDNA2.
Writing `v_readlane_b32 v0, v1, s2` fails; use `"=s"` in inline asm.

## Instructions used by compute kernels

### Vector ALU

| instruction | operation |
| --- | --- |
| `v_mov_b32` | move, including inline constants and SGPR to VGPR |
| `v_add_f32`, `v_sub_f32`, `v_mul_f32` | fp32 arithmetic |
| `v_fma_f32`, `v_mac_f32` | fused multiply-add (`v_mac` accumulates into the destination) |
| `v_min_f32`, `v_max_f32` | fp32 min/max |
| `v_cndmask_b32` | select between two VGPRs using VCC |
| `v_cmp_*` | compare into VCC (`v_cmp_gt_f32`, `v_cmp_eq_u32`, ...) |
| `v_rcp_f32`, `v_rsq_f32`, `v_sqrt_f32` | reciprocal, reciprocal square root, square root (MUFU) |
| `v_exp_f32`, `v_log_f32` | exponential and logarithm (MUFU) |
| `v_sin_f32`, `v_cos_f32` | sine and cosine (MUFU) |
| `v_cvt_f32_i32`, `v_cvt_i32_f32` | integer/fp32 conversion |
| `v_cvt_f16_f32`, `v_cvt_f32_f16` | fp16 conversion |
| `v_readlane_b32` | broadcast one lane to an SGPR (lane select in an SGPR) |
| `v_writelane_b32` | write an SGPR into one lane |
| `v_readfirstlane_b32` | broadcast the first active lane to an SGPR |
| `v_and_b32`, `v_or_b32`, `v_xor_b32`, `v_not_b32` | bitwise |
| `v_lshlrev_b32`, `v_lshrrev_b32`, `v_ashrrev_i32` | shifts (source order is reversed: `v_lshlrev_b32 dst, shift, value`) |
| `v_add_u32`, `v_sub_u32`, `v_mul_lo_u32` | integer arithmetic |
| `v_mad_u32_u24`, `v_mad_i32_i24` | integer multiply-add with 24-bit inputs |
| `v_perm_b32`, `v_alignbit_b32`, `v_alignbyte_b32` | byte/bit permute and funnel shifts |
| `v_bfe_u32`, `v_bfi_b32` | bit-field extract and insert |
| `v_cvt_pkrtz_f16_f32` | pack two fp32 values into one half2 with round-to-zero |

### Packed fp16 and dot products

RDNA2 has packed fp16 math and two dot-product instructions. It has **no MFMA and no WMMA**;
those are CDNA or RDNA3+.

| instruction | operation |
| --- | --- |
| `v_pk_add_f16`, `v_pk_mul_f16`, `v_pk_fma_f16` | two fp16 operations per 32-bit register |
| `v_pk_add_f32` | packed fp32 add (two fp32 values in a pair of registers) |
| `v_dot2_f32_f16` | VOP3P: `dst = src0.x*src1.x + src0.y*src1.y + src2` in fp32; two MACs per instruction |
| `v_dot2c_f32_f16` | VOP2 form of the same dot; the destination is also the accumulator |
| `v_dot2_i32_i16` | two signed 16-bit products into an integer accumulator |
| `v_dot4c_i32_i8` | VOP2: four signed int8 products into an int32 accumulator; `c` means the destination is also the accumulator |
| `v_dot8_i32_i4` | eight signed int4 products into an int32 accumulator |

`v_dot2_f32_f16` is the reason the fp16 GEMM reaches 9.5-10 TFLOP/s while the fp32 path stays at
5.25. The fp16 dot2 peak is `36 CU * 64 lanes * 4 flop * clock`, about 24 TFLOP/s at 2.6 GHz.
`v_dot4c_i32_i8` is emitted automatically for the Q8_0 dot product.

### Scalar ALU and control

| instruction | operation |
| --- | --- |
| `s_mov_b32`, `s_mov_b64` | move and 64-bit address materialization |
| `s_add_u32`, `s_sub_u32`, `s_mul_i32`, `s_mul_hi_u32` | scalar arithmetic |
| `s_lshl_b32`, `s_lshr_b32`, `s_ashr_i32` | scalar shifts |
| `s_and_b32`, `s_or_b32`, `s_xor_b32`, `s_andn2_b32` | scalar bitwise |
| `s_cmp_*` | scalar compare into SCC |
| `s_cbranch_scc0/1`, `s_cbranch_vccz/vccnz`, `s_cbranch_execz/execnz` | branches |
| `s_setpc_b64`, `s_swappc_b64` | indirect branch and call/return |
| `s_getpc_b64` | address of the next instruction |
| `s_barrier` | work-group barrier |
| `s_waitcnt` | wait for memory/exp counts (see hazards) |
| `s_sleep` | de-schedule a wave for a number of cycles |
| `s_nop` | alignment and hazard padding |
| `s_endpgm` | end the wave |

### Memory

| instruction | operation |
| --- | --- |
| `s_load_dword`, `s_load_dwordx2/x4` | scalar load through the constant/global path |
| `s_buffer_load_dword`, `s_buffer_load_dwordx4` | scalar buffer load |
| `buffer_load_dword`, `buffer_load_dwordx2/x4` | vector buffer load |
| `buffer_store_dword`, `buffer_store_dwordx2/x4` | vector buffer store |
| `global_load_dword`, `global_store_dword` | global memory with a 64-bit address |
| `flat_load_dword`, `flat_store_dword` | generic address (can hit LDS or global) |
| `scratch_load_dword`, `scratch_store_dword` | private memory |
| `ds_read_b32`, `ds_read_b64`, `ds_read_b128` | LDS load of 1, 2 or 4 dwords |
| `ds_read2_b32`, `ds_read2st64_b32` | two LDS reads with one base and two offsets |
| `ds_write_b32`, `ds_write_b64`, `ds_write_b128` | LDS stores |
| `ds_add_u32`, `ds_min_u32`, `ds_max_u32` | LDS atomics |
| `ds_bpermute_b32`, `ds_permute_b32` | arbitrary lane permutation through LDS |
| `ds_swizzle_b32` | fixed lane swizzle (butterfly, quad, etc.) |
| `ds_gws_*` | work-group synchronization primitives |

Cache modifiers on loads/stores:

- `glc`, `slc` and `dlc` are the RDNA2 cache-policy bits. Their exact per-level meaning differs
  between loads, stores and atomics; see the RDNA2 ISA guide before relying on one.
- `sc0`, `sc1` and `nt` are CDNA3 (gfx940+) bits and `da` is a GCN image bit; none exist on gfx1031.
- Prefer the compiler's choices unless a measured cache-policy experiment justifies changing them.

## Hazards and ordering

- **`s_waitcnt`** is the scoreboard wait. `vmcnt(N)` waits until at most N VMEM operations are
  outstanding; `lgkmcnt(N)` waits for LDS/constant/scalar memory; `expcnt(N)` waits for exports.
  The compiler inserts these automatically. Hand-written asm must add them before using a loaded
  value.
- `s_barrier` synchronizes all waves in a work-group and is required around shared-memory
  producer/consumer phases. `__syncthreads()` lowers to it.
- LDS has no cache; a load is visible to other waves only after the store completes. A barrier
  orders the phases.
- VALU results have a read-after-write latency (measured ~6 cycles for `v_fma_f32`,
  `v_dot2_f32_f16` and `v_dot4c_i32_i8` on this card). Interleave at least four independent
  accumulator chains to keep the pipeline busy.
- MUFU results (`v_exp_f32`, `v_rcp_f32`, `v_sqrt_f32`) have a longer latency; measured ~11 cycles
  for `v_exp_f32`. Do not put a MUFU result on the critical path of every iteration if it can be
  hoisted or overlapped.
- One wave issues independent VALU work at ~1.75 cycles against an ideal of one. Run at least two
  waves per SIMD32 to fill issue slots; more waves also hide memory and MUFU latency.
- Inline asm is opaque to the scheduler. Put independent instructions in separate asm blocks, or
  write the whole unrolled sequence in one block and schedule it yourself.
- `asm volatile` keeps its order against other volatile asm and side effects, which in practice
  stops LLVM from hoisting LDS reads past it. A pure-math wrapper such as `dot2` should be plain
  `asm` (no `volatile`) or a builtin. On the fp16 GEMM, dropping `volatile` from `dot2` made the
  128x64 tiles 4-5% faster.
- `ds_read_b64` needs an 8-byte-aligned address and `ds_read_b128` a 16-byte-aligned one. A
  misaligned `ds_read_b128`, produced when the compiler merged reads over a 68-byte row pitch, made
  the fp16 GEMM 4.5x slower. Pad shared rows to a multiple of 16 bytes and declare `__align__(16)`.
- Inline asm that touches a register implicitly must declare it. `s_add_u32`, `s_subb_u32` and every
  VALU compare write flags the compiler does not see. At minimum `s_add_u32`/`s_sub_u32` need an
  `"scc"` clobber, otherwise the scheduler will move a loop compare across them and the loop will
  execute once. This was observed on this project: 32 `s_add_u32` in a loop body broke the loop
  condition until the clobber was added.
- A VALU instruction with a memory operand (none on RDNA2) or an inline asm that reads or writes
  memory needs a `"memory"` clobber; DS instructions in asm need `"memory"` if the compiler also
  accesses the same shared array.

## Measured costs on gfx1031

`tools/IsaBench` runs one wave on one CU, measures `s_memtime` cycles, and reports cycles per
instruction. Latency rows are a dependent chain of 16; throughput rows issue 32 independent
operations per iteration. The loop overhead is included and is visible in the `v_mov_b32` row.

### Vector and scalar ALU

| instruction | latency | issue interval | note |
| --- | ---: | ---: | --- |
| `v_mov_b32` | - | 1.75 | issue-rate floor |
| `v_fma_f32` | 6.13 | 1.78 | VOP3 only |
| `v_fmac_f32` | 6.06 | 1.76 | VOP2 accumulator form |
| `v_add_f32` / `v_mul_f32` | 6.13 | 1.75 | |
| `v_dot2_f32_f16` | 6.28 | 1.76 | two MACs per instruction |
| `v_dot4c_i32_i8` | 6.13 | 1.72 | four int8 MACs per instruction |
| `v_pk_fma_f16` | 6.13 | 1.78 | two fp16 MACs per instruction |
| `v_add_nc_u32` | 6.13 | 1.72 | |
| `v_lshlrev_b32` | 6.13 | 1.72 | |
| `v_mad_u32_u24` | 6.13 | 1.72 | |
| `v_bfe_u32` | 6.13 | 1.78 | |
| `v_mul_lo_u32` | 8.94 | 4.59 | half rate, avoid in hot loops |
| `v_cmp_gt_f32` + `v_cndmask_b32` | 4.66 | 2.05 | per instruction of the pair |
| `v_exp_f32`, `v_log_f32`, `v_rcp_f32`, `v_sqrt_f32`, `v_rsq_f32` | 10.81 | 4.63 | MUFU, half rate |
| `s_add_u32` | 3.88 | 1.97 | scalar unit, half rate |

The issue interval is flat at ~1.75 cycles for every VALU encoding, VOP1 through VOP3P. There is no
throughput advantage to a 32-bit encoding over a 64-bit one; pick instructions by work done, not by
encoding width. A SIMD32 can issue one wave32 VALU instruction per cycle, so a single wave leaves
roughly 40% of its SIMD idle.

Four waves showed the same per-wave interval as one (1.78 / 1.79 / 1.75). A 128-thread block
spreads over the WGP's four SIMD32s, so this does not show whether two waves on one SIMD fill the
gap. Measure that with a block of at least eight waves before drawing an occupancy conclusion.

### LDS

| measurement | cycles | note |
| --- | ---: | --- |
| `ds_read_b32` dependent chain | 34.0 | pure address chase, no conversion |
| `ds_read_b64` dependent chain | 34.0 | same latency as b32 |
| `ds_read_b128` dependent chain | 44.0 | +10 cycles for 4x the data |
| `ds_read_b32` issue | 3.50 | no waitcnt |
| `ds_read_b64` issue | 3.50 | same as b32 |
| `ds_read_b128` issue | 6.50 | 2x issue for 4x data, the efficient read |
| `ds_read2_b32` | 3.00 | per dword, 6.0 per instruction |
| `ds_write_b32` issue | 3.50 | |
| `ds_bpermute_b32` | 3.50 | same cost as a load |
| `ds_read_b32`, stride 1..32 floats | 7.63 | with waitcnt; no bank penalty |
| `ds_write_b32`, stride 1 | 8.75 | with waitcnt |
| `ds_write_b32`, stride 32 | 38.9 | 4.4x, 32-way bank conflict |

LDS **reads are conflict-free** on this card for every stride tested, including all 32 lanes hitting
bank 0 with different addresses. The read crossbar handles an arbitrary permutation of 32 lanes.
LDS **writes are banked**: the same stride-32 pattern costs 4.4x. Do not waste time padding a
read-only tile; do avoid same-bank write patterns.

This was measured the hard way on the fp16 GEMM. Widening the shared reads to `ds_read2_b64`
required a 16-byte-aligned tile pitch (24 halves instead of 18), which changed the transposed B
stores from 4-way to 32-way bank conflicts. The kernel fell from 8.52 to 4.99 TFLOP/s even though
the read path got shorter. Check the store pattern before changing a shared pitch for a wider read.

The 34-cycle chain is the real load-to-use number including the loop and the scoreboard wait. A
`ds_read_b128` costs 10 more cycles of latency but moves 4x the data at 1.9x the issue cost, so tile
loads should use 128-bit reads where the layout allows it.

### Memory hierarchy (pointer chase, one 128-byte line per step)

| level | cycles | note |
| --- | ---: | --- |
| L1, 4 KB working set | 116 | L0/L1 hit |
| L2, 1 MB working set | 283 | |
| 8 MB working set | 470 | no separate infinity-cache step |
| 16 MB working set | 462 | |
| 32 MB working set | 560 | |
| DRAM, 256 MB working set | 470 | |

The knee is between 1 MB and 8 MB. The RX 6700 has 80 MB of infinity cache, but this chase shows
no step for it and is not monotonic (32 MB costs more than 256 MB). A fixed 128-byte stride is
open to prefetch and TLB effects, so this table bounds L1 and L2 only. Rerun with a randomized
chase before using the numbers above L2.

### fp8 on gfx1031

There is no fp8 instruction in the RDNA2 opcode list and no fp8 ALU. `__hip_fp8_e4m3`
compiles to software emulation with NaN and denormal branches (dozens of scalar and
vector instructions per element). The fast path is a six-integer-op decode per two
values:

```
h0 = ((b & 0x80) << 8) | (((b & 0x7f) << 7) + 0x2000)   // E4M3 normal -> fp16
```

Measured, one wave, dependent chains (`tools/IsaBench`):

| sequence | cycles |
| --- | ---: |
| `v_dot2_f32_f16`, native fp16 | 1.76 |
| E4M3 decode + `v_dot2_f32_f16` (2 MACs) | 12.28 |
| E4M3 -> fp16 decode, 2 values | 7.59 |

So fp8 is a storage format here, not a compute format: a weight-only fp8 GEMM pays
roughly three instructions per element to decode, which is about 4x the issue cost of
the fp16 dot2 it feeds. It only pays when the weights stream from DRAM and the kernel
is bandwidth-bound; the FFN weights (2 MB each, Infinity-Cache resident) are not.

### Issue mixing

| sequence | cycles | note |
| --- | ---: | --- |
| 8 `ds_read_b32` + wait | 61 | 7.63 per load with the wait |
| 8 `ds_read_b32` + wait + 16 `v_fma_f32` | 128 | load, wait, compute is additive |
| 32 `v_fma_f32` | 57 | 1.78 per FMA |
| 32 `v_fma_f32` + 32 `s_add_u32` | 230 | scalar ops do not hide vector issue |

A load-wait-compute loop pays the full LDS latency every tile. Software pipelining (issue the next
tile's loads before the current tile's arithmetic) is worth more than any instruction-selection
change. The scalar unit shares issue bandwidth with the vector path closely enough that SALU
address arithmetic is not free; hoist it out of the loop.

`tools/IsaBench` calibrates `s_memtime` against wall time; runs report 2.64-2.65 GHz, so the table is
in core clocks. The boost clock moves with power state. Compare instruction sequences within one
run, not across runs.

## gfx1031 specifics

- No MFMA, no WMMA, no fp8. Packed fp16 (`v_pk_*`) and `v_dot2_f32_f16` are the matrix-ish path.
- `v_dot4c_i32_i8` is available and is what the compiler uses for int8 dot products.
- No fp16 atomic add in global or shared memory. Accumulate fp16 gradients in fp32.
- 128 KB LDS per WGP, 64 KB per CU; a work-group cannot exceed the per-CU limit.
- The runtime reports 18 WGPs, not 36 CUs. Peak fp32 is `36 CU * 64 SP * 2 * clock`; do not use
  `multiProcessorCount * 64 * 2 * clock`.
- RGP cannot trace the pip HIP runtime on Windows. Use the OpenCL hsaco path in the
  `profile-kernel` skill for instruction-level captures.

## Reading the disassembly

`scripts/Disassemble.ps1` compiles one translation unit to gfx1031 assembly. The instruction mix
script counts VALU, SALU, VMEM, LDS and branch instructions per kernel. For a kernel:

1. Find the hot loop in the `.s` file (the backward branch target).
2. Count the instructions inside the loop, not the whole function.
3. Check `s_waitcnt` placement: a wait inside the loop that is not required is a stall.
4. Check `ds_read`/`ds_read2`/`ds_read_b128`: scalar reads where a 128-bit read is possible are a
   common waste, but vectorizing can raise VGPR pressure and lose more than it gains.
5. Check `.amdhsa_next_free_vgpr` and `scratch_*`: spills mean the schedule is already lost.
6. Compare two variants with the benchmark harness and, if needed, RGP.

## Further reading

- `opcodes.md` in this directory: all 1153 RDNA2 instructions grouped by functional unit, with
  encodings, opcodes, operands and descriptions.
- AMD RDNA2 ISA Reference Guide (PDF) for bit-level encoding tables.
- AMD RDNA Performance Guide for LDS banking, occupancy and cache guidance.
- `profile-kernel` skill for RGP captures and instruction timing.

