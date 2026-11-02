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
- A CU has four SIMD16 units and 64 stream processors. A wave32 instruction occupies a SIMD16 for
  two cycles, so one wave32 issues at most one instruction per two cycles per SIMD. Several waves
  per SIMD are needed to fill issue slots.
- RDNA2 is wave32-native and also supports wave64. Kernels in this repository are wave32.
- LDS is 64 KB per CU (128 KB per WGP). It is 32 banks of 32 bits. Two lanes hitting the same bank
  with different addresses serialize.
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
| `VOP3` | 64 | three sources, modifiers, or a VGPR destination that VOP1/VOP2 cannot encode |
| `VOP3P` | 64 | packed fp16 and dot products: `v_pk_fma_f16`, `v_dot2_f32_f16`, `v_dot4c_i32_i8` |
| `VOPC` | 32 | vector compare producing VCC |
| `SOP1` / `SOP2` | 32 | scalar ALU and moves |
| `SOPC` | 32 | scalar compare |
| `SOPK` | 32 | scalar with a 16-bit immediate |
| `SOPP` | 32 | control: branch, waitcnt, barrier, endpgm |
| `SMEM` | 64 | scalar memory: `s_load_dword`, `s_buffer_load_dword` |
| `MUBUF` / `MTBUF` | 64 | buffer load/store (the usual global path) |
| `FLAT` / `GLOBAL` / `SCRATCH` | 64 | flat, global and private memory |
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
| `v_dot2_f32_f16` | `dst = src0.x*src1.x + src0.y*src1.y + dst` in fp32; two MACs per instruction |
| `v_dot2_i32_i16` | two signed 16-bit products into an integer accumulator |
| `v_dot4c_i32_i8` | four signed int8 products into an int32 accumulator; `c` means the destination is also the accumulator |
| `v_dot8_i32_i4` | eight signed int4 products into an int32 accumulator |

`v_dot2_f32_f16` is the reason the fp16 GEMM reaches 8.5 TFLOP/s while the fp32 path stays at
5.25. `v_dot4c_i32_i8` is emitted automatically for the Q8_0 dot product.

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

- `glc` (global level coherent): bypass L0 for that access.
- `slc` (system level coherent): bypass L1/L2 where applicable.
- `dlc` (device level coherent): force L2 visibility.
- `sc0/sc1`, `nt` and `da` exist on some forms; prefer the compiler's choices unless a measured
  cache-policy experiment justifies changing them.

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
- A wave32 instruction occupies a SIMD16 for two cycles. Two waves per SIMD are the minimum to
  hide the issue interval; more waves hide memory and MUFU latency.
- Inline asm is opaque to the scheduler. Put independent instructions in separate asm blocks, or
  write the whole unrolled sequence in one block and schedule it yourself. `asm volatile` prevents
  the compiler from deleting or reordering it.

## Measured costs on gfx1031

`tools/IsaBench` runs one wave on one CU, measures `s_memtime` cycles, and reports cycles per
instruction. The numbers below are the first stable set; they include the loop and address-update
instructions where noted. Use them for relative decisions, then confirm with the real kernel.

| measurement | cycles/instruction | note |
| --- | ---: | --- |
| `v_fma_f32` latency | 6.13 | dependent chain |
| `v_fma_f32` throughput | 1.72 | 8 independent chains, one wave |
| `v_dot2_f32_f16` latency | 6.13 | dependent chain |
| `v_dot2_f32_f16` throughput | 1.72 | 8 independent chains |
| `v_dot4c_i32_i8` latency | 6.13 | dependent chain |
| `v_dot4c_i32_i8` throughput | 1.72 | 8 independent chains |
| `v_exp_f32` latency | 10.81 | dependent chain, MUFU |
| `ds_read_b32` dependent chain | 65.0 | load plus address update and waitcnt |
| `ds_read_b32` throughput | 3.13 | independent loads, 32 lanes |

External reference points measured on RDNA2 by Chips and Cheese: LDS load-to-use about 19.5 ns
(roughly 42 cycles at 2.17 GHz), scalar cache about 17.4 ns. Those are pure load-to-use numbers;
the 65-cycle chain above also contains `v_cvt_i32_f32`, the address mask and the waitcnt.

`tools/IsaBench` calibrates `s_memtime` against wall time; the first run reported 2.65 GHz, so the
table is in core clocks. The boost clock moves with power state, and the loop overhead is included
in every row. Compare instruction sequences on the same run, not across runs.

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
