---
name: write-isa
description: Write and tune gfx1031 (RDNA2) kernel code at the instruction level. Use when a kernel is issue-, LDS-, or latency-bound and C++ or HIP intrinsics cannot express the instruction you need, or when comparing hand-written VALU/VMEM/LDS sequences against the compiler's schedule. Covers disassembly, the gfx1031 execution model, inline GCN assembly, hazard rules, measured instruction costs and the verification loop.
---

# Write ISA

The compiler is good at the common cases. Reach for inline assembly only when the disassembly shows
a specific instruction the compiler cannot select, or when a measured experiment says a different
sequence is faster. Every hand-written path keeps the scalar C++ reference and is verified by the
same tests and benchmark as the compiler path.

Full references:

- `references/rdna2-isa.md` - execution model, encodings, instruction families, hazards, measured
  costs, gfx1031 specifics.
- `references/opcodes.md` - all 1153 RDNA2 instructions from AMD's machine-readable spec, grouped by
  functional unit, with encodings, opcodes and operands.

## Workflow

### 1. Establish the baseline

Run the kernel's test and benchmark case before touching the ISA.

```powershell
ctest --preset local-release
build/local-release/ROCmKernels.Benchmark.exe --filter "<kernel>" --samples 10 --warmup 5 --min-sample-ms 50
```

A change is only interesting if it moves the median and the test still passes.

### 2. Disassemble and find the hot loop

```powershell
pwsh scripts/Disassemble.ps1 -Source ROCmKernels.Kernels/src/Generic/Linear/LinearActFast.hip -Output build/isa/LinearActFast.s
```

Or use the project target that writes every kernel:

```powershell
cmake --build --preset local-release --target ROCmKernels.Assembly
```

Find the function symbol, then the backward branch that closes the main loop. Count only the loop
body. Check:

- VALU:SALU ratio; excessive SALU is address arithmetic that could be hoisted or scalarized.
- `ds_read2_b32` / `ds_read_b128`; scalar LDS reads where a vector read fits.
- `s_waitcnt` inside the loop; an unnecessary wait is a stall.
- `.amdhsa_next_free_vgpr` and `scratch_*`; spills invalidate the schedule.
- MUFU instructions (`v_exp_f32`, `v_rcp_f32`) on the critical path.

`cmake --build --preset local-release --target ROCmKernels.InstructionMix` prints VALU, SALU, VMEM,
LDS and branch counts per kernel.

### 3. Decide whether the ISA is the answer

Use C++/HIP first. The compiler already selects `v_dot2_f32_f16` for fp16 dot products and
`v_dot4c_i32_i8` for int8 dots when the data layout allows it. An inline-asm path is justified when:

- the instruction exists but the compiler does not select it, for example a packed dot that needs a
  particular register layout;
- a dependent chain is latency-bound and you can supply independent accumulators the compiler
  cannot infer;
- a shared-memory layout needs a specific `ds_read_b128`/`ds_read2` pattern;
- the compiler inserts a wait or a move that a hand-written sequence removes.

Do not rewrite a kernel that is already bandwidth-bound. Measure first.

### 4. Write the asm

Inline assembly uses the standard GCC-style constraints:

| constraint | operand |
| --- | --- |
| `"v"` | VGPR |
| `"s"` | SGPR |
| `"a"` | SGPR or inline constant |
| `"I"`, `"J"`, `"K"` | immediate ranges |
| `"+v"` | read-write VGPR |
| `"=v"` | write-only VGPR |
| `"=s"` | write-only SGPR |

`asm volatile` prevents deletion and reordering. A few rules learned on gfx1031:

- `v_readlane_b32` writes an **SGPR**, so use `"=s"` for the destination. `"=v"` assembles in the
  `.s` file but fails the final device link.
- The lane-select operand of `v_readlane_b32` must be an SGPR, not a literal. Make it come from a
  runtime value, not a compile-time constant.
- `v_dot2_f32_f16` is `dst = src0.x*src1.x + src0.y*src1.y + dst`. Use `"+v"` for the accumulator.
- `v_dot4c_i32_i8` has three operands and accumulates into the destination: `"+v"`.
- Do not split a dependent chain across several `asm` statements and expect the compiler to
  schedule it. Put the whole unrolled block in one `asm`, or use independent `"+v"` accumulators.
- The compiler does not insert `s_waitcnt` for values produced inside an asm block. Add it yourself
  when the next instruction consumes a memory result.

Two working examples from this repository:

```cpp
// LinearActFast fp16: two MACs per instruction, fp32 accumulate.
__device__ __forceinline__ float dot2(const __half2 a, const __half2 b, const float accumulator) {
    const unsigned aBits = *reinterpret_cast<const unsigned *>(&a);
    const unsigned bBits = *reinterpret_cast<const unsigned *>(&b);
    float result = accumulator;
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(result) : "v"(aBits), "v"(bBits));
    return result;
}
```

```cpp
// Q8_0: four int8 products per instruction, int32 accumulate.
asm volatile("v_dot4c_i32_i8 %0, %1, %2" : "+v"(accumulator) : "v"(aBits), "v"(bBits));
```

### 5. Verify and measure

```powershell
cmake --build --preset local-release
ctest --preset local-release
build/local-release/ROCmKernels.Benchmark.exe --filter "<kernel>" --samples 10 --warmup 5 --min-sample-ms 50
```

Then capture the kernel with the `profile-kernel` skill if the change is large or the result is
surprising. Instruction tracing shows whether the hand-written sequence actually removed the stall.

### 6. Keep the fallback

Leave the C++/compiler path in place behind a variant or a separate entry point until the new path
has passed the full test suite and the benchmark on more than one shape. The annotation kernel in
`AttentionFlash32.h` keeps both: `attention_fp32_forward_flash_tuned` selects the variant, and the
default entry point is the measured winner.

## Microbenchmark

`tools/IsaBench` measures latency and throughput of individual instructions with `s_memtime` on one
wave.

```powershell
cmake --build --preset local-release --target ROCmKernels.IsaBench
build/local-release/ROCmKernels.IsaBench.exe
```

Current results are in `references/rdna2-isa.md`. The harness is deliberately small; add a kernel
for each instruction you need. Keep one wave for latency and use independent chains for
throughput. The absolute numbers are less important than the comparison between two sequences.

## Pitfalls

- **The runtime reports 18 compute units, but the card has 36 CUs.** `multiProcessorCount` is the
  WGP count. Peak fp32 uses 36 CUs.
- **Wave32 issue is two cycles per instruction per SIMD16.** One wave cannot saturate a CU; use at
  least two waves per SIMD.
- **LDS bank conflicts.** Pad a row-major array by one element, or make the stride odd. A 33-float
  pitch is the standard trick; a 65-float pitch works for 64-wide tiles.
- **Vectorizing shared reads is not always faster.** A K-contiguous `float4` layout on the fp32
  GEMM raised VGPRs to 117 and regressed from 5.25 to 3.2 TFLOP/s because the compiler still emitted
  `ds_read2` and occupancy dropped. Measure.
- **Instruction-level fusion has a register cost.** The same lesson as the attention kernel: the
  fp16 dot2 path won because it lowered both the instruction count and the shared-memory footprint.
- **Do not trust sub-microsecond benchmark cases.** The harness divides a batched event window by
  the iteration count; a kernel that runs in 0.1 us is dominated by launch gaps. Use a larger shape
  before concluding anything.

## Files

- `scripts/Disassemble.ps1` compiles one translation unit to gfx1031 assembly.
- `scripts/InstructionMix.ps1` counts instruction classes per kernel.
- `tools/IsaBench/IsaBench.hip` measures instruction latency and throughput.
- `references/rdna2-isa.md` is the architecture and measured-cost reference.
- `references/opcodes.md` is the complete RDNA2 opcode list.
- `.opencode/skills/profile-kernel/SKILL.md` captures real kernels with RGP.
