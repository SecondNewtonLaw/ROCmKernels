# ROCmKernels

[![License](https://img.shields.io/github/license/SecondNewtonLaw/ROCmKernels)](LICENSE)

### Background Knowledge

The RX 6700 is Navi 22, gfx1031, and ROCm does not officially support it on Windows. Getting it to work means a pip ROCm build with the gfx1031 device libraries and a matching PyTorch build. That setup runs, but every kernel experiment ends up tangled in Python packaging, JIT wrappers and torch's extension build.

This repository exists to get that out of the way. Kernels are written in plain HIP, compiled by the Clang that ships with ROCm, and checked by a C++ test binary. gfx1031 is the only target, and the build says so explicitly.

### State of Development

The kernels from the embedding training project are imported: SwiGLU, residual add plus LayerNorm, rotary, cross entropy, embedding-gradient conversion, Muon and two attention widths. Each family has a Catch2 reference test and benchmark cases.

The test binary is a normal executable. The build copies the HIP runtime DLLs from the selected ROCm next to it, so it starts without a prepared `PATH`, even when a different HIP SDK is installed.

### Why there is no CMake HIP language

CMake's HIP language support wants the `hip-lang` CMake package. The ROCm builds that drive this card do not ship it, and without it CMake silently borrows an unrelated HIP SDK for includes and libraries.

Instead, `.hip` files are marked as C++ and compiled with `clang++ -x hip --offload-arch=gfx1031`, pointing at the device libraries of the selected ROCm. Host code and kernels use one compiler, one runtime and one ABI.

### Kernels

| Family | Source | Contents |
| --- | --- | --- |
| Elementwise | `Elementwise/Vector.hip`, `Copy.hip` | `VectorAdd` and `Copy` for float, plus double `VectorAdd` |
| Activation | `Activation/SwiGLU.hip` | Fused SiLU gate, forward and backward |
| Norm | `Norm/AddNorm.hip` | Residual add and 384-wide LayerNorm, with fused backward |
| Positional | `Positional/Rotary.hip` | 32-wide rotary embedding, forward and backward |
| Loss | `Loss/CrossEntropy.hip` | Fused softmax cross entropy, statistics and gradient |
| Embedding | `Embedding/EmbeddingGrad.hip` | Sorted sparse row list to dense gradient matrix, scalar and float4 |
| Optimizer | `Optimizer/Muon.hip` | Momentum, Gram matrix, Newton-Schulz multiply and parameter update |
| Attention | `Attention/Attention.hip`, `Attention64.hip` | 32- and 64-wide FP32 attention, simple and tiled, with masks and local radius |

The C entry points keep the names the training code already loads, and each header adds a namespaced wrapper that throws on a HIP error. Kernels compile with `-ffp-contract=off`, matching how they were validated.

### Adding a kernel

1. Put the kernel and a launcher in `ROCmKernels.Kernels/src`.
2. Declare the launcher in `ROCmKernels.Kernels/include/ROCmKernels/Kernels`.
3. Add the source to `ROCMKERNELS_KERNEL_SOURCES` in `CMakeLists.txt`.
4. Add a test in `ROCmKernels.Tests`.
5. Add a benchmark case in `ROCmKernels.Benchmark/src`.

### Benchmarking

`ROCmKernels.Benchmark` times every registered kernel with HIP events. Each case warms up until the clocks settle, then measures batches until the sample is long enough, and reports median, minimum and p95 time per launch. Bandwidth and compute throughput come from the bytes and operations the case declares.

```powershell
./build/local-release/ROCmKernels.Benchmark.exe
./build/local-release/ROCmKernels.Benchmark.exe --filter VectorAdd --csv results.csv
```

A 256 MiB memset runs between samples to evict the caches; `--no-flush` keeps them warm. `--list` shows the registered cases, `--help` the remaining flags.

The header prints streaming copy ceilings at three working sets: 32 MiB and 96 MiB, which sit inside or at the edge of the Infinity Cache, and 512 MiB, which is HBM. The 10 GB RX 6700 has 80 MB of Infinity Cache (the XT has 96 MB), so the middle number is the one to watch here. Kernels whose footprint fits the cache can legitimately exceed the 512 MiB ceiling.

Each row reports the unique footprint and the issued traffic separately. `Unique MiB` is the data a launch touches once. `Traffic MiB` counts every load and store, including re-reads served by cache, and `GB/s` divides that traffic by the median time. A kernel with a small footprint and large traffic is not HBM-bound; it is spending time on loads and instructions.

Attention makes the distinction visible: the simple 32-wide kernel re-reads K/V for each four-query block and issues 1542 MiB for a 12 MiB footprint, while the tiled kernel re-reads per sixteen queries and issues 390 MiB. Both fit the cache. The tiled kernel wins because it issues fewer loads and instructions, not because it uses less DRAM bandwidth. It also declares four FLOPs per key, query and feature forward and eight backward, which shows the compute side: roughly 2.3 TFLOP/s tiled against 0.8 simple.

The flash kernel in `src/Attention/AttentionFlash32.h` is the measured result of the profiling loop. It keeps the transposed lane-per-key mapping, hoists the tensor base pointers out of the key loop, uses 32-key shared tiles instead of the imported kernel's reference-order bookkeeping, and replaces `expf` with `__expf` in the online softmax. On the same 12 MiB workload it runs at **0.96 ms / 3.36 TFLOP/s** against **1.36 ms / 2.37 TFLOP/s** for the tiled entry, a 29% improvement. RGP measures the compute submit at 115 µs against 158 µs, the same direction and magnitude. The sweep that produced it is exposed as `attention_fp32_forward_flash_tuned`, variants 0 through 18; `AttentionForwardFlash` is variant 13, sixteen warps by four queries with 32-key tiles. Vectorising the weight broadcast was the main regression: it raised the kernel to 127 VGPRs and spilled to scratch.

Cases live next to the harness in `ROCmKernels.Benchmark/src`. Register a case, set `bytesMoved`, `bytesTouched` and `operations`, and the columns follow.

### Generic kernels

`ROCmKernels.Kernels/src/Generic` holds dtype-generic kernels. Storage is selected at runtime with `ROCmKernels::Kernels::DType` (`Float32` or `Float16`); every kernel accumulates in fp32 and writes the selected type. `DType.hpp` provides the scalar and pair loads/stores. New files in `src/Generic`, `Tests/Generic` and `Benchmark/src/Generic` are globbed by CMake, so a kernel package is four files: header, `.hip`, test, benchmark.

The first group is `LayerNormScaleAdd` (fused residual, LayerScale and LayerNorm, forward and backward), `AdamW` (one launch for a descriptor array of tensors), `LinearAct` (16x16 tiled GEMM with bias and SiLU fused into the epilogue, plus standalone SiLU), and `AttentionGeneric` (head widths 16, 32 and 64 with a windowed online softmax).

The second group is ported from llama.cpp `ggml-cuda` (MIT, see `THIRD_PARTY_NOTICES.md`): `RMSNorm` (residual add and scale fused into one pass), `Rope` (Normal and NeoX, in-place safe), `TopK` (k up to 8 over 55k classes), `LinearActVec` (batch-1 GEMV with bias and SiLU) and `Softmax` (optional mask, scale, fully masked rows).

| kernel | case | result |
| --- | --- | --- |
| `LayerNormScaleAdd` | fp32 fwd 4096x1024 | 658 GB/s |
| `RMSNorm` | fp32 fwd 4096x1024, residual+scale | 866 GB/s |
| `SiLU` | fp32 / fp16 4M | 688 / 479 GB/s |
| `Softmax` | fp32 r4096 c1024 | 338 GB/s |
| `Rope` | fp32 normal b2 h12 t1024 d128 | 215 GB/s |
| `LinearActVec` | fp32 c4096 k4096 | 557 GB/s, 0.12 ms |
| `AdamW` | fp32 / fp16 8M elements | 191 / 238 GB/s |
| `TopK` | fp32 r512 c55765 k8 | 0.77 ms |
| `AttentionGeneric` | 16/32/64, fp32/fp16 | 1.0 to 1.5 TFLOP/s |
| `LinearAct` | fp32 4096x4096x1024 | 1.4 TFLOP/s |

`LinearAct` is the weak point: the tiled GEMM reloads both tiles from global memory and has no fp16 vector path. The next port is the RDNA2 quantized-matmul configuration from llama.cpp `mmq-config-rdna2.cuh` plus `vecdotq.cuh`, which is where the FFN cost actually goes at inference time.

### Profiling

`ROCmKernels.Assembly` writes gfx1031 assembly for every kernel, and `ROCmKernels.InstructionMix` counts instructions per kernel, split into VALU, SALU, VMEM, LDS, branches and other. That is the static side of attributing time to instructions.

```powershell
cmake --build --preset release --target ROCmKernels.InstructionMix
```

Dynamic per-instruction timing is still possible. RGP cannot trace the pip HIP runtime, but the HIP code object itself is portable: compile the kernel with the HIP front-end, extract the gfx1031 hsaco, load it with `clCreateProgramWithBinary`, and dispatch it through the OpenCL runtime while RGP captures. The instruction stream is the production HIP code; only the launcher is OpenCL.

That path is verified on this card. A HIP kernel dispatched through the OpenCL runtime produced correct results, and RGP captured it with instruction tracing into a `.rgp` that contains the HIP kernel symbol. The attention kernels use a profiling-only shim that takes plain pointers and reconstructs `Input`, because the OpenCL runtime cannot patch `cl_mem` handles inside a by-value struct. `scripts/ExtractHsaco.ps1 -Define ROCMKERNELS_PROFILE_SHIM` builds that variant, and the probe dispatches `attention_flash_profile` or `attention_tiled_profile` at the same shape for an A/B capture. The full workflow, the argument rules and the current limits are in the project skill at `.opencode/skills/profile-kernel/SKILL.md`.

The captures can be inspected without the GUI: `rgp-analyzer inspect <file>.rgp` reports each queue event's GPU duration, the code objects and the SQTT byte count; `resource-summary` reports VGPR and SGPR counts. That is enough to compare two variants from the same capture conditions. If the RGP panel stalls after `Capture API initialized` or reports `Failed to create router`, kill the stale panel process and reset the driver; a failed trace leaves the capture API wedged.

Direct HIP capture still crashes the process, and the Windows `HSA_OVERRIDE_GFX_VERSION=10.3.0` trick does not change that. The official 7.1 runtime still reports no ROCm-capable device, the pip runtime keeps reporting gfx1031, and a gfx1030-compiled binary is rejected with an invalid kernel image.

### Instruction-level work

`scripts/Disassemble.ps1` compiles one translation unit to gfx1031 assembly, and `ROCmKernels.IsaBench` measures individual instructions on the card with `s_memtime`.

```powershell
pwsh scripts/Disassemble.ps1 -Source ROCmKernels.Kernels/src/Generic/Linear/LinearActFast.hip -Output build/isa/LinearActFast.s
cmake --build --preset release --target ROCmKernels.IsaBench
build/release/ROCmKernels.IsaBench.exe
```

The first measured set on this card: `v_fma_f32`, `v_dot2_f32_f16` and `v_dot4c_i32_i8` each have a 6.13-cycle dependent latency and a 1.72-cycle issue interval for one wave; `v_exp_f32` is 10.81 cycles; a dependent `ds_read_b32` chain is 65 cycles including the address update, and LDS throughput is 3.13 cycles per 32-lane load. `s_memtime` calibrates to 2.65 GHz. The full ISA reference, the complete opcode list from AMD's machine-readable RDNA2 spec, the hazard rules and the inline-assembly guide are in the project skill at `.opencode/skills/write-isa/SKILL.md`.

The worked example is the fp16 GEMM: the compiler's fp32 path is LDS-bound at 5.25 TFLOP/s, and a hand-written `v_dot2_f32_f16` path with K-contiguous half tiles reaches 8.53 TFLOP/s. The same instruction on the wrong layout is slower, so every hand-written sequence is benchmarked against the compiler's.

### Disclaimer

Not affiliated with AMD. The RX 6700 is unsupported by ROCm, so driver and runtime behaviour can change between versions. Nothing here is guaranteed to work on another GPU or another ROCm build. MIT licensed, no warranty.

### Credits

Built on AMD's ROCm, Clang and HIP runtime. Tests use [Catch2](https://github.com/catchorg/Catch2).

#### Usage of AI

Fully done using AI.

### License

MIT, see [LICENSE](LICENSE).

### Building

#### Requirements

- CMake 3.31 or newer, Ninja
- ROCm with Clang, the HIP runtime and the gfx1031 device libraries. Both the AMD HIP SDK and the pip `_rocm_sdk_core` bundle work
- A Radeon RX 6700 to run the tests

#### Configure and build

```powershell
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Without presets:

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

#### Device assembly

```powershell
cmake --build --preset release --target ROCmKernels.Assembly
```

writes `build/release/assembly/<kernel>.s` for every `.hip` file, ready for disassembly or a diff after an optimisation change. `ROCmKernels.InstructionMix` runs the per-kernel instruction counts over that output.

#### Selecting ROCm

CMake picks the installation in this order:

1. the `ROCM_PATH` cache variable
2. the `ROCM_PATH` environment variable
3. an active Python environment (`VIRTUAL_ENV`, `CONDA_PREFIX`) holding the pip ROCm wheels
4. the `HIP_PATH` environment variable
5. `hipconfig --path` from `PATH`
6. the default AMD HIP SDK directories

With more than one installation, set it yourself:

```powershell
$env:ROCM_PATH = "C:/Program Files/AMD/ROCm/7.1"
cmake --preset release
```

For the pip wheels, either activate the virtual environment or point at it:

```powershell
$env:ROCM_PATH = "F:/rocm-test/venv/Lib/site-packages/_rocm_sdk_core"
```

`CMakeUserPresets.json` is gitignored, so a local install can live there instead of in the environment:

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "local-release",
      "inherits": "release",
      "cacheVariables": { "ROCM_PATH": "<path/to/_rocm_sdk_core>" }
    }
  ]
}
```

The toolchain resolves before `project()`, so changing `ROCM_PATH` needs a fresh build directory.

#### Runtime libraries

The build copies the HIP runtime DLLs from the selected installation next to each executable, and `ctest` puts that installation's `bin` first on `PATH`. A different HIP SDK on `PATH` therefore cannot be loaded by accident. Turn the copy off with `-DROCMKERNELS_COPY_HIP_RUNTIME=OFF`.

An installation can compile gfx1031 code and still fail to detect the card. The device test reports that as a failure, not a skip.

#### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `ROCMKERNELS_COPY_HIP_RUNTIME` | `ON` | Copy HIP runtime DLLs next to executables |
| `USE_CCACHE` | `ON` | Use ccache as the compiler launcher when available |
| `ALLOW_NON_STANDARD_MATH` | `OFF` | Enable `-ffast-math` for host and device code |
| `DEBUG_INFORMATION_EXTENSIVE` | `ON` | Emit `-g3` debug information |
| `DEBUG_NO_INLINING` | `ON` | Disable inlining in Debug builds |
| `BUILD_TESTING` | `ON` | Build the Catch2 test binary |
| `BUILD_BENCHMARKS` | `ON` | Build the benchmark executable |
| `BUILD_PROFILING_TOOLS` | `OFF` | Build `OpenClProbe` for the profiling skill |
| `BUILD_ISA_BENCH` | `ON` | Build the RDNA2 instruction microbenchmark |
| `ROCMKERNELS_EMIT_ASSEMBLY` | `ON` | Provide the `ROCmKernels.Assembly` target |
