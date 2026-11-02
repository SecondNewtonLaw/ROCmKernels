---
name: profile-kernel
description: Profile a gfx1031 HIP kernel by loading its HIP-compiled code object through the OpenCL runtime and capturing instruction-level SQTT data with Radeon GPU Profiler. Use when per-instruction timing, stall attribution or hardware counters are needed and direct HIP capture crashes.
---

# Profile Kernel

Radeon GPU Profiler cannot trace the HIP runtime used by this project: enabling tracing kills the
process with an access violation. It can trace OpenCL on the same card. Both runtimes consume the
same AMDGPU code object, and the OpenCL runtime accepts a HIP-compiled `.hsaco` directly.

So the profiling path is:

1. Compile the kernel with the HIP front-end, exactly as production does.
2. Extract the gfx1031 code object.
3. Load that code object with `clCreateProgramWithBinary`.
4. Dispatch it with OpenCL while RGP captures.
5. Read the instruction timing from the `.rgp`.

No kernel port. The ISA, register allocation and scheduling are the production HIP build. Only the
host launcher changes, and that launcher exists only for profiling.

## Verified state

- A HIP kernel compiled with clang 23 for gfx1031 loads and runs through the OpenCL runtime, with
  correct results.
- RGP captured it with `--rgp-instruction-tracing`: the `.rgp` contains the HIP kernel name, and the
  process survived 2,000,000 dispatches. The capture script also produced a 326 KB trace.
- A real kernel translation unit (`Attention.hip`) loads as an OpenCL binary and lists all of its
  mangled kernel variants, including the tiled ones.
- Pointer-struct kernels such as attention dispatch through a profiling-only shim that takes plain
  pointers. The shim is compiled into the same translation unit with `-Define ROCMKERNELS_PROFILE_SHIM`
  and exposes `attention_flash_profile` and `attention_tiled_profile`. Both produce correct output
  through OpenCL and both captured with instruction tracing: 115 us of compute-submit time for the
  tuned flash kernel against 158 us for the tiled kernel at b2 h12 l1024.
- The explicit handshake is verified. The probe creates its context, writes a ready file, then waits
  for the signal file; the panel sends `c` between the two. This removes the race that made the first
  capture attempt fail with result -2.
- `rgp-analyzer-cli` reads these captures offline: `inspect` prints queue-event GPU durations, code
  objects and SQTT sizes, and `resource-summary` prints VGPR and SGPR counts.
- Repeated crashed HIP traces can wedge the driver's profiling state. If the panel CLI stalls after
  `Capture API initialized successfully` or reports `Failed to create router`, kill stale
  `RadeonDeveloperPanelCLI` processes and reset the driver (`schtasks /run /tn ResetGpu`) before
  capturing again.

## Prerequisites

- `ROCM_PATH` pointing at the working ROCm bundle.
- RGP installed. Set `RGP_CLI` to `RadeonDeveloperPanelCLI.exe`.
- `BUILD_PROFILING_TOOLS=ON` for the probe executable.
- The card must be idle. Close other GPU work, and expect 5 to 10 percent run-to-run variance.

## Workflow

### 1. Extract the code object

```powershell
pwsh scripts/ExtractHsaco.ps1 -Source tools/ProfileKernel/SampleKernel.hip -Output build/profile/add_arrays.hsaco
```

The script compiles the TU with the production flags (`-O3`, `-std=c++23`, `-ffp-contract=off`,
`--offload-arch=gfx1031`), dumps `.hip_fatbin` from the object, and unbundles
`hip-amdgcn-amd-amdhsa--gfx1031`.

For a production kernel, pass its source file and add include paths with `-IncludePath` if needed.
Keep the same flags as the real build or the instruction mix will lie.

### 2. Build and sanity-check the probe

```powershell
cmake --preset local-release -DBUILD_PROFILING_TOOLS=ON
cmake --build --preset local-release --target ROCmKernels.OpenClProbe
```

List the kernels in a code object:

```powershell
build/local-release/ROCmKernels.OpenClProbe.exe --hsaco build/profile/add_arrays.hsaco --list
```

Run it once and check the output:

```powershell
build/local-release/ROCmKernels.OpenClProbe.exe --hsaco build/profile/add_arrays.hsaco --kernel add_arrays
```

`OpenClProbe.cpp` binds four arguments: `(const float*, const float*, float*, int)`. Adapt
`BindArguments` in that file for other signatures.

### 3. Capture

```powershell
scripts/CaptureRgp.ps1 `
  -Application build/local-release/ROCmKernels.OpenClProbe.exe `
  -ArgumentList @("--hsaco", "build/profile/add_arrays.hsaco", "--kernel", "add_arrays", "--iterations", "2000000") `
  -Output build/profile/add_arrays.rgp
```

The script starts the RGP CLI first, waits for it to attach, starts the probe, waits for the capture
file, then stops both. Logs are written next to the `.rgp`.

The panel CLI is attach-only. It has no launch, application or executable option, and a positional
path is ignored. Only the GUI panel launches processes. Start the probe yourself after the CLI is
listening; that is what the script does.

For deterministic timing, use the explicit handshake instead of auto-capture:

```powershell
scripts/CaptureRgp.ps1 `
  -Application build/local-release/ROCmKernels.OpenClProbe.exe `
  -ArgumentList @("--hsaco", "build/profile/add_arrays.hsaco", "--iterations", "3000000") `
  -HandshakeFile build/profile/capture.signal `
  -Output build/profile/add_arrays.rgp
```

The probe creates its OpenCL context first, so the driver has already seen the process, then blocks
on the signal file before the first dispatch. The script sends the panel's `c` capture command and
only then creates the signal file. Capture is armed before any kernel is queued, which removes the
race that auto-capture has when the application starts dispatching immediately.

Useful capture options are in `RadeonDeveloperPanelCLI.exe --help=profiling`:

- `--rgp-auto-capture=dispatch:N` to pick a later dispatch. The probe runs the kernel repeatedly,
  so `dispatch:1` is the first iteration.
- `--rgp-instruction-tracing` for instruction timing and stall data.
- `--rgp-counter-collection` for hardware counters.
- `--rgp-sqtt-buffer-size high` if the capture is truncated.

### 4. Read the capture

Open the `.rgp` in RGP. Select the dispatch in the wave/dispatch list and open Instruction Timing.
Look at:

- Issued versus stalled cycles per instruction.
- Long-scoreboard stalls on global loads (VMEM) and LDS bank conflicts.
- VALU and SALU occupancy, and whether wave issue is limited by dependencies.
- Divergence for masked branches.

Compare variants by capturing each separately with the same shapes. Use the benchmark harness for
median and p95 wall time, and RGP for the instruction-level explanation.

## Adapting to a real kernel

### Kernel names

Kernel symbols are mangled. List them with `--list` and pass the exact symbol to `--kernel`. For
example, the simple 32-wide attention forward kernel is
`_ZL15attention_queryILb0EEv5InputS0_S0_PKbS0_PKfS0_PfS5_iiiiN11ROCmKernels7Kernels7StridesE`.

### Arguments

- Scalars: `clSetKernelArg(kernel, index, sizeof(type), &value)`.
- Plain pointers: pass a `cl_mem` handle with `sizeof(cl_mem)`. The runtime patches the kernel
  argument with the device address. This works for swiglu, rotary, add-norm, cross entropy,
  embedding conversion, muon and vector add.
- Structs containing pointers (attention `Input`): the runtime copies the struct bytes verbatim, so
  the pointer inside must already be a device address. OpenCL `cl_mem` handles do not work here.

### Attention shim

For pointer-struct kernels, compile a profiling-only shim that takes plain pointers and reconstructs
the structs inside the kernel. A `__global__` kernel cannot call another `__global__` kernel, so each
profiled kernel needs a `__device__` body plus a thin `__global__` wrapper:

```cpp
extern "C" __global__ void attention_flash_profile(
        const float *q, const float *k, const float *v, const bool *mask,
        float *output, float *lse, int heads, int queries, int keys, int radius,
        Strides qs, Strides ks, Strides vs, Strides os) {
    attention_query_flash_body<16, 4, 32, true, false>(
            Input{q, qs}, Input{k, ks}, Input{v, vs}, mask, output, lse,
            heads, queries, keys, radius, os);
}
```

Guard it with `#ifdef ROCMKERNELS_PROFILE_SHIM` in the kernel TU, compile a second hsaco with that
define, and dispatch the shim. The wrapper disappears after inlining; the hot loop is unchanged.
State clearly in any results that the shim was used.

```powershell
pwsh scripts/ExtractHsaco.ps1 `
  -Source ROCmKernels.Kernels/src/Attention/Attention.hip `
  -Output build/profile/attention.hsaco `
  -Define ROCMKERNELS_PROFILE_SHIM
```

The probe selects the shim with `--kernel` and takes the launch geometry explicitly. The flash kernel
uses 16 warps and 64 query rows per block; the imported tiled kernel uses 4 warps and 16 rows:

```powershell
build/local-release/ROCmKernels.OpenClProbe.exe --hsaco build/profile/attention.hsaco `
  --kernel attention_flash_profile --warps 16 --block-rows 64 --iterations 4
build/local-release/ROCmKernels.OpenClProbe.exe --hsaco build/profile/attention.hsaco `
  --kernel attention_tiled_profile --warps 4 --block-rows 16 --iterations 4
```

The OpenCL global size must keep the HIP grid shape. A flat `global = blocks * heads * batch * local`
leaves `blockIdx.y` and `blockIdx.z` at zero and silently computes only batch 0, head 0. Enqueue a
three-dimensional range with `global = {blocks * local, heads, batch}` and `local = {warps * 32, 1, 1}`.

If the OpenCL runtime exposes SVM (`clSVMAlloc`), SVM pointers can be placed in the structs instead
and the shim is unnecessary. Check `clinfo` first; this driver currently reports no SVM capability.

## Reading a capture without the GUI

The Radeon Developer Tool Suite installed here (`F:\Tools\RadeonDeveloperToolSuite-2026-10-02-1872`)
ships the panel, the CLI capture tool and RGA, but **no analyzer CLI**. There is no
`rgp-analyzer-cli.exe` in the bundle and it is not on `PATH`.

The third-party analyzer is https://github.com/taowen/rgp-analyzer-cli (GPL-2.0, Python, installed
with `pip install -e .`). It targets Linux/RADV Vulkan captures and its SQTT decoding needs the
vendored `native/rocm_sqtt_decoder_helper`, built on Linux; the README states its limits directly
(no exec-mask data, no fine-grained stall taxonomy, no cacheline metrics). Treat it as a structural
inspector for linux captures, not as a replacement for the RGP GUI on these Windows OpenCL captures.

For the Windows OpenCL path the practical fallback is the static side: `ROCmKernels.Assembly` plus
the InstructionMix script, and A/B wall time from the benchmark harness. The `.rgp` can still be
opened in the RGP GUI when one is available; `build/profile/*.rgp` holds the captured dispatches.

The container starts with the magic `42 30 30 50` ("B00P") and is not a zip archive, so do not
expect `Expand-Archive` to read it.

`rgp-analyzer-cli` reads the container and prints the queue events, code objects and SQTT sizes:

```powershell
rgp-analyzer inspect build/profile/attention_flash.rgp
rgp-analyzer resource-summary build/profile/attention_flash.rgp
```

Put the ROCm LLVM directory on `PATH` first; `stitch-report` and `resource-summary` need
`llvm-readelf`. The SQTT decoder commands (`decode-sqtt`, `shader-focus`) expect rocprof `.att`
metadata and a Linux helper, so they do not run on these Windows OpenCL captures. `inspect` is still
enough for an A/B comparison, because both variants are captured under the same conditions.

### Work sizes

OpenCL global size is the total number of work items, not the grid dimension. A HIP launch of
`blocks` blocks with `threads` threads becomes:

```cpp
size_t local = threads;
size_t global = blocks * threads;
```

A global size that is not divisible by the local size returns `CL_INVALID_WORK_GROUP_SIZE`.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `CL_INVALID_KERNEL_ARGS` (-52) | Wrong argument size, or a struct containing a host pointer |
| `CL_INVALID_WORK_GROUP_SIZE` (-54) | `global` is not a multiple of `local` |
| `clCreateProgramWithBinary` fails | Wrong target, truncated hsaco, or missing metadata |
| No capture file | Probe exits before attach; raise `--iterations`, increase `-DelayMs` |
| RGP never connects | Process name filter mismatch; pass `-ProcessName` |
| `Failed to create router` | A stale panel CLI, or a separately started `RadeonDeveloperServiceCLI`, holds the local router. Do not run the service CLI alongside the panel CLI. Kill both and retry. |
| Stalls after `Capture API initialized` | The driver's profiling state is wedged, usually after a crashed HIP trace. Reboot. |
| HIP capture crashes | Expected; do not capture HIP directly |

## Limitations

- The dispatch runs on an OpenCL queue, not a HIP stream. Kernel ISA is identical, but queue
  scheduling and occupancy around the dispatch can differ slightly.
- OpenCL may flush caches differently between dispatches. Keep the benchmark harness for final
  wall-time numbers.
- Instruction tracing adds overhead. Use it to explain bottlenecks, not to measure absolute time.
- RGP's ETW warning concerns D3D12 signal/wait and does not block OpenCL or instruction tracing.

## Files

- `scripts/ExtractHsaco.ps1` compiles a TU and extracts the gfx1031 code object.
- `scripts/CaptureRgp.ps1` drives the RGP CLI capture.
- `tools/ProfileKernel/OpenClProbe.cpp` loads a hsaco and dispatches it.
- `tools/ProfileKernel/SampleKernel.hip` is the four-argument smoke test.

