# What torch runs for the training GEMMs

`F.linear` goes `at::linear` -> `addmm`/`mm` -> `at::cuda::blas::gemm` -> `rocblas_gemm_ex`.
TunableOp is off in this build (`torch.cuda.tunable.is_enabled()` is `False`), and the trace shows
`rocblas_gemm_ex` directly, so the reference kernel is rocBLAS/Tensile, not hipBLASLt.

Measured with torch 2.13.0+rocm10.1.0a20260819, rocBLAS 5.7.0.

## The two calls

At 8192 tokens, d=128, ff=8192:

| case | rocBLAS call | strides |
| --- | --- | --- |
| ffn up `F.silu(F.linear(x, w1))` | `gemm_ex(T, N, m=8192, n=8192, k=128)` | A lda=128, B ldb=128, C ldc=8192 |
| ffn down `F.linear(h, w2)` | `gemm_ex(T, N, m=128, n=8192, k=8192)` | A lda=8192, B ldb=8192, C ldc=128 |

Both use `f16_r` inputs and `f32_r` compute, `beta=0`.

## The selected Tensile kernels

`TENSILE_DB=0xffff` prints the name of every dispatched kernel:

- up: `Cijk_Alik_Bljk_HHS_BH_MT128x128x16_SN_..._WG16_8_1_..._TT8_16_..._VW8_..._WGM1`
- down: `Cijk_Alik_Bljk_HHS_BH_MT128x256x16_SN_..._WG16_16_1_..._TT8_16_..._VW8_..._WGM4`

Decoding the fields that matter:

| field | meaning | up | down |
| --- | --- | --- | --- |
| `MT` | macro tile (M x N x depth) | 128x128x16 | 128x256x16 |
| `WG` | work-group dims | 16x8 = 128 threads | 16x16 = 256 threads |
| `TT` | thread tile, outputs per thread | 8x16 = 128 | 8x16 = 128 |
| `VW` | vector width, halves per load | 8 (16 bytes) | 8 (16 bytes) |
| `WGM` | work-group mapping (L2 swizzle) | 1 | 4 |
| `GSU` | global split-U | 1 | 1 |
| `HHS` | half in, half out, fp32 accumulate | | |

The down shape is the interesting one: with `m=128` there is exactly one macro tile in M, so Tensile
launches **32 blocks of 256 threads** and gives each thread 128 outputs. That is the opposite of the
usual "more blocks" instinct, and it is why our 64x64 tile loses there: 256 blocks of 16 outputs
per thread pay the per-block prologue (global loads, two barriers per K tile) eight times more often.

## Our kernel next to it

`LinearActFast` fp16 uses MT64x64x32, 16x16 threads, thread tile 4x4 (16 outputs), 2-half loads.

Same process, 256 MiB cold flush, median of 20, fp16:

| case | torch | LinearActFast (K-major weights) | ratio |
| --- | ---: | ---: | ---: |
| ffn up 8192x8192x128 | 2.434 ms | 2.462 ms | 1.01x |
| ffn down 8192x128x8192 | 1.864 ms | 2.626 ms | 1.41x |

The up shape is at parity. The down shape needs the 128x256 macro tile, 8x16 thread tile and
16-byte vector loads to match rocBLAS.

Weight layout: `F.linear` stores weights as `[out, in]`. `linear_nt` takes that layout directly but
is 1.7x slower than the K-major path (its coalescing is worse at K=128), so the practical wiring is
a one-time transpose at model load and `linear_act_fast` / `linear_fast` at step time.

## Reproducing

```powershell
$env:ROCBLAS_LAYER = "1"          # gemm_ex trace, writes rocblas_trace.log
$env:ROCBLAS_LAYER = "2"          # rocblas-bench command line, writes rocblas_bench.log
$env:TENSILE_DB = "0xffff"        # prints the selected kernel name
```

The Python comparison harness is `tools/TorchCompare/compare_ops.py`; it loads
`ROCmKernels.Kernels.Shared.dll` (build with `-DBUILD_SHARED_KERNELS=ON`) and runs torch and our
kernels back to back in one process, which is the only way to compare across the clock drift.
