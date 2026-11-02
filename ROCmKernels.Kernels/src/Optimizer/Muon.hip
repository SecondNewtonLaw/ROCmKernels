#include <ROCmKernels/Kernels/Optimizer/Muon.hpp>

#include <hip/hip_runtime.h>
#include <cstdint>

// Newton-Schulz runs on the tall orientation y (m x n, m >= n): wide parameters are stored transposed in the workspace.
// The per-matrix scale s = 1 / (||z||_F + 1e-7) is folded into the first iteration instead of a separate pass:
// gram(s z) = s^2 gram(z), and s (a z + z B') equals the step taken on s z.

constexpr int TILE = 128;
constexpr int KSTEP = 8;

static __device__ __forceinline__ float wave_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down(value, offset, 32);
    return value;
}

static __global__ void nonzero_kernel(const int64_t* gradients, int* flags, int64_t elements) {
    const float* g = reinterpret_cast<const float*>(gradients[blockIdx.y]);
    bool any = false;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < elements && !any;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        any = g[i] != 0.0f;
    if (any) flags[blockIdx.y] = 1;
}

// Momentum update, Nesterov combine, sum of squares. The transposed variant stages 32x32 tiles in LDS.
template<bool TRANSPOSE>
static __global__ void prep_kernel(const int64_t* gradients, const int64_t* buffers, float* y, const int* flags, float* sumsq,
                                   int rows, int columns, float momentum) {
    const int matrix = blockIdx.z;
    const float* g = reinterpret_cast<const float*>(gradients[matrix]);
    float* buffer = reinterpret_cast<float*>(buffers[matrix]);
    float* out = y + static_cast<int64_t>(matrix) * rows * columns;
    const bool live = flags[matrix] != 0;
    __shared__ float tile[32][33];
    __shared__ float partial[8];
    float local = 0.0f;
    if constexpr (TRANSPOSE) {
        const int r0 = blockIdx.y * 32, c0 = blockIdx.x * 32;
        for (int k = threadIdx.y; k < 32; k += 8) {
            const int64_t index = static_cast<int64_t>(r0 + k) * columns + c0 + threadIdx.x;
            const float gv = g[index], bv = buffer[index];
            // reactive.MuonNS: buf = buf + (1-m)(g - buf), then g = g + m(buf - g).
            const float next = live ? bv + (1.0f - momentum) * (gv - bv) : bv;
            if (live) buffer[index] = next;
            const float z = gv + momentum * (next - gv);
            tile[k][threadIdx.x] = z;
            local += z * z;
        }
        __syncthreads();
        for (int k = threadIdx.y; k < 32; k += 8)
            out[static_cast<int64_t>(c0 + k) * rows + r0 + threadIdx.x] = tile[threadIdx.x][k];
    } else {
        const int64_t total = static_cast<int64_t>(rows) * columns;
        for (int64_t index = (static_cast<int64_t>(blockIdx.x) * 8 + threadIdx.y) * 32 + threadIdx.x; index < total;
             index += static_cast<int64_t>(gridDim.x) * 256) {
            const float gv = g[index], bv = buffer[index];
            // reactive.MuonNS: buf = buf + (1-m)(g - buf), then g = g + m(buf - g).
            const float next = live ? bv + (1.0f - momentum) * (gv - bv) : bv;
            if (live) buffer[index] = next;
            const float z = gv + momentum * (next - gv);
            out[index] = z;
            local += z * z;
        }
    }
    local = wave_sum(local);
    if (threadIdx.x == 0) partial[threadIdx.y] = local;
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < 8; ++i) total_sum += partial[i];
        atomicAdd(&sumsq[matrix], total_sum);
    }
}

// 128x128 output tile, 256 threads, 8x8 per thread as four 4x4 quadrants, register prefetch of the next k-slab.
// GRAM: C = scale * A^T A + beta * Cin for symmetric C (n x n), A is m x n; only upper tiles are computed and mirrored.
//       diag is added to the diagonal of C (folds a*I into B).
// else: O = s * (Y @ B) with Y m x n, B n x n. ADD applies O to the parameters in the epilogue instead of storing it:
//       p = p * keep - move * O (p row-major rows x columns; transposed when the workspace is p^T).
template<bool GRAM, bool ADD>
static __global__ void gemm_kernel(const float* A, const float* B, float* C, const float* sumsq, int m, int n, float alpha, float beta, int mode,
                                   float diag, const int64_t* params, const int* flags, float decay_factor, float step, int transposed) {
    const int bj = blockIdx.x, bi = blockIdx.y, matrix = blockIdx.z;
    if (GRAM && bj < bi) return;
    const int64_t a_stride = static_cast<int64_t>(m) * n, c_stride = GRAM ? static_cast<int64_t>(n) * n : a_stride;
    A += matrix * a_stride;
    C += matrix * c_stride;
    if constexpr (!GRAM) B += static_cast<int64_t>(matrix) * n * n;
    const float* Cin = B;
    if constexpr (GRAM && ADD) Cin = B + matrix * c_stride;

    // transposed scalar stores of the A tile hit 2-way bank conflicts at stride 128; stride 132 spreads aq=1 onto the other 16 banks
    constexpr int LEFT = GRAM ? TILE : TILE + 4;
    __shared__ __attribute__((aligned(16))) float left[KSTEP][LEFT], right[KSTEP][TILE];
    const int t = threadIdx.x, tx = t % 16, ty = t / 16;
    const int kr = t / 32, c4 = t % 32;
    float acc[8][8] = {};
    const int k_end = GRAM ? m : n;
    constexpr int LOADS = KSTEP / 8;
    float4 la[LOADS], lb[LOADS];
    const int arow = t / 2, aq = t % 2;

    auto fetch = [&](int k0) {
#pragma unroll
        for (int s = 0; s < LOADS; ++s) {
            if constexpr (GRAM) {
                la[s] = *reinterpret_cast<const float4*>(A + static_cast<int64_t>(k0 + s * 8 + kr) * n + bi * TILE + c4 * 4);
                lb[s] = *reinterpret_cast<const float4*>(A + static_cast<int64_t>(k0 + s * 8 + kr) * n + bj * TILE + c4 * 4);
            } else {
                la[s] = *reinterpret_cast<const float4*>(A + static_cast<int64_t>(bi * TILE + arow) * n + k0 + s * 8 + aq * 4);
                lb[s] = *reinterpret_cast<const float4*>(B + static_cast<int64_t>(k0 + s * 8 + kr) * n + bj * TILE + c4 * 4);
            }
        }
    };
    fetch(0);
    for (int k0 = 0; k0 < k_end; k0 += KSTEP) {
#pragma unroll
        for (int s = 0; s < LOADS; ++s) {
            if constexpr (GRAM) {
                *reinterpret_cast<float4*>(&left[s * 8 + kr][c4 * 4]) = la[s];
            } else {
                left[s * 8 + aq * 4 + 0][arow] = la[s].x;
                left[s * 8 + aq * 4 + 1][arow] = la[s].y;
                left[s * 8 + aq * 4 + 2][arow] = la[s].z;
                left[s * 8 + aq * 4 + 3][arow] = la[s].w;
            }
            *reinterpret_cast<float4*>(&right[s * 8 + kr][c4 * 4]) = lb[s];
        }
        __syncthreads();
        if (k0 + KSTEP < k_end) fetch(k0 + KSTEP);
#pragma unroll
        for (int kk = 0; kk < KSTEP; ++kk) {
            const float4 a0 = *reinterpret_cast<const float4*>(&left[kk][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4*>(&left[kk][ty * 4 + 64]);
            const float4 b0 = *reinterpret_cast<const float4*>(&right[kk][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4*>(&right[kk][tx * 4 + 64]);
            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }

    float scale = GRAM ? alpha : 1.0f;
    if (mode) {
        const float s = 1.0f / (sqrtf(sumsq[matrix]) + 1e-7f);
        scale = GRAM ? s * s : s;
    }
    float keep = 1.0f, move = 0.0f;
    float* p = nullptr;
    if constexpr (!GRAM && ADD) {
        p = reinterpret_cast<float*>(params[matrix]);
        const bool live = flags[matrix] != 0;
        keep = live ? 1.0f - decay_factor : 1.0f;
        move = live ? step : 0.0f;
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            const int row0 = bi * TILE + ty * 4 + h * 64, col0 = bj * TILE + tx * 4 + w * 64;
            float out[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t at = static_cast<int64_t>(row0 + i) * n + col0;
                if constexpr (GRAM) {
                    float4 add = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    if constexpr (ADD) add = *reinterpret_cast<const float4*>(Cin + at);
                    out[i][0] = fmaf(beta, add.x, scale * acc[h * 4 + i][w * 4 + 0]);
                    out[i][1] = fmaf(beta, add.y, scale * acc[h * 4 + i][w * 4 + 1]);
                    out[i][2] = fmaf(beta, add.z, scale * acc[h * 4 + i][w * 4 + 2]);
                    out[i][3] = fmaf(beta, add.w, scale * acc[h * 4 + i][w * 4 + 3]);
                    if (bi == bj) {
#pragma unroll
                        for (int k = 0; k < 4; ++k)
                            if (row0 + i == col0 + k) out[i][k] += diag;
                    }
                } else {
#pragma unroll
                    for (int k = 0; k < 4; ++k) out[i][k] = scale * acc[h * 4 + i][w * 4 + k];
                }
                if constexpr (!(!GRAM && ADD)) *reinterpret_cast<float4*>(C + at) = make_float4(out[i][0], out[i][1], out[i][2], out[i][3]);
            }
            if constexpr (!GRAM && ADD) {
                if (!transposed) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float4* q = reinterpret_cast<float4*>(p + static_cast<int64_t>(row0 + i) * n + col0);
                        const float4 v = *q;
                        *q = make_float4(fmaf(-move, out[i][0], v.x * keep), fmaf(-move, out[i][1], v.y * keep),
                                         fmaf(-move, out[i][2], v.z * keep), fmaf(-move, out[i][3], v.w * keep));
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float4* q = reinterpret_cast<float4*>(p + static_cast<int64_t>(col0 + j) * m + row0);
                        const float4 v = *q;
                        *q = make_float4(fmaf(-move, out[0][j], v.x * keep), fmaf(-move, out[1][j], v.y * keep),
                                         fmaf(-move, out[2][j], v.z * keep), fmaf(-move, out[3][j], v.w * keep));
                    }
                }
            }
            if (GRAM && bi != bj) {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    *reinterpret_cast<float4*>(C + static_cast<int64_t>(col0 + j) * n + row0) = make_float4(out[0][j], out[1][j], out[2][j], out[3][j]);
            }
        }
    }
}

// B = c3 * I + c4 * a + c5 * a^2, the Newton-Schulz quintic denominator.
// Building B elementwise removes the second y^T y of every iteration.
static __global__ void make_b_kernel(const float* a, const float* a2, float* b, int n, float c3, float c4, float c5) {
    const int matrix = blockIdx.z;
    const int64_t total = static_cast<int64_t>(n) * n;
    const int64_t base = static_cast<int64_t>(matrix) * total;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index) / n, column = static_cast<int>(index) % n;
        float value = c4 * a[base + index] + c5 * a2[base + index];
        if (row == column) value += c3;
        b[base + index] = value;
    }
}

extern "C" ROCMKERNELS_EXPORT int muon_fp32_make_b(const float* a, const float* a2, float* b, int count, int n, float c3, float c4, float c5,
                                        hipStream_t stream) {
    if (count < 1 || n <= 0 || (n % TILE)) return hipErrorInvalidValue;
    const int blocks = static_cast<int>((static_cast<int64_t>(n) * n + 255) / 256 < 64 ? (static_cast<int64_t>(n) * n + 255) / 256 : 64);
    hipLaunchKernelGGL(make_b_kernel, dim3(blocks, 1, count), dim3(256), 0, stream, a, a2, b, n, c3, c4, c5);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int muon_fp32_supported(int rows, int columns) {
    const int m = rows >= columns ? rows : columns, n = rows >= columns ? columns : rows;
    return n % TILE == 0 && m % TILE == 0;
}

extern "C" ROCMKERNELS_EXPORT int muon_fp32_prepare(const int64_t* gradients, const int64_t* buffers, float* y, int* flags, float* sumsq,
                                         int count, int rows, int columns, float momentum, hipStream_t stream) {
    if (count < 1 || !muon_fp32_supported(rows, columns)) return hipErrorInvalidValue;
    const int64_t elements = static_cast<int64_t>(rows) * columns;
    if (hipMemsetAsync(flags, 0, sizeof(int) * count, stream) != hipSuccess) return hipErrorUnknown;
    if (hipMemsetAsync(sumsq, 0, sizeof(float) * count, stream) != hipSuccess) return hipErrorUnknown;
    hipLaunchKernelGGL(nonzero_kernel, dim3(64, count), dim3(256), 0, stream, gradients, flags, elements);
    if (rows >= columns) {
        const int blocks = static_cast<int>(elements / 2048 < 1 ? 1 : (elements / 2048 > 64 ? 64 : elements / 2048));
        hipLaunchKernelGGL((prep_kernel<false>), dim3(blocks, 1, count), dim3(32, 8), 0, stream, gradients, buffers, y, flags, sumsq, rows, columns, momentum);
    } else {
        hipLaunchKernelGGL((prep_kernel<true>), dim3(columns / 32, rows / 32, count), dim3(32, 8), 0, stream, gradients, buffers, y, flags, sumsq, rows, columns, momentum);
    }
    return hipGetLastError();
}

// g (n x n) = scale * y^T y + beta * gin; y is m x n. gin may be null (beta ignored).
extern "C" ROCMKERNELS_EXPORT int muon_fp32_gram(const float* y, const float* gin, float* g, const float* sumsq, int count, int m, int n,
                                      float alpha, float beta, float diag, int mode, hipStream_t stream) {
    if (count < 1 || n % TILE || m % KSTEP) return hipErrorInvalidValue;
    const dim3 grid(n / TILE, n / TILE, count);
    if (gin) hipLaunchKernelGGL((gemm_kernel<true, true>), grid, dim3(256), 0, stream, y, gin, g, sumsq, m, n, alpha, beta, mode, diag, nullptr, nullptr, 0.0f, 0.0f, 0);
    else hipLaunchKernelGGL((gemm_kernel<true, false>), grid, dim3(256), 0, stream, y, gin, g, sumsq, m, n, alpha, beta, mode, diag, nullptr, nullptr, 0.0f, 0.0f, 0);
    return hipGetLastError();
}

// out = s * (y @ b)
extern "C" ROCMKERNELS_EXPORT int muon_fp32_multiply(const float* y, const float* b, float* out, const float* sumsq, int count, int m, int n,
                                          int mode, hipStream_t stream) {
    if (count < 1 || n % TILE || m % TILE) return hipErrorInvalidValue;
    hipLaunchKernelGGL((gemm_kernel<false, false>), dim3(n / TILE, m / TILE, count), dim3(256), 0, stream, y, b, out, sumsq, m, n, 0.0f, 0.0f, mode,
                       0.0f, nullptr, nullptr, 0.0f, 0.0f, 0);
    return hipGetLastError();
}

// p = p * (1 - decay_factor) - step * s * (y @ b) for live matrices; transposed when y is the workspace of a wide parameter.
extern "C" ROCMKERNELS_EXPORT int muon_fp32_apply(const float* y, const float* b, const int64_t* params, const int* flags, const float* sumsq, int count,
                                       int m, int n, int mode, int transposed, float decay_factor, float step, hipStream_t stream) {
    if (count < 1 || n % TILE || m % TILE) return hipErrorInvalidValue;
    hipLaunchKernelGGL((gemm_kernel<false, true>), dim3(n / TILE, m / TILE, count), dim3(256), 0, stream, y, b, static_cast<float*>(nullptr), sumsq, m, n,
                       0.0f, 0.0f, mode, 0.0f, params, flags, decay_factor, step, transposed);
    return hipGetLastError();
}



