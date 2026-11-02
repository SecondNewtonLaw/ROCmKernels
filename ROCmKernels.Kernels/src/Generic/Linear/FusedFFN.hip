#include <ROCmKernels/Kernels/Generic/FusedFFN.hpp>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

using ROCmKernels::Kernels::DType;

namespace {
    // One block owns 64 positions and the full 128-wide output. The hidden layer is
    // walked in 64-wide chunks; each chunk's activation is written to shared memory
    // and consumed by the down GEMM inside the same block, so the 128 MB
    // intermediate never reaches global memory.
    //
    // Both weights use torch's [out][in] layout: w1[hidden][width] and w2[out][hidden].
    // That makes each staged tile contiguous along k, so the load and the shared
    // store are both vectorised and bank clean, with no transpose.
    constexpr int kRows = 64;
    constexpr int kWidth = 128;
    constexpr int kOutput = 128;
    constexpr int kChunk = 64;
    constexpr int kThreads = 256;
    constexpr int kTX = 16;
    constexpr int kTY = 16;

    constexpr int kPitchA = kWidth + 8;    // x tile: [row][width]
    constexpr int kPitchH = kChunk + 8;    // activation tile: [row][chunk]
    constexpr int kPitchW1 = kWidth + 8;   // w1 tile: [hidden chunk][width]
    constexpr int kPitchW2 = kChunk + 8;   // w2 tile: [output][chunk]

    __device__ __forceinline__ float dot2(const __half2 a, const __half2 b, const float accumulator) {
        const unsigned aBits = *reinterpret_cast<const unsigned *>(&a);
        const unsigned bBits = *reinterpret_cast<const unsigned *>(&b);
        float result = accumulator;
        asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(result) : "v"(aBits), "v"(bBits));
        return result;
    }

    __device__ __forceinline__ float silu(const float value) {
        return value / (1.0f + __expf(-value));
    }

    __global__ __launch_bounds__(kThreads) void ffn_up_silu_down_kernel(const __half *x, const __half *w1, const __half *b1, const __half *w2,
            const __half *b2, __half *out, const int rows, const int hidden) {
        __shared__ __half xs[kRows][kPitchA];
        __shared__ __half hs[kRows][kPitchH];
        __shared__ __half bs1[kChunk][kPitchW1];
        __shared__ __half bs2[kOutput][kPitchW2];

        const int tid = threadIdx.y * kTX + threadIdx.x;
        const int rowBase = blockIdx.x * kRows;

        // Stage x once per block; every hidden chunk re-reads it from LDS.
        for (int index = tid; index < kRows * (kWidth / 8); index += kThreads) {
            const int row = index / (kWidth / 8);
            const int kOffset = (index % (kWidth / 8)) * 8;
            const int globalRow = rowBase + row;
            if (globalRow < rows) {
                const uint4 value = *reinterpret_cast<const uint4 *>(x + static_cast<std::size_t>(globalRow) * kWidth + kOffset);
                *reinterpret_cast<uint4 *>(&xs[row][kOffset]) = value;
            } else {
                *reinterpret_cast<uint4 *>(&xs[row][kOffset]) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        float accumulators[4][8] = {};

        for (int chunk = 0; chunk < hidden; chunk += kChunk) {
            // w1[k][chunk .. chunk+63] stays in its own layout: contiguous along k.
            for (int index = tid; index < kChunk * (kWidth / 8); index += kThreads) {
                const int row = index / (kWidth / 8);
                const int kOffset = (index % (kWidth / 8)) * 8;
                const uint4 value = *reinterpret_cast<const uint4 *>(w1 + static_cast<std::size_t>(chunk + row) * kWidth + kOffset);
                *reinterpret_cast<uint4 *>(&bs1[row][kOffset]) = value;
            }
            __syncthreads();

            float tile[4][4] = {};
            #pragma unroll 2
            for (int k = 0; k < kWidth; k += 2) {
                __half2 a[4];
                __half2 b[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<const __half2 *>(&xs[threadIdx.y + i * kTY][k]);
                #pragma unroll
                for (int j = 0; j < 4; ++j) b[j] = *reinterpret_cast<const __half2 *>(&bs1[threadIdx.x + j * kTX][k]);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) tile[i][j] = dot2(a[i], b[j], tile[i][j]);
                }
            }

            // SiLU and store as fp16 into the shared activation tile.
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = threadIdx.y + i * kTY;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int hiddenColumn = chunk + threadIdx.x + j * kTX;
                    const float bias = b1 != nullptr ? __half2float(b1[hiddenColumn]) : 0.0f;
                    hs[row][threadIdx.x + j * kTX] = __float2half_rn(silu(tile[i][j] + bias));
                }
            }
            __syncthreads();

            // w2[0 .. 127][chunk .. chunk+63] is contiguous along the hidden axis.
            for (int index = tid; index < kOutput * (kChunk / 8); index += kThreads) {
                const int row = index / (kChunk / 8);
                const int kOffset = (index % (kChunk / 8)) * 8;
                const uint4 value = *reinterpret_cast<const uint4 *>(w2 + static_cast<std::size_t>(row) * hidden + chunk + kOffset);
                *reinterpret_cast<uint4 *>(&bs2[row][kOffset]) = value;
            }
            __syncthreads();

            #pragma unroll 2
            for (int k = 0; k < kChunk; k += 2) {
                __half2 a[4];
                __half2 b[8];
                #pragma unroll
                for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<const __half2 *>(&hs[threadIdx.y + i * kTY][k]);
                #pragma unroll
                for (int j = 0; j < 8; ++j) b[j] = *reinterpret_cast<const __half2 *>(&bs2[threadIdx.x + j * kTX][k]);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) accumulators[i][j] = dot2(a[i], b[j], accumulators[i][j]);
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = rowBase + threadIdx.y + i * kTY;
            if (row >= rows) continue;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int column = threadIdx.x + j * kTX;
                const float bias = b2 != nullptr ? __half2float(b2[column]) : 0.0f;
                out[static_cast<std::size_t>(row) * kOutput + column] = __float2half_rn(accumulators[i][j] + bias);
            }
        }
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int ffn_up_silu_down(const void *x, const void *w1, const void *b1, const void *w2, const void *b2, void *out,
        int rows, int hidden, ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    if (dtype != DType::Float16) return hipErrorInvalidValue;
    if (!x || !w1 || !w2 || !out || rows <= 0 || hidden <= 0 || (hidden % 64) != 0) return hipErrorInvalidValue;

    const dim3 grid((rows + kRows - 1) / kRows);
    hipLaunchKernelGGL((ffn_up_silu_down_kernel), grid, dim3(kTX, kTY), 0, stream, static_cast<const __half *>(x),
            static_cast<const __half *>(w1), static_cast<const __half *>(b1), static_cast<const __half *>(w2),
            static_cast<const __half *>(b2), static_cast<__half *>(out), rows, hidden);
    return hipGetLastError();
}
