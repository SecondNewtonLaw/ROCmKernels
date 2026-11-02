#include <ROCmKernels/Kernels/Generic/LinearActFast.hpp>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

using ROCmKernels::Kernels::DType;
using ROCmKernels::Kernels::LoadScalar;
using ROCmKernels::Kernels::StoreScalar;

namespace {
    constexpr int kBlockRows = 64;
    constexpr int kBlockColumns = 64;
    constexpr int kKTile = 16;
    constexpr int kThreads = 256;
    constexpr int kThreadsX = 16;
    constexpr int kThreadsY = 16;
    constexpr int kVectorBytes = 16;
    // The extra float per row offsets each shared row by one bank, so the strided reads below stay conflict free.
    constexpr int kSharedPitch = kBlockColumns + 1;

    __device__ __forceinline__ float silu(const float value) {
        return value / (1.0f + __expf(-value));
    }

    // The fp16 path keeps K contiguous so one LDS delivers a half2 pair and one
    // instruction performs two MACs. RDNA2 has v_dot2_f32_f16 and no MFMA.
    constexpr int kHalfPitch = kKTile + 2;

    __device__ __forceinline__ float dot2(const __half2 a, const __half2 b, const float accumulator) {
        const unsigned aBits = *reinterpret_cast<const unsigned *>(&a);
        const unsigned bBits = *reinterpret_cast<const unsigned *>(&b);
        float result = accumulator;
        asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(result) : "v"(aBits), "v"(bBits));
        return result;
    }

    template<typename T> constexpr int VectorElements() {
        return kVectorBytes / static_cast<int>(sizeof(T));
    }

    template<typename T> __device__ __forceinline__ bool Vectorizable(const T *pointer, const int stride) {
        const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
        return (address & (kVectorBytes - 1)) == 0 && stride % VectorElements<T>() == 0;
    }

    template<typename T>
    __device__ __forceinline__ void LoadATile(const T *input, float (*tile)[kSharedPitch], const int rowBase, const int kBase, const int rows,
            const int inner, const bool vectorized, const int tid) {
        constexpr int kElements = VectorElements<T>();
        constexpr int kVectorsPerRow = kKTile / kElements;

        if (vectorized) {
            if (tid >= kBlockRows * kVectorsPerRow) return;

            const int row = tid / kVectorsPerRow;
            const int kOffset = (tid % kVectorsPerRow) * kElements;
            const int globalRow = rowBase + row;

            if (globalRow < rows && kBase + kOffset < inner) {
                const T *source = input + static_cast<std::size_t>(globalRow) * inner + kBase + kOffset;
                if constexpr (std::is_same_v<T, float>) {
                    const float4 value = *reinterpret_cast<const float4 *>(source);
                    tile[kOffset + 0][row] = value.x;
                    tile[kOffset + 1][row] = value.y;
                    tile[kOffset + 2][row] = value.z;
                    tile[kOffset + 3][row] = value.w;
                } else {
                    const uint4 value = *reinterpret_cast<const uint4 *>(source);
                    const __half *values = reinterpret_cast<const __half *>(&value);
#pragma unroll
                    for (int element = 0; element < kElements; ++element) tile[kOffset + element][row] = __half2float(values[element]);
                }
            } else {
#pragma unroll
                for (int element = 0; element < kElements; ++element) tile[kOffset + element][row] = 0.0f;
            }
            return;
        }

        for (int index = tid; index < kBlockRows * kKTile; index += kThreads) {
            const int row = index / kKTile;
            const int kOffset = index % kKTile;
            const int globalRow = rowBase + row;
            const int k = kBase + kOffset;
            tile[kOffset][row] = globalRow < rows && k < inner
                    ? LoadScalar(input + static_cast<std::size_t>(globalRow) * inner + k)
                    : 0.0f;
        }
    }

    template<typename T>
    __device__ __forceinline__ void LoadBTile(const T *weight, float (*tile)[kSharedPitch], const int columnBase, const int kBase, const int columns,
            const int inner, const bool vectorized, const int tid) {
        constexpr int kElements = VectorElements<T>();
        constexpr int kVectorsPerRow = kBlockColumns / kElements;

        if (vectorized) {
            if (tid >= kKTile * kVectorsPerRow) return;

            const int kOffset = tid / kVectorsPerRow;
            const int column = (tid % kVectorsPerRow) * kElements;
            const int k = kBase + kOffset;
            const int globalColumn = columnBase + column;

            if (k < inner && globalColumn < columns) {
                const T *source = weight + static_cast<std::size_t>(k) * columns + globalColumn;
                if constexpr (std::is_same_v<T, float>) {
                    const float4 value = *reinterpret_cast<const float4 *>(source);
                    tile[kOffset][column + 0] = value.x;
                    tile[kOffset][column + 1] = value.y;
                    tile[kOffset][column + 2] = value.z;
                    tile[kOffset][column + 3] = value.w;
                } else {
                    const uint4 value = *reinterpret_cast<const uint4 *>(source);
                    const __half *values = reinterpret_cast<const __half *>(&value);
#pragma unroll
                    for (int element = 0; element < kElements; ++element) tile[kOffset][column + element] = __half2float(values[element]);
                }
            } else {
#pragma unroll
                for (int element = 0; element < kElements; ++element) tile[kOffset][column + element] = 0.0f;
            }
            return;
        }

        for (int index = tid; index < kKTile * kBlockColumns; index += kThreads) {
            const int kOffset = index / kBlockColumns;
            const int column = index % kBlockColumns;
            const int k = kBase + kOffset;
            const int globalColumn = columnBase + column;
            tile[kOffset][column] = k < inner && globalColumn < columns
                    ? LoadScalar(weight + static_cast<std::size_t>(k) * columns + globalColumn)
                    : 0.0f;
        }
    }

    template<typename T>
    __global__ __launch_bounds__(kThreads) void linear_act_fast_kernel(const T *input, const T *weight, const T *bias, T *output, const int rows,
            const int columns, const int inner) {
        __shared__ float tileA[kKTile][kSharedPitch];
        __shared__ float tileB[kKTile][kSharedPitch];

        const int tid = threadIdx.y * kThreadsX + threadIdx.x;
        const int rowBase = blockIdx.y * kBlockRows;
        const int columnBase = blockIdx.x * kBlockColumns;
        const bool vectorizedA = Vectorizable(input, inner);
        const bool vectorizedB = Vectorizable(weight, columns);

        float accumulators[4][4] = {};

        for (int kBase = 0; kBase < inner; kBase += kKTile) {
            LoadATile(input, tileA, rowBase, kBase, rows, inner, vectorizedA, tid);
            LoadBTile(weight, tileB, columnBase, kBase, columns, inner, vectorizedB, tid);
            __syncthreads();

#pragma unroll
            for (int k = 0; k < kKTile; ++k) {
                float a[4];
                float b[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) a[i] = tileA[k][threadIdx.y + i * kThreadsY];
#pragma unroll
                for (int j = 0; j < 4; ++j) b[j] = tileB[k][threadIdx.x + j * kThreadsX];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) accumulators[i][j] = fmaf(a[i], b[j], accumulators[i][j]);
                }
            }
            __syncthreads();
        }

        float biasValues[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int column = columnBase + threadIdx.x + j * kThreadsX;
            biasValues[j] = bias != nullptr && column < columns ? LoadScalar(bias + column) : 0.0f;
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = rowBase + threadIdx.y + i * kThreadsY;
            if (row >= rows) continue;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int column = columnBase + threadIdx.x + j * kThreadsX;
                if (column >= columns) continue;
                StoreScalar(output + static_cast<std::size_t>(row) * columns + column, silu(accumulators[i][j] + biasValues[j]));
            }
        }
    }

    __device__ __forceinline__ void load_a_half(const __half *input, __half (*tile)[kHalfPitch], const int rowBase, const int kBase, const int rows,
            const int inner, const bool vectorized, const int tid) {
        constexpr int kElements = 8;
        constexpr int kVectorsPerRow = kKTile / kElements;

        if (vectorized) {
            if (tid >= kBlockRows * kVectorsPerRow) return;
            const int row = tid / kVectorsPerRow;
            const int kOffset = (tid % kVectorsPerRow) * kElements;
            const int globalRow = rowBase + row;
            if (globalRow < rows && kBase + kOffset < inner) {
                const uint4 value = *reinterpret_cast<const uint4 *>(input + static_cast<std::size_t>(globalRow) * inner + kBase + kOffset);
                const __half *values = reinterpret_cast<const __half *>(&value);
                #pragma unroll
                for (int element = 0; element < kElements; ++element) tile[row][kOffset + element] = values[element];
            } else {
                #pragma unroll
                for (int element = 0; element < kElements; ++element) tile[row][kOffset + element] = __float2half_rn(0.0f);
            }
            return;
        }

        for (int index = tid; index < kBlockRows * kKTile; index += kThreads) {
            const int row = index / kKTile;
            const int kOffset = index % kKTile;
            const int globalRow = rowBase + row;
            const int k = kBase + kOffset;
            tile[row][kOffset] = globalRow < rows && k < inner ? input[static_cast<std::size_t>(globalRow) * inner + k] : __float2half_rn(0.0f);
        }
    }

    __device__ __forceinline__ void load_b_half(const __half *weight, __half (*tile)[kHalfPitch], const int columnBase, const int kBase,
            const int columns, const int inner, const bool vectorized, const int tid) {
        constexpr int kElements = 8;
        constexpr int kVectorsPerRow = kBlockColumns / kElements;

        if (vectorized) {
            if (tid >= kKTile * kVectorsPerRow) return;
            const int kOffset = tid / kVectorsPerRow;
            const int column = (tid % kVectorsPerRow) * kElements;
            const int k = kBase + kOffset;
            const int globalColumn = columnBase + column;
            if (k < inner && globalColumn < columns) {
                const uint4 value = *reinterpret_cast<const uint4 *>(weight + static_cast<std::size_t>(k) * columns + globalColumn);
                const __half *values = reinterpret_cast<const __half *>(&value);
                #pragma unroll
                for (int element = 0; element < kElements; ++element) tile[column + element][kOffset] = values[element];
            } else {
                #pragma unroll
                for (int element = 0; element < kElements; ++element) tile[column + element][kOffset] = __float2half_rn(0.0f);
            }
            return;
        }

        for (int index = tid; index < kKTile * kBlockColumns; index += kThreads) {
            const int kOffset = index / kBlockColumns;
            const int column = index % kBlockColumns;
            const int k = kBase + kOffset;
            const int globalColumn = columnBase + column;
            tile[column][kOffset] = k < inner && globalColumn < columns
                    ? weight[static_cast<std::size_t>(k) * columns + globalColumn]
                    : __float2half_rn(0.0f);
        }
    }

    __global__ __launch_bounds__(kThreads) void linear_act_fast_half_kernel(const __half *input, const __half *weight, const __half *bias,
            __half *output, const int rows, const int columns, const int inner) {
        __shared__ __half tileA[kBlockRows][kHalfPitch];
        __shared__ __half tileB[kBlockColumns][kHalfPitch];

        const int tid = threadIdx.y * kThreadsX + threadIdx.x;
        const int rowBase = blockIdx.y * kBlockRows;
        const int columnBase = blockIdx.x * kBlockColumns;
        const bool vectorizedA = Vectorizable(input, inner);
        const bool vectorizedB = Vectorizable(weight, columns);

        float accumulators[4][4] = {};

        for (int kBase = 0; kBase < inner; kBase += kKTile) {
            load_a_half(input, tileA, rowBase, kBase, rows, inner, vectorizedA, tid);
            load_b_half(weight, tileB, columnBase, kBase, columns, inner, vectorizedB, tid);
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < kKTile; k += 2) {
                __half2 a[4];
                __half2 b[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<const __half2 *>(&tileA[threadIdx.y + i * kThreadsY][k]);
                #pragma unroll
                for (int j = 0; j < 4; ++j) b[j] = *reinterpret_cast<const __half2 *>(&tileB[threadIdx.x + j * kThreadsX][k]);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) accumulators[i][j] = dot2(a[i], b[j], accumulators[i][j]);
                }
            }
            __syncthreads();
        }

        float biasValues[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int column = columnBase + threadIdx.x + j * kThreadsX;
            biasValues[j] = bias != nullptr && column < columns ? __half2float(bias[column]) : 0.0f;
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = rowBase + threadIdx.y + i * kThreadsY;
            if (row >= rows) continue;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int column = columnBase + threadIdx.x + j * kThreadsX;
                if (column >= columns) continue;
                output[static_cast<std::size_t>(row) * columns + column] = __float2half_rn(silu(accumulators[i][j] + biasValues[j]));
            }
        }
    }

    template<typename T>
    int launch_linear(const void *input, const void *weight, const void *bias, void *output, const int rows, const int columns, const int inner,
            const hipStream_t stream) {
        const dim3 grid((columns + kBlockColumns - 1) / kBlockColumns, (rows + kBlockRows - 1) / kBlockRows);
        if constexpr (std::is_same_v<T, __half>) {
            hipLaunchKernelGGL((linear_act_fast_half_kernel), grid, dim3(kThreadsX, kThreadsY), 0, stream, static_cast<const __half *>(input),
                    static_cast<const __half *>(weight), static_cast<const __half *>(bias), static_cast<__half *>(output), rows, columns, inner);
        } else {
            hipLaunchKernelGGL((linear_act_fast_kernel<T>), grid, dim3(kThreadsX, kThreadsY), 0, stream, static_cast<const T *>(input),
                    static_cast<const T *>(weight), static_cast<const T *>(bias), static_cast<T *>(output), rows, columns, inner);
        }
        return hipGetLastError();
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int linear_act_fast(const void *input, const void *weight, const void *bias, void *output, int rows, int columns,
        int inner, ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    if (dtype != DType::Float32 && dtype != DType::Float16) return hipErrorInvalidValue;
    if (!input || !weight || !output || rows <= 0 || columns <= 0 || inner <= 0) return hipErrorInvalidValue;

    switch (dtype) {
        case DType::Float32:
            return launch_linear<float>(input, weight, bias, output, rows, columns, inner, stream);
        case DType::Float16:
            return launch_linear<__half>(input, weight, bias, output, rows, columns, inner, stream);
    }

    return hipErrorInvalidValue;
}
