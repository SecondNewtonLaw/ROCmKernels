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

    // B stored K-major: weight[k][column]. Each thread reads one k for eight columns.
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

    // B stored N-major, the layout torch uses: weight[column][k]. Each thread reads
    // one column for four k values, so both the global read and the store stay in
    // one direction.
    template<typename T>
    __device__ __forceinline__ void LoadBTileTransposed(const T *weight, float (*tile)[kSharedPitch], const int columnBase, const int kBase,
            const int columns, const int inner, const bool vectorized, const int tid) {
        constexpr int kElements = VectorElements<T>();
        constexpr int kVectorsPerRow = kKTile / kElements;

        if (vectorized) {
            if (tid >= kBlockColumns * kVectorsPerRow) return;

            const int column = tid / kVectorsPerRow;
            const int kOffset = (tid % kVectorsPerRow) * kElements;
            const int k = kBase + kOffset;
            const int globalColumn = columnBase + column;

            if (k < inner && globalColumn < columns) {
                const T *source = weight + static_cast<std::size_t>(globalColumn) * inner + k;
                if constexpr (std::is_same_v<T, float>) {
                    const float4 value = *reinterpret_cast<const float4 *>(source);
                    tile[kOffset + 0][column] = value.x;
                    tile[kOffset + 1][column] = value.y;
                    tile[kOffset + 2][column] = value.z;
                    tile[kOffset + 3][column] = value.w;
                } else {
                    const uint4 value = *reinterpret_cast<const uint4 *>(source);
                    const __half *values = reinterpret_cast<const __half *>(&value);
#pragma unroll
                    for (int element = 0; element < kElements; ++element) tile[kOffset + element][column] = __half2float(values[element]);
                }
            } else {
#pragma unroll
                for (int element = 0; element < kElements; ++element) tile[kOffset + element][column] = 0.0f;
            }
            return;
        }

        for (int index = tid; index < kKTile * kBlockColumns; index += kThreads) {
            const int column = index / kKTile;
            const int kOffset = index % kKTile;
            const int k = kBase + kOffset;
            const int globalColumn = columnBase + column;
            tile[kOffset][column] = k < inner && globalColumn < columns
                    ? LoadScalar(weight + static_cast<std::size_t>(globalColumn) * inner + k)
                    : 0.0f;
        }
    }

    template<typename T, bool Activate, bool Transposed>
    __global__ __launch_bounds__(kThreads) void linear_act_fast_kernel(const T *input, const T *weight, const T *bias, T *output, const int rows,
            const int columns, const int inner) {
        __shared__ float tileA[kKTile][kSharedPitch];
        __shared__ float tileB[kKTile][kSharedPitch];

        const int tid = threadIdx.y * kThreadsX + threadIdx.x;
        const int rowBase = blockIdx.y * kBlockRows;
        const int columnBase = blockIdx.x * kBlockColumns;
        const bool vectorizedA = Vectorizable(input, inner);
        const bool vectorizedB = Vectorizable(weight, Transposed ? inner : columns);

        float accumulators[4][4] = {};

        for (int kBase = 0; kBase < inner; kBase += kKTile) {
            LoadATile(input, tileA, rowBase, kBase, rows, inner, vectorizedA, tid);
            if constexpr (Transposed) {
                LoadBTileTransposed(weight, tileB, columnBase, kBase, columns, inner, vectorizedB, tid);
            } else {
                LoadBTile(weight, tileB, columnBase, kBase, columns, inner, vectorizedB, tid);
            }
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
                const float value = accumulators[i][j] + biasValues[j];
                StoreScalar(output + static_cast<std::size_t>(row) * columns + column, Activate ? silu(value) : value);
            }
        }
    }

    template<int KTile, int Pitch>
    __device__ __forceinline__ void load_a_half(const __half *input, __half (*tile)[Pitch], const int rowBase, const int kBase, const int rows,
            const int inner, const bool vectorized, const int tid) {
        constexpr int kElements = 8;
        constexpr int kVectorsPerRow = KTile / kElements;

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

        for (int index = tid; index < kBlockRows * KTile; index += kThreads) {
            const int row = index / KTile;
            const int kOffset = index % KTile;
            const int globalRow = rowBase + row;
            const int k = kBase + kOffset;
            tile[row][kOffset] = globalRow < rows && k < inner ? input[static_cast<std::size_t>(globalRow) * inner + k] : __float2half_rn(0.0f);
        }
    }

    // B stored K-major: weight[k][column], scattered into the column-major tile.
    template<int KTile, int Pitch>
    __device__ __forceinline__ void load_b_half(const __half *weight, __half (*tile)[Pitch], const int columnBase, const int kBase,
            const int columns, const int inner, const bool vectorized, const int tid) {
        constexpr int kElements = 8;
        constexpr int kVectorsPerRow = kBlockColumns / kElements;

        if (vectorized) {
            if (tid >= KTile * kVectorsPerRow) return;
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

        for (int index = tid; index < KTile * kBlockColumns; index += kThreads) {
            const int kOffset = index / kBlockColumns;
            const int column = index % kBlockColumns;
            const int k = kBase + kOffset;
            const int globalColumn = columnBase + column;
            tile[column][kOffset] = k < inner && globalColumn < columns
                    ? weight[static_cast<std::size_t>(k) * columns + globalColumn]
                    : __float2half_rn(0.0f);
        }
    }

    // B stored N-major: weight[column][k]. One column per thread, contiguous in K,
    // so the source read and the shared store are both vectorised.
    template<int KTile>
    __device__ __forceinline__ void load_b_half_transposed(const __half *weight, __half (*tile)[KTile], const int columnBase,
            const int kBase, const int columns, const int inner, const bool vectorized, const int tid) {
        constexpr int kElements = 8;
        constexpr int kVectorsPerRow = KTile / kElements;

        if (vectorized) {
            if (tid >= kBlockColumns * kVectorsPerRow) return;
            const int column = tid / kVectorsPerRow;
            const int kOffset = (tid % kVectorsPerRow) * kElements;
            const int k = kBase + kOffset;
            const int globalColumn = columnBase + column;
            if (k < inner && globalColumn < columns) {
                const uint4 value = *reinterpret_cast<const uint4 *>(weight + static_cast<std::size_t>(globalColumn) * inner + k);
                *reinterpret_cast<uint4 *>(&tile[column][kOffset]) = value;
            } else {
                #pragma unroll
                for (int element = 0; element < kElements; ++element) tile[column][kOffset + element] = __float2half_rn(0.0f);
            }
            return;
        }

        for (int index = tid; index < kBlockColumns * KTile; index += kThreads) {
            const int column = index / KTile;
            const int kOffset = index % KTile;
            const int k = kBase + kOffset;
            const int globalColumn = columnBase + column;
            tile[column][kOffset] = k < inner && globalColumn < columns
                    ? weight[static_cast<std::size_t>(globalColumn) * inner + k]
                    : __float2half_rn(0.0f);
        }
    }

    template<int KTile, bool Activate, bool Transposed>
    __global__ __launch_bounds__(kThreads) void linear_act_fast_half_kernel(const __half *input, const __half *weight, const __half *bias,
            __half *output, const int rows, const int columns, const int inner) {
        __shared__ __half tileA[kBlockRows][KTile + 2];
        __shared__ __half tileB[kBlockColumns][Transposed ? KTile : KTile + 2];

        const int tid = threadIdx.y * kThreadsX + threadIdx.x;
        const int rowBase = blockIdx.y * kBlockRows;
        const int columnBase = blockIdx.x * kBlockColumns;
        const bool vectorizedA = Vectorizable(input, inner);
        const bool vectorizedB = Vectorizable(weight, Transposed ? inner : columns);

        float accumulators[4][4] = {};

        for (int kBase = 0; kBase < inner; kBase += KTile) {
            load_a_half<KTile, KTile + 2>(input, tileA, rowBase, kBase, rows, inner, vectorizedA, tid);
            if constexpr (Transposed) {
                load_b_half_transposed<KTile>(weight, tileB, columnBase, kBase, columns, inner, vectorizedB, tid);
            } else {
                load_b_half<KTile, KTile + 2>(weight, tileB, columnBase, kBase, columns, inner, vectorizedB, tid);
            }
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < KTile; k += 2) {
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
            }            __syncthreads();
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
                const float value = accumulators[i][j] + biasValues[j];
                output[static_cast<std::size_t>(row) * columns + column] = __float2half_rn(Activate ? silu(value) : value);
            }
        }
    }

    // Wide fp16 tiles. Both keep 32 outputs per thread and 256 threads:
    //   Columns > 128: 128 rows x 64 columns, 8x4 per thread.
    //   Columns <= 128: 64 rows x 128 columns, 4x8 per thread, so the whole
    //   output width is covered by one block and A is read once.
    // rocBLAS uses 128x128x16 and 128x256x16 Tensile tiles for these shapes.
    template<int Rows, int KTile, int Pitch>
    __device__ __forceinline__ void load_a_half_wide(const __half *input, __half (*tile)[Pitch], const int rowBase, const int kBase,
            const int rows, const int inner, const bool vectorized, const int tid) {
        constexpr int kElements = 8;
        constexpr int kVectorsPerRow = KTile / kElements;

        if (vectorized) {
            for (int index = tid; index < Rows * kVectorsPerRow; index += kThreads) {
                const int row = index / kVectorsPerRow;
                const int kOffset = (index % kVectorsPerRow) * kElements;
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
            }
            return;
        }

        for (int index = tid; index < Rows * KTile; index += kThreads) {
            const int row = index / KTile;
            const int kOffset = index % KTile;
            const int globalRow = rowBase + row;
            const int k = kBase + kOffset;
            tile[row][kOffset] = globalRow < rows && k < inner
                    ? input[static_cast<std::size_t>(globalRow) * inner + k]
                    : __float2half_rn(0.0f);
        }
    }

    template<int Columns, int KTile, int Pitch>
    __device__ __forceinline__ void load_b_half_wide(const __half *weight, __half (*tile)[Pitch], const int columnBase, const int kBase,
            const int columns, const int inner, const bool vectorized, const int tid) {
        constexpr int kElements = 8;
        constexpr int kVectorsPerRow = Columns / kElements;

        if (vectorized) {
            for (int index = tid; index < KTile * kVectorsPerRow; index += kThreads) {
                const int kOffset = index / kVectorsPerRow;
                const int column = (index % kVectorsPerRow) * kElements;
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
            }
            return;
        }

        for (int index = tid; index < KTile * Columns; index += kThreads) {
            const int kOffset = index / Columns;
            const int column = index % Columns;
            const int k = kBase + kOffset;
            const int globalColumn = columnBase + column;
            tile[column][kOffset] = k < inner && globalColumn < columns
                    ? weight[static_cast<std::size_t>(k) * columns + globalColumn]
                    : __float2half_rn(0.0f);
        }
    }

    template<int Rows, int Columns, int TileRows, int TileColumns, int KTile, bool Activate, bool FloatOut, bool EmitS>
    __global__ __launch_bounds__(kThreads) void linear_act_fast_half_wide_kernel(const __half *input, const __half *weight, const __half *bias,
            void *output, __half *sigmoidOut, const int rows, const int columns, const int inner) {
        __shared__ __half tileA[Rows][KTile + 2];
        __shared__ __half tileB[Columns][KTile + 2];

        const int tid = threadIdx.y * kThreadsX + threadIdx.x;
        const int rowBase = blockIdx.y * Rows;
        const int columnBase = blockIdx.x * Columns;
        const bool vectorizedA = Vectorizable(input, inner);
        const bool vectorizedB = Vectorizable(weight, columns);

        float accumulators[TileRows][TileColumns] = {};

        for (int kBase = 0; kBase < inner; kBase += KTile) {
            load_a_half_wide<Rows, KTile, KTile + 2>(input, tileA, rowBase, kBase, rows, inner, vectorizedA, tid);
            load_b_half_wide<Columns, KTile, KTile + 2>(weight, tileB, columnBase, kBase, columns, inner, vectorizedB, tid);
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < KTile; k += 2) {
                __half2 a[TileRows];
                __half2 b[TileColumns];
                #pragma unroll
                for (int i = 0; i < TileRows; ++i) a[i] = *reinterpret_cast<const __half2 *>(&tileA[threadIdx.y + i * kThreadsY][k]);
                #pragma unroll
                for (int j = 0; j < TileColumns; ++j) b[j] = *reinterpret_cast<const __half2 *>(&tileB[threadIdx.x + j * kThreadsX][k]);
                #pragma unroll
                for (int i = 0; i < TileRows; ++i) {
                    #pragma unroll
                    for (int j = 0; j < TileColumns; ++j) accumulators[i][j] = dot2(a[i], b[j], accumulators[i][j]);
                }
            }
            __syncthreads();
        }

        float biasValues[TileColumns];
        #pragma unroll
        for (int j = 0; j < TileColumns; ++j) {
            const int column = columnBase + threadIdx.x + j * kThreadsX;
            biasValues[j] = bias != nullptr && column < columns ? __half2float(bias[column]) : 0.0f;
        }

        #pragma unroll
        for (int i = 0; i < TileRows; ++i) {
            const int row = rowBase + threadIdx.y + i * kThreadsY;
            if (row >= rows) continue;
            #pragma unroll
            for (int j = 0; j < TileColumns; ++j) {
                const int column = columnBase + threadIdx.x + j * kThreadsX;
                if (column >= columns) continue;
                const float value = accumulators[i][j] + biasValues[j];
                const std::size_t index = static_cast<std::size_t>(row) * columns + column;
                if constexpr (EmitS) {
                    // One exp for both the activation and the sigmoid the backward reuses.
                    const float sigmoid = 1.0f / (1.0f + __expf(-value));
                    static_cast<__half *>(output)[index] = __float2half_rn(value * sigmoid);
                    sigmoidOut[index] = __float2half_rn(sigmoid);
                } else if constexpr (FloatOut) static_cast<float *>(output)[index] = Activate ? silu(value) : value;
                else static_cast<__half *>(output)[index] = __float2half_rn(Activate ? silu(value) : value);
            }
        }
    }
    template<typename T, bool Activate, bool Transposed>
    int launch_linear(const void *input, const void *weight, const void *bias, void *output, const int rows, const int columns, const int inner,
            const hipStream_t stream) {
        if constexpr (std::is_same_v<T, __half> && !Transposed) {
            if (columns > 128) {
                const dim3 grid((columns + 63) / 64, (rows + 127) / 128);
                hipLaunchKernelGGL((linear_act_fast_half_wide_kernel<128, 64, 8, 4, 64, Activate, false, false>), grid, dim3(kThreadsX, kThreadsY), 0, stream,
                        static_cast<const __half *>(input), static_cast<const __half *>(weight), static_cast<const __half *>(bias),
                        static_cast<__half *>(output), nullptr, rows, columns, inner);
            } else {
                const dim3 grid((columns + 127) / 128, (rows + 63) / 64);
                hipLaunchKernelGGL((linear_act_fast_half_wide_kernel<64, 128, 4, 8, 32, Activate, false, false>), grid, dim3(kThreadsX, kThreadsY), 0, stream,
                        static_cast<const __half *>(input), static_cast<const __half *>(weight), static_cast<const __half *>(bias),
                        static_cast<__half *>(output), nullptr, rows, columns, inner);
            }        } else if constexpr (std::is_same_v<T, __half>) {
            const dim3 grid((columns + kBlockColumns - 1) / kBlockColumns, (rows + kBlockRows - 1) / kBlockRows);
            hipLaunchKernelGGL((linear_act_fast_half_kernel<16, Activate, true>), grid, dim3(kThreadsX, kThreadsY), 0, stream,
                    static_cast<const __half *>(input), static_cast<const __half *>(weight), static_cast<const __half *>(bias),
                    static_cast<__half *>(output), rows, columns, inner);
        } else {
            const dim3 grid((columns + kBlockColumns - 1) / kBlockColumns, (rows + kBlockRows - 1) / kBlockRows);
            hipLaunchKernelGGL((linear_act_fast_kernel<T, Activate, Transposed>), grid, dim3(kThreadsX, kThreadsY), 0, stream,
                    static_cast<const T *>(input), static_cast<const T *>(weight), static_cast<const T *>(bias), static_cast<T *>(output), rows,
                    columns, inner);
        }
        return hipGetLastError();
    }

    template<typename T, bool Activate, bool Transposed>
    int dispatch_linear(const void *input, const void *weight, const void *bias, void *output, const int rows, const int columns, const int inner,
            const DType dtype, const hipStream_t stream) {
        if (dtype != DType::Float32 && dtype != DType::Float16) return hipErrorInvalidValue;
        if (!input || !weight || !output || rows <= 0 || columns <= 0 || inner <= 0) return hipErrorInvalidValue;
        if (dtype == DType::Float32) return launch_linear<float, Activate, Transposed>(input, weight, bias, output, rows, columns, inner, stream);
        return launch_linear<__half, Activate, Transposed>(input, weight, bias, output, rows, columns, inner, stream);
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int linear_act_fast(const void *input, const void *weight, const void *bias, void *output, int rows, int columns,
        int inner, ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    return dispatch_linear<float, true, false>(input, weight, bias, output, rows, columns, inner, dtype, stream);
}

extern "C" ROCMKERNELS_EXPORT int linear_fast(const void *input, const void *weight, const void *bias, void *output, int rows, int columns,
        int inner, ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    return dispatch_linear<float, false, false>(input, weight, bias, output, rows, columns, inner, dtype, stream);
}

extern "C" ROCMKERNELS_EXPORT int linear_act_nt(const void *input, const void *weight, const void *bias, void *output, int rows, int columns,
        int inner, ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    return dispatch_linear<float, true, true>(input, weight, bias, output, rows, columns, inner, dtype, stream);
}

extern "C" ROCMKERNELS_EXPORT int linear_nt(const void *input, const void *weight, const void *bias, void *output, int rows, int columns,
        int inner, ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    return dispatch_linear<float, false, true>(input, weight, bias, output, rows, columns, inner, dtype, stream);
}

// Same K-major fp16 GEMM with an fp32 output, for weight gradients.
// Same as linear_act_fast and additionally writes sigmoid(z) so the backward
// never has to evaluate exp again.
extern "C" ROCMKERNELS_EXPORT int linear_act_fast_s(const void *input, const void *weight, const void *bias, void *output, void *sigmoidOut,
        int rows, int columns, int inner, hipStream_t stream) {
    if (!input || !weight || !output || !sigmoidOut || rows <= 0 || columns <= 0 || inner <= 0) return hipErrorInvalidValue;
    if (columns > 128) {
        const dim3 grid((columns + 63) / 64, (rows + 127) / 128);
        hipLaunchKernelGGL((linear_act_fast_half_wide_kernel<128, 64, 8, 4, 64, true, false, true>), grid, dim3(kThreadsX, kThreadsY), 0, stream,
                static_cast<const __half *>(input), static_cast<const __half *>(weight), static_cast<const __half *>(bias),
                static_cast<__half *>(output), static_cast<__half *>(sigmoidOut), rows, columns, inner);
    } else {
        const dim3 grid((columns + 127) / 128, (rows + 63) / 64);
        hipLaunchKernelGGL((linear_act_fast_half_wide_kernel<64, 128, 4, 8, 32, true, false, true>), grid, dim3(kThreadsX, kThreadsY), 0, stream,
                static_cast<const __half *>(input), static_cast<const __half *>(weight), static_cast<const __half *>(bias),
                static_cast<__half *>(output), static_cast<__half *>(sigmoidOut), rows, columns, inner);
    }
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int linear_fast_fp32_out(const void *input, const void *weight, const void *bias, void *output, int rows,
        int columns, int inner, hipStream_t stream) {
    if (!input || !weight || !output || rows <= 0 || columns <= 0 || inner <= 0) return hipErrorInvalidValue;
    if (columns > 128) {
        const dim3 grid((columns + 63) / 64, (rows + 127) / 128);
        hipLaunchKernelGGL((linear_act_fast_half_wide_kernel<128, 64, 8, 4, 64, false, true, false>), grid, dim3(kThreadsX, kThreadsY), 0, stream,
                static_cast<const __half *>(input), static_cast<const __half *>(weight), static_cast<const __half *>(bias), output, nullptr, rows,
                columns, inner);
    } else {
        const dim3 grid((columns + 127) / 128, (rows + 63) / 64);
        hipLaunchKernelGGL((linear_act_fast_half_wide_kernel<64, 128, 4, 8, 32, false, true, false>), grid, dim3(kThreadsX, kThreadsY), 0, stream,
                static_cast<const __half *>(input), static_cast<const __half *>(weight), static_cast<const __half *>(bias), output, nullptr, rows,
                columns, inner);
    }
    return hipGetLastError();
}





















