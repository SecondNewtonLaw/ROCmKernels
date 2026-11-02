// Ported from llama.cpp ggml-cuda softmax.cu (MIT, Copyright (c) 2023-2026 The ggml authors).
#include <ROCmKernels/Kernels/Generic/Softmax.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <math.h>

namespace {
    constexpr int kThreads = 256;
    constexpr int kWarps = kThreads / 32;
    // Rows up to this width are staged in shared memory, so the row is read from
    // global memory once instead of once per pass.
    constexpr int kStageColumns = 4096;

    __device__ __forceinline__ float wave_max(float value) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_down(value, offset, 32));
        return value;
    }

    __device__ __forceinline__ float wave_sum(float value) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down(value, offset, 32);
        return value;
    }

    // Reduces value across the block and returns the total to every thread.
    __device__ __forceinline__ float block_max(float value, float *scratch) {
        const int lane = threadIdx.x % 32;
        const int wave = threadIdx.x / 32;
        value = wave_max(value);
        if (lane == 0) scratch[wave] = value;
        __syncthreads();
        if (wave == 0) {
            float total = lane < kWarps ? scratch[lane] : -INFINITY;
            total = wave_max(total);
            if (lane == 0) scratch[0] = total;
        }
        __syncthreads();
        return scratch[0];
    }

    __device__ __forceinline__ float block_sum(float value, float *scratch) {
        const int lane = threadIdx.x % 32;
        const int wave = threadIdx.x / 32;
        value = wave_sum(value);
        if (lane == 0) scratch[wave] = value;
        __syncthreads();
        if (wave == 0) {
            float total = lane < kWarps ? scratch[lane] : 0.0f;
            total = wave_sum(total);
            if (lane == 0) scratch[0] = total;
        }
        __syncthreads();
        return scratch[0];
    }

    __device__ __forceinline__ float wave_all_max(float value) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor(value, offset, 32));
        return value;
    }

    __device__ __forceinline__ float wave_all_sum(float value) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor(value, offset, 32);
        return value;
    }

    // One wave per row while the row fits in registers: no shared memory or barriers.
    template<typename T, bool HasMask, int PerLane>
    __global__ __launch_bounds__(kThreads) void softmax_forward_wave_kernel(const T *input, const bool *mask, T *output, const int rows,
            const int columns, const float scale) {
        const int lane = threadIdx.x % 32;
        const int row = blockIdx.x * kWarps + threadIdx.x / 32;
        if (row >= rows) return;
        const std::size_t base = std::size_t(row) * columns;

        float values[PerLane];
        float maximum = -INFINITY;
        #pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            const int column = i * 32 + lane;
            float value = -INFINITY;
            if (column < columns) {
                value = ROCmKernels::Kernels::LoadScalar(input + base + column) * scale;
                if constexpr (HasMask) {
                    if (!mask[base + column]) value = -INFINITY;
                }
            }
            values[i] = value;
            maximum = fmaxf(maximum, value);
        }
        maximum = wave_all_max(maximum);

        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            values[i] = values[i] != -INFINITY ? __expf(values[i] - maximum) : 0.0f;
            sum += values[i];
        }
        sum = wave_all_sum(sum);

        const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
        #pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            const int column = i * 32 + lane;
            if (column >= columns) break;
            ROCmKernels::Kernels::StoreScalar(output + base + column, values[i] * inverse);
        }
    }

    template<typename T, bool HasMask>
    __global__ void softmax_forward_staged_kernel(const T *input, const bool *mask, T *output, const int columns, const float scale) {
        // One global read per element; the max, sum and output passes read the row from LDS,
        // where reads are conflict-free and much cheaper than a second global pass.
        __shared__ float staged[kStageColumns];
        __shared__ float maxScratch[kWarps];
        __shared__ float sumScratch[kWarps];

        const std::size_t base = std::size_t(blockIdx.x) * columns;
        const bool paired = columns % 2 == 0;

        if (paired) {
            for (int column = threadIdx.x * 2; column < columns; column += kThreads * 2) {
                float2 value = ROCmKernels::Kernels::LoadPair(input + base + column);
                value.x *= scale;
                value.y *= scale;
                if constexpr (HasMask) {
                    if (!mask[base + column]) value.x = -INFINITY;
                    if (!mask[base + column + 1]) value.y = -INFINITY;
                }
                *reinterpret_cast<float2 *>(staged + column) = value;
            }
        } else {
            for (int column = threadIdx.x; column < columns; column += kThreads) {
                float value = ROCmKernels::Kernels::LoadScalar(input + base + column) * scale;
                if constexpr (HasMask) {
                    if (!mask[base + column]) value = -INFINITY;
                }
                staged[column] = value;
            }
        }
        __syncthreads();

        float maximum = -INFINITY;
        for (int column = threadIdx.x; column < columns; column += kThreads) maximum = fmaxf(maximum, staged[column]);
        maximum = block_max(maximum, maxScratch);

        float sum = 0.0f;
        for (int column = threadIdx.x; column < columns; column += kThreads) {
            const float value = staged[column];
            if (value != -INFINITY) sum += __expf(value - maximum);
        }
        sum = block_sum(sum, sumScratch);

        const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
        if (paired) {
            for (int column = threadIdx.x * 2; column < columns; column += kThreads * 2) {
                const float2 value = *reinterpret_cast<const float2 *>(staged + column);
                const float2 probability = sum > 0.0f ? make_float2(__expf(value.x - maximum) * inverse, __expf(value.y - maximum) * inverse)
                                                      : make_float2(0.0f, 0.0f);
                ROCmKernels::Kernels::StorePair(output + base + column, probability);
            }
            return;
        }
        for (int column = threadIdx.x; column < columns; column += kThreads) {
            const float value = staged[column];
            const float probability = sum > 0.0f ? __expf(value - maximum) * inverse : 0.0f;
            ROCmKernels::Kernels::StoreScalar(output + base + column, probability);
        }
    }

    template<typename T, bool HasMask>
    __global__ void softmax_forward_kernel(const T *input, const bool *mask, T *output, const int columns, const float scale) {
        __shared__ float maxScratch[kWarps];
        __shared__ float sumScratch[kWarps];

        const std::size_t row = blockIdx.x;
        const std::size_t base = row * columns;

        float maximum = -INFINITY;
        for (int column = threadIdx.x; column < columns; column += kThreads) {
            if constexpr (HasMask) {
                if (!mask[base + column]) continue;
            }
            maximum = fmaxf(maximum, ROCmKernels::Kernels::LoadScalar(input + base + column) * scale);
        }
        maximum = block_max(maximum, maxScratch);

        float sum = 0.0f;
        for (int column = threadIdx.x; column < columns; column += kThreads) {
            if constexpr (HasMask) {
                if (!mask[base + column]) continue;
            }
            const float value = ROCmKernels::Kernels::LoadScalar(input + base + column) * scale;
            // A fully masked row leaves maximum at -INFINITY; skipping keeps the sum at zero instead of NaN.
            if (value != -INFINITY) sum += __expf(value - maximum);
        }
        sum = block_sum(sum, sumScratch);

        const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int column = threadIdx.x; column < columns; column += kThreads) {
            float value = ROCmKernels::Kernels::LoadScalar(input + base + column) * scale;
            if constexpr (HasMask) {
                if (!mask[base + column]) value = -INFINITY;
            }
            const float probability = sum > 0.0f ? __expf(value - maximum) * inverse : 0.0f;
            ROCmKernels::Kernels::StoreScalar(output + base + column, probability);
        }
    }

    template<typename T>
    int launch_forward(const void *input, const bool *mask, void *output, const int rows, const int columns, const float scale,
            hipStream_t stream) {
        const dim3 grid(rows);
        const dim3 block(kThreads);

        const auto wave = [&](auto masked, auto unmasked) {
            hipLaunchKernelGGL(mask != nullptr ? masked : unmasked, dim3((rows + kWarps - 1) / kWarps), block, 0, stream,
                    static_cast<const T *>(input), mask, static_cast<T *>(output), rows, columns, scale);
            return hipGetLastError();
        };
        if (columns <= 128) return wave(softmax_forward_wave_kernel<T, true, 4>, softmax_forward_wave_kernel<T, false, 4>);
        if (columns <= 256) return wave(softmax_forward_wave_kernel<T, true, 8>, softmax_forward_wave_kernel<T, false, 8>);
        if (columns <= 512) return wave(softmax_forward_wave_kernel<T, true, 16>, softmax_forward_wave_kernel<T, false, 16>);
        if (columns <= 1024) return wave(softmax_forward_wave_kernel<T, true, 32>, softmax_forward_wave_kernel<T, false, 32>);

        // The mask becomes a template parameter so the loops stay branch-free.
        if (columns <= kStageColumns) {
            if (mask != nullptr) {
                hipLaunchKernelGGL((softmax_forward_staged_kernel<T, true>), grid, block, 0, stream, static_cast<const T *>(input), mask,
                        static_cast<T *>(output), columns, scale);
            } else {
                hipLaunchKernelGGL((softmax_forward_staged_kernel<T, false>), grid, block, 0, stream, static_cast<const T *>(input), mask,
                        static_cast<T *>(output), columns, scale);
            }
            return hipGetLastError();
        }

        if (mask != nullptr) {
            hipLaunchKernelGGL((softmax_forward_kernel<T, true>), grid, block, 0, stream, static_cast<const T *>(input), mask,
                    static_cast<T *>(output), columns, scale);
        } else {
            hipLaunchKernelGGL((softmax_forward_kernel<T, false>), grid, block, 0, stream, static_cast<const T *>(input), mask,
                    static_cast<T *>(output), columns, scale);
        }

        return hipGetLastError();
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int softmax_forward(const void *input, const bool *mask, void *output, const int rows, const int columns,
        const float scale, ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    if (!input || !output || rows <= 0 || columns <= 0) return hipErrorInvalidValue;

    switch (dtype) {
        case ROCmKernels::Kernels::DType::Float32:
            return launch_forward<float>(input, mask, output, rows, columns, scale, stream);
        case ROCmKernels::Kernels::DType::Float16:
            return launch_forward<__half>(input, mask, output, rows, columns, scale, stream);
    }

    return hipErrorInvalidValue;
}
