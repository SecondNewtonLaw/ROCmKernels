// Ported from llama.cpp ggml-cuda softmax.cu (MIT, Copyright (c) 2023-2026 The ggml authors).
#include <ROCmKernels/Kernels/Generic/Softmax.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <math.h>

namespace {
    constexpr int kThreads = 256;
    constexpr int kWarps = kThreads / 32;

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

        // The mask becomes a template parameter so the loops stay branch-free.
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
