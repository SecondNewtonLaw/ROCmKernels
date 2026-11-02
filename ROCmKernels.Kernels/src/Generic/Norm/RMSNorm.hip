// Fused residual + RMS norm + optional scale, ported from llama.cpp ggml-cuda norm.cu (MIT, Copyright (c) 2023-2026 The ggml authors).
#include <ROCmKernels/Kernels/Generic/RMSNorm.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <math.h>

namespace {
    constexpr int kThreads = 256;
    constexpr int kWarps = kThreads / 32;
    // Rows up to this width are staged in shared memory so the second pass reads
    // LDS instead of repeating the global loads.
    constexpr int kStageColumns = 4096;

    __device__ __forceinline__ float wave_sum(float value) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down(value, offset, 32);
        return value;
    }

    // Reduces value across the block and returns the total to every thread.
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

    template<typename T, bool HasResidual, bool HasScale>
    __global__ void rms_norm_forward_staged_kernel(const T *x, const T *residual, const T *scale, T *output, float *rstd, const int columns,
            const float epsilon) {
        __shared__ float staged[kStageColumns];
        __shared__ float scratch[kWarps];

        const std::size_t base = std::size_t(blockIdx.x) * columns;

        float squareSum = 0.0f;
        for (int column = threadIdx.x; column < columns; column += kThreads) {
            float value = ROCmKernels::Kernels::LoadScalar(x + base + column);
            if constexpr (HasResidual) value += ROCmKernels::Kernels::LoadScalar(residual + base + column);
            staged[column] = value;
            squareSum += value * value;
        }

        const float mean = block_sum(squareSum, scratch) / static_cast<float>(columns);
        const float inverse = rsqrtf(mean + epsilon);

        for (int column = threadIdx.x; column < columns; column += kThreads) {
            float value = staged[column] * inverse;
            if constexpr (HasScale) value *= ROCmKernels::Kernels::LoadScalar(scale + column);
            ROCmKernels::Kernels::StoreScalar(output + base + column, value);
        }

        if (rstd != nullptr && threadIdx.x == 0) rstd[blockIdx.x] = inverse;
    }

    template<typename T, bool HasResidual, bool HasScale>
    __global__ void rms_norm_forward_kernel(const T *x, const T *residual, const T *scale, T *output, float *rstd, const int columns,
            const float epsilon) {
        __shared__ float scratch[kWarps];

        const std::size_t row = blockIdx.x;
        const std::size_t base = row * columns;

        // s = x + residual is formed twice so the second pass only reads and writes.
        float squareSum = 0.0f;
        for (int column = threadIdx.x; column < columns; column += kThreads) {
            float value = ROCmKernels::Kernels::LoadScalar(x + base + column);
            if constexpr (HasResidual) value += ROCmKernels::Kernels::LoadScalar(residual + base + column);
            squareSum += value * value;
        }

        const float mean = block_sum(squareSum, scratch) / static_cast<float>(columns);
        const float inverse = rsqrtf(mean + epsilon);

        for (int column = threadIdx.x; column < columns; column += kThreads) {
            float value = ROCmKernels::Kernels::LoadScalar(x + base + column);
            if constexpr (HasResidual) value += ROCmKernels::Kernels::LoadScalar(residual + base + column);
            value *= inverse;
            if constexpr (HasScale) value *= ROCmKernels::Kernels::LoadScalar(scale + column);
            ROCmKernels::Kernels::StoreScalar(output + base + column, value);
        }

        if (rstd != nullptr && threadIdx.x == 0) rstd[row] = inverse;
    }

    template<typename T>
    int launch_forward(const void *x, const void *residual, const void *scale, void *output, float *rstd, const int rows, const int columns,
            const float epsilon, hipStream_t stream) {
        const dim3 grid(rows);
        const dim3 block(kThreads);

        // The optional terms become template parameters so the loops stay branch-free.
        if (columns <= kStageColumns) {
            if (residual != nullptr && scale != nullptr) {
                hipLaunchKernelGGL((rms_norm_forward_staged_kernel<T, true, true>), grid, block, 0, stream, static_cast<const T *>(x),
                        static_cast<const T *>(residual), static_cast<const T *>(scale), static_cast<T *>(output), rstd, columns, epsilon);
            } else if (residual != nullptr) {
                hipLaunchKernelGGL((rms_norm_forward_staged_kernel<T, true, false>), grid, block, 0, stream, static_cast<const T *>(x),
                        static_cast<const T *>(residual), static_cast<const T *>(scale), static_cast<T *>(output), rstd, columns, epsilon);
            } else if (scale != nullptr) {
                hipLaunchKernelGGL((rms_norm_forward_staged_kernel<T, false, true>), grid, block, 0, stream, static_cast<const T *>(x),
                        static_cast<const T *>(residual), static_cast<const T *>(scale), static_cast<T *>(output), rstd, columns, epsilon);
            } else {
                hipLaunchKernelGGL((rms_norm_forward_staged_kernel<T, false, false>), grid, block, 0, stream, static_cast<const T *>(x),
                        static_cast<const T *>(residual), static_cast<const T *>(scale), static_cast<T *>(output), rstd, columns, epsilon);
            }
            return hipGetLastError();
        }

        if (residual != nullptr && scale != nullptr) {
            hipLaunchKernelGGL((rms_norm_forward_kernel<T, true, true>), grid, block, 0, stream, static_cast<const T *>(x),
                    static_cast<const T *>(residual), static_cast<const T *>(scale), static_cast<T *>(output), rstd, columns, epsilon);
        } else if (residual != nullptr) {
            hipLaunchKernelGGL((rms_norm_forward_kernel<T, true, false>), grid, block, 0, stream, static_cast<const T *>(x),
                    static_cast<const T *>(residual), static_cast<const T *>(scale), static_cast<T *>(output), rstd, columns, epsilon);
        } else if (scale != nullptr) {
            hipLaunchKernelGGL((rms_norm_forward_kernel<T, false, true>), grid, block, 0, stream, static_cast<const T *>(x),
                    static_cast<const T *>(residual), static_cast<const T *>(scale), static_cast<T *>(output), rstd, columns, epsilon);
        } else {
            hipLaunchKernelGGL((rms_norm_forward_kernel<T, false, false>), grid, block, 0, stream, static_cast<const T *>(x),
                    static_cast<const T *>(residual), static_cast<const T *>(scale), static_cast<T *>(output), rstd, columns, epsilon);
        }

        return hipGetLastError();
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int rms_norm_forward(const void *x, const void *residual, const void *scale, void *output, float *rstd,
        const int rows, const int columns, const float epsilon, ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    if (!x || !output || rows <= 0 || columns <= 0 || epsilon < 0.0f) return hipErrorInvalidValue;

    switch (dtype) {
        case ROCmKernels::Kernels::DType::Float32:
            return launch_forward<float>(x, residual, scale, output, rstd, rows, columns, epsilon, stream);
        case ROCmKernels::Kernels::DType::Float16:
            return launch_forward<__half>(x, residual, scale, output, rstd, rows, columns, epsilon, stream);
    }

    return hipErrorInvalidValue;
}
