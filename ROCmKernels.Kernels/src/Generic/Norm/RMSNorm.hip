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

    __device__ __forceinline__ float wave_all_sum(float value) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor(value, offset, 32);
        return value;
    }

    // One wave per row while the row fits in registers: no shared memory or barriers.
    template<typename T, bool HasResidual, bool HasScale, int PerLane>
    __global__ __launch_bounds__(kThreads) void rms_norm_forward_wave_kernel(const T *x, const T *residual, const T *scale, T *output,
            float *rstd, const int rows, const int columns, const float epsilon) {
        const int lane = threadIdx.x % 32;
        const int row = blockIdx.x * kWarps + threadIdx.x / 32;
        if (row >= rows) return;
        const std::size_t base = std::size_t(row) * columns;

        float values[PerLane];
        float squareSum = 0.0f;
        #pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            const int column = i * 32 + lane;
            float value = 0.0f;
            if (column < columns) {
                value = ROCmKernels::Kernels::LoadScalar(x + base + column);
                if constexpr (HasResidual) value += ROCmKernels::Kernels::LoadScalar(residual + base + column);
            }
            values[i] = value;
            squareSum += value * value;
        }
        const float inverse = rsqrtf(wave_all_sum(squareSum) / static_cast<float>(columns) + epsilon);

        #pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            const int column = i * 32 + lane;
            if (column >= columns) break;
            float value = values[i] * inverse;
            if constexpr (HasScale) value *= ROCmKernels::Kernels::LoadScalar(scale + column);
            ROCmKernels::Kernels::StoreScalar(output + base + column, value);
        }

        if (rstd != nullptr && lane == 0) rstd[row] = inverse;
    }

    template<typename T, bool HasResidual, bool HasScale>
    __global__ void rms_norm_forward_staged_kernel(const T *x, const T *residual, const T *scale, T *output, float *rstd, const int columns,
            const float epsilon) {
        __shared__ float staged[kStageColumns];
        __shared__ float scratch[kWarps];

        const std::size_t base = std::size_t(blockIdx.x) * columns;
        const bool paired = columns % 2 == 0;

        float squareSum = 0.0f;
        if (paired) {
            for (int column = threadIdx.x * 2; column < columns; column += kThreads * 2) {
                float2 value = ROCmKernels::Kernels::LoadPair(x + base + column);
                if constexpr (HasResidual) {
                    const float2 added = ROCmKernels::Kernels::LoadPair(residual + base + column);
                    value.x += added.x;
                    value.y += added.y;
                }
                *reinterpret_cast<float2 *>(staged + column) = value;
                squareSum += value.x * value.x + value.y * value.y;
            }
        } else {
            for (int column = threadIdx.x; column < columns; column += kThreads) {
                float value = ROCmKernels::Kernels::LoadScalar(x + base + column);
                if constexpr (HasResidual) value += ROCmKernels::Kernels::LoadScalar(residual + base + column);
                staged[column] = value;
                squareSum += value * value;
            }
        }

        const float mean = block_sum(squareSum, scratch) / static_cast<float>(columns);
        const float inverse = rsqrtf(mean + epsilon);

        if (paired) {
            for (int column = threadIdx.x * 2; column < columns; column += kThreads * 2) {
                float2 value = *reinterpret_cast<const float2 *>(staged + column);
                value.x *= inverse;
                value.y *= inverse;
                if constexpr (HasScale) {
                    const float2 factor = ROCmKernels::Kernels::LoadPair(scale + column);
                    value.x *= factor.x;
                    value.y *= factor.y;
                }
                ROCmKernels::Kernels::StorePair(output + base + column, value);
            }
            if (rstd != nullptr && threadIdx.x == 0) rstd[blockIdx.x] = inverse;
            return;
        }
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

        const auto wave = [&]<int PerLane>() {
            const dim3 waveGrid((rows + kWarps - 1) / kWarps);
            const auto launch = [&](auto kernel) {
                hipLaunchKernelGGL(kernel, waveGrid, block, 0, stream, static_cast<const T *>(x), static_cast<const T *>(residual),
                        static_cast<const T *>(scale), static_cast<T *>(output), rstd, rows, columns, epsilon);
            };
            if (residual != nullptr && scale != nullptr) launch(rms_norm_forward_wave_kernel<T, true, true, PerLane>);
            else if (residual != nullptr) launch(rms_norm_forward_wave_kernel<T, true, false, PerLane>);
            else if (scale != nullptr) launch(rms_norm_forward_wave_kernel<T, false, true, PerLane>);
            else launch(rms_norm_forward_wave_kernel<T, false, false, PerLane>);
            return hipGetLastError();
        };
        if (columns <= 128) return wave.template operator()<4>();
        if (columns <= 256) return wave.template operator()<8>();
        // Measured: at 1024 columns one wave per row is slower than the staged kernel.
        if (columns <= 512) return wave.template operator()<16>();

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
