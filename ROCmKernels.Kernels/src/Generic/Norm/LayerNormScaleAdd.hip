#include <ROCmKernels/Kernels/Generic/LayerNormScaleAdd.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <math.h>

namespace {
    constexpr int kThreads = 256;
    constexpr int kWarps = kThreads / 32;
    // Rows up to this width keep the fused x + scale*z value in shared memory,
    // so the normalization pass does not repeat the three global loads.
    constexpr int kStageColumns = 4096;
    // Enough blocks to fill 36 CUs a few times over while each wave still covers several rows.
    constexpr int kBackwardBlocks = 144;

    __device__ __forceinline__ float wave_sum(float value) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down(value, offset, 32);
        return value;
    }

    __device__ __forceinline__ float wave_all_sum(float value) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor(value, offset, 32);
        return value;
    }

    // Reduces both values across the block and returns the totals to every thread.
    // scratch holds one partial per value per warp.
    __device__ __forceinline__ void block_reduce_pair(float &first, float &second, float *scratch) {
        const int lane = threadIdx.x % 32;
        const int wave = threadIdx.x / 32;
        first = wave_sum(first);
        second = wave_sum(second);
        if (lane == 0) {
            scratch[wave] = first;
            scratch[kWarps + wave] = second;
        }
        __syncthreads();
        if (wave == 0) {
            float firstTotal = lane < kWarps ? scratch[lane] : 0.0f;
            float secondTotal = lane < kWarps ? scratch[kWarps + lane] : 0.0f;
            firstTotal = wave_sum(firstTotal);
            secondTotal = wave_sum(secondTotal);
            if (lane == 0) {
                scratch[0] = firstTotal;
                scratch[kWarps] = secondTotal;
            }
        }
        __syncthreads();
        first = scratch[0];
        second = scratch[kWarps];
    }

    // One wave per row while the row fits in registers: no shared memory or barriers, and every wave works on narrow rows.
    template<typename T, int PerLane>
    __global__ __launch_bounds__(kThreads) void layernorm_scale_add_forward_wave_kernel(const T *x, const T *z, const T *scale,
            const T *weight, const T *bias, T *output, float *statistics, const int rows, const int columns, const float epsilon) {
        const int lane = threadIdx.x % 32;
        const int row = blockIdx.x * kWarps + threadIdx.x / 32;
        if (row >= rows) return;
        const std::size_t base = std::size_t(row) * columns;

        float values[PerLane];
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            const int column = i * 32 + lane;
            values[i] = column < columns
                    ? ROCmKernels::Kernels::LoadScalar(x + base + column) +
                      ROCmKernels::Kernels::LoadScalar(scale + column) * ROCmKernels::Kernels::LoadScalar(z + base + column)
                    : 0.0f;
            sum += values[i];
        }
        const float mean = wave_all_sum(sum) / static_cast<float>(columns);

        float squareSum = 0.0f;
        #pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            const float centered = i * 32 + lane < columns ? values[i] - mean : 0.0f;
            squareSum += centered * centered;
        }
        const float rstd = rsqrtf(wave_all_sum(squareSum) / static_cast<float>(columns) + epsilon);

        #pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            const int column = i * 32 + lane;
            if (column >= columns) break;
            float normalized = (values[i] - mean) * rstd;
            if (weight != nullptr) normalized *= ROCmKernels::Kernels::LoadScalar(weight + column);
            if (bias != nullptr) normalized += ROCmKernels::Kernels::LoadScalar(bias + column);
            ROCmKernels::Kernels::StoreScalar(output + base + column, normalized);
        }

        if (lane == 0) {
            statistics[row * 2] = mean;
            statistics[row * 2 + 1] = rstd;
        }
    }

    template<typename T>
    __global__ void layernorm_scale_add_forward_staged_kernel(const T *x, const T *z, const T *scale, const T *weight, const T *bias,
            T *output, float *statistics, const int columns, const float epsilon) {
        __shared__ float staged[kStageColumns];
        __shared__ float scratch[2 * kWarps];

        const std::size_t base = std::size_t(blockIdx.x) * columns;

        float sum = 0.0f;
        float squareSum = 0.0f;
        for (int column = threadIdx.x; column < columns; column += kThreads) {
            const float value = ROCmKernels::Kernels::LoadScalar(x + base + column) +
                                ROCmKernels::Kernels::LoadScalar(scale + column) * ROCmKernels::Kernels::LoadScalar(z + base + column);
            staged[column] = value;
            sum += value;
            squareSum += value * value;
        }
        block_reduce_pair(sum, squareSum, scratch);

        const float mean = sum / static_cast<float>(columns);
        const float variance = fmaxf(squareSum / static_cast<float>(columns) - mean * mean, 0.0f);
        const float rstd = rsqrtf(variance + epsilon);

        for (int column = threadIdx.x; column < columns; column += kThreads) {
            float normalized = (staged[column] - mean) * rstd;
            if (weight != nullptr) normalized *= ROCmKernels::Kernels::LoadScalar(weight + column);
            if (bias != nullptr) normalized += ROCmKernels::Kernels::LoadScalar(bias + column);
            ROCmKernels::Kernels::StoreScalar(output + base + column, normalized);
        }

        if (threadIdx.x == 0) {
            statistics[blockIdx.x * 2] = mean;
            statistics[blockIdx.x * 2 + 1] = rstd;
        }
    }

    template<typename T>
    __global__ void layernorm_scale_add_forward_kernel(const T *x, const T *z, const T *scale, const T *weight, const T *bias, T *output,
            float *statistics, const int columns, const float epsilon) {
        __shared__ float scratch[2 * kWarps];

        const std::size_t row = blockIdx.x;
        const std::size_t base = row * columns;

        // Both moments come from one pass over the inputs; the second pass only writes.
        float sum = 0.0f;
        float squareSum = 0.0f;
        for (int column = threadIdx.x; column < columns; column += kThreads) {
            const float value = ROCmKernels::Kernels::LoadScalar(x + base + column) +
                                ROCmKernels::Kernels::LoadScalar(scale + column) * ROCmKernels::Kernels::LoadScalar(z + base + column);
            sum += value;
            squareSum += value * value;
        }
        block_reduce_pair(sum, squareSum, scratch);

        const float mean = sum / static_cast<float>(columns);
        // Clamped because the two-moment form can go slightly negative on constant rows.
        const float variance = fmaxf(squareSum / static_cast<float>(columns) - mean * mean, 0.0f);
        const float rstd = rsqrtf(variance + epsilon);

        for (int column = threadIdx.x; column < columns; column += kThreads) {
            const float value = ROCmKernels::Kernels::LoadScalar(x + base + column) +
                                ROCmKernels::Kernels::LoadScalar(scale + column) * ROCmKernels::Kernels::LoadScalar(z + base + column);
            float normalized = (value - mean) * rstd;
            if (weight != nullptr) normalized *= ROCmKernels::Kernels::LoadScalar(weight + column);
            if (bias != nullptr) normalized += ROCmKernels::Kernels::LoadScalar(bias + column);
            ROCmKernels::Kernels::StoreScalar(output + base + column, normalized);
        }

        if (threadIdx.x == 0) {
            statistics[row * 2] = mean;
            statistics[row * 2 + 1] = rstd;
        }
    }

    // Waves stride over rows and keep the parameter gradients in registers, so the global atomics drop from three per
    // element to three per column per block.
    template<typename T, int PerLane>
    __global__ __launch_bounds__(kThreads) void layernorm_scale_add_backward_wave_kernel(const T *x, const T *z, const T *scale,
            const T *weight, const float *statistics, const T *upstream, T *gradientX, T *gradientZ, float *gradientScale,
            float *gradientWeight, float *gradientBias, const int rows, const int columns) {
        __shared__ float partial[3][PerLane * 32];
        for (int index = threadIdx.x; index < 3 * PerLane * 32; index += kThreads) (&partial[0][0])[index] = 0.0f;
        __syncthreads();

        const int lane = threadIdx.x % 32;
        float scales[PerLane], weights[PerLane];
        float scaleGradient[PerLane] = {}, weightGradient[PerLane] = {}, biasGradient[PerLane] = {};
        #pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            const int column = i * 32 + lane;
            scales[i] = column < columns ? ROCmKernels::Kernels::LoadScalar(scale + column) : 0.0f;
            weights[i] = weight != nullptr && column < columns ? ROCmKernels::Kernels::LoadScalar(weight + column) : 1.0f;
        }

        for (int row = blockIdx.x * kWarps + threadIdx.x / 32; row < rows; row += gridDim.x * kWarps) {
            const std::size_t base = std::size_t(row) * columns;
            const float mean = statistics[row * 2];
            const float rstd = statistics[row * 2 + 1];

            float normalized[PerLane], weighted[PerLane], zValues[PerLane];
            float plainSum = 0.0f;
            float centeredSum = 0.0f;
            #pragma unroll
            for (int i = 0; i < PerLane; ++i) {
                const int column = i * 32 + lane;
                normalized[i] = weighted[i] = zValues[i] = 0.0f;
                if (column >= columns) continue;
                zValues[i] = ROCmKernels::Kernels::LoadScalar(z + base + column);
                normalized[i] = (ROCmKernels::Kernels::LoadScalar(x + base + column) + scales[i] * zValues[i] - mean) * rstd;
                const float dy = ROCmKernels::Kernels::LoadScalar(upstream + base + column);
                weighted[i] = dy * weights[i];
                plainSum += weighted[i];
                centeredSum += weighted[i] * normalized[i];
                weightGradient[i] += dy * normalized[i];
                biasGradient[i] += dy;
            }
            const float plainMean = wave_all_sum(plainSum) / static_cast<float>(columns);
            const float centeredMean = wave_all_sum(centeredSum) / static_cast<float>(columns);

            #pragma unroll
            for (int i = 0; i < PerLane; ++i) {
                const int column = i * 32 + lane;
                if (column >= columns) break;
                const float ds = rstd * (weighted[i] - plainMean - normalized[i] * centeredMean);
                ROCmKernels::Kernels::StoreScalar(gradientX + base + column, ds);
                ROCmKernels::Kernels::StoreScalar(gradientZ + base + column, ds * scales[i]);
                scaleGradient[i] += ds * zValues[i];
            }
        }

        #pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            const int column = i * 32 + lane;
            if (column >= columns) break;
            atomicAdd(&partial[0][column], scaleGradient[i]);
            atomicAdd(&partial[1][column], weightGradient[i]);
            atomicAdd(&partial[2][column], biasGradient[i]);
        }
        __syncthreads();
        for (int column = threadIdx.x; column < columns; column += kThreads) {
            if (gradientScale != nullptr) atomicAdd(gradientScale + column, partial[0][column]);
            if (gradientWeight != nullptr) atomicAdd(gradientWeight + column, partial[1][column]);
            if (gradientBias != nullptr) atomicAdd(gradientBias + column, partial[2][column]);
        }
    }

    // PerThread > 0: blocks stride over rows and keep parameter gradients in registers; 0: one block per row with per-element atomics.
    template<typename T, int PerThread>
    __global__ void layernorm_scale_add_backward_kernel(const T *x, const T *z, const T *scale, const T *weight, const float *statistics,
            const T *upstream, T *gradientX, T *gradientZ, float *gradientScale, float *gradientWeight, float *gradientBias,
            const int rows, const int columns) {
        __shared__ float scratch[2 * kWarps];

        float scaleGradient[PerThread > 0 ? PerThread : 1] = {};
        float weightGradient[PerThread > 0 ? PerThread : 1] = {};
        float biasGradient[PerThread > 0 ? PerThread : 1] = {};

        for (std::size_t row = blockIdx.x; row < static_cast<std::size_t>(rows); row += PerThread > 0 ? gridDim.x : rows) {
            const std::size_t base = row * columns;
            const float mean = statistics[row * 2];
            const float rstd = statistics[row * 2 + 1];

            float plainSum = 0.0f;
            float centeredSum = 0.0f;
            for (int column = threadIdx.x; column < columns; column += kThreads) {
                const float value = ROCmKernels::Kernels::LoadScalar(x + base + column) +
                                    ROCmKernels::Kernels::LoadScalar(scale + column) * ROCmKernels::Kernels::LoadScalar(z + base + column);
                const float normalized = (value - mean) * rstd;
                float dy = ROCmKernels::Kernels::LoadScalar(upstream + base + column);
                if (weight != nullptr) dy *= ROCmKernels::Kernels::LoadScalar(weight + column);
                plainSum += dy;
                centeredSum += dy * normalized;
            }
            block_reduce_pair(plainSum, centeredSum, scratch);

            const float plainMean = plainSum / static_cast<float>(columns);
            const float centeredMean = centeredSum / static_cast<float>(columns);

            // Returns this element's contributions to the scale, weight and bias gradients.
            const auto backward = [&](const int column, float &scaleTerm, float &weightTerm, float &biasTerm) {
                const float zValue = ROCmKernels::Kernels::LoadScalar(z + base + column);
                const float scaleValue = ROCmKernels::Kernels::LoadScalar(scale + column);
                const float normalized = (ROCmKernels::Kernels::LoadScalar(x + base + column) + scaleValue * zValue - mean) * rstd;
                const float dy = ROCmKernels::Kernels::LoadScalar(upstream + base + column);
                float dyWeighted = dy;
                if (weight != nullptr) dyWeighted *= ROCmKernels::Kernels::LoadScalar(weight + column);

                const float ds = rstd * (dyWeighted - plainMean - normalized * centeredMean);
                ROCmKernels::Kernels::StoreScalar(gradientX + base + column, ds);
                ROCmKernels::Kernels::StoreScalar(gradientZ + base + column, ds * scaleValue);
                scaleTerm = ds * zValue;
                weightTerm = dy * normalized;
                biasTerm = dy;
            };

            if constexpr (PerThread > 0) {
                #pragma unroll
                for (int k = 0; k < PerThread; ++k) {
                    const int column = threadIdx.x + k * kThreads;
                    if (column >= columns) break;
                    float scaleTerm, weightTerm, biasTerm;
                    backward(column, scaleTerm, weightTerm, biasTerm);
                    scaleGradient[k] += scaleTerm;
                    weightGradient[k] += weightTerm;
                    biasGradient[k] += biasTerm;
                }
                // block_reduce_pair reuses scratch on the next row.
                __syncthreads();
            } else {
                for (int column = threadIdx.x; column < columns; column += kThreads) {
                    float scaleTerm, weightTerm, biasTerm;
                    backward(column, scaleTerm, weightTerm, biasTerm);
                    if (gradientScale != nullptr) atomicAdd(gradientScale + column, scaleTerm);
                    if (gradientWeight != nullptr) atomicAdd(gradientWeight + column, weightTerm);
                    if (gradientBias != nullptr) atomicAdd(gradientBias + column, biasTerm);
                }
            }
        }

        if constexpr (PerThread > 0) {
            #pragma unroll
            for (int k = 0; k < PerThread; ++k) {
                const int column = threadIdx.x + k * kThreads;
                if (column >= columns) break;
                if (gradientScale != nullptr) atomicAdd(gradientScale + column, scaleGradient[k]);
                if (gradientWeight != nullptr) atomicAdd(gradientWeight + column, weightGradient[k]);
                if (gradientBias != nullptr) atomicAdd(gradientBias + column, biasGradient[k]);
            }
        }
    }

    template<typename T>
    int launch_forward(const void *x, const void *z, const void *scale, const void *weight, const void *bias, void *output, float *statistics,
            const int rows, const int columns, const float epsilon, hipStream_t stream) {
        const auto wave = [&](auto kernel) {
            hipLaunchKernelGGL(kernel, dim3((rows + kWarps - 1) / kWarps), dim3(kThreads), 0, stream, static_cast<const T *>(x),
                    static_cast<const T *>(z), static_cast<const T *>(scale), static_cast<const T *>(weight), static_cast<const T *>(bias),
                    static_cast<T *>(output), statistics, rows, columns, epsilon);
            return hipGetLastError();
        };
        if (columns <= 128) return wave(layernorm_scale_add_forward_wave_kernel<T, 4>);
        if (columns <= 256) return wave(layernorm_scale_add_forward_wave_kernel<T, 8>);
        // Measured: at 1024 fp32 columns one wave per row is slower than the block-per-row kernel.
        if (columns <= 512) return wave(layernorm_scale_add_forward_wave_kernel<T, 16>);

        if (columns <= kStageColumns) {
            hipLaunchKernelGGL((layernorm_scale_add_forward_staged_kernel<T>), dim3(rows), dim3(kThreads), 0, stream,
                    static_cast<const T *>(x), static_cast<const T *>(z), static_cast<const T *>(scale), static_cast<const T *>(weight),
                    static_cast<const T *>(bias), static_cast<T *>(output), statistics, columns, epsilon);
            return hipGetLastError();
        }

        hipLaunchKernelGGL((layernorm_scale_add_forward_kernel<T>), dim3(rows), dim3(kThreads), 0, stream,
                static_cast<const T *>(x), static_cast<const T *>(z), static_cast<const T *>(scale), static_cast<const T *>(weight),
                static_cast<const T *>(bias), static_cast<T *>(output), statistics, columns, epsilon);
        return hipGetLastError();
    }

    template<typename T>
    int launch_backward(const void *x, const void *z, const void *scale, const void *weight, const float *statistics, const void *upstream,
            void *gradientX, void *gradientZ, float *gradientScale, float *gradientWeight, float *gradientBias, const int rows,
            const int columns, hipStream_t stream) {
        const std::size_t columnBytes = static_cast<std::size_t>(columns) * sizeof(float);
        if (gradientScale != nullptr) {
            const hipError_t status = hipMemsetAsync(gradientScale, 0, columnBytes, stream);
            if (status != hipSuccess) return status;
        }
        if (gradientWeight != nullptr) {
            const hipError_t status = hipMemsetAsync(gradientWeight, 0, columnBytes, stream);
            if (status != hipSuccess) return status;
        }
        if (gradientBias != nullptr) {
            const hipError_t status = hipMemsetAsync(gradientBias, 0, columnBytes, stream);
            if (status != hipSuccess) return status;
        }

        const auto wave = [&](auto kernel) {
            const int blocks = std::min((rows + kWarps - 1) / kWarps, kBackwardBlocks);
            hipLaunchKernelGGL(kernel, dim3(blocks), dim3(kThreads), 0, stream, static_cast<const T *>(x), static_cast<const T *>(z),
                    static_cast<const T *>(scale), static_cast<const T *>(weight), statistics, static_cast<const T *>(upstream),
                    static_cast<T *>(gradientX), static_cast<T *>(gradientZ), gradientScale, gradientWeight, gradientBias, rows, columns);
            return hipGetLastError();
        };
        if (columns <= 128) return wave(layernorm_scale_add_backward_wave_kernel<T, 4>);
        // Measured: from 512 columns the block kernel below beats one wave per row.
        if (columns <= 256) return wave(layernorm_scale_add_backward_wave_kernel<T, 8>);

        const auto block = [&](auto kernel, const int blocks) {
            hipLaunchKernelGGL(kernel, dim3(blocks), dim3(kThreads), 0, stream, static_cast<const T *>(x), static_cast<const T *>(z),
                    static_cast<const T *>(scale), static_cast<const T *>(weight), statistics, static_cast<const T *>(upstream),
                    static_cast<T *>(gradientX), static_cast<T *>(gradientZ), gradientScale, gradientWeight, gradientBias, rows, columns);
            return hipGetLastError();
        };
        const int strided = std::min(rows, kBackwardBlocks);
        if (columns <= 4 * kThreads) return block(layernorm_scale_add_backward_kernel<T, 4>, strided);
        if (columns <= 8 * kThreads) return block(layernorm_scale_add_backward_kernel<T, 8>, strided);
        return block(layernorm_scale_add_backward_kernel<T, 0>, rows);
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int layernorm_scale_add_forward(const void *x, const void *z, const void *scale, const void *weight,
        const void *bias, void *output, float *statistics, const int rows, const int columns, const float epsilon,
        ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    if (!x || !z || !scale || !output || !statistics || rows <= 0 || columns <= 0 || epsilon < 0.0f) return hipErrorInvalidValue;

    switch (dtype) {
        case ROCmKernels::Kernels::DType::Float32:
            return launch_forward<float>(x, z, scale, weight, bias, output, statistics, rows, columns, epsilon, stream);
        case ROCmKernels::Kernels::DType::Float16:
            return launch_forward<__half>(x, z, scale, weight, bias, output, statistics, rows, columns, epsilon, stream);
    }

    return hipErrorInvalidValue;
}

extern "C" ROCMKERNELS_EXPORT int layernorm_scale_add_backward(const void *x, const void *z, const void *scale, const void *weight,
        const float *statistics, const void *upstream, void *gradientX, void *gradientZ, float *gradientScale, float *gradientWeight,
        float *gradientBias, const int rows, const int columns, ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    if (!x || !z || !scale || !statistics || !upstream || !gradientX || !gradientZ || !gradientScale || rows <= 0 || columns <= 0)
        return hipErrorInvalidValue;

    // dweight needs weight to form dy * weight; dbias is independent and follows its own pointer.
    float *const weightGradient = weight != nullptr ? gradientWeight : nullptr;

    switch (dtype) {
        case ROCmKernels::Kernels::DType::Float32:
            return launch_backward<float>(x, z, scale, weight, statistics, upstream, gradientX, gradientZ, gradientScale, weightGradient,
                    gradientBias, rows, columns, stream);
        case ROCmKernels::Kernels::DType::Float16:
            return launch_backward<__half>(x, z, scale, weight, statistics, upstream, gradientX, gradientZ, gradientScale, weightGradient,
                    gradientBias, rows, columns, stream);
    }

    return hipErrorInvalidValue;
}
