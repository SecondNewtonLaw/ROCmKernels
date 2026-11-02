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

    __device__ __forceinline__ float wave_sum(float value) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down(value, offset, 32);
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

    template<typename T>
    __global__ void layernorm_scale_add_backward_kernel(const T *x, const T *z, const T *scale, const T *weight, const float *statistics,
            const T *upstream, T *gradientX, T *gradientZ, float *gradientScale, float *gradientWeight, float *gradientBias,
            const int columns) {
        __shared__ float scratch[2 * kWarps];

        const std::size_t row = blockIdx.x;
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

        for (int column = threadIdx.x; column < columns; column += kThreads) {
            const float value = ROCmKernels::Kernels::LoadScalar(x + base + column) +
                                ROCmKernels::Kernels::LoadScalar(scale + column) * ROCmKernels::Kernels::LoadScalar(z + base + column);
            const float normalized = (value - mean) * rstd;
            const float dy = ROCmKernels::Kernels::LoadScalar(upstream + base + column);
            float dyWeighted = dy;
            if (weight != nullptr) dyWeighted *= ROCmKernels::Kernels::LoadScalar(weight + column);

            const float ds = rstd * (dyWeighted - plainMean - normalized * centeredMean);
            ROCmKernels::Kernels::StoreScalar(gradientX + base + column, ds);
            ROCmKernels::Kernels::StoreScalar(gradientZ + base + column, ds * ROCmKernels::Kernels::LoadScalar(scale + column));

            // Parameter gradients stay fp32; one atomically accumulated row contribution per column.
            if (gradientScale != nullptr) atomicAdd(gradientScale + column, ds * ROCmKernels::Kernels::LoadScalar(z + base + column));
            if (gradientWeight != nullptr) atomicAdd(gradientWeight + column, dy * normalized);
            if (gradientBias != nullptr) atomicAdd(gradientBias + column, dy);
        }
    }

    template<typename T>
    int launch_forward(const void *x, const void *z, const void *scale, const void *weight, const void *bias, void *output, float *statistics,
            const int rows, const int columns, const float epsilon, hipStream_t stream) {
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

        hipLaunchKernelGGL((layernorm_scale_add_backward_kernel<T>), dim3(rows), dim3(kThreads), 0, stream,
                static_cast<const T *>(x), static_cast<const T *>(z), static_cast<const T *>(scale), static_cast<const T *>(weight),
                statistics, static_cast<const T *>(upstream), static_cast<T *>(gradientX), static_cast<T *>(gradientZ), gradientScale,
                gradientWeight, gradientBias, columns);
        return hipGetLastError();
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
