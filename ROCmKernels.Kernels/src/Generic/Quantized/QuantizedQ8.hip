// Q8_0 blocks and GEMV, ported from llama.cpp ggml-cuda dequantize.cuh, vecdotq.cuh and quantize.cu (MIT, Copyright (c) 2023-2026 The ggml authors).
#include <ROCmKernels/Kernels/Generic/QuantizedQ8.hpp>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {
    using ROCmKernels::Kernels::BlockQ8_0;
    using ROCmKernels::Kernels::DType;
    using ROCmKernels::Kernels::kQ8Block;

    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerBlock = 8;
    constexpr int kThreads = kWarpSize * kWarpsPerBlock;
    constexpr int kDequantizeThreads = 256;
    // Smallest positive fp16 scale, used when a nonzero block would otherwise round to zero.
    constexpr float kMinimumScale = 5.9604644775390625e-8f;

    __device__ __forceinline__ float warp_sum(float value) {
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) value += __shfl_down(value, offset, kWarpSize);
        return value;
    }

    __device__ __forceinline__ float warp_max(float value) {
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor(value, offset, kWarpSize));
        return value;
    }

    __device__ __forceinline__ int quantized_dot(const BlockQ8_0 &a, const BlockQ8_0 &b) {
        int total = 0;
        #pragma unroll
        for (int index = 0; index < kQ8Block; ++index) total += static_cast<int>(a.values[index]) * static_cast<int>(b.values[index]);
        return total;
    }

    /// One warp per block so the abs-max reduction stays inside the warp.
    __global__ void quantize_q8_0_kernel(const float *input, BlockQ8_0 *output, const long long blocks) {
        const int lane = threadIdx.x % kWarpSize;
        const long long block = static_cast<long long>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpSize;
        if (block >= blocks) return;

        const float value = input[block * kQ8Block + lane];
        const float maximum = warp_max(fabsf(value));
        __half scale = __float2half_rn(maximum / 127.0f);
        if (maximum > 0.0f && __half2float(scale) == 0.0f) scale = __float2half_rn(kMinimumScale);

        // Quantize against the stored scale so dequantization reproduces the block.
        const float storedScale = __half2float(scale);
        const float inverse = storedScale > 0.0f ? 1.0f / storedScale : 0.0f;
        int quantized = static_cast<int>(roundf(value * inverse));
        quantized = quantized > 127 ? 127 : (quantized < -127 ? -127 : quantized);

        BlockQ8_0 &destination = output[block];
        if (lane == 0) destination.scale = scale;
        destination.values[lane] = static_cast<std::int8_t>(quantized);
    }

    __global__ void dequantize_q8_0_kernel(const BlockQ8_0 *input, float *output, const long long elements) {
        const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= elements) return;

        const BlockQ8_0 &block = input[index / kQ8Block];
        output[index] = static_cast<float>(block.values[index % kQ8Block]) * __half2float(block.scale);
    }

    /// One warp per output column; each lane owns a subset of the block range of that row.
    template<typename T>
    __global__ void quantized_gemv_q8_0_kernel(const BlockQ8_0 *input, const BlockQ8_0 *weights, const T *bias, T *output, const int columns,
            const int blocks) {
        const int lane = threadIdx.x % kWarpSize;
        const int column = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpSize;
        if (column >= columns) return;

        const BlockQ8_0 *row = weights + static_cast<std::size_t>(column) * blocks;

        float total = 0.0f;
        for (int block = lane; block < blocks; block += kWarpSize) {
            const BlockQ8_0 &a = input[block];
            const BlockQ8_0 &b = row[block];
            total += static_cast<float>(quantized_dot(a, b)) * __half2float(a.scale) * __half2float(b.scale);
        }

        total = warp_sum(total);
        if (lane == 0) {
            if (bias != nullptr) total += ROCmKernels::Kernels::LoadScalar(bias + column);
            ROCmKernels::Kernels::StoreScalar(output + column, total);
        }
    }

    int launch_quantize(const float *input, BlockQ8_0 *output, const long long elements, const hipStream_t stream) {
        const long long blocks = elements / kQ8Block;
        const unsigned int grid = static_cast<unsigned int>((blocks + kWarpsPerBlock - 1) / kWarpsPerBlock);
        hipLaunchKernelGGL((quantize_q8_0_kernel), dim3(grid), dim3(kThreads), 0, stream, input, output, blocks);
        return hipGetLastError();
    }

    int launch_dequantize(const BlockQ8_0 *input, float *output, const long long elements, const hipStream_t stream) {
        const unsigned int grid = static_cast<unsigned int>((elements + kDequantizeThreads - 1) / kDequantizeThreads);
        hipLaunchKernelGGL((dequantize_q8_0_kernel), dim3(grid), dim3(kDequantizeThreads), 0, stream, input, output, elements);
        return hipGetLastError();
    }

    template<typename T>
    int launch_gemv(const BlockQ8_0 *input, const BlockQ8_0 *weights, const void *bias, void *output, const int columns, const int inner,
            const hipStream_t stream) {
        const int blocks = inner / kQ8Block;
        const unsigned int grid = static_cast<unsigned int>((columns + kWarpsPerBlock - 1) / kWarpsPerBlock);
        hipLaunchKernelGGL((quantized_gemv_q8_0_kernel<T>), dim3(grid), dim3(kThreads), 0, stream, input, weights,
                static_cast<const T *>(bias), static_cast<T *>(output), columns, blocks);
        return hipGetLastError();
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int quantize_q8_0(const float *input, ROCmKernels::Kernels::BlockQ8_0 *output, const long long elements,
        hipStream_t stream) {
    using ROCmKernels::Kernels::kQ8Block;

    if (!input || !output || elements <= 0 || elements % kQ8Block != 0) return hipErrorInvalidValue;
    return launch_quantize(input, output, elements, stream);
}

extern "C" ROCMKERNELS_EXPORT int dequantize_q8_0(const ROCmKernels::Kernels::BlockQ8_0 *input, float *output, const long long elements,
        hipStream_t stream) {
    using ROCmKernels::Kernels::kQ8Block;

    if (!input || !output || elements <= 0 || elements % kQ8Block != 0) return hipErrorInvalidValue;
    return launch_dequantize(input, output, elements, stream);
}

extern "C" ROCMKERNELS_EXPORT int quantized_gemv_q8_0(const ROCmKernels::Kernels::BlockQ8_0 *input,
        const ROCmKernels::Kernels::BlockQ8_0 *weights, const void *bias, void *output, const int columns, const int inner,
        const ROCmKernels::Kernels::DType outputDtype, hipStream_t stream) {
    using ROCmKernels::Kernels::DType;
    using ROCmKernels::Kernels::kQ8Block;

    if (!input || !weights || !output || columns <= 0 || inner <= 0 || inner % kQ8Block != 0) return hipErrorInvalidValue;

    switch (outputDtype) {
        case DType::Float32:
            return launch_gemv<float>(input, weights, bias, output, columns, inner, stream);
        case DType::Float16:
            return launch_gemv<__half>(input, weights, bias, output, columns, inner, stream);
    }

    return hipErrorInvalidValue;
}
