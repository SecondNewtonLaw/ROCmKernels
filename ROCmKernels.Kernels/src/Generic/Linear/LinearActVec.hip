// Warp-per-row matrix-vector product, ported from llama.cpp ggml-cuda mmvf.cu (MIT, Copyright (c) 2023-2026 The ggml authors).
#include <ROCmKernels/Kernels/Generic/LinearActVec.hpp>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>

namespace {
    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerBlock = 8;
    constexpr int kThreads = kWarpSize * kWarpsPerBlock;

    __device__ __forceinline__ float silu(const float value) {
        return value / (1.0f + __expf(-value));
    }

    __device__ __forceinline__ float warp_sum(float value) {
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) value += __shfl_down(value, offset, kWarpSize);
        return value;
    }

    template<typename T>
    __global__ void linear_act_vec_kernel(const T *input, const T *weight, const T *bias, T *output, const int columns, const int inner) {
        const int lane = threadIdx.x % kWarpSize;
        const int column = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpSize;
        if (column >= columns) return;

        const T *row = weight + static_cast<std::size_t>(column) * inner;

        float accumulator = 0.0f;
        for (int k = lane; k < inner; k += kWarpSize) {
            accumulator += ROCmKernels::Kernels::LoadScalar(row + k) * ROCmKernels::Kernels::LoadScalar(input + k);
        }

        accumulator = warp_sum(accumulator);
        if (lane == 0) {
            if (bias != nullptr) accumulator += ROCmKernels::Kernels::LoadScalar(bias + column);
            ROCmKernels::Kernels::StoreScalar(output + column, silu(accumulator));
        }
    }

    template<typename T>
    int launch_linear_act_vec(const void *input, const void *weight, const void *bias, void *output, const int columns, const int inner,
            const hipStream_t stream) {
        const dim3 grid((columns + kWarpsPerBlock - 1) / kWarpsPerBlock);
        hipLaunchKernelGGL((linear_act_vec_kernel<T>), grid, dim3(kThreads), 0, stream, static_cast<const T *>(input),
                static_cast<const T *>(weight), static_cast<const T *>(bias), static_cast<T *>(output), columns, inner);
        return hipGetLastError();
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int linear_act_vec(const void *input, const void *weight, const void *bias, void *output, int columns, int inner,
        ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    using ROCmKernels::Kernels::DType;

    if (dtype != DType::Float32 && dtype != DType::Float16) return hipErrorInvalidValue;
    if (!input || !weight || !output || columns <= 0 || inner <= 0) return hipErrorInvalidValue;

    switch (dtype) {
        case DType::Float32:
            return launch_linear_act_vec<float>(input, weight, bias, output, columns, inner, stream);
        case DType::Float16:
            return launch_linear_act_vec<__half>(input, weight, bias, output, columns, inner, stream);
    }

    return hipErrorInvalidValue;
}
