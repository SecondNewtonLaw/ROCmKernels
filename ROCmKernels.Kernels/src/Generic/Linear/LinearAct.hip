#include <ROCmKernels/Kernels/Generic/LinearAct.hpp>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>

namespace {
    constexpr int kTile = 16;
    constexpr int kSiluThreads = 256;
    constexpr int kMaxBlocks = 4096;

    __device__ __forceinline__ float silu(const float value) {
        return value / (1.0f + __expf(-value));
    }

    template<typename T>
    __global__ void linear_act_kernel(const T *input, const T *weight, const T *bias, T *output, const int rows, const int columns,
            const int inner) {
        __shared__ float tileA[kTile][kTile];
        __shared__ float tileB[kTile][kTile];

        const int row = blockIdx.y * kTile + threadIdx.y;
        const int column = blockIdx.x * kTile + threadIdx.x;

        float accumulator = 0.0f;
        for (int base = 0; base < inner; base += kTile) {
            const int innerFromA = base + threadIdx.x;
            const int innerFromB = base + threadIdx.y;

            // Out-of-range lanes stage zeros so the reduction never reads the tails.
            tileA[threadIdx.y][threadIdx.x] = row < rows && innerFromA < inner
                    ? ROCmKernels::Kernels::LoadScalar(input + static_cast<std::size_t>(row) * inner + innerFromA)
                    : 0.0f;
            tileB[threadIdx.y][threadIdx.x] = column < columns && innerFromB < inner
                    ? ROCmKernels::Kernels::LoadScalar(weight + static_cast<std::size_t>(innerFromB) * columns + column)
                    : 0.0f;
            __syncthreads();

#pragma unroll
            for (int step = 0; step < kTile; ++step) {
                accumulator = fmaf(tileA[threadIdx.y][step], tileB[step][threadIdx.x], accumulator);
            }
            __syncthreads();
        }

        if (row < rows && column < columns) {
            if (bias != nullptr) accumulator += ROCmKernels::Kernels::LoadScalar(bias + column);
            ROCmKernels::Kernels::StoreScalar(output + static_cast<std::size_t>(row) * columns + column, silu(accumulator));
        }
    }

    template<typename T>
    __global__ void silu_kernel(const T *input, T *output, const long long elements) {
        const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
        for (long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; index < elements; index += stride) {
            ROCmKernels::Kernels::StoreScalar(output + index, silu(ROCmKernels::Kernels::LoadScalar(input + index)));
        }
    }

    template<typename T>
    int launch_linear(const void *input, const void *weight, const void *bias, void *output, const int rows, const int columns,
            const int inner, const hipStream_t stream) {
        const dim3 grid((columns + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
        hipLaunchKernelGGL((linear_act_kernel<T>), grid, dim3(kTile, kTile), 0, stream, static_cast<const T *>(input),
                static_cast<const T *>(weight), static_cast<const T *>(bias), static_cast<T *>(output), rows, columns, inner);
        return hipGetLastError();
    }

    template<typename T>
    int launch_silu(const void *input, void *output, const long long elements, const hipStream_t stream) {
        const long long requestedBlocks = (elements + kSiluThreads - 1) / kSiluThreads;
        const unsigned int blocks = static_cast<unsigned int>(requestedBlocks < kMaxBlocks ? requestedBlocks : kMaxBlocks);
        hipLaunchKernelGGL((silu_kernel<T>), dim3(blocks), dim3(kSiluThreads), 0, stream, static_cast<const T *>(input),
                static_cast<T *>(output), elements);
        return hipGetLastError();
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int linear_act_forward(const void *input, const void *weight, const void *bias, void *output, int rows,
        int columns, int inner, ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    using ROCmKernels::Kernels::DType;

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

extern "C" ROCMKERNELS_EXPORT int silu_forward(const void *input, void *output, long long elements, ROCmKernels::Kernels::DType dtype,
        hipStream_t stream) {
    using ROCmKernels::Kernels::DType;

    if (dtype != DType::Float32 && dtype != DType::Float16) return hipErrorInvalidValue;
    if (!input || !output || elements < 0) return hipErrorInvalidValue;
    if (elements == 0) return hipSuccess;

    switch (dtype) {
        case DType::Float32:
            return launch_silu<float>(input, output, elements, stream);
        case DType::Float16:
            return launch_silu<__half>(input, output, elements, stream);
    }

    return hipErrorInvalidValue;
}
