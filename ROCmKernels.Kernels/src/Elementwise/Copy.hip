//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Kernels/Elementwise/Copy.hpp>

#include <ROCmKernels/Common/Error.hpp>

#include <cstdint>

namespace {
    constexpr unsigned int kThreadsPerBlock = 256;

    __global__ void CopyKernel(const float *source, float *destination, const std::size_t count) {
        const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

        for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
                index < count; index += stride) {
            destination[index] = source[index];
        }
    }

    __global__ void CopyVectorizedKernel(const float4 *source, float4 *destination, const std::size_t count) {
        const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

        for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
                index < count; index += stride) {
            destination[index] = source[index];
        }
    }

    bool IsAligned(const void *pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4) == 0; }
} // namespace

namespace ROCmKernels::Kernels {
    void Copy(const float *source, float *destination, const std::size_t count, hipStream_t stream) {
        if (count == 0) {
            return;
        }

        if (count % 4 == 0 && IsAligned(source) && IsAligned(destination)) {
            const std::size_t vectorCount = count / 4;
            const auto blocks = static_cast<unsigned int>((vectorCount + kThreadsPerBlock - 1) / kThreadsPerBlock);
            CopyVectorizedKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
                    reinterpret_cast<const float4 *>(source), reinterpret_cast<float4 *>(destination), vectorCount);
        } else {
            const auto blocks = static_cast<unsigned int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
            CopyKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(source, destination, count);
        }

        ROCMKERNELS_HIP_CHECK(hipGetLastError());
    }
} // namespace ROCmKernels::Kernels
