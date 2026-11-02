//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Kernels/Elementwise/Vector.hpp>

#include <ROCmKernels/Common/Error.hpp>

namespace {
    constexpr unsigned int kThreadsPerBlock = 256;

    template <typename T> __global__ void VectorAddKernel(const T *lhs, const T *rhs, T *out, const std::size_t count) {
        const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

        for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
             index < count; index += stride) {
            out[index] = lhs[index] + rhs[index];
        }
    }

    template <typename T> void LaunchVectorAdd(const T *lhs, const T *rhs, T *out, const std::size_t count, hipStream_t stream) {
        if (count == 0) {
            return;
        }

        const auto blocks = static_cast<unsigned int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
        VectorAddKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(lhs, rhs, out, count);
        ROCMKERNELS_HIP_CHECK(hipGetLastError());
    }
} // namespace

namespace ROCmKernels::Kernels {
    void VectorAdd(const float *lhs, const float *rhs, float *out, const std::size_t count, hipStream_t stream) {
        LaunchVectorAdd(lhs, rhs, out, count, stream);
    }

    void VectorAdd(const double *lhs, const double *rhs, double *out, const std::size_t count, hipStream_t stream) {
        LaunchVectorAdd(lhs, rhs, out, count, stream);
    }
} // namespace ROCmKernels::Kernels
