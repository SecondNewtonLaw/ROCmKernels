#pragma once

#include <ROCmKernels/Kernels/DType.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

// Weight layout: the plain names take weight[k][column] (K-major, this library's
// original layout). The _nt names take weight[column][k], which is what
// torch.nn.Linear stores, so a torch weight can be passed without a transpose.
extern "C" ROCMKERNELS_EXPORT int linear_act_fast(const void *input, const void *weight, const void *bias, void *output, int rows, int columns,
        int inner, ROCmKernels::Kernels::DType dtype, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int linear_fast(const void *input, const void *weight, const void *bias, void *output, int rows, int columns,
        int inner, ROCmKernels::Kernels::DType dtype, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int linear_act_nt(const void *input, const void *weight, const void *bias, void *output, int rows, int columns,
        int inner, ROCmKernels::Kernels::DType dtype, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int linear_nt(const void *input, const void *weight, const void *bias, void *output, int rows, int columns,
        int inner, ROCmKernels::Kernels::DType dtype, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// output[r][c] = silu(sum_k input[r][k] * weight[k][c] + bias[c]); bias may be null.
    inline void LinearActFast(const void *input, const void *weight, const void *bias, void *output, const int rows, const int columns,
            const int inner, const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(linear_act_fast(input, weight, bias, output, rows, columns, inner, dtype, stream));
    }

    /// output[r][c] = sum_k input[r][k] * weight[k][c] + bias[c].
    inline void LinearFast(const void *input, const void *weight, const void *bias, void *output, const int rows, const int columns,
            const int inner, const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(linear_fast(input, weight, bias, output, rows, columns, inner, dtype, stream));
    }

    /// Same as LinearActFast with weight[column][k], the torch.nn.Linear layout.
    inline void LinearActNT(const void *input, const void *weight, const void *bias, void *output, const int rows, const int columns,
            const int inner, const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(linear_act_nt(input, weight, bias, output, rows, columns, inner, dtype, stream));
    }

    /// Same as LinearFast with weight[column][k], the torch.nn.Linear layout.
    inline void LinearNT(const void *input, const void *weight, const void *bias, void *output, const int rows, const int columns,
            const int inner, const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(linear_nt(input, weight, bias, output, rows, columns, inner, dtype, stream));
    }
} // namespace ROCmKernels::Kernels
