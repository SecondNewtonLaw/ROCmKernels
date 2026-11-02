#pragma once

#include <ROCmKernels/Kernels/DType.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int top_k_forward(const void *scores, float *values, int *indices, int rows, int classes, int k,
        ROCmKernels::Kernels::DType dtype, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// Row-wise top-k: values and indices receive the k largest scores of each row, ordered by value descending.
    ///
    /// scores is [rows][classes]; values is [rows][k] float32 and indices is [rows][k] int32. 1 <= k <= 8 and k <= classes
    /// must hold. Ties may come back in any order.
    inline void TopKForward(const void *scores, float *values, int *indices, const int rows, const int classes, const int k,
            const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(top_k_forward(scores, values, indices, rows, classes, k, dtype, stream));
    }
} // namespace ROCmKernels::Kernels
