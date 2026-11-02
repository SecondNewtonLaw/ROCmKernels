#pragma once

#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int cross_entropy_fp32_forward(const float *logits, const std::int64_t *targets, float *losses, float *statistics,
        int rows, int classes, int stride, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int cross_entropy_fp32_backward(const float *logits, const std::int64_t *targets, const float *statistics,
        const float *upstream, float *gradient, int rows, int classes, int stride, int grad_stride, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// Row-wise cross entropy over logits[row * stride + class].
    ///
    /// statistics receives {row maximum, log-sum-exp} per row. A target of -100 contributes no loss.
    inline void CrossEntropyForward(const float *logits, const std::int64_t *targets, float *losses, float *statistics, const int rows,
            const int classes, const int stride, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(cross_entropy_fp32_forward(logits, targets, losses, statistics, rows, classes, stride, stream));
    }

    /// upstream holds one scale per row, read at row * gradStride. gradient is [rows][classes].
    inline void CrossEntropyBackward(const float *logits, const std::int64_t *targets, const float *statistics, const float *upstream,
            float *gradient, const int rows, const int classes, const int stride, const int gradStride, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(
                cross_entropy_fp32_backward(logits, targets, statistics, upstream, gradient, rows, classes, stride, gradStride, stream));
    }
} // namespace ROCmKernels::Kernels
