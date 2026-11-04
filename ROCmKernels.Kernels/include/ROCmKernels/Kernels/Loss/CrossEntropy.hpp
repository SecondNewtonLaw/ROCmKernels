#pragma once

#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int cross_entropy_fp32_forward(const float *logits, const std::int64_t *targets, float *losses, float *statistics,
        int rows, int classes, int stride, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int cross_entropy_fp32_backward(const float *logits, const std::int64_t *targets, const float *statistics,
        const float *upstream, float *gradient, int rows, int classes, int stride, int upstream_stride, hipStream_t stream);

extern "C" ROCMKERNELS_EXPORT int cross_entropy_fp16_forward(const void *logits, const std::int64_t *targets, float *losses, float *statistics,
        int rows, int classes, int stride, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int cross_entropy_fp16_backward(const void *logits, const std::int64_t *targets, const float *statistics,
        const float *upstream, void *gradient, int rows, int classes, int stride, int upstream_stride, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// Row-wise cross entropy over logits[row * stride + class].
    ///
    /// statistics receives {row maximum, log(sum exp(x - maximum))} per row. The full
    /// log-sum-exp is the sum of the two slots. A target of -100 contributes no loss.
    inline void CrossEntropyForward(const float *logits, const std::int64_t *targets, float *losses, float *statistics, const int rows,
            const int classes, const int stride, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(cross_entropy_fp32_forward(logits, targets, losses, statistics, rows, classes, stride, stream));
    }

    /// upstream holds one scale per row, read at upstream[row * upstreamStride].
    /// upstreamStride is in elements: 1 for a contiguous [rows] array, 0 to broadcast a scalar.
    /// gradient is overwritten at gradient[row * classes + column] and must be contiguous [rows][classes].
    inline void CrossEntropyBackward(const float *logits, const std::int64_t *targets, const float *statistics, const float *upstream,
            float *gradient, const int rows, const int classes, const int stride, const int upstreamStride, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(
                cross_entropy_fp32_backward(logits, targets, statistics, upstream, gradient, rows, classes, stride, upstreamStride, stream));
    }

    /// Same forward over fp16 logits. Losses and statistics stay fp32.
    inline void CrossEntropyForwardFp16(const void *logits, const std::int64_t *targets, float *losses, float *statistics, const int rows,
            const int classes, const int stride, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(cross_entropy_fp16_forward(logits, targets, losses, statistics, rows, classes, stride, stream));
    }

    /// fp16 logits in, fp16 gradient out, fp32 statistics and upstream. This is the path that
    /// removes the fp16 -> fp32 cast in front of the loss and halves the loss traffic.
    ///
    /// gradient entries are p * upstream - onehot * upstream. With a mean reduction the softmax
    /// term is upstream / rows, which falls below the fp16 subnormal minimum once classes is large
    /// (about 6e-8, so classes > ~16k at upstream = 1/rows). The stored gradient is then the one-hot
    /// term, which is exactly what a fp32 gradient cast down to fp16 would produce. Scale the loss
    /// if the softmax term has to survive the store.
    inline void CrossEntropyBackwardFp16(const void *logits, const std::int64_t *targets, const float *statistics, const float *upstream,
            void *gradient, const int rows, const int classes, const int stride, const int upstreamStride, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(
                cross_entropy_fp16_backward(logits, targets, statistics, upstream, gradient, rows, classes, stride, upstreamStride, stream));
    }
} // namespace ROCmKernels::Kernels
