#pragma once

#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>
#include <ROCmKernels/Kernels/Strides.hpp>

#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int attention64_fp32_forward(const float *q, const float *k, const float *v, const bool *mask, float *output, float *lse,
        int batch, int heads, int queries, int keys, int radius, ROCmKernels::Kernels::Strides qs, ROCmKernels::Kernels::Strides ks,
        ROCmKernels::Kernels::Strides vs, ROCmKernels::Kernels::Strides os, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int attention64_fp32_backward(const float *q, const float *k, const float *v, const bool *mask, const float *output,
        const float *lse, const float *upstream, float *gradient_q, float *gradient_k, float *gradient_v, float *delta, int batch,
        int heads, int queries, int keys, int radius, ROCmKernels::Kernels::Strides qs, ROCmKernels::Kernels::Strides ks,
        ROCmKernels::Kernels::Strides vs, ROCmKernels::Kernels::Strides os, ROCmKernels::Kernels::Strides us, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// FP32 attention with 64-wide heads. Layouts match the 32-wide variant.
    inline void Attention64Forward(const float *q, const float *k, const float *v, const bool *mask, float *output, float *lse,
            const int batch, const int heads, const int queries, const int keys, const int radius, const Strides &qs, const Strides &ks,
            const Strides &vs, const Strides &os, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(
                attention64_fp32_forward(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream));
    }

    inline void Attention64Backward(const float *q, const float *k, const float *v, const bool *mask, const float *output,
            const float *lse, const float *upstream, float *gradient_q, float *gradient_k, float *gradient_v, float *delta, const int batch,
            const int heads, const int queries, const int keys, const int radius, const Strides &qs, const Strides &ks, const Strides &vs,
            const Strides &os, const Strides &us, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(attention64_fp32_backward(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v, delta, batch,
                heads, queries, keys, radius, qs, ks, vs, os, us, stream));
    }
} // namespace ROCmKernels::Kernels
