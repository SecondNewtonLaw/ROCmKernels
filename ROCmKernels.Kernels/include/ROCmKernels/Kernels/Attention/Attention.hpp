#pragma once

#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>
#include <ROCmKernels/Kernels/Strides.hpp>

#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int attention_fp32_forward(const float *q, const float *k, const float *v, const bool *mask, float *output, float *lse,
        int batch, int heads, int queries, int keys, int radius, ROCmKernels::Kernels::Strides qs, ROCmKernels::Kernels::Strides ks,
        ROCmKernels::Kernels::Strides vs, ROCmKernels::Kernels::Strides os, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int attention_fp32_forward_tiled(const float *q, const float *k, const float *v, const bool *mask, float *output,
        float *lse, int batch, int heads, int queries, int keys, int radius, ROCmKernels::Kernels::Strides qs,
        ROCmKernels::Kernels::Strides ks, ROCmKernels::Kernels::Strides vs, ROCmKernels::Kernels::Strides os, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int attention_fp32_backward(const float *q, const float *k, const float *v, const bool *mask, const float *output,
        const float *lse, const float *upstream, float *gradient_q, float *gradient_k, float *gradient_v, float *delta, int batch,
        int heads, int queries, int keys, int radius, ROCmKernels::Kernels::Strides qs, ROCmKernels::Kernels::Strides ks,
        ROCmKernels::Kernels::Strides vs, ROCmKernels::Kernels::Strides os, ROCmKernels::Kernels::Strides us, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int attention_fp32_backward_tiled(const float *q, const float *k, const float *v, const bool *mask, const float *output,
        const float *lse, const float *upstream, float *gradient_q, float *gradient_k, float *gradient_v, float *delta, int batch,
        int heads, int queries, int keys, int radius, ROCmKernels::Kernels::Strides qs, ROCmKernels::Kernels::Strides ks,
        ROCmKernels::Kernels::Strides vs, ROCmKernels::Kernels::Strides os, ROCmKernels::Kernels::Strides us, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int attention_fp32_statistics(const float *q, const float *k, const bool *mask, float *lse, int batch, int heads,
        int queries, int keys, int radius, ROCmKernels::Kernels::Strides qs, ROCmKernels::Kernels::Strides ks, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int attention_fp32_forward_tuned(const float *q, const float *k, const float *v, const bool *mask, float *output,
        float *lse, int batch, int heads, int queries, int keys, int radius, int group, ROCmKernels::Kernels::Strides qs,
        ROCmKernels::Kernels::Strides ks, ROCmKernels::Kernels::Strides vs, ROCmKernels::Kernels::Strides os, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int attention_fp32_forward_flash(const float *q, const float *k, const float *v, const bool *mask, float *output,
        float *lse, int batch, int heads, int queries, int keys, int radius, ROCmKernels::Kernels::Strides qs,
        ROCmKernels::Kernels::Strides ks, ROCmKernels::Kernels::Strides vs, ROCmKernels::Kernels::Strides os, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int attention_fp32_forward_flash_tuned(const float *q, const float *k, const float *v, const bool *mask, float *output,
        float *lse, int batch, int heads, int queries, int keys, int radius, int variant, ROCmKernels::Kernels::Strides qs,
        ROCmKernels::Kernels::Strides ks, ROCmKernels::Kernels::Strides vs, ROCmKernels::Kernels::Strides os, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int attention_fp32_backward_tuned(const float *q, const float *k, const float *v, const bool *mask, const float *output,
        const float *lse, const float *upstream, float *gradient_q, float *gradient_k, float *gradient_v, float *delta, int batch,
        int heads, int queries, int keys, int radius, int group, ROCmKernels::Kernels::Strides qs, ROCmKernels::Kernels::Strides ks,
        ROCmKernels::Kernels::Strides vs, ROCmKernels::Kernels::Strides os, ROCmKernels::Kernels::Strides us, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// FP32 attention with 32-wide heads.
    ///
    /// mask is [batch][keys]. radius < 0 attends globally. lse receives {row maximum, exponent sum} per (batch, head, query).
    /// output writes q at os.batch, os.head and os.token; lse and the gradients are packed [batch][head][query|key][32].
    inline void AttentionForward(const float *q, const float *k, const float *v, const bool *mask, float *output, float *lse, const int batch,
            const int heads, const int queries, const int keys, const int radius, const Strides &qs, const Strides &ks, const Strides &vs,
            const Strides &os, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(attention_fp32_forward(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream));
    }

    inline void AttentionForwardTiled(const float *q, const float *k, const float *v, const bool *mask, float *output, float *lse,
            const int batch, const int heads, const int queries, const int keys, const int radius, const Strides &qs, const Strides &ks,
            const Strides &vs, const Strides &os, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(
                attention_fp32_forward_tiled(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream));
    }

    /// delta receives sum(upstream * output) per query row.
    inline void AttentionBackward(const float *q, const float *k, const float *v, const bool *mask, const float *output, const float *lse,
            const float *upstream, float *gradient_q, float *gradient_k, float *gradient_v, float *delta, const int batch, const int heads,
            const int queries, const int keys, const int radius, const Strides &qs, const Strides &ks, const Strides &vs, const Strides &os,
            const Strides &us, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(attention_fp32_backward(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v, delta, batch,
                heads, queries, keys, radius, qs, ks, vs, os, us, stream));
    }

    inline void AttentionBackwardTiled(const float *q, const float *k, const float *v, const bool *mask, const float *output,
            const float *lse, const float *upstream, float *gradient_q, float *gradient_k, float *gradient_v, float *delta, const int batch,
            const int heads, const int queries, const int keys, const int radius, const Strides &qs, const Strides &ks, const Strides &vs,
            const Strides &os, const Strides &us, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(attention_fp32_backward_tiled(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v, delta,
                batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream));
    }

    /// Softmax statistics only; skips value reads and the output write.
    inline void AttentionStatistics(const float *q, const float *k, const bool *mask, float *lse, const int batch, const int heads,
            const int queries, const int keys, const int radius, const Strides &qs, const Strides &ks, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(attention_fp32_statistics(q, k, mask, lse, batch, heads, queries, keys, radius, qs, ks, stream));
    }

    inline void AttentionForwardTuned(const float *q, const float *k, const float *v, const bool *mask, float *output, float *lse,
            const int batch, const int heads, const int queries, const int keys, const int radius, const int group, const Strides &qs,
            const Strides &ks, const Strides &vs, const Strides &os, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(
                attention_fp32_forward_tuned(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, group, qs, ks, vs, os, stream));
    }

    /// Flash attention forward with online softmax and 64-key shared-memory tiles.
    inline void AttentionForwardFlash(const float *q, const float *k, const float *v, const bool *mask, float *output, float *lse,
            const int batch, const int heads, const int queries, const int keys, const int radius, const Strides &qs, const Strides &ks,
            const Strides &vs, const Strides &os, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(attention_fp32_forward_flash(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream));
    }

    /// Variant selector used to reproduce the tuning sweep: 0 through 8.
    inline void AttentionForwardFlashTuned(const float *q, const float *k, const float *v, const bool *mask, float *output, float *lse,
            const int batch, const int heads, const int queries, const int keys, const int radius, const int variant, const Strides &qs,
            const Strides &ks, const Strides &vs, const Strides &os, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(
                attention_fp32_forward_flash_tuned(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, variant, qs, ks, vs, os, stream));
    }

    inline void AttentionBackwardTuned(const float *q, const float *k, const float *v, const bool *mask, const float *output,
            const float *lse, const float *upstream, float *gradient_q, float *gradient_k, float *gradient_v, float *delta, const int batch,
            const int heads, const int queries, const int keys, const int radius, const int group, const Strides &qs, const Strides &ks,
            const Strides &vs, const Strides &os, const Strides &us, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(attention_fp32_backward_tuned(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v, delta,
                batch, heads, queries, keys, radius, group, qs, ks, vs, os, us, stream));
    }
} // namespace ROCmKernels::Kernels
