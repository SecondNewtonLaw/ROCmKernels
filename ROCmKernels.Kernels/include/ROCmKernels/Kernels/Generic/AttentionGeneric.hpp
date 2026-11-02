#pragma once

#include <ROCmKernels/Kernels/DType.hpp>
#include <ROCmKernels/Kernels/Strides.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int attention_generic_forward(const void *q, const void *k, const void *v, const bool *mask, void *output,
        float *lse, int batch, int heads, int queries, int keys, int headWidth, int radius, ROCmKernels::Kernels::Strides qs,
        ROCmKernels::Kernels::Strides ks, ROCmKernels::Kernels::Strides vs, ROCmKernels::Kernels::Strides os,
        ROCmKernels::Kernels::DType dtype, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// Windowed flash attention forward for 16-, 32- or 64-wide heads.
    ///
    /// mask is [batch][keys]. radius < 0 attends globally. lse receives {row maximum, exponent sum} per (batch, head, query).
    /// q, k and v use qs, ks and vs; output uses os.
    inline void AttentionGenericForward(const void *q, const void *k, const void *v, const bool *mask, void *output, float *lse,
            const int batch, const int heads, const int queries, const int keys, const int headWidth, const int radius, const Strides &qs,
            const Strides &ks, const Strides &vs, const Strides &os, const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(attention_generic_forward(q, k, v, mask, output, lse, batch, heads, queries, keys, headWidth, radius, qs, ks,
                vs, os, dtype, stream));
    }
} // namespace ROCmKernels::Kernels
