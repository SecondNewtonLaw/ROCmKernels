#pragma once

#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int muon_fp32_supported(int rows, int columns);
extern "C" ROCMKERNELS_EXPORT int muon_fp32_prepare(const std::int64_t *gradients, const std::int64_t *buffers, float *y, int *flags, float *sumsq,
        int count, int rows, int columns, float momentum, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int muon_fp32_gram(const float *y, const float *gin, float *g, const float *sumsq, int count, int m, int n, float alpha,
        float beta, float diag, int mode, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int muon_fp32_multiply(const float *y, const float *b, float *out, const float *sumsq, int count, int m, int n, int mode,
        hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int muon_fp32_apply(const float *y, const float *b, const std::int64_t *params, const int *flags, const float *sumsq, int count,
        int m, int n, int mode, int transposed, float decay_factor, float step, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// True when both dimensions are multiples of 128, which every Muon kernel requires.
    inline bool MuonSupported(const int rows, const int columns) { return muon_fp32_supported(rows, columns) != 0; }

    /// Momentum update, Nesterov combination and sum of squares for count matrices.
    ///
    /// gradients and buffers hold one device address per matrix. y is count * rows * columns floats.
    inline void MuonPrepare(const std::int64_t *gradients, const std::int64_t *buffers, float *y, int *flags, float *sumsq,
            const int count, const int rows, const int columns, const float momentum, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(muon_fp32_prepare(gradients, buffers, y, flags, sumsq, count, rows, columns, momentum, stream));
    }

    /// g = scale * y^T y + beta * gin, with the 1 / (Frobenius norm + 1e-7) scale when mode is set.
    inline void MuonGram(const float *y, const float *gin, float *g, const float *sumsq, const int count, const int m, const int n,
            const float alpha, const float beta, const float diag, const bool scaled, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(muon_fp32_gram(y, gin, g, sumsq, count, m, n, alpha, beta, diag, scaled ? 1 : 0, stream));
    }

    /// out = scale * y @ b.
    inline void MuonMultiply(const float *y, const float *b, float *out, const float *sumsq, const int count, const int m, const int n,
            const bool scaled, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(muon_fp32_multiply(y, b, out, sumsq, count, m, n, scaled ? 1 : 0, stream));
    }

    /// Parameter update for live matrices: p = p * (1 - decay) - step * scale * (y @ b).
    inline void MuonApply(const float *y, const float *b, const std::int64_t *params, const int *flags, const float *sumsq,
            const int count, const int m, const int n, const bool scaled, const bool transposed, const float decay, const float step,
            hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(muon_fp32_apply(y, b, params, flags, sumsq, count, m, n, scaled ? 1 : 0, transposed ? 1 : 0, decay, step,
                stream));
    }
} // namespace ROCmKernels::Kernels
