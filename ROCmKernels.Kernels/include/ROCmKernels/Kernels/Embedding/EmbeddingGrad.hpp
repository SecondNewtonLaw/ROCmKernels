#pragma once

#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int embedding_grad_fp32(const std::int64_t *keys, const std::int64_t *order, const float *values, float *dense, int nnz,
        int rows, int columns, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int embedding_grad_fp32_tuned(const std::int64_t *keys, const std::int64_t *order, const float *values, float *dense,
        int nnz, int rows, int columns, int vectorized, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// Converts a sorted sparse row list into a dense [rows][columns] buffer.
    ///
    /// keys is nondecreasing and holds nnz row indices; order permutes the nnz rows of values.
    inline void EmbeddingGrad(const std::int64_t *keys, const std::int64_t *order, const float *values, float *dense, const int nnz,
            const int rows, const int columns, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(embedding_grad_fp32(keys, order, values, dense, nnz, rows, columns, stream));
    }

    /// Same conversion, using float4 traffic when columns is a multiple of four and both buffers are 16-byte aligned.
    inline void EmbeddingGradTuned(const std::int64_t *keys, const std::int64_t *order, const float *values, float *dense, const int nnz,
            const int rows, const int columns, const bool vectorized, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(embedding_grad_fp32_tuned(keys, order, values, dense, nnz, rows, columns, vectorized ? 1 : 0, stream));
    }
} // namespace ROCmKernels::Kernels
