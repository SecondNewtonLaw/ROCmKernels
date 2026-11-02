#pragma once

#include <ROCmKernels/Kernels/DType.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

namespace ROCmKernels::Kernels {
    /// Arithmetic applied to every broadcast element pair.
    enum class BinaryOp : unsigned char { Add, Sub, Mul, Div };

    /// Shape of the second operand. The first operand is always [rows][columns].
    enum class BroadcastMode : unsigned char { None, Row, Column, Scalar };
} // namespace ROCmKernels::Kernels

extern "C" ROCMKERNELS_EXPORT int broadcast_binary_forward(const void *a, const void *b, void *out, int rows, int columns,
        ROCmKernels::Kernels::BinaryOp op, ROCmKernels::Kernels::BroadcastMode mode, ROCmKernels::Kernels::DType dtype, hipStream_t stream);

extern "C" ROCMKERNELS_EXPORT int copy_2d_forward(const void *src, void *dst, int rows, int columns, int sourceRowStride,
        int destinationRowStride, ROCmKernels::Kernels::DType dtype, hipStream_t stream);

extern "C" ROCMKERNELS_EXPORT int fill_forward(void *dst, float value, long long elements, ROCmKernels::Kernels::DType dtype, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// out[r][c] = op(a[r][c], b[...]), where b is [rows][columns], [columns], [rows] or one element.
    ///
    /// a may equal out for exact in-place operation; any other overlap with out returns hipErrorInvalidValue.
    /// b is reused across rows, so b must not overlap out.
    inline void BroadcastBinary(const void *a, const void *b, void *out, const int rows, const int columns, const BinaryOp op,
            const BroadcastMode mode, const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(broadcast_binary_forward(a, b, out, rows, columns, op, mode, dtype, stream));
    }

    /// Copies rows of columns contiguous elements; row strides are in elements and must be at least columns.
    ///
    /// src and dst must not overlap.
    inline void Copy2D(const void *src, void *dst, const int rows, const int columns, const int sourceRowStride,
            const int destinationRowStride, const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(copy_2d_forward(src, dst, rows, columns, sourceRowStride, destinationRowStride, dtype, stream));
    }

    /// Writes value to every element.
    inline void Fill(void *dst, const float value, const long long elements, const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(fill_forward(dst, value, elements, dtype, stream));
    }
} // namespace ROCmKernels::Kernels
