#pragma once

#include <ROCmKernels/Kernels/DType.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int get_rows_forward(const void *table, const int *indices, void *output, int rows, int columns,
        ROCmKernels::Kernels::DType dtype, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int set_rows_forward(const void *input, const int *indices, void *table, int rows, int columns,
        ROCmKernels::Kernels::DType dtype, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int add_rows_forward(const void *input, const int *indices, void *table, int rows, int columns,
        ROCmKernels::Kernels::DType dtype, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// Gathers rows: output[r][c] = table[indices[r]][c].
    ///
    /// table, input and output are dense with a row stride of columns. indices must hold rows values in [0, tableRows).
    inline void GetRowsForward(const void *table, const int *indices, void *output, const int rows, const int columns, const DType dtype,
            hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(get_rows_forward(table, indices, output, rows, columns, dtype, stream));
    }

    /// Overwrites rows: table[indices[r]][c] = input[r][c]. indices must be unique; duplicate targets race and the last write is unspecified.
    inline void SetRowsForward(const void *input, const int *indices, void *table, const int rows, const int columns, const DType dtype,
            hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(set_rows_forward(input, indices, table, rows, columns, dtype, stream));
    }

    /// Accumulates rows: table[indices[r]][c] += input[r][c]. Duplicate indices are safe, but atomic order is unspecified, so rounded sums can vary.
    ///
    /// Only Float32 is supported; Float16 throws because the target has no half atomic add.
    inline void AddRowsForward(const void *input, const int *indices, void *table, const int rows, const int columns, const DType dtype,
            hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(add_rows_forward(input, indices, table, rows, columns, dtype, stream));
    }
} // namespace ROCmKernels::Kernels
