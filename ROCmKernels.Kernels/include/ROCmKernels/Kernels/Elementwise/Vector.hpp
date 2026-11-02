//
// Created by Dottik on 2/11/2026.
//

#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

namespace ROCmKernels::Kernels {
    /// out[i] = lhs[i] + rhs[i] for count elements.
    void VectorAdd(const float *lhs, const float *rhs, float *out, std::size_t count, hipStream_t stream = nullptr);
    void VectorAdd(const double *lhs, const double *rhs, double *out, std::size_t count, hipStream_t stream = nullptr);
} // namespace ROCmKernels::Kernels
