#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>

namespace ROCmKernels::Kernels {
    /// destination[i] = source[i] for count elements.
    void Copy(const float *source, float *destination, std::size_t count, hipStream_t stream = nullptr);
} // namespace ROCmKernels::Kernels
