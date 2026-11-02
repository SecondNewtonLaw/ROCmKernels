#pragma once

namespace ROCmKernels::Kernels {
    /// Element strides of a [batch][head][token] tensor.
    struct Strides {
        int batch;
        int head;
        int token;
    };
} // namespace ROCmKernels::Kernels
