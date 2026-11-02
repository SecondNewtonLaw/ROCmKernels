//
// Created by Dottik on 2/11/2026.
//

#pragma once

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace ROCmKernels {
    /// Raised when a HIP API call does not return hipSuccess.
    class HipError final : public std::runtime_error {
        hipError_t m_status;

      public:
        HipError(hipError_t status, const std::string &message) : std::runtime_error(message), m_status(status) {}

        [[nodiscard]] hipError_t Status() const noexcept { return this->m_status; }
    };

    /// Builds and throws a HipError describing the failing expression and where it was evaluated.
    [[noreturn]] inline void ThrowHipError(hipError_t status, const char *expression, const char *file, const int line) {
        std::string message = "HIP call failed: ";
        message += expression;
        message += " -> ";
        message += hipGetErrorString(status);
        message += " (";
        message += file;
        message += ":";
        message += std::to_string(line);
        message += ")";

        throw HipError(status, message);
    }
} // namespace ROCmKernels

#define ROCMKERNELS_HIP_CHECK(expression)                                                                                                                      \
    do {                                                                                                                                                       \
        const hipError_t rocmkernelsHipStatus = static_cast<hipError_t>(expression);                                                                                                  \
        if (rocmkernelsHipStatus != hipSuccess) {                                                                                                              \
            ::ROCmKernels::ThrowHipError(rocmkernelsHipStatus, #expression, __FILE__, __LINE__);                                                               \
        }                                                                                                                                                      \
    } while (false)
