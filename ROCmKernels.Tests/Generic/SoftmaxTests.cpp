#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/Softmax.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../TestSupport.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {
    enum class MaskMode { None, Sparse, FullyMaskedRow };

    template<typename T> T Narrow(const float value) {
        if constexpr (std::is_same_v<T, __half>) return __float2half_rn(value);
        else return value;
    }

    template<typename T> float Widen(const T value) {
        if constexpr (std::is_same_v<T, __half>) return __half2float(value);
        else return value;
    }

    template<typename T> std::vector<T> Narrow(const std::vector<float> &values) {
        std::vector<T> narrowed(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) narrowed[index] = Narrow<T>(values[index]);
        return narrowed;
    }

    template<typename T> std::vector<float> Widen(const std::vector<T> &values) {
        std::vector<float> widened(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) widened[index] = Widen<T>(values[index]);
        return widened;
    }

    std::vector<std::uint8_t> BuildMask(const int rows, const int columns, const MaskMode mode) {
        const std::size_t elements = static_cast<std::size_t>(rows) * columns;
        std::vector<std::uint8_t> mask(elements, 1);

        if (mode == MaskMode::Sparse) {
            mask[0] = 0;
            mask[static_cast<std::size_t>(columns) / 2] = 0;
            mask[elements - 1] = 0;
            if (rows > 3) mask[static_cast<std::size_t>(3) * columns + columns / 3] = 0;
        } else if (mode == MaskMode::FullyMaskedRow) {
            const int maskedRow = rows > 2 ? 2 : 0;
            for (int column = 0; column < columns; ++column) mask[static_cast<std::size_t>(maskedRow) * columns + column] = 0;
        }

        return mask;
    }

    struct Reference {
        std::vector<float> output;
        std::vector<float> rowSums;
    };

    /// Double-precision reference over the exact values the device reads.
    Reference ComputeReference(const std::vector<float> &input, const std::vector<std::uint8_t> &mask, const int rows, const int columns,
            const bool useMask, const float scale) {
        Reference reference;
        reference.output.assign(static_cast<std::size_t>(rows) * columns, 0.0f);
        reference.rowSums.assign(rows, 0.0f);

        constexpr double kNegativeInfinity = -std::numeric_limits<double>::infinity();
        for (int row = 0; row < rows; ++row) {
            const std::size_t base = static_cast<std::size_t>(row) * columns;

            double maximum = kNegativeInfinity;
            for (int column = 0; column < columns; ++column) {
                if (useMask && mask[base + column] == 0) continue;
                maximum = std::max(maximum, static_cast<double>(input[base + column]) * scale);
            }
            if (maximum == kNegativeInfinity) continue;

            double total = 0.0;
            for (int column = 0; column < columns; ++column) {
                if (useMask && mask[base + column] == 0) continue;
                total += std::exp(static_cast<double>(input[base + column]) * scale - maximum);
            }

            for (int column = 0; column < columns; ++column) {
                if (useMask && mask[base + column] == 0) continue;
                reference.output[base + column] =
                    static_cast<float>(std::exp(static_cast<double>(input[base + column]) * scale - maximum) / total);
            }
            reference.rowSums[row] = 1.0f;
        }

        return reference;
    }

    template<typename T> void RunCase(const int rows, const int columns, const MaskMode mode, const unsigned int seed) {
        const std::size_t elements = static_cast<std::size_t>(rows) * columns;
        const bool useMask = mode != MaskMode::None;

        const std::vector<float> hostInput = ROCmKernels::Tests::RandomValues<float>(elements, seed, -1.0f, 1.0f);
        const std::vector<T> storageInput = Narrow<T>(hostInput);
        const std::vector<std::uint8_t> mask = BuildMask(rows, columns, mode);
        const float scale = 1.0f / std::sqrt(static_cast<float>(columns));

        const Reference reference = ComputeReference(Widen<T>(storageInput), mask, rows, columns, useMask, scale);

        ROCmKernels::DeviceBuffer<T> deviceInput(elements);
        ROCmKernels::DeviceBuffer<std::uint8_t> deviceMask(useMask ? elements : 0);
        ROCmKernels::DeviceBuffer<T> deviceOutput(elements);

        deviceInput.CopyFromHost(storageInput.data(), storageInput.size());
        if (useMask) deviceMask.CopyFromHost(mask.data(), mask.size());

        const bool *maskPointer = useMask ? reinterpret_cast<const bool *>(deviceMask.Data()) : nullptr;
        ROCmKernels::Kernels::SoftmaxForward(deviceInput.Data(), maskPointer, deviceOutput.Data(), rows, columns, scale,
                ROCmKernels::Kernels::DTypeOf<T>::value);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actualOutput(elements);
        deviceOutput.CopyToHost(actualOutput.data(), actualOutput.size());
        const std::vector<float> widenedOutput = Widen<T>(actualOutput);

        const double tolerance = std::is_same_v<T, __half> ? 2e-3 : 5e-4;
        ROCmKernels::Tests::CheckClose(reference.output, widenedOutput, tolerance);

        std::vector<float> actualSums(rows);
        for (int row = 0; row < rows; ++row) {
            const std::size_t base = static_cast<std::size_t>(row) * columns;
            double sum = 0.0;
            for (int column = 0; column < columns; ++column) sum += static_cast<double>(widenedOutput[base + column]);
            actualSums[row] = static_cast<float>(sum);
        }
        ROCmKernels::Tests::CheckClose(reference.rowSums, actualSums, tolerance);
    }
} // namespace

TEST_CASE("Softmax matches the reference", "[generic][softmax]") {
    SECTION("fp32") {
        SECTION("no mask") { RunCase<float>(7, 100, MaskMode::None, 101); }
        SECTION("sparse mask") { RunCase<float>(7, 100, MaskMode::Sparse, 201); }
        SECTION("fully masked row") { RunCase<float>(7, 100, MaskMode::FullyMaskedRow, 301); }
        SECTION("single row") { RunCase<float>(1, 100, MaskMode::None, 401); }
        SECTION("wide row") { RunCase<float>(7, 1000, MaskMode::Sparse, 501); }
    }

    SECTION("fp16") {
        SECTION("no mask") { RunCase<__half>(7, 100, MaskMode::None, 601); }
        SECTION("sparse mask") { RunCase<__half>(7, 100, MaskMode::Sparse, 701); }
        SECTION("fully masked row") { RunCase<__half>(7, 100, MaskMode::FullyMaskedRow, 801); }
        SECTION("single row") { RunCase<__half>(1, 100, MaskMode::None, 901); }
        SECTION("wide row") { RunCase<__half>(7, 1000, MaskMode::Sparse, 1001); }
    }
}
