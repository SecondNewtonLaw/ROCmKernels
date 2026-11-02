#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/RMSNorm.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../TestSupport.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace {
    constexpr float kEpsilon = 1e-5f;

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

    struct Reference {
        std::vector<float> output;
        std::vector<float> rstd;
    };

    /// Double-precision reference over the exact values the device reads.
    Reference ComputeReference(const std::vector<float> &x, const std::vector<float> &residual, const std::vector<float> &scale,
            const int rows, const int columns, const bool useResidual, const bool useScale) {
        Reference reference;
        const std::size_t elements = static_cast<std::size_t>(rows) * columns;
        reference.output.resize(elements);
        reference.rstd.resize(rows);

        for (int row = 0; row < rows; ++row) {
            const std::size_t base = static_cast<std::size_t>(row) * columns;

            double squareSum = 0.0;
            for (int column = 0; column < columns; ++column) {
                const double value = static_cast<double>(x[base + column]) +
                                     (useResidual ? static_cast<double>(residual[base + column]) : 0.0);
                squareSum += value * value;
            }

            const double inverse = 1.0 / std::sqrt(squareSum / columns + kEpsilon);
            reference.rstd[row] = static_cast<float>(inverse);

            for (int column = 0; column < columns; ++column) {
                const double value = static_cast<double>(x[base + column]) +
                                     (useResidual ? static_cast<double>(residual[base + column]) : 0.0);
                double normalized = value * inverse;
                if (useScale) normalized *= scale[column];
                reference.output[base + column] = static_cast<float>(normalized);
            }
        }

        return reference;
    }

    template<typename T> void RunCase(const int rows, const int columns, const bool useResidual, const bool useScale, const unsigned int seed) {
        const std::size_t elements = static_cast<std::size_t>(rows) * columns;

        const std::vector<float> hostX = ROCmKernels::Tests::RandomValues<float>(elements, seed, -1.0f, 1.0f);
        const std::vector<float> hostResidual = ROCmKernels::Tests::RandomValues<float>(elements, seed + 1, -1.0f, 1.0f);
        const std::vector<float> hostScale = ROCmKernels::Tests::RandomValues<float>(columns, seed + 2, 0.5f, 1.5f);

        const std::vector<T> storageX = Narrow<T>(hostX);
        const std::vector<T> storageResidual = Narrow<T>(hostResidual);
        const std::vector<T> storageScale = Narrow<T>(hostScale);

        const Reference reference = ComputeReference(Widen<T>(storageX), Widen<T>(storageResidual), Widen<T>(storageScale), rows, columns,
                useResidual, useScale);

        ROCmKernels::DeviceBuffer<T> deviceX(elements);
        ROCmKernels::DeviceBuffer<T> deviceResidual(useResidual ? elements : 0);
        ROCmKernels::DeviceBuffer<T> deviceScale(useScale ? columns : 0);
        ROCmKernels::DeviceBuffer<T> deviceOutput(elements);
        ROCmKernels::DeviceBuffer<float> deviceRstd(rows);

        deviceX.CopyFromHost(storageX.data(), storageX.size());
        if (useResidual) deviceResidual.CopyFromHost(storageResidual.data(), storageResidual.size());
        if (useScale) deviceScale.CopyFromHost(storageScale.data(), storageScale.size());

        ROCmKernels::Kernels::RMSNormForward(deviceX.Data(), deviceResidual.Data(), deviceScale.Data(), deviceOutput.Data(), deviceRstd.Data(),
                rows, columns, kEpsilon, ROCmKernels::Kernels::DTypeOf<T>::value);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actualOutput(elements);
        std::vector<float> actualRstd(rows);
        deviceOutput.CopyToHost(actualOutput.data(), actualOutput.size());
        deviceRstd.CopyToHost(actualRstd.data(), actualRstd.size());

        const double tolerance = std::is_same_v<T, __half> ? 2e-3 : 5e-4;
        ROCmKernels::Tests::CheckClose(reference.output, Widen<T>(actualOutput), tolerance);
        ROCmKernels::Tests::CheckClose(reference.rstd, actualRstd, tolerance);
    }
} // namespace

TEST_CASE("RMSNorm matches the reference", "[generic][norm]") {
    SECTION("fp32") {
        SECTION("residual and scale") { RunCase<float>(7, 100, true, true, 101); }
        SECTION("no residual") { RunCase<float>(7, 100, false, true, 201); }
        SECTION("no scale") { RunCase<float>(7, 100, true, false, 301); }
        SECTION("no residual or scale") { RunCase<float>(7, 100, false, false, 401); }
        SECTION("single row") { RunCase<float>(1, 100, true, true, 501); }
    }

    SECTION("fp16") {
        SECTION("residual and scale") { RunCase<__half>(7, 100, true, true, 601); }
        SECTION("no residual") { RunCase<__half>(7, 100, false, true, 701); }
        SECTION("no scale") { RunCase<__half>(7, 100, true, false, 801); }
        SECTION("no residual or scale") { RunCase<__half>(7, 100, false, false, 901); }
        SECTION("single row") { RunCase<__half>(1, 100, true, true, 1001); }
    }
}
