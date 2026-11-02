#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/LinearActFast.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../TestSupport.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace {
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

    double ReferenceSiLU(const double value) {
        return value / (1.0 + std::exp(-value));
    }

    template<typename T> void RunFastLinearCase(const int rows, const int columns, const int inner, const bool withBias,
            const unsigned int seed) {
        const std::size_t inputElements = static_cast<std::size_t>(rows) * inner;
        const std::size_t weightElements = static_cast<std::size_t>(inner) * columns;
        const std::size_t outputElements = static_cast<std::size_t>(rows) * columns;

        const std::vector<float> hostInput = ROCmKernels::Tests::RandomValues<float>(inputElements, seed, -1.0f, 1.0f);
        const std::vector<float> hostWeight = ROCmKernels::Tests::RandomValues<float>(weightElements, seed + 1, -1.0f, 1.0f);
        const std::vector<float> hostBias = ROCmKernels::Tests::RandomValues<float>(withBias ? columns : 0, seed + 2, -1.0f, 1.0f);

        const std::vector<T> storageInput = Narrow<T>(hostInput);
        const std::vector<T> storageWeight = Narrow<T>(hostWeight);
        const std::vector<T> storageBias = Narrow<T>(hostBias);

        const std::vector<float> referenceInput = Widen<T>(storageInput);
        const std::vector<float> referenceWeight = Widen<T>(storageWeight);
        const std::vector<float> referenceBias = Widen<T>(storageBias);

        std::vector<float> expected(outputElements);
        for (int row = 0; row < rows; ++row) {
            for (int column = 0; column < columns; ++column) {
                double sum = withBias ? referenceBias[column] : 0.0;
                for (int k = 0; k < inner; ++k) {
                    sum += static_cast<double>(referenceInput[static_cast<std::size_t>(row) * inner + k]) *
                           referenceWeight[static_cast<std::size_t>(k) * columns + column];
                }
                expected[static_cast<std::size_t>(row) * columns + column] = static_cast<float>(ReferenceSiLU(sum));
            }
        }

        ROCmKernels::DeviceBuffer<T> deviceInput(inputElements);
        ROCmKernels::DeviceBuffer<T> deviceWeight(weightElements);
        ROCmKernels::DeviceBuffer<T> deviceBias(withBias ? columns : 0);
        ROCmKernels::DeviceBuffer<T> deviceOutput(outputElements);

        deviceInput.CopyFromHost(storageInput.data(), storageInput.size());
        deviceWeight.CopyFromHost(storageWeight.data(), storageWeight.size());
        if (withBias) deviceBias.CopyFromHost(storageBias.data(), storageBias.size());

        ROCmKernels::Kernels::LinearActFast(deviceInput.Data(), deviceWeight.Data(), withBias ? deviceBias.Data() : nullptr, deviceOutput.Data(),
                rows, columns, inner, ROCmKernels::Kernels::DTypeOf<T>::value);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actual(outputElements);
        deviceOutput.CopyToHost(actual.data(), actual.size());

        const double tolerance = std::is_same_v<T, __half> ? 2e-3 : 5e-4;
        ROCmKernels::Tests::CheckClose(expected, Widen<T>(actual), tolerance);
    }
} // namespace

TEST_CASE("LinearActFast matches the reference", "[generic][linear][fast]") {
    SECTION("fp32") {
        SECTION("tailed matrix with bias") { RunFastLinearCase<float>(17, 19, 23, true, 101); }
        SECTION("tailed matrix without bias") { RunFastLinearCase<float>(17, 19, 23, false, 102); }
        SECTION("full tile with bias") { RunFastLinearCase<float>(64, 64, 16, true, 103); }
        SECTION("full tile without bias") { RunFastLinearCase<float>(64, 64, 16, false, 104); }
        SECTION("multiple tiles with tails") { RunFastLinearCase<float>(65, 67, 31, true, 105); }
    }

    SECTION("fp16") {
        SECTION("tailed matrix with bias") { RunFastLinearCase<__half>(17, 19, 23, true, 201); }
        SECTION("tailed matrix without bias") { RunFastLinearCase<__half>(17, 19, 23, false, 202); }
        SECTION("full tile with bias") { RunFastLinearCase<__half>(64, 64, 16, true, 203); }
        SECTION("full tile without bias") { RunFastLinearCase<__half>(64, 64, 16, false, 204); }
        SECTION("multiple tiles with tails") { RunFastLinearCase<__half>(65, 67, 31, true, 205); }
    }
}
