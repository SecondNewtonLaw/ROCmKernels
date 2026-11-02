#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/LinearActVec.hpp>

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

    template<typename T> void RunLinearVecCase(const int columns, const int inner, const bool withBias, const unsigned int seed) {
        const std::size_t inputElements = static_cast<std::size_t>(inner);
        const std::size_t weightElements = static_cast<std::size_t>(columns) * inner;
        const std::size_t outputElements = static_cast<std::size_t>(columns);

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
        for (int column = 0; column < columns; ++column) {
            double sum = withBias ? static_cast<double>(referenceBias[column]) : 0.0;
            for (int k = 0; k < inner; ++k) {
                sum += static_cast<double>(referenceInput[k]) *
                       static_cast<double>(referenceWeight[static_cast<std::size_t>(column) * inner + k]);
            }
            expected[column] = static_cast<float>(ReferenceSiLU(sum));
        }

        ROCmKernels::DeviceBuffer<T> deviceInput(inputElements);
        ROCmKernels::DeviceBuffer<T> deviceWeight(weightElements);
        ROCmKernels::DeviceBuffer<T> deviceBias(withBias ? columns : 0);
        ROCmKernels::DeviceBuffer<T> deviceOutput(outputElements);

        deviceInput.CopyFromHost(storageInput.data(), storageInput.size());
        deviceWeight.CopyFromHost(storageWeight.data(), storageWeight.size());
        if (withBias) deviceBias.CopyFromHost(storageBias.data(), storageBias.size());

        ROCmKernels::Kernels::LinearActVec(deviceInput.Data(), deviceWeight.Data(), withBias ? deviceBias.Data() : nullptr,
                deviceOutput.Data(), columns, inner, ROCmKernels::Kernels::DTypeOf<T>::value);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actual(outputElements);
        deviceOutput.CopyToHost(actual.data(), actual.size());

        const double tolerance = std::is_same_v<T, __half> ? 2e-3 : 5e-4;
        ROCmKernels::Tests::CheckClose(expected, Widen<T>(actual), tolerance);
    }
} // namespace

TEST_CASE("LinearActVec matches the reference", "[generic][linear]") {
    SECTION("fp32") {
        SECTION("tailed matrix with bias") { RunLinearVecCase<float>(37, 100, true, 301); }
        SECTION("tailed matrix without bias") { RunLinearVecCase<float>(37, 100, false, 302); }
        SECTION("single element") { RunLinearVecCase<float>(1, 1, true, 303); }
        SECTION("inner warp tail") { RunLinearVecCase<float>(37, 33, true, 304); }
    }

    SECTION("fp16") {
        SECTION("tailed matrix with bias") { RunLinearVecCase<__half>(37, 100, true, 401); }
        SECTION("tailed matrix without bias") { RunLinearVecCase<__half>(37, 100, false, 402); }
        SECTION("single element") { RunLinearVecCase<__half>(1, 1, true, 403); }
        SECTION("inner warp tail") { RunLinearVecCase<__half>(37, 33, true, 404); }
    }
}
