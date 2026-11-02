#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/AdamW.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../TestSupport.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace {
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.999f;
    constexpr float kEpsilon = 1e-8f;
    constexpr float kLearningRate = 1e-3f;
    constexpr float kWeightDecay = 0.01f;

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

    /// Double-precision reference over the values the device actually reads.
    template<typename T> void RunCase(const std::int64_t step, const unsigned int seed) {
        constexpr int tensorCount = 3;
        const std::int64_t sizes[tensorCount] = {1000, 33, 4096};

        const double firstCorrection = 1.0 - std::pow(static_cast<double>(kBeta1), static_cast<double>(step));
        const double secondCorrection = 1.0 - std::pow(static_cast<double>(kBeta2), static_cast<double>(step));

        std::vector<std::vector<float>> referenceParameter(tensorCount);
        std::vector<std::vector<float>> referenceFirst(tensorCount);
        std::vector<std::vector<float>> referenceSecond(tensorCount);
        std::vector<ROCmKernels::DeviceBuffer<T>> deviceParameter;
        std::vector<ROCmKernels::DeviceBuffer<T>> deviceGradient;
        std::vector<ROCmKernels::DeviceBuffer<float>> deviceFirst;
        std::vector<ROCmKernels::DeviceBuffer<float>> deviceSecond;
        std::vector<ROCmKernels::Kernels::AdamWTensor> descriptors(tensorCount);

        deviceParameter.reserve(tensorCount);
        deviceGradient.reserve(tensorCount);
        deviceFirst.reserve(tensorCount);
        deviceSecond.reserve(tensorCount);

        for (int tensor = 0; tensor < tensorCount; ++tensor) {
            const std::size_t elements = static_cast<std::size_t>(sizes[tensor]);
            const unsigned int tensorSeed = seed + static_cast<unsigned int>(tensor) * 4;

            const std::vector<float> randomParameter = ROCmKernels::Tests::RandomValues<float>(elements, tensorSeed, -1.0f, 1.0f);
            const std::vector<float> randomGradient = ROCmKernels::Tests::RandomValues<float>(elements, tensorSeed + 1, -1.0f, 1.0f);
            const std::vector<float> initialFirst = ROCmKernels::Tests::RandomValues<float>(elements, tensorSeed + 2, 0.0f, 1.0f);
            const std::vector<float> initialSecond = ROCmKernels::Tests::RandomValues<float>(elements, tensorSeed + 3, 0.0f, 1.0f);

            const std::vector<T> storageParameter = Narrow<T>(randomParameter);
            const std::vector<T> storageGradient = Narrow<T>(randomGradient);
            const std::vector<float> inputParameter = Widen<T>(storageParameter);
            const std::vector<float> inputGradient = Widen<T>(storageGradient);

            referenceParameter[tensor].resize(elements);
            referenceFirst[tensor].resize(elements);
            referenceSecond[tensor].resize(elements);
            for (std::size_t index = 0; index < elements; ++index) {
                const double gradient = inputGradient[index];
                const double first = static_cast<double>(kBeta1) * initialFirst[index] + (1.0 - kBeta1) * gradient;
                const double second = static_cast<double>(kBeta2) * initialSecond[index] + (1.0 - kBeta2) * gradient * gradient;
                const double firstHat = first / firstCorrection;
                const double secondHat = second / secondCorrection;

                double parameter = inputParameter[index];
                parameter -= kLearningRate * kWeightDecay * parameter;
                parameter -= kLearningRate * firstHat / (std::sqrt(secondHat) + kEpsilon);

                referenceParameter[tensor][index] = static_cast<float>(parameter);
                referenceFirst[tensor][index] = static_cast<float>(first);
                referenceSecond[tensor][index] = static_cast<float>(second);
            }

            deviceParameter.emplace_back(elements);
            deviceGradient.emplace_back(elements);
            deviceFirst.emplace_back(elements);
            deviceSecond.emplace_back(elements);

            deviceParameter[tensor].CopyFromHost(storageParameter.data(), elements);
            deviceGradient[tensor].CopyFromHost(storageGradient.data(), elements);
            deviceFirst[tensor].CopyFromHost(initialFirst.data(), elements);
            deviceSecond[tensor].CopyFromHost(initialSecond.data(), elements);

            descriptors[tensor] = ROCmKernels::Kernels::AdamWTensor{
                deviceParameter[tensor].Data(), deviceGradient[tensor].Data(), deviceFirst[tensor].Data(), deviceSecond[tensor].Data(), sizes[tensor]
            };
        }

        ROCmKernels::Kernels::AdamWMulti(descriptors.data(), tensorCount, step, kLearningRate, kBeta1, kBeta2, kEpsilon, kWeightDecay,
                ROCmKernels::Kernels::DTypeOf<T>::value);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        const double parameterTolerance = std::is_same_v<T, __half> ? 2e-3 : 5e-4;
        for (int tensor = 0; tensor < tensorCount; ++tensor) {
            const std::size_t elements = static_cast<std::size_t>(sizes[tensor]);
            std::vector<T> actualParameter(elements);
            std::vector<float> actualFirst(elements);
            std::vector<float> actualSecond(elements);

            deviceParameter[tensor].CopyToHost(actualParameter.data(), elements);
            deviceFirst[tensor].CopyToHost(actualFirst.data(), elements);
            deviceSecond[tensor].CopyToHost(actualSecond.data(), elements);

            ROCmKernels::Tests::CheckClose(referenceParameter[tensor], Widen<T>(actualParameter), parameterTolerance);
            ROCmKernels::Tests::CheckClose(referenceFirst[tensor], actualFirst, 5e-4);
            ROCmKernels::Tests::CheckClose(referenceSecond[tensor], actualSecond, 5e-4);
        }
    }
} // namespace

TEST_CASE("AdamW multi-tensor matches the reference", "[generic][optimizer]") {
    SECTION("fp32") {
        SECTION("step 1") { RunCase<float>(1, 101); }
        SECTION("step 7") { RunCase<float>(7, 201); }
    }

    SECTION("fp16") {
        SECTION("step 1") { RunCase<__half>(1, 301); }
        SECTION("step 7") { RunCase<__half>(7, 401); }
    }
}

TEST_CASE("AdamW rejects invalid arguments", "[generic][optimizer]") {
    using ROCmKernels::Kernels::AdamWTensor;
    using ROCmKernels::Kernels::DType;

    const AdamWTensor empty{nullptr, nullptr, nullptr, nullptr, 0};
    const auto invoke = [](const AdamWTensor *tensors, const int count, const long long step, const float learningRate, const float beta1,
                               const DType dtype) {
        return adamw_multi(tensors, count, step, learningRate, beta1, 0.999f, 1e-8f, 0.01f, dtype, nullptr);
    };

    CHECK(invoke(nullptr, 1, 1, 1e-3f, 0.9f, DType::Float32) == static_cast<int>(hipErrorInvalidValue));
    CHECK(invoke(&empty, 0, 1, 1e-3f, 0.9f, DType::Float32) == static_cast<int>(hipErrorInvalidValue));
    CHECK(invoke(&empty, 1, 0, 1e-3f, 0.9f, DType::Float32) == static_cast<int>(hipErrorInvalidValue));
    CHECK(invoke(&empty, 1, 1, -1e-3f, 0.9f, DType::Float32) == static_cast<int>(hipErrorInvalidValue));
    CHECK(invoke(&empty, 1, 1, 1e-3f, 1.0f, DType::Float32) == static_cast<int>(hipErrorInvalidValue));
    CHECK(invoke(&empty, 1, 1, 1e-3f, 0.9f, static_cast<DType>(7)) == static_cast<int>(hipErrorInvalidValue));

    // A tensor with no elements is skipped, so a null payload is never dereferenced.
    CHECK(invoke(&empty, 1, 1, 1e-3f, 0.9f, DType::Float32) == static_cast<int>(hipSuccess));
}
