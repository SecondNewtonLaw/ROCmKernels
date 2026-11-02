#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/Rope.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../TestSupport.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace {
    constexpr int kBatch = 1;
    constexpr int kHeads = 2;
    constexpr int kTokens = 5;
    constexpr int kHeadDim = 16;
    constexpr float kTheta = 10000.0f;

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

    /// Double-precision reference over the exact values the device reads.
    std::vector<float> ComputeReference(const std::vector<float> &input, const std::vector<int> &positions, const int rotaryDim,
            const ROCmKernels::Kernels::RopeStyle style) {
        const int half = rotaryDim / 2;
        std::vector<float> output(input.size());

        for (int batch = 0; batch < kBatch; ++batch) {
            for (int head = 0; head < kHeads; ++head) {
                for (int token = 0; token < kTokens; ++token) {
                    const std::size_t base = ((static_cast<std::size_t>(batch) * kHeads + head) * kTokens + token) * kHeadDim;
                    const int position = positions[batch * kTokens + token];

                    for (int index = 0; index < half; ++index) {
                        const int low = style == ROCmKernels::Kernels::RopeStyle::Normal ? 2 * index : index;
                        const int high = style == ROCmKernels::Kernels::RopeStyle::Normal ? 2 * index + 1 : index + half;
                        const double angle = static_cast<double>(position) * std::pow(static_cast<double>(kTheta), -2.0 * index / rotaryDim);
                        const double cosine = std::cos(angle);
                        const double sine = std::sin(angle);
                        const double x0 = input[base + low];
                        const double x1 = input[base + high];

                        output[base + low] = static_cast<float>(x0 * cosine - x1 * sine);
                        output[base + high] = static_cast<float>(x0 * sine + x1 * cosine);
                    }

                    for (int index = rotaryDim; index < kHeadDim; ++index) output[base + index] = input[base + index];
                }
            }
        }

        return output;
    }

    template<typename T> void RunCase(const int rotaryDim, const ROCmKernels::Kernels::RopeStyle style, const bool inPlace,
            const unsigned int seed) {
        constexpr std::size_t elements = static_cast<std::size_t>(kBatch) * kHeads * kTokens * kHeadDim;

        const std::vector<int> positions{0, 1, 2, 3, 4};
        const std::vector<float> hostInput = ROCmKernels::Tests::RandomValues<float>(elements, seed, -2.0f, 2.0f);
        const std::vector<T> storageInput = Narrow<T>(hostInput);
        const std::vector<float> expected = ComputeReference(Widen<T>(storageInput), positions, rotaryDim, style);

        const ROCmKernels::Kernels::Strides strides{kHeads * kTokens * kHeadDim, kTokens * kHeadDim, kHeadDim};

        ROCmKernels::DeviceBuffer<T> deviceInput(elements);
        ROCmKernels::DeviceBuffer<T> deviceOutput(elements);
        ROCmKernels::DeviceBuffer<int> devicePositions(positions.size());

        deviceInput.CopyFromHost(storageInput.data(), storageInput.size());
        devicePositions.CopyFromHost(positions.data(), positions.size());

        ROCmKernels::Kernels::RopeForward(deviceInput.Data(), inPlace ? deviceInput.Data() : deviceOutput.Data(), devicePositions.Data(), kBatch,
                kHeads, kTokens, kHeadDim, rotaryDim, kTheta, style, strides, strides, ROCmKernels::Kernels::DTypeOf<T>::value);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actualStorage(elements);
        if (inPlace) deviceInput.CopyToHost(actualStorage.data(), actualStorage.size());
        else deviceOutput.CopyToHost(actualStorage.data(), actualStorage.size());

        const double tolerance = std::is_same_v<T, __half> ? 2e-3 : 5e-4;
        ROCmKernels::Tests::CheckClose(expected, Widen<T>(actualStorage), tolerance);
    }
} // namespace

TEST_CASE("Rope matches the reference", "[generic][rope]") {
    using ROCmKernels::Kernels::RopeStyle;

    SECTION("fp32") {
        SECTION("normal") {
            SECTION("rotaryDim 8") { RunCase<float>(8, RopeStyle::Normal, false, 101); }
            SECTION("rotaryDim 16") { RunCase<float>(16, RopeStyle::Normal, false, 102); }
        }
        SECTION("neox") {
            SECTION("rotaryDim 8") { RunCase<float>(8, RopeStyle::NeoX, false, 103); }
            SECTION("rotaryDim 16") { RunCase<float>(16, RopeStyle::NeoX, false, 104); }
        }
    }

    SECTION("fp16") {
        SECTION("normal") {
            SECTION("rotaryDim 8") { RunCase<__half>(8, RopeStyle::Normal, false, 201); }
            SECTION("rotaryDim 16") { RunCase<__half>(16, RopeStyle::Normal, false, 202); }
        }
        SECTION("neox") {
            SECTION("rotaryDim 8") { RunCase<__half>(8, RopeStyle::NeoX, false, 203); }
            SECTION("rotaryDim 16") { RunCase<__half>(16, RopeStyle::NeoX, false, 204); }
        }
    }

    SECTION("in-place") {
        SECTION("fp32 normal rotaryDim 8") { RunCase<float>(8, RopeStyle::Normal, true, 301); }
        SECTION("fp32 neox rotaryDim 8") { RunCase<float>(8, RopeStyle::NeoX, true, 302); }
        SECTION("fp32 normal rotaryDim 16") { RunCase<float>(16, RopeStyle::Normal, true, 303); }
        SECTION("fp32 neox rotaryDim 16") { RunCase<float>(16, RopeStyle::NeoX, true, 304); }
        SECTION("fp16 normal rotaryDim 8") { RunCase<__half>(8, RopeStyle::Normal, true, 401); }
        SECTION("fp16 neox rotaryDim 8") { RunCase<__half>(8, RopeStyle::NeoX, true, 402); }
    }
}
