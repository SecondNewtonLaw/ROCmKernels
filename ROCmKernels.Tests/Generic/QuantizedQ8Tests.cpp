#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/QuantizedQ8.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../TestSupport.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace {
    using ROCmKernels::Kernels::BlockQ8_0;
    using ROCmKernels::Kernels::kQ8Block;

    template<typename T> T Narrow(const float value) {
        if constexpr (std::is_same_v<T, __half>) return __float2half_rn(value);
        else return value;
    }

    template<typename T> float Widen(const T value) {
        if constexpr (std::is_same_v<T, __half>) return __half2float(value);
        else return value;
    }

    BlockQ8_0 QuantizeBlock(const float *values) {
        float maximum = 0.0f;
        for (int index = 0; index < kQ8Block; ++index) maximum = std::max(maximum, std::abs(values[index]));

        const float scale = maximum / 127.0f;
        const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;

        BlockQ8_0 block{};
        block.scale = __float2half_rn(scale);
        for (int index = 0; index < kQ8Block; ++index) block.values[index] = static_cast<std::int8_t>(std::round(values[index] * inverse));
        return block;
    }

    std::vector<BlockQ8_0> QuantizeHost(const std::vector<float> &values) {
        std::vector<BlockQ8_0> blocks(values.size() / kQ8Block);
        for (std::size_t block = 0; block < blocks.size(); ++block) blocks[block] = QuantizeBlock(values.data() + block * kQ8Block);
        return blocks;
    }

    template<typename T> void CheckRelative(const std::vector<T> &expected, const std::vector<float> &actual, const double tolerance) {
        REQUIRE(expected.size() == actual.size());

        for (std::size_t index = 0; index < expected.size(); ++index) {
            const double expectedValue = static_cast<double>(expected[index]);
            const double difference = std::abs(expectedValue - static_cast<double>(actual[index]));
            const double allowed = tolerance * std::max(1.0, std::abs(expectedValue));
            INFO("index " << index << " expected " << expectedValue << " got " << actual[index] << " allowed " << allowed);
            CHECK(difference <= allowed);
        }
    }

    void RunRoundTrip(const int elements, const unsigned int seed) {
        using namespace ROCmKernels::Kernels;

        const std::vector<float> host = ROCmKernels::Tests::RandomValues<float>(elements, seed, -1.0f, 1.0f);

        ROCmKernels::DeviceBuffer<float> deviceInput(elements);
        ROCmKernels::DeviceBuffer<BlockQ8_0> deviceBlocks(elements / kQ8Block);
        ROCmKernels::DeviceBuffer<float> deviceOutput(elements);
        deviceInput.CopyFromHost(host.data(), host.size());

        QuantizeQ8_0(deviceInput.Data(), deviceBlocks.Data(), elements);
        DequantizeQ8_0(deviceBlocks.Data(), deviceOutput.Data(), elements);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> actual(elements);
        deviceOutput.CopyToHost(actual.data(), actual.size());
        CheckRelative(host, actual, 2e-2);
    }

    void RunSmallMagnitudeRoundTrip(const unsigned int seed) {
        using namespace ROCmKernels::Kernels;

        const std::vector<float> host = ROCmKernels::Tests::RandomValues<float>(kQ8Block, seed, -1.0e-6f, 1.0e-6f);

        ROCmKernels::DeviceBuffer<float> deviceInput(kQ8Block);
        ROCmKernels::DeviceBuffer<BlockQ8_0> deviceBlocks(1);
        ROCmKernels::DeviceBuffer<float> deviceOutput(kQ8Block);
        deviceInput.CopyFromHost(host.data(), host.size());

        QuantizeQ8_0(deviceInput.Data(), deviceBlocks.Data(), kQ8Block);
        DequantizeQ8_0(deviceBlocks.Data(), deviceOutput.Data(), kQ8Block);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<BlockQ8_0> blocks(1);
        std::vector<float> actual(kQ8Block);
        deviceBlocks.CopyToHost(blocks.data(), blocks.size());
        deviceOutput.CopyToHost(actual.data(), actual.size());

        const float storedScale = __half2float(blocks[0].scale);
        REQUIRE(storedScale > 0.0f);
        CHECK(std::any_of(actual.begin(), actual.end(), [](const float value) { return value != 0.0f; }));

        for (int index = 0; index < kQ8Block; ++index) {
            INFO("index " << index << " expected " << host[index] << " got " << actual[index]);
            CHECK(std::abs(actual[index] - host[index]) <= storedScale);
        }
    }

    void RunZeroBlock() {
        using namespace ROCmKernels::Kernels;

        const std::vector<float> host(kQ8Block, 0.0f);
        ROCmKernels::DeviceBuffer<float> deviceInput(kQ8Block);
        ROCmKernels::DeviceBuffer<BlockQ8_0> deviceBlocks(1);
        ROCmKernels::DeviceBuffer<float> deviceOutput(kQ8Block);
        deviceInput.CopyFromHost(host.data(), host.size());

        QuantizeQ8_0(deviceInput.Data(), deviceBlocks.Data(), kQ8Block);
        DequantizeQ8_0(deviceBlocks.Data(), deviceOutput.Data(), kQ8Block);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<BlockQ8_0> blocks(1);
        std::vector<float> actual(kQ8Block);
        deviceBlocks.CopyToHost(blocks.data(), blocks.size());
        deviceOutput.CopyToHost(actual.data(), actual.size());

        CHECK(__half2float(blocks[0].scale) == 0.0f);
        CheckRelative(std::vector<float>(kQ8Block, 0.0f), actual, 0.0);
    }

    template<typename T> void RunGemvCase(const int columns, const int inner, const unsigned int seed, const bool useBias) {
        using namespace ROCmKernels::Kernels;

        const std::size_t blocksPerRow = inner / kQ8Block;
        const std::vector<float> hostInput = ROCmKernels::Tests::RandomValues<float>(inner, seed, -1.0f, 1.0f);
        const std::vector<float> hostWeights =
                ROCmKernels::Tests::RandomValues<float>(static_cast<std::size_t>(columns) * inner, seed + 1, -1.0f, 1.0f);
        const std::vector<float> hostBias = ROCmKernels::Tests::RandomValues<float>(columns, seed + 2, -0.5f, 0.5f);

        const std::vector<BlockQ8_0> inputBlocks = QuantizeHost(hostInput);
        const std::vector<BlockQ8_0> weightBlocks = QuantizeHost(hostWeights);

        // Reference over the exact values the device dequantizes, accumulated in double.
        std::vector<double> reference(columns, 0.0);
        for (int column = 0; column < columns; ++column) {
            double total = 0.0;
            for (std::size_t block = 0; block < blocksPerRow; ++block) {
                const BlockQ8_0 &a = inputBlocks[block];
                const BlockQ8_0 &b = weightBlocks[static_cast<std::size_t>(column) * blocksPerRow + block];

                int dot = 0;
                for (int index = 0; index < kQ8Block; ++index) dot += static_cast<int>(a.values[index]) * static_cast<int>(b.values[index]);
                total += static_cast<double>(dot) * static_cast<double>(__half2float(a.scale)) * static_cast<double>(__half2float(b.scale));
            }

            if (useBias) total += static_cast<double>(Widen<T>(Narrow<T>(hostBias[column])));
            reference[column] = total;
        }

        ROCmKernels::DeviceBuffer<BlockQ8_0> deviceInput(inputBlocks.size());
        ROCmKernels::DeviceBuffer<BlockQ8_0> deviceWeights(weightBlocks.size());
        ROCmKernels::DeviceBuffer<T> deviceBias(useBias ? columns : 0);
        ROCmKernels::DeviceBuffer<T> deviceOutput(columns);

        deviceInput.CopyFromHost(inputBlocks.data(), inputBlocks.size());
        deviceWeights.CopyFromHost(weightBlocks.data(), weightBlocks.size());
        if (useBias) {
            std::vector<T> storageBias(columns);
            for (int column = 0; column < columns; ++column) storageBias[column] = Narrow<T>(hostBias[column]);
            deviceBias.CopyFromHost(storageBias.data(), storageBias.size());
        }

        QuantizedGEMVQ8_0(deviceInput.Data(), deviceWeights.Data(), useBias ? deviceBias.Data() : nullptr, deviceOutput.Data(), columns, inner,
                DTypeOf<T>::value);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actual(columns);
        deviceOutput.CopyToHost(actual.data(), actual.size());

        std::vector<float> widened(columns);
        for (int column = 0; column < columns; ++column) widened[column] = Widen<T>(actual[column]);
        CheckRelative(reference, widened, 1e-2);
    }
} // namespace

TEST_CASE("Q8_0 quantize and dequantize round trip", "[generic][quantized]") {
    SECTION("32 elements") { RunRoundTrip(32, 1101); }
    SECTION("128 elements") { RunRoundTrip(128, 1201); }
    SECTION("small magnitude block") { RunSmallMagnitudeRoundTrip(1301); }
    SECTION("zero block") { RunZeroBlock(); }
}

TEST_CASE("Q8_0 GEMV matches the dequantized reference", "[generic][quantized]") {
    SECTION("fp32 output") {
        SECTION("columns 37, inner 64") {
            RunGemvCase<float>(37, 64, 2101, true);
            RunGemvCase<float>(37, 64, 2201, false);
        }
        SECTION("columns 5, inner 128") {
            RunGemvCase<float>(5, 128, 2301, true);
            RunGemvCase<float>(5, 128, 2401, false);
        }
    }

    SECTION("fp16 output") {
        SECTION("columns 37, inner 64") {
            RunGemvCase<__half>(37, 64, 3101, true);
            RunGemvCase<__half>(37, 64, 3201, false);
        }
        SECTION("columns 5, inner 128") {
            RunGemvCase<__half>(5, 128, 3301, true);
            RunGemvCase<__half>(5, 128, 3401, false);
        }
    }
}

TEST_CASE("Q8_0 rejects invalid shapes", "[generic][quantized]") {
    using namespace ROCmKernels::Kernels;

    float values[64] = {};
    BlockQ8_0 blocks[2] = {};

    CHECK(quantize_q8_0(values, blocks, 33, nullptr) == hipErrorInvalidValue);
    CHECK(dequantize_q8_0(blocks, values, 0, nullptr) == hipErrorInvalidValue);
    CHECK(quantized_gemv_q8_0(blocks, blocks, nullptr, values, 1, 33, DType::Float32, nullptr) == hipErrorInvalidValue);
    CHECK(quantized_gemv_q8_0(blocks, blocks, nullptr, values, 0, 32, DType::Float32, nullptr) == hipErrorInvalidValue);

    constexpr long long kTooManyElements = 1LL << 40;
    CHECK(quantize_q8_0(values, blocks, kTooManyElements, nullptr) == hipErrorInvalidValue);
    CHECK(dequantize_q8_0(blocks, values, kTooManyElements, nullptr) == hipErrorInvalidValue);
}
