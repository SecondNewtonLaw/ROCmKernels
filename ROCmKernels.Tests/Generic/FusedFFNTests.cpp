#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/FusedFFN.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../TestSupport.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace {
    double ReferenceSiLU(const double value) {
        return value / (1.0 + std::exp(-value));
    }

    void RunFusedCase(const int rows, const int hidden, const bool withBias, const unsigned int seed) {
        constexpr int kWidth = 128;
        constexpr int kOutput = 128;

        const std::vector<float> hostInput = ROCmKernels::Tests::RandomValues<float>(static_cast<std::size_t>(rows) * kWidth, seed, -1.0f, 1.0f);
        const std::vector<float> hostW1 = ROCmKernels::Tests::RandomValues<float>(static_cast<std::size_t>(hidden) * kWidth, seed + 1, -0.2f, 0.2f);
        const std::vector<float> hostW2 = ROCmKernels::Tests::RandomValues<float>(static_cast<std::size_t>(kOutput) * hidden, seed + 2, -0.2f, 0.2f);
        const std::vector<float> hostB1 = ROCmKernels::Tests::RandomValues<float>(withBias ? hidden : 0, seed + 3, -0.1f, 0.1f);
        const std::vector<float> hostB2 = ROCmKernels::Tests::RandomValues<float>(withBias ? kOutput : 0, seed + 4, -0.1f, 0.1f);

        std::vector<__half> input(hostInput.size());
        std::vector<__half> w1(hostW1.size());
        std::vector<__half> w2(hostW2.size());
        std::vector<__half> b1(withBias ? hidden : 0);
        std::vector<__half> b2(withBias ? kOutput : 0);
        for (std::size_t index = 0; index < hostInput.size(); ++index) input[index] = __float2half_rn(hostInput[index]);
        for (std::size_t index = 0; index < hostW1.size(); ++index) w1[index] = __float2half_rn(hostW1[index]);
        for (std::size_t index = 0; index < hostW2.size(); ++index) w2[index] = __float2half_rn(hostW2[index]);
        for (std::size_t index = 0; index < b1.size(); ++index) b1[index] = __float2half_rn(hostB1[index]);
        for (std::size_t index = 0; index < b2.size(); ++index) b2[index] = __float2half_rn(hostB2[index]);

        ROCmKernels::DeviceBuffer<__half> deviceInput;
        ROCmKernels::DeviceBuffer<__half> deviceW1;
        ROCmKernels::DeviceBuffer<__half> deviceW2;
        ROCmKernels::DeviceBuffer<__half> deviceB1;
        ROCmKernels::DeviceBuffer<__half> deviceB2;
        ROCmKernels::DeviceBuffer<__half> deviceOutput;
        deviceInput.Resize(input.size());
        deviceW1.Resize(w1.size());
        deviceW2.Resize(w2.size());
        deviceInput.CopyFromHost(input.data(), input.size());
        deviceW1.CopyFromHost(w1.data(), w1.size());
        deviceW2.CopyFromHost(w2.data(), w2.size());
        if (withBias) {
            deviceB1.Resize(b1.size());
            deviceB2.Resize(b2.size());
            deviceB1.CopyFromHost(b1.data(), b1.size());
            deviceB2.CopyFromHost(b2.data(), b2.size());
        }
        deviceOutput.Resize(static_cast<std::size_t>(rows) * kOutput);

        ROCmKernels::Kernels::FfnUpSiluDown(deviceInput.Data(), deviceW1.Data(), withBias ? deviceB1.Data() : nullptr, deviceW2.Data(),
                withBias ? deviceB2.Data() : nullptr, deviceOutput.Data(), rows, hidden, ROCmKernels::Kernels::DType::Float16);

        std::vector<__half> actual(static_cast<std::size_t>(rows) * kOutput);
        deviceOutput.CopyToHost(actual.data(), actual.size());

        // Compare against the fp16-rounded inputs the device actually sees.
        std::vector<float> refInput(input.size()), refW1(w1.size()), refW2(w2.size());
        std::vector<float> refB1(b1.size()), refB2(b2.size());
        for (std::size_t index = 0; index < input.size(); ++index) refInput[index] = __half2float(input[index]);
        for (std::size_t index = 0; index < w1.size(); ++index) refW1[index] = __half2float(w1[index]);
        for (std::size_t index = 0; index < w2.size(); ++index) refW2[index] = __half2float(w2[index]);
        for (std::size_t index = 0; index < b1.size(); ++index) refB1[index] = __half2float(b1[index]);
        for (std::size_t index = 0; index < b2.size(); ++index) refB2[index] = __half2float(b2[index]);

        double worst = 0.0;
        for (int row = 0; row < rows; ++row) {
            for (int column = 0; column < kOutput; ++column) {
                double expected = 0.0;
                for (int hiddenIndex = 0; hiddenIndex < hidden; ++hiddenIndex) {
                    double sum = 0.0;
                    for (int k = 0; k < kWidth; ++k) {
                        sum += static_cast<double>(refInput[static_cast<std::size_t>(row) * kWidth + k]) *
                               static_cast<double>(refW1[static_cast<std::size_t>(hiddenIndex) * kWidth + k]);
                    }
                    if (withBias) sum += static_cast<double>(refB1[hiddenIndex]);
                    expected += ReferenceSiLU(sum) * static_cast<double>(refW2[static_cast<std::size_t>(column) * hidden + hiddenIndex]);
                }
                if (withBias) expected += static_cast<double>(refB2[column]);
                worst = std::max(worst, std::abs(__half2float(actual[static_cast<std::size_t>(row) * kOutput + column]) - expected));
            }
        }
        REQUIRE(worst < 0.05);
    }
} // namespace

TEST_CASE("Fused FFN matches the reference", "[generic][ffn]") {
    RunFusedCase(128, 256, true, 7);
    RunFusedCase(64, 64, false, 11);
    RunFusedCase(192, 128, true, 13);
}


