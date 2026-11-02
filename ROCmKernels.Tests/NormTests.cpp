//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Norm/AddNorm.hpp>

#include <catch2/catch_test_macros.hpp>

#include "TestSupport.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace {
    constexpr int kWidth = 384;
}

TEST_CASE("AddNorm forward matches the reference", "[addnorm][norm]") {
    constexpr int rows = 33;
    constexpr float epsilon = 1e-5f;

    const std::vector<float> x = ROCmKernels::Tests::RandomValues<float>(rows * kWidth, 41, -1.0f, 1.0f);
    const std::vector<float> residual = ROCmKernels::Tests::RandomValues<float>(rows * kWidth, 42, -1.0f, 1.0f);
    const std::vector<float> weight = ROCmKernels::Tests::RandomValues<float>(kWidth, 43, 0.5f, 1.5f);
    const std::vector<float> bias = ROCmKernels::Tests::RandomValues<float>(kWidth, 44, -0.5f, 0.5f);

    std::vector<float> expectedSummed(rows * kWidth);
    std::vector<float> expectedNormalized(rows * kWidth);
    std::vector<float> expectedStatistics(rows * 2);

    for (int row = 0; row < rows; ++row) {
        double mean = 0;
        for (int column = 0; column < kWidth; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) * kWidth + column;
            expectedSummed[index] = x[index] + residual[index];
            mean += expectedSummed[index];
        }
        mean /= kWidth;

        double variance = 0;
        for (int column = 0; column < kWidth; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) * kWidth + column;
            const double centered = expectedSummed[index] - mean;
            variance += centered * centered;
        }
        variance /= kWidth;

        const float reciprocal = static_cast<float>(1.0 / std::sqrt(variance + epsilon));
        expectedStatistics[row * 2] = static_cast<float>(mean);
        expectedStatistics[row * 2 + 1] = reciprocal;

        for (int column = 0; column < kWidth; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) * kWidth + column;
            expectedNormalized[index] = static_cast<float>((expectedSummed[index] - mean) * reciprocal) * weight[column] + bias[column];
        }
    }

    ROCmKernels::DeviceBuffer<float> deviceX(rows * kWidth);
    ROCmKernels::DeviceBuffer<float> deviceResidual(rows * kWidth);
    ROCmKernels::DeviceBuffer<float> deviceWeight(kWidth);
    ROCmKernels::DeviceBuffer<float> deviceBias(kWidth);
    ROCmKernels::DeviceBuffer<float> deviceSummed(rows * kWidth);
    ROCmKernels::DeviceBuffer<float> deviceNormalized(rows * kWidth);
    ROCmKernels::DeviceBuffer<float> deviceStatistics(rows * 2);

    deviceX.CopyFromHost(x.data(), x.size());
    deviceResidual.CopyFromHost(residual.data(), residual.size());
    deviceWeight.CopyFromHost(weight.data(), weight.size());
    deviceBias.CopyFromHost(bias.data(), bias.size());

    ROCmKernels::Kernels::AddNormForward(
        deviceX.Data(), deviceResidual.Data(), deviceWeight.Data(), deviceBias.Data(), deviceSummed.Data(), deviceNormalized.Data(), deviceStatistics.Data(),
        rows, epsilon
    );
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> actualSummed(rows * kWidth);
    std::vector<float> actualNormalized(rows * kWidth);
    std::vector<float> actualStatistics(rows * 2);
    deviceSummed.CopyToHost(actualSummed.data(), actualSummed.size());
    deviceNormalized.CopyToHost(actualNormalized.data(), actualNormalized.size());
    deviceStatistics.CopyToHost(actualStatistics.data(), actualStatistics.size());

    ROCmKernels::Tests::CheckClose(expectedSummed, actualSummed, 1e-6);
    ROCmKernels::Tests::CheckClose(expectedNormalized, actualNormalized, 1e-5);
    ROCmKernels::Tests::CheckClose(expectedStatistics, actualStatistics, 1e-6);
}

TEST_CASE("AddNorm backward matches the reference", "[addnorm][norm]") {
    constexpr int rows = 17;

    const std::vector<float> summed = ROCmKernels::Tests::RandomValues<float>(rows * kWidth, 51, -1.0f, 1.0f);
    const std::vector<float> upstream = ROCmKernels::Tests::RandomValues<float>(rows * kWidth, 52, -1.0f, 1.0f);
    const std::vector<float> residualUpstream = ROCmKernels::Tests::RandomValues<float>(rows * kWidth, 53, -1.0f, 1.0f);
    const std::vector<float> weight = ROCmKernels::Tests::RandomValues<float>(kWidth, 54, 0.5f, 1.5f);

    std::vector<float> statistics(rows * 2);
    std::vector<float> centered(rows * kWidth);
    for (int row = 0; row < rows; ++row) {
        double mean = 0;
        for (int column = 0; column < kWidth; ++column) {
            mean += summed[static_cast<std::size_t>(row) * kWidth + column];
        }
        mean /= kWidth;

        double variance = 0;
        for (int column = 0; column < kWidth; ++column) {
            const double difference = summed[static_cast<std::size_t>(row) * kWidth + column] - mean;
            variance += difference * difference;
        }
        variance /= kWidth;

        const float reciprocal = static_cast<float>(1.0 / std::sqrt(variance + 1e-5));
        statistics[row * 2] = static_cast<float>(mean);
        statistics[row * 2 + 1] = reciprocal;

        for (int column = 0; column < kWidth; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) * kWidth + column;
            centered[index] = static_cast<float>((summed[index] - mean) * reciprocal);
        }
    }

    std::vector<float> expectedGradient(rows * kWidth);
    std::vector<float> expectedWeightGradient(kWidth, 0.0f);
    std::vector<float> expectedBiasGradient(kWidth, 0.0f);

    for (int row = 0; row < rows; ++row) {
        double plainMean = 0;
        double centeredMean = 0;
        for (int column = 0; column < kWidth; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) * kWidth + column;
            const double value = upstream[index] * weight[column];
            plainMean += value;
            centeredMean += value * centered[index];
        }
        plainMean /= kWidth;
        centeredMean /= kWidth;

        const float reciprocal = statistics[row * 2 + 1];
        for (int column = 0; column < kWidth; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) * kWidth + column;
            const double value = upstream[index] * weight[column];
            expectedGradient[index] = static_cast<float>((value - plainMean) - centered[index] * centeredMean) * reciprocal + residualUpstream[index];
            expectedWeightGradient[column] += static_cast<float>(upstream[index] * centered[index]);
            expectedBiasGradient[column] += upstream[index];
        }
    }

    ROCmKernels::DeviceBuffer<float> deviceSummed(rows * kWidth);
    ROCmKernels::DeviceBuffer<float> deviceUpstream(rows * kWidth);
    ROCmKernels::DeviceBuffer<float> deviceResidualUpstream(rows * kWidth);
    ROCmKernels::DeviceBuffer<float> deviceWeight(kWidth);
    ROCmKernels::DeviceBuffer<float> deviceStatistics(rows * 2);
    ROCmKernels::DeviceBuffer<float> deviceGradient(rows * kWidth);
    ROCmKernels::DeviceBuffer<float> deviceWeightGradient(kWidth);
    ROCmKernels::DeviceBuffer<float> deviceBiasGradient(kWidth);

    deviceSummed.CopyFromHost(summed.data(), summed.size());
    deviceUpstream.CopyFromHost(upstream.data(), upstream.size());
    deviceResidualUpstream.CopyFromHost(residualUpstream.data(), residualUpstream.size());
    deviceWeight.CopyFromHost(weight.data(), weight.size());
    deviceStatistics.CopyFromHost(statistics.data(), statistics.size());

    ROCmKernels::Kernels::AddNormBackward(
        deviceSummed.Data(), deviceUpstream.Data(), deviceResidualUpstream.Data(), deviceWeight.Data(), deviceStatistics.Data(), deviceGradient.Data(),
        deviceWeightGradient.Data(), deviceBiasGradient.Data(), rows
    );
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> actualGradient(rows * kWidth);
    std::vector<float> actualWeightGradient(kWidth);
    std::vector<float> actualBiasGradient(kWidth);
    deviceGradient.CopyToHost(actualGradient.data(), actualGradient.size());
    deviceWeightGradient.CopyToHost(actualWeightGradient.data(), actualWeightGradient.size());
    deviceBiasGradient.CopyToHost(actualBiasGradient.data(), actualBiasGradient.size());

    ROCmKernels::Tests::CheckClose(expectedGradient, actualGradient, 1e-4);
    ROCmKernels::Tests::CheckClose(expectedWeightGradient, actualWeightGradient, 1e-3);
    ROCmKernels::Tests::CheckClose(expectedBiasGradient, actualBiasGradient, 1e-3);
}
