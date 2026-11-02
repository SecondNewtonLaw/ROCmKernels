//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Loss/CrossEntropy.hpp>

#include <catch2/catch_test_macros.hpp>

#include "TestSupport.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

TEST_CASE("Cross entropy forward and backward match the reference", "[crossentropy][loss]") {
    constexpr int rows = 4;
    constexpr int classes = 37;
    constexpr int stride = 45;
    constexpr int gradStride = 3;

    const std::vector<float> logits = ROCmKernels::Tests::RandomValues<float>(rows * stride, 61, -3.0f, 3.0f);
    const std::vector<std::int64_t> targets{3, -100, 10, 36};
    const std::vector<float> upstream = ROCmKernels::Tests::RandomValues<float>(rows * gradStride, 62, -1.0f, 1.0f);

    std::vector<float> expectedLosses(rows);
    std::vector<float> expectedStatistics(rows * 2);
    for (int row = 0; row < rows; ++row) {
        float maximum = -INFINITY;
        for (int column = 0; column < classes; ++column) {
            maximum = std::max(maximum, logits[static_cast<std::size_t>(row) * stride + column]);
        }

        double total = 0;
        for (int column = 0; column < classes; ++column) {
            total += std::exp(logits[static_cast<std::size_t>(row) * stride + column] - maximum);
        }
        const float logarithm = static_cast<float>(std::log(total));

        expectedStatistics[row * 2] = maximum;
        expectedStatistics[row * 2 + 1] = logarithm;
        expectedLosses[row] = targets[row] == -100 ? 0.0f : logarithm - (logits[static_cast<std::size_t>(row) * stride + targets[row]] - maximum);
    }

    ROCmKernels::DeviceBuffer<float> deviceLogits(rows * stride);
    ROCmKernels::DeviceBuffer<std::int64_t> deviceTargets(rows);
    ROCmKernels::DeviceBuffer<float> deviceLosses(rows);
    ROCmKernels::DeviceBuffer<float> deviceStatistics(rows * 2);
    deviceLogits.CopyFromHost(logits.data(), logits.size());
    deviceTargets.CopyFromHost(targets.data(), targets.size());

    ROCmKernels::Kernels::CrossEntropyForward(deviceLogits.Data(), deviceTargets.Data(), deviceLosses.Data(), deviceStatistics.Data(), rows, classes, stride);
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> actualLosses(rows);
    std::vector<float> actualStatistics(rows * 2);
    deviceLosses.CopyToHost(actualLosses.data(), actualLosses.size());
    deviceStatistics.CopyToHost(actualStatistics.data(), actualStatistics.size());

    ROCmKernels::Tests::CheckClose(expectedLosses, actualLosses, 1e-6);
    ROCmKernels::Tests::CheckClose(expectedStatistics, actualStatistics, 1e-6);

    std::vector<float> expectedGradient(rows * classes, 0.0f);
    for (int row = 0; row < rows; ++row) {
        const float scale = upstream[static_cast<std::size_t>(row) * gradStride];
        if (targets[row] == -100) {
            continue;
        }

        const float maximum = expectedStatistics[row * 2];
        const float logarithm = expectedStatistics[row * 2 + 1];
        for (int column = 0; column < classes; ++column) {
            const float probability = static_cast<float>(std::exp((logits[static_cast<std::size_t>(row) * stride + column] - maximum) - logarithm));
            expectedGradient[static_cast<std::size_t>(row) * classes + column] = probability * scale - (column == targets[row] ? scale : 0.0f);
        }
    }

    ROCmKernels::DeviceBuffer<float> deviceUpstream(rows * gradStride);
    ROCmKernels::DeviceBuffer<float> deviceGradient(rows * classes);
    deviceUpstream.CopyFromHost(upstream.data(), upstream.size());

    ROCmKernels::Kernels::CrossEntropyBackward(
        deviceLogits.Data(), deviceTargets.Data(), deviceStatistics.Data(), deviceUpstream.Data(), deviceGradient.Data(), rows, classes, stride, gradStride
    );
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> actualGradient(rows * classes);
    deviceGradient.CopyToHost(actualGradient.data(), actualGradient.size());

    ROCmKernels::Tests::CheckClose(expectedGradient, actualGradient, 1e-6);
}

TEST_CASE("Cross entropy backward reads the upstream stride in elements", "[crossentropy][loss]") {
    constexpr int rows = 64;
    constexpr int classes = 1024;
    constexpr int stride = 1024;

    const std::vector<float> logits = ROCmKernels::Tests::RandomValues<float>(rows * stride, 71, -4.0f, 4.0f);
    std::vector<std::int64_t> targets(rows);
    for (int row = 0; row < rows; ++row) targets[row] = (row * 37) % classes;
    const std::vector<float> upstream(rows, 1.0f / rows);

    ROCmKernels::DeviceBuffer<float> deviceLogits(rows * stride);
    ROCmKernels::DeviceBuffer<std::int64_t> deviceTargets(rows);
    ROCmKernels::DeviceBuffer<float> deviceLosses(rows);
    ROCmKernels::DeviceBuffer<float> deviceStatistics(rows * 2);
    ROCmKernels::DeviceBuffer<float> deviceUpstream(rows);
    ROCmKernels::DeviceBuffer<float> deviceScalarUpstream(1);
    ROCmKernels::DeviceBuffer<float> deviceGradient(rows * classes);
    deviceLogits.CopyFromHost(logits.data(), logits.size());
    deviceTargets.CopyFromHost(targets.data(), targets.size());
    deviceUpstream.CopyFromHost(upstream.data(), upstream.size());
    const float scalar = 1.0f / rows;
    deviceScalarUpstream.CopyFromHost(&scalar, 1);

    ROCmKernels::Kernels::CrossEntropyForward(deviceLogits.Data(), deviceTargets.Data(), deviceLosses.Data(), deviceStatistics.Data(), rows, classes, stride);
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> statistics(rows * 2);
    deviceStatistics.CopyToHost(statistics.data(), statistics.size());

    std::vector<float> expected(rows * classes);
    for (int row = 0; row < rows; ++row) {
        const float maximum = statistics[row * 2];
        const float logarithm = statistics[row * 2 + 1];
        for (int column = 0; column < classes; ++column) {
            const float probability = std::exp((logits[static_cast<std::size_t>(row) * stride + column] - maximum) - logarithm);
            expected[static_cast<std::size_t>(row) * classes + column] =
                    probability * upstream[row] - (column == targets[row] ? upstream[row] : 0.0f);
        }
    }

    ROCmKernels::Kernels::CrossEntropyBackward(deviceLogits.Data(), deviceTargets.Data(), deviceStatistics.Data(), deviceUpstream.Data(),
            deviceGradient.Data(), rows, classes, stride, 1);
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> actual(rows * classes);
    deviceGradient.CopyToHost(actual.data(), actual.size());
    ROCmKernels::Tests::CheckClose(expected, actual, 1e-6);

    std::vector<float> prefilled(rows * classes, 123.0f);
    deviceGradient.CopyFromHost(prefilled.data(), prefilled.size());
    ROCmKernels::Kernels::CrossEntropyBackward(deviceLogits.Data(), deviceTargets.Data(), deviceStatistics.Data(), deviceScalarUpstream.Data(),
            deviceGradient.Data(), rows, classes, stride, 0);
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());
    deviceGradient.CopyToHost(actual.data(), actual.size());
    ROCmKernels::Tests::CheckClose(expected, actual, 1e-6);
}
