//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Loss/CrossEntropy.hpp>

#include <catch2/catch_test_macros.hpp>

#include "TestSupport.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
    std::vector<__half> NarrowToHalf(const std::vector<float> &values) {
        std::vector<__half> narrowed(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) narrowed[index] = __float2half_rn(values[index]);
        return narrowed;
    }

    // Widened fp16 logits, so both paths see exactly the same numbers.
    std::vector<float> WidenHalf(const std::vector<__half> &values) {
        std::vector<float> widened(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) widened[index] = __half2float(values[index]);
        return widened;
    }
} // namespace

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

TEST_CASE("Cross entropy fp16 matches the fp32 path on widened logits", "[crossentropy][loss]") {
    constexpr int rows = 12;
    constexpr int classes = 1003; // not a multiple of eight, so the scalar tail runs
    constexpr int stride = 1016;  // multiple of eight, so every row is 16-byte aligned

    const std::vector<float> hostLogits = ROCmKernels::Tests::RandomValues<float>(rows * stride, 131, -6.0f, 6.0f);
    const std::vector<__half> narrowedLogits = NarrowToHalf(hostLogits);
    std::vector<std::int64_t> targets(rows);
    for (int row = 0; row < rows; ++row) targets[row] = (row * 73) % classes;
    targets[3] = -100;
    const std::vector<float> upstream(rows, 0.25f);

    ROCmKernels::DeviceBuffer<__half> deviceLogits(rows * stride);
    ROCmKernels::DeviceBuffer<std::int64_t> deviceTargets(rows);
    ROCmKernels::DeviceBuffer<float> deviceLosses(rows);
    ROCmKernels::DeviceBuffer<float> deviceStatistics(rows * 2);
    ROCmKernels::DeviceBuffer<float> deviceUpstream(rows);
    ROCmKernels::DeviceBuffer<__half> deviceGradient(rows * classes);
    deviceLogits.CopyFromHost(narrowedLogits.data(), narrowedLogits.size());
    deviceTargets.CopyFromHost(targets.data(), targets.size());
    deviceUpstream.CopyFromHost(upstream.data(), upstream.size());

    ROCmKernels::Kernels::CrossEntropyForwardFp16(deviceLogits.Data(), deviceTargets.Data(), deviceLosses.Data(), deviceStatistics.Data(), rows,
            classes, stride);
    ROCmKernels::Kernels::CrossEntropyBackwardFp16(deviceLogits.Data(), deviceTargets.Data(), deviceStatistics.Data(), deviceUpstream.Data(),
            deviceGradient.Data(), rows, classes, stride, 1);
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    // fp32 reference over the same widened values.
    const std::vector<float> widenedLogits = WidenHalf(narrowedLogits);
    ROCmKernels::DeviceBuffer<float> referenceLogits(rows * stride);
    ROCmKernels::DeviceBuffer<float> referenceLosses(rows);
    ROCmKernels::DeviceBuffer<float> referenceStatistics(rows * 2);
    ROCmKernels::DeviceBuffer<float> referenceGradient(rows * classes);
    referenceLogits.CopyFromHost(widenedLogits.data(), widenedLogits.size());
    ROCmKernels::Kernels::CrossEntropyForward(referenceLogits.Data(), deviceTargets.Data(), referenceLosses.Data(), referenceStatistics.Data(),
            rows, classes, stride);
    ROCmKernels::Kernels::CrossEntropyBackward(referenceLogits.Data(), deviceTargets.Data(), referenceStatistics.Data(), deviceUpstream.Data(),
            referenceGradient.Data(), rows, classes, stride, 1);
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> losses(rows), statistics(rows * 2), expectedLosses(rows), expectedStatistics(rows * 2);
    std::vector<float> expectedGradient(rows * classes);
    std::vector<__half> actualGradient(rows * classes);
    deviceLosses.CopyToHost(losses.data(), losses.size());
    deviceStatistics.CopyToHost(statistics.data(), statistics.size());
    deviceGradient.CopyToHost(actualGradient.data(), actualGradient.size());
    referenceLosses.CopyToHost(expectedLosses.data(), expectedLosses.size());
    referenceStatistics.CopyToHost(expectedStatistics.data(), expectedStatistics.size());
    referenceGradient.CopyToHost(expectedGradient.data(), expectedGradient.size());

    ROCmKernels::Tests::CheckClose(expectedLosses, losses, 1e-5);
    ROCmKernels::Tests::CheckClose(expectedStatistics, statistics, 1e-5);

    // fp16 storage is a 2^-11 relative rounding, but small gradients fall into the subnormal
    // range where the quantum is 2^-24; the bound allows one quantum there.
    double worstExcess = 0.0;
    for (std::size_t index = 0; index < expectedGradient.size(); ++index) {
        const double expected = expectedGradient[index];
        const double actual = __half2float(actualGradient[index]);
        worstExcess = std::max(worstExcess, std::abs(actual - expected) / (1e-3 * std::abs(expected) + 3.1e-8));
    }
    CHECK(worstExcess < 1.0);

    // A target of -100 contributes no loss and no gradient.
    CHECK(losses[3] == 0.0f);
    for (int column = 0; column < classes; ++column) CHECK(__half2float(actualGradient[3 * classes + column]) == 0.0f);
}

TEST_CASE("Cross entropy fp16 handles a misaligned row stride", "[crossentropy][loss]") {
    constexpr int rows = 10;
    constexpr int classes = 777;
    constexpr int alignedStride = 784;
    constexpr int unalignedStride = 777; // rows start at odd 16-byte boundaries

    const std::vector<float> hostLogits = ROCmKernels::Tests::RandomValues<float>(rows * alignedStride, 141, -5.0f, 5.0f);
    const std::vector<__half> narrowedLogits = NarrowToHalf(hostLogits);
    std::vector<__half> repacked(static_cast<std::size_t>(rows) * unalignedStride);
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < classes; ++column)
            repacked[static_cast<std::size_t>(row) * unalignedStride + column] = narrowedLogits[static_cast<std::size_t>(row) * alignedStride + column];

    std::vector<std::int64_t> targets(rows);
    for (int row = 0; row < rows; ++row) targets[row] = (row * 31) % classes;
    const std::vector<float> upstream(rows, -0.5f);

    auto run = [&](const std::vector<__half> &logits, const int stride, std::vector<float> &losses, std::vector<float> &statistics,
                   std::vector<__half> &gradient) {
        ROCmKernels::DeviceBuffer<__half> deviceLogits(logits.size());
        ROCmKernels::DeviceBuffer<std::int64_t> deviceTargets(rows);
        ROCmKernels::DeviceBuffer<float> deviceLosses(rows);
        ROCmKernels::DeviceBuffer<float> deviceStatistics(rows * 2);
        ROCmKernels::DeviceBuffer<float> deviceUpstream(rows);
        ROCmKernels::DeviceBuffer<__half> deviceGradient(static_cast<std::size_t>(rows) * classes);
        deviceLogits.CopyFromHost(logits.data(), logits.size());
        deviceTargets.CopyFromHost(targets.data(), targets.size());
        deviceUpstream.CopyFromHost(upstream.data(), upstream.size());

        ROCmKernels::Kernels::CrossEntropyForwardFp16(deviceLogits.Data(), deviceTargets.Data(), deviceLosses.Data(), deviceStatistics.Data(),
                rows, classes, stride);
        ROCmKernels::Kernels::CrossEntropyBackwardFp16(deviceLogits.Data(), deviceTargets.Data(), deviceStatistics.Data(), deviceUpstream.Data(),
                deviceGradient.Data(), rows, classes, stride, 1);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        losses.resize(rows);
        statistics.resize(rows * 2);
        gradient.resize(static_cast<std::size_t>(rows) * classes);
        deviceLosses.CopyToHost(losses.data(), losses.size());
        deviceStatistics.CopyToHost(statistics.data(), statistics.size());
        deviceGradient.CopyToHost(gradient.data(), gradient.size());
    };

    std::vector<float> alignedLosses, alignedStatistics, unalignedLosses, unalignedStatistics;
    std::vector<__half> alignedGradient, unalignedGradient;
    run(narrowedLogits, alignedStride, alignedLosses, alignedStatistics, alignedGradient);
    run(repacked, unalignedStride, unalignedLosses, unalignedStatistics, unalignedGradient);

    ROCmKernels::Tests::CheckClose(alignedLosses, unalignedLosses, 1e-5);
    ROCmKernels::Tests::CheckClose(alignedStatistics, unalignedStatistics, 1e-5);
    double worstExcess = 0.0;
    for (std::size_t index = 0; index < alignedGradient.size(); ++index) {
        const double expected = __half2float(alignedGradient[index]);
        const double actual = __half2float(unalignedGradient[index]);
        // Two independent fp16 stores can land one quantum apart in the subnormal range.
        worstExcess = std::max(worstExcess, std::abs(actual - expected) / (1e-3 * std::abs(expected) + 6.2e-8));
    }
    CHECK(worstExcess < 1.0);
}
