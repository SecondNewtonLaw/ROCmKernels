//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Optimizer/Muon.hpp>

#include <catch2/catch_test_macros.hpp>

#include "TestSupport.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
    std::vector<float>
    ReferenceGram(const std::vector<float> &y, const std::vector<float> &gin, const int m, const int n, const float alpha, const float beta, const float diag) {
        std::vector<float> result(static_cast<std::size_t>(n) * n);
        for (int row = 0; row < n; ++row) {
            for (int column = 0; column < n; ++column) {
                double total = 0;
                for (int k = 0; k < m; ++k) {
                    total += static_cast<double>(y[static_cast<std::size_t>(k) * n + row]) * y[static_cast<std::size_t>(k) * n + column];
                }

                result[static_cast<std::size_t>(row) * n + column] =
                    static_cast<float>(alpha * total + beta * gin[static_cast<std::size_t>(row) * n + column] + (row == column ? diag : 0.0f));
            }
        }

        return result;
    }

    std::vector<float> ReferenceMultiply(const std::vector<float> &y, const std::vector<float> &b, const int m, const int n, const float scale) {
        std::vector<float> result(static_cast<std::size_t>(m) * n);
        for (int row = 0; row < m; ++row) {
            for (int column = 0; column < n; ++column) {
                double total = 0;
                for (int k = 0; k < n; ++k) {
                    total += static_cast<double>(y[static_cast<std::size_t>(row) * n + k]) * b[static_cast<std::size_t>(k) * n + column];
                }
                result[static_cast<std::size_t>(row) * n + column] = static_cast<float>(scale * total);
            }
        }

        return result;
    }
} // namespace

TEST_CASE("Muon reports the supported shapes", "[muon][optimizer]") {
    CHECK(ROCmKernels::Kernels::MuonSupported(128, 128));
    CHECK(ROCmKernels::Kernels::MuonSupported(256, 128));
    CHECK_FALSE(ROCmKernels::Kernels::MuonSupported(127, 128));
    CHECK_FALSE(ROCmKernels::Kernels::MuonSupported(128, 64));
}

TEST_CASE("Muon prepare, gram, multiply and apply match the reference", "[muon][optimizer]") {
    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int elements = m * n;
    constexpr float momentum = 0.9f;
    constexpr float decay = 0.1f;
    constexpr float step = 0.01f;

    const std::vector<float> gradient = ROCmKernels::Tests::RandomValues<float>(elements, 81, -1.0f, 1.0f);
    std::vector<float> initialBuffer(elements, 0.25f);

    ROCmKernels::DeviceBuffer<float> deviceGradient(elements);
    ROCmKernels::DeviceBuffer<float> deviceBuffer(elements);
    ROCmKernels::DeviceBuffer<float> deviceY(elements);
    ROCmKernels::DeviceBuffer<float> deviceSumsq(1);
    ROCmKernels::DeviceBuffer<int> deviceFlags(1);
    ROCmKernels::DeviceBuffer<std::int64_t> deviceGradientPointer(1);
    ROCmKernels::DeviceBuffer<std::int64_t> deviceBufferPointer(1);

    deviceGradient.CopyFromHost(gradient.data(), gradient.size());
    deviceBuffer.CopyFromHost(initialBuffer.data(), initialBuffer.size());

    const std::int64_t gradientAddress = reinterpret_cast<std::int64_t>(deviceGradient.Data());
    const std::int64_t bufferAddress = reinterpret_cast<std::int64_t>(deviceBuffer.Data());
    deviceGradientPointer.CopyFromHost(&gradientAddress, 1);
    deviceBufferPointer.CopyFromHost(&bufferAddress, 1);

    ROCmKernels::Kernels::MuonPrepare(
        deviceGradientPointer.Data(), deviceBufferPointer.Data(), deviceY.Data(), deviceFlags.Data(), deviceSumsq.Data(), 1, m, n, momentum
    );
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> expectedBuffer(elements);
    std::vector<float> expectedY(elements);
    double expectedSumsq = 0;
    for (int index = 0; index < elements; ++index) {
        expectedBuffer[index] = initialBuffer[index] * momentum + gradient[index];
        expectedY[index] = gradient[index] + momentum * expectedBuffer[index];
        expectedSumsq += static_cast<double>(expectedY[index]) * expectedY[index];
    }

    std::vector<float> actualBuffer(elements);
    std::vector<float> actualY(elements);
    float actualSumsq = 0;
    int actualFlags = 0;
    deviceBuffer.CopyToHost(actualBuffer.data(), actualBuffer.size());
    deviceY.CopyToHost(actualY.data(), actualY.size());
    deviceSumsq.CopyToHost(&actualSumsq, 1);
    deviceFlags.CopyToHost(&actualFlags, 1);

    ROCmKernels::Tests::CheckClose(expectedBuffer, actualBuffer, 1e-5);
    ROCmKernels::Tests::CheckClose(expectedY, actualY, 1e-5);
    CHECK(actualFlags == 1);
    CHECK(std::abs(static_cast<double>(actualSumsq) - expectedSumsq) <= 1.0);

    const std::vector<float> gin = ROCmKernels::Tests::RandomValues<float>(n * n, 82, -1.0f, 1.0f);
    ROCmKernels::DeviceBuffer<float> deviceGin(n * n);
    ROCmKernels::DeviceBuffer<float> deviceGram(n * n);
    deviceGin.CopyFromHost(gin.data(), gin.size());

    ROCmKernels::Kernels::MuonGram(deviceY.Data(), deviceGin.Data(), deviceGram.Data(), deviceSumsq.Data(), 1, m, n, 0.5f, 0.25f, 0.75f, false);
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> actualGram(n * n);
    deviceGram.CopyToHost(actualGram.data(), actualGram.size());
    ROCmKernels::Tests::CheckClose(ReferenceGram(expectedY, gin, m, n, 0.5f, 0.25f, 0.75f), actualGram, 0.05);

    const std::vector<float> b = ROCmKernels::Tests::RandomValues<float>(n * n, 83, -1.0f, 1.0f);
    ROCmKernels::DeviceBuffer<float> deviceB(n * n);
    ROCmKernels::DeviceBuffer<float> deviceOut(elements);
    deviceB.CopyFromHost(b.data(), b.size());

    const float scale = static_cast<float>(1.0 / (std::sqrt(expectedSumsq) + 1e-7));
    ROCmKernels::Kernels::MuonMultiply(deviceY.Data(), deviceB.Data(), deviceOut.Data(), deviceSumsq.Data(), 1, m, n, true);
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> actualOut(elements);
    deviceOut.CopyToHost(actualOut.data(), actualOut.size());
    ROCmKernels::Tests::CheckClose(ReferenceMultiply(expectedY, b, m, n, scale), actualOut, 0.05);

    const std::vector<float> initialParameter = ROCmKernels::Tests::RandomValues<float>(elements, 84, -1.0f, 1.0f);
    ROCmKernels::DeviceBuffer<float> deviceParameter(elements);
    ROCmKernels::DeviceBuffer<std::int64_t> deviceParameterPointer(1);
    deviceParameter.CopyFromHost(initialParameter.data(), initialParameter.size());

    const std::int64_t parameterAddress = reinterpret_cast<std::int64_t>(deviceParameter.Data());
    deviceParameterPointer.CopyFromHost(&parameterAddress, 1);

    ROCmKernels::Kernels::MuonApply(
        deviceY.Data(), deviceB.Data(), deviceParameterPointer.Data(), deviceFlags.Data(), deviceSumsq.Data(), 1, m, n, true, false, decay, step
    );
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    const std::vector<float> expectedParameter = ReferenceMultiply(expectedY, b, m, n, scale);
    std::vector<float> expectedUpdated(elements);
    for (int index = 0; index < elements; ++index) {
        expectedUpdated[index] = initialParameter[index] * (1.0f - decay) - step * expectedParameter[index];
    }

    std::vector<float> actualUpdated(elements);
    deviceParameter.CopyToHost(actualUpdated.data(), actualUpdated.size());
    ROCmKernels::Tests::CheckClose(expectedUpdated, actualUpdated, 0.05);
}

TEST_CASE("Muon leaves dead matrices untouched", "[muon][optimizer]") {
    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int elements = m * n;

    std::vector<float> gradient(elements, 0.0f);
    std::vector<float> buffer = ROCmKernels::Tests::RandomValues<float>(elements, 91, -1.0f, 1.0f);
    const std::vector<float> initialBuffer = buffer;

    ROCmKernels::DeviceBuffer<float> deviceGradient(elements);
    ROCmKernels::DeviceBuffer<float> deviceBuffer(elements);
    ROCmKernels::DeviceBuffer<float> deviceY(elements);
    ROCmKernels::DeviceBuffer<float> deviceSumsq(1);
    ROCmKernels::DeviceBuffer<int> deviceFlags(1);
    ROCmKernels::DeviceBuffer<std::int64_t> deviceGradientPointer(1);
    ROCmKernels::DeviceBuffer<std::int64_t> deviceBufferPointer(1);

    deviceGradient.CopyFromHost(gradient.data(), gradient.size());
    deviceBuffer.CopyFromHost(buffer.data(), buffer.size());

    const std::int64_t gradientAddress = reinterpret_cast<std::int64_t>(deviceGradient.Data());
    const std::int64_t bufferAddress = reinterpret_cast<std::int64_t>(deviceBuffer.Data());
    deviceGradientPointer.CopyFromHost(&gradientAddress, 1);
    deviceBufferPointer.CopyFromHost(&bufferAddress, 1);

    ROCmKernels::Kernels::MuonPrepare(
        deviceGradientPointer.Data(), deviceBufferPointer.Data(), deviceY.Data(), deviceFlags.Data(), deviceSumsq.Data(), 1, m, n, 0.9f
    );
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    int actualFlags = 1;
    deviceFlags.CopyToHost(&actualFlags, 1);
    CHECK(actualFlags == 0);

    std::vector<float> actualBuffer(elements);
    deviceBuffer.CopyToHost(actualBuffer.data(), actualBuffer.size());
    ROCmKernels::Tests::CheckClose(initialBuffer, actualBuffer, 0.0);
}
