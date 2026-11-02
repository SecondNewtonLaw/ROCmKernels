//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Activation/SwiGLU.hpp>

#include <catch2/catch_test_macros.hpp>

#include "TestSupport.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

TEST_CASE("SwiGLU forward matches the reference", "[swiglu][activation]") {
    constexpr int rows = 5;
    constexpr int width = 7;
    constexpr int count = rows * width;

    const std::vector<float> input = ROCmKernels::Tests::RandomValues<float>(2 * count, 11, -4.0f, 4.0f);
    std::vector<float> expected(count);
    for (int index = 0; index < count; ++index) {
        const int source = (index / width) * (2 * width) + index % width;
        const float value = input[source];
        expected[index] = (value / (1.0f + std::exp(-value))) * input[source + width];
    }

    ROCmKernels::DeviceBuffer<float> deviceInput(input.size());
    ROCmKernels::DeviceBuffer<float> deviceOutput(count);
    deviceInput.CopyFromHost(input.data(), input.size());

    ROCmKernels::Kernels::SwiGLUForward(deviceInput.Data(), deviceOutput.Data(), count, width);
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> actual(count);
    deviceOutput.CopyToHost(actual.data(), count);

    ROCmKernels::Tests::CheckClose(expected, actual, 1e-5);
}

TEST_CASE("SwiGLU backward matches the reference", "[swiglu][activation]") {
    constexpr int rows = 3;
    constexpr int width = 5;
    constexpr int count = rows * width;

    const std::vector<float> input = ROCmKernels::Tests::RandomValues<float>(2 * count, 12, -4.0f, 4.0f);
    const std::vector<float> upstream = ROCmKernels::Tests::RandomValues<float>(count, 13, -1.0f, 1.0f);

    std::vector<float> expected(2 * count, 0.0f);
    for (int index = 0; index < count; ++index) {
        const int source = (index / width) * (2 * width) + index % width;
        const float value = input[source];
        const float sigmoid = 1.0f / (1.0f + std::exp(-value));

        expected[source] = upstream[index] * input[source + width] * sigmoid * (1.0f + value * (1.0f - sigmoid));
        expected[source + width] = upstream[index] * (value * sigmoid);
    }

    ROCmKernels::DeviceBuffer<float> deviceInput(input.size());
    ROCmKernels::DeviceBuffer<float> deviceUpstream(count);
    ROCmKernels::DeviceBuffer<float> deviceGradient(2 * count);
    deviceInput.CopyFromHost(input.data(), input.size());
    deviceUpstream.CopyFromHost(upstream.data(), count);

    ROCmKernels::Kernels::SwiGLUBackward(deviceInput.Data(), deviceUpstream.Data(), deviceGradient.Data(), count, width);
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> actual(2 * count);
    deviceGradient.CopyToHost(actual.data(), 2 * count);

    ROCmKernels::Tests::CheckClose(expected, actual, 1e-5);
}
