//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Positional/Rotary.hpp>

#include <catch2/catch_test_macros.hpp>

#include "TestSupport.hpp"

#include <cstddef>
#include <vector>

namespace {
    constexpr int kHeadWidth = 32;

    void ReferenceRotary(
        const std::vector<float> &input, const std::vector<float> &cosine, const std::vector<float> &sine, std::vector<float> &output, const int batchSize,
        const int heads, const int length, const bool backward
    ) {
        for (int batch = 0; batch < batchSize; ++batch) {
            for (int token = 0; token < length; ++token) {
                const std::size_t bufferBase = (static_cast<std::size_t>(batch) * length + token) * kHeadWidth;

                for (int head = 0; head < heads; ++head) {
                    const std::size_t inputBase =
                        (static_cast<std::size_t>(batch) * length + token) * heads * kHeadWidth + static_cast<std::size_t>(head) * kHeadWidth;
                    const std::size_t outputBase = inputBase;

                    for (int column = 0; column < kHeadWidth / 2; ++column) {
                        const float low = input[inputBase + column];
                        const float high = input[inputBase + column + 16];
                        const float cosineLow = cosine[bufferBase + column];
                        const float cosineHigh = cosine[bufferBase + column + 16];
                        const float sineLow = sine[bufferBase + column];
                        const float sineHigh = sine[bufferBase + column + 16];

                        if (backward) {
                            output[outputBase + column] = low * cosineLow + high * sineHigh;
                            output[outputBase + column + 16] = high * cosineHigh - low * sineLow;
                        } else {
                            output[outputBase + column] = low * cosineLow - high * sineLow;
                            output[outputBase + column + 16] = high * cosineHigh + low * sineHigh;
                        }
                    }
                }
            }
        }
    }
} // namespace

TEST_CASE("Rotary forward matches the reference", "[rotary][positional]") {
    constexpr int batchSize = 2;
    constexpr int heads = 3;
    constexpr int length = 4;
    constexpr int elements = batchSize * heads * length * kHeadWidth;

    const std::vector<float> q = ROCmKernels::Tests::RandomValues<float>(elements, 21, -2.0f, 2.0f);
    const std::vector<float> k = ROCmKernels::Tests::RandomValues<float>(elements, 22, -2.0f, 2.0f);
    const std::vector<float> cosine = ROCmKernels::Tests::RandomValues<float>(batchSize * length * kHeadWidth, 23, -1.0f, 1.0f);
    const std::vector<float> sine = ROCmKernels::Tests::RandomValues<float>(batchSize * length * kHeadWidth, 24, -1.0f, 1.0f);

    std::vector<float> expectedQ(elements);
    std::vector<float> expectedK(elements);
    ReferenceRotary(q, cosine, sine, expectedQ, batchSize, heads, length, false);
    ReferenceRotary(k, cosine, sine, expectedK, batchSize, heads, length, false);

    ROCmKernels::DeviceBuffer<float> deviceQ(elements);
    ROCmKernels::DeviceBuffer<float> deviceK(elements);
    ROCmKernels::DeviceBuffer<float> deviceCosine(cosine.size());
    ROCmKernels::DeviceBuffer<float> deviceSine(sine.size());
    ROCmKernels::DeviceBuffer<float> deviceOutQ(elements);
    ROCmKernels::DeviceBuffer<float> deviceOutK(elements);
    deviceQ.CopyFromHost(q.data(), elements);
    deviceK.CopyFromHost(k.data(), elements);
    deviceCosine.CopyFromHost(cosine.data(), cosine.size());
    deviceSine.CopyFromHost(sine.data(), sine.size());

    const ROCmKernels::Kernels::Strides qkStrides{length * heads * kHeadWidth, kHeadWidth, heads * kHeadWidth};
    const ROCmKernels::Kernels::Strides tableStrides{length * kHeadWidth, 0, kHeadWidth};

    ROCmKernels::Kernels::Rotary(
        deviceQ.Data(), deviceK.Data(), deviceCosine.Data(), deviceSine.Data(), deviceOutQ.Data(), deviceOutK.Data(), batchSize, heads, length, false,
        qkStrides, qkStrides, tableStrides, tableStrides
    );
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> actualQ(elements);
    std::vector<float> actualK(elements);
    deviceOutQ.CopyToHost(actualQ.data(), elements);
    deviceOutK.CopyToHost(actualK.data(), elements);

    ROCmKernels::Tests::CheckClose(expectedQ, actualQ, 1e-6);
    ROCmKernels::Tests::CheckClose(expectedK, actualK, 1e-6);
}

TEST_CASE("Rotary backward matches the reference", "[rotary][positional]") {
    constexpr int batchSize = 1;
    constexpr int heads = 2;
    constexpr int length = 3;
    constexpr int elements = batchSize * heads * length * kHeadWidth;

    const std::vector<float> q = ROCmKernels::Tests::RandomValues<float>(elements, 31, -2.0f, 2.0f);
    const std::vector<float> k = ROCmKernels::Tests::RandomValues<float>(elements, 32, -2.0f, 2.0f);
    const std::vector<float> cosine = ROCmKernels::Tests::RandomValues<float>(batchSize * length * kHeadWidth, 33, -1.0f, 1.0f);
    const std::vector<float> sine = ROCmKernels::Tests::RandomValues<float>(batchSize * length * kHeadWidth, 34, -1.0f, 1.0f);

    std::vector<float> expectedQ(elements);
    std::vector<float> expectedK(elements);
    ReferenceRotary(q, cosine, sine, expectedQ, batchSize, heads, length, true);
    ReferenceRotary(k, cosine, sine, expectedK, batchSize, heads, length, true);

    ROCmKernels::DeviceBuffer<float> deviceQ(elements);
    ROCmKernels::DeviceBuffer<float> deviceK(elements);
    ROCmKernels::DeviceBuffer<float> deviceCosine(cosine.size());
    ROCmKernels::DeviceBuffer<float> deviceSine(sine.size());
    ROCmKernels::DeviceBuffer<float> deviceOutQ(elements);
    ROCmKernels::DeviceBuffer<float> deviceOutK(elements);
    deviceQ.CopyFromHost(q.data(), elements);
    deviceK.CopyFromHost(k.data(), elements);
    deviceCosine.CopyFromHost(cosine.data(), cosine.size());
    deviceSine.CopyFromHost(sine.data(), sine.size());

    const ROCmKernels::Kernels::Strides qkStrides{length * heads * kHeadWidth, kHeadWidth, heads * kHeadWidth};
    const ROCmKernels::Kernels::Strides tableStrides{length * kHeadWidth, 0, kHeadWidth};

    ROCmKernels::Kernels::Rotary(
        deviceQ.Data(), deviceK.Data(), deviceCosine.Data(), deviceSine.Data(), deviceOutQ.Data(), deviceOutK.Data(), batchSize, heads, length, true, qkStrides,
        qkStrides, tableStrides, tableStrides
    );
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> actualQ(elements);
    std::vector<float> actualK(elements);
    deviceOutQ.CopyToHost(actualQ.data(), elements);
    deviceOutK.CopyToHost(actualK.data(), elements);

    ROCmKernels::Tests::CheckClose(expectedQ, actualQ, 1e-6);
    ROCmKernels::Tests::CheckClose(expectedK, actualK, 1e-6);
}
