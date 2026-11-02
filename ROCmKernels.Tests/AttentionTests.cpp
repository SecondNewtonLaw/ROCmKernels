//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Attention/Attention.hpp>
#include <ROCmKernels/Kernels/Attention/Attention64.hpp>

#include <catch2/catch_test_macros.hpp>

#include "TestSupport.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
    struct Reference {
        std::vector<float> output;
        std::vector<float> lse;
        std::vector<float> gradientQ;
        std::vector<float> gradientK;
        std::vector<float> gradientV;
    };

    Reference ReferenceAttention(
        const std::vector<float> &q, const std::vector<float> &k, const std::vector<float> &v, const std::vector<std::uint8_t> &mask,
        const std::vector<float> &upstream, const int headWidth, const int batchSize, const int heads, const int queries, const int keys,
        const int radius = -1
    ) {
        const double scale = 1.0 / std::sqrt(static_cast<double>(headWidth));

        Reference reference{};
        reference.output.assign(static_cast<std::size_t>(batchSize) * heads * queries * headWidth, 0.0f);
        reference.lse.assign(static_cast<std::size_t>(batchSize) * heads * queries * 2, 0.0f);
        reference.gradientQ.assign(static_cast<std::size_t>(batchSize) * heads * queries * headWidth, 0.0f);
        reference.gradientK.assign(static_cast<std::size_t>(batchSize) * heads * keys * headWidth, 0.0f);
        reference.gradientV.assign(static_cast<std::size_t>(batchSize) * heads * keys * headWidth, 0.0f);

        std::vector<double> scores(keys);
        std::vector<double> probabilities(keys);

        for (int batch = 0; batch < batchSize; ++batch) {
            for (int head = 0; head < heads; ++head) {
                for (int query = 0; query < queries; ++query) {
                    const std::size_t qBase =
                        (static_cast<std::size_t>(batch) * heads + head) * queries * headWidth + static_cast<std::size_t>(query) * headWidth;

                    double maximum = -INFINITY;
                    for (int key = 0; key < keys; ++key) {
                        scores[key] = -INFINITY;
                        if (!mask[static_cast<std::size_t>(batch) * keys + key]) {
                            continue;
                        }
                        if (radius >= 0 && std::abs(query - key) > radius) {
                            continue;
                        }

                        const std::size_t kBase =
                            (static_cast<std::size_t>(batch) * heads + head) * keys * headWidth + static_cast<std::size_t>(key) * headWidth;
                        double dot = 0;
                        for (int column = 0; column < headWidth; ++column) {
                            dot += static_cast<double>(q[qBase + column]) * k[kBase + column];
                        }
                        scores[key] = dot * scale;
                        maximum = std::max(maximum, scores[key]);
                    }

                    double total = 0;
                    for (int key = 0; key < keys; ++key) {
                        probabilities[key] = maximum == -INFINITY ? 0.0 : std::exp(scores[key] - maximum);
                        total += probabilities[key];
                    }

                    const std::size_t row = (static_cast<std::size_t>(batch) * heads + head) * queries + query;
                    reference.lse[row * 2] = maximum == -INFINITY ? -INFINITY : static_cast<float>(maximum);
                    reference.lse[row * 2 + 1] = static_cast<float>(total);

                    for (int column = 0; column < headWidth; ++column) {
                        double value = 0;
                        for (int key = 0; key < keys; ++key) {
                            if (probabilities[key] == 0.0) {
                                continue;
                            }
                            const std::size_t vBase =
                                (static_cast<std::size_t>(batch) * heads + head) * keys * headWidth + static_cast<std::size_t>(key) * headWidth;
                            value += probabilities[key] * v[vBase + column];
                        }
                        reference.output[qBase + column] = static_cast<float>(total > 0 ? value / total : 0.0);
                    }

                    double delta = 0;
                    for (int column = 0; column < headWidth; ++column) {
                        delta += static_cast<double>(upstream[qBase + column]) * reference.output[qBase + column];
                    }

                    for (int key = 0; key < keys; ++key) {
                        if (probabilities[key] == 0.0) {
                            continue;
                        }

                        const std::size_t kBase =
                            (static_cast<std::size_t>(batch) * heads + head) * keys * headWidth + static_cast<std::size_t>(key) * headWidth;
                        const double probability = probabilities[key] / total;

                        double valueScore = 0;
                        for (int column = 0; column < headWidth; ++column) {
                            valueScore += static_cast<double>(upstream[qBase + column]) * v[kBase + column];
                        }
                        const double derivative = probability * (valueScore - delta);

                        for (int column = 0; column < headWidth; ++column) {
                            reference.gradientQ[qBase + column] += static_cast<float>(derivative * k[kBase + column] * scale);
                            reference.gradientK[kBase + column] += static_cast<float>(derivative * q[qBase + column] * scale);
                            reference.gradientV[kBase + column] += static_cast<float>(probability * upstream[qBase + column]);
                        }
                    }
                }
            }
        }

        return reference;
    }

    template <int HeadWidth> void RunAttentionCase(const bool tiled, const int batchSize, const int heads, const int queries, const int keys) {
        const std::size_t qElements = static_cast<std::size_t>(batchSize) * heads * queries * HeadWidth;
        const std::size_t kElements = static_cast<std::size_t>(batchSize) * heads * keys * HeadWidth;

        const std::vector<float> q = ROCmKernels::Tests::RandomValues<float>(qElements, 101, -1.0f, 1.0f);
        const std::vector<float> k = ROCmKernels::Tests::RandomValues<float>(kElements, 102, -1.0f, 1.0f);
        const std::vector<float> v = ROCmKernels::Tests::RandomValues<float>(kElements, 103, -1.0f, 1.0f);
        const std::vector<float> upstream = ROCmKernels::Tests::RandomValues<float>(qElements, 104, -1.0f, 1.0f);

        std::vector<std::uint8_t> mask(static_cast<std::size_t>(batchSize) * keys, 1);
        mask[3] = 0;

        const Reference reference = ReferenceAttention(q, k, v, mask, upstream, HeadWidth, batchSize, heads, queries, keys);

        ROCmKernels::DeviceBuffer<float> deviceQ(qElements);
        ROCmKernels::DeviceBuffer<float> deviceK(kElements);
        ROCmKernels::DeviceBuffer<float> deviceV(kElements);
        ROCmKernels::DeviceBuffer<std::uint8_t> deviceMask(mask.size());
        ROCmKernels::DeviceBuffer<float> deviceUpstream(qElements);
        ROCmKernels::DeviceBuffer<float> deviceOutput(qElements);
        ROCmKernels::DeviceBuffer<float> deviceLse(static_cast<std::size_t>(batchSize) * heads * queries * 2);
        ROCmKernels::DeviceBuffer<float> deviceGradientQ(qElements);
        ROCmKernels::DeviceBuffer<float> deviceGradientK(kElements);
        ROCmKernels::DeviceBuffer<float> deviceGradientV(kElements);
        ROCmKernels::DeviceBuffer<float> deviceDelta(static_cast<std::size_t>(batchSize) * heads * queries);

        deviceQ.CopyFromHost(q.data(), q.size());
        deviceK.CopyFromHost(k.data(), k.size());
        deviceV.CopyFromHost(v.data(), v.size());
        deviceMask.CopyFromHost(mask.data(), mask.size());
        deviceUpstream.CopyFromHost(upstream.data(), upstream.size());

        const ROCmKernels::Kernels::Strides qStrides{heads * queries * HeadWidth, queries * HeadWidth, HeadWidth};
        const ROCmKernels::Kernels::Strides kStrides{heads * keys * HeadWidth, keys * HeadWidth, HeadWidth};
        const auto boolMask = reinterpret_cast<const bool *>(deviceMask.Data());

        if constexpr (HeadWidth == 32) {
            if (tiled) {
                ROCmKernels::Kernels::AttentionForwardTiled(
                    deviceQ.Data(), deviceK.Data(), deviceV.Data(), boolMask, deviceOutput.Data(), deviceLse.Data(), batchSize, heads, queries, keys, -1,
                    qStrides, kStrides, kStrides, qStrides
                );
            } else {
                ROCmKernels::Kernels::AttentionForward(
                    deviceQ.Data(), deviceK.Data(), deviceV.Data(), boolMask, deviceOutput.Data(), deviceLse.Data(), batchSize, heads, queries, keys, -1,
                    qStrides, kStrides, kStrides, qStrides
                );
            }
        } else {
            ROCmKernels::Kernels::Attention64Forward(
                deviceQ.Data(), deviceK.Data(), deviceV.Data(), boolMask, deviceOutput.Data(), deviceLse.Data(), batchSize, heads, queries, keys, -1, qStrides,
                kStrides, kStrides, qStrides
            );
        }
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> actualOutput(qElements);
        std::vector<float> actualLse(reference.lse.size());
        deviceOutput.CopyToHost(actualOutput.data(), actualOutput.size());
        deviceLse.CopyToHost(actualLse.data(), actualLse.size());

        ROCmKernels::Tests::CheckClose(reference.output, actualOutput, 5e-4);
        ROCmKernels::Tests::CheckClose(reference.lse, actualLse, 5e-4);

        if constexpr (HeadWidth == 32) {
            if (tiled) {
                ROCmKernels::Kernels::AttentionBackwardTiled(
                    deviceQ.Data(), deviceK.Data(), deviceV.Data(), boolMask, deviceOutput.Data(), deviceLse.Data(), deviceUpstream.Data(),
                    deviceGradientQ.Data(), deviceGradientK.Data(), deviceGradientV.Data(), deviceDelta.Data(), batchSize, heads, queries, keys, -1, qStrides,
                    kStrides, kStrides, qStrides, qStrides
                );
            } else {
                ROCmKernels::Kernels::AttentionBackward(
                    deviceQ.Data(), deviceK.Data(), deviceV.Data(), boolMask, deviceOutput.Data(), deviceLse.Data(), deviceUpstream.Data(),
                    deviceGradientQ.Data(), deviceGradientK.Data(), deviceGradientV.Data(), deviceDelta.Data(), batchSize, heads, queries, keys, -1, qStrides,
                    kStrides, kStrides, qStrides, qStrides
                );
            }
        } else {
            ROCmKernels::Kernels::Attention64Backward(
                deviceQ.Data(), deviceK.Data(), deviceV.Data(), boolMask, deviceOutput.Data(), deviceLse.Data(), deviceUpstream.Data(), deviceGradientQ.Data(),
                deviceGradientK.Data(), deviceGradientV.Data(), deviceDelta.Data(), batchSize, heads, queries, keys, -1, qStrides, kStrides, kStrides, qStrides,
                qStrides
            );
        }
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> actualGradientQ(qElements);
        std::vector<float> actualGradientK(kElements);
        std::vector<float> actualGradientV(kElements);
        deviceGradientQ.CopyToHost(actualGradientQ.data(), actualGradientQ.size());
        deviceGradientK.CopyToHost(actualGradientK.data(), actualGradientK.size());
        deviceGradientV.CopyToHost(actualGradientV.data(), actualGradientV.size());

        ROCmKernels::Tests::CheckClose(reference.gradientQ, actualGradientQ, 5e-4);
        ROCmKernels::Tests::CheckClose(reference.gradientK, actualGradientK, 5e-4);
        ROCmKernels::Tests::CheckClose(reference.gradientV, actualGradientV, 5e-4);
    }
    void RunFlashCase(const int variant, const int batchSize, const int heads, const int queries, const int keys, const int radius) {
        const std::size_t qElements = static_cast<std::size_t>(batchSize) * heads * queries * 32;
        const std::size_t kElements = static_cast<std::size_t>(batchSize) * heads * keys * 32;
        const std::vector<float> q = ROCmKernels::Tests::RandomValues<float>(qElements, 101, -1.0f, 1.0f);
        const std::vector<float> k = ROCmKernels::Tests::RandomValues<float>(kElements, 102, -1.0f, 1.0f);
        const std::vector<float> v = ROCmKernels::Tests::RandomValues<float>(kElements, 103, -1.0f, 1.0f);
        const std::vector<float> upstream(qElements, 0.0f);
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(batchSize) * keys, 1);
        mask[3] = 0;
        const Reference reference = ReferenceAttention(q, k, v, mask, upstream, 32, batchSize, heads, queries, keys, radius);

        ROCmKernels::DeviceBuffer<float> deviceQ(qElements);
        ROCmKernels::DeviceBuffer<float> deviceK(kElements);
        ROCmKernels::DeviceBuffer<float> deviceV(kElements);
        ROCmKernels::DeviceBuffer<std::uint8_t> deviceMask(mask.size());
        ROCmKernels::DeviceBuffer<float> deviceOutput(qElements);
        ROCmKernels::DeviceBuffer<float> deviceLse(static_cast<std::size_t>(batchSize) * heads * queries * 2);
        deviceQ.CopyFromHost(q.data(), q.size());
        deviceK.CopyFromHost(k.data(), k.size());
        deviceV.CopyFromHost(v.data(), v.size());
        deviceMask.CopyFromHost(mask.data(), mask.size());

        const ROCmKernels::Kernels::Strides qStrides{heads * queries * 32, queries * 32, 32};
        const ROCmKernels::Kernels::Strides kStrides{heads * keys * 32, keys * 32, 32};
        const auto boolMask = reinterpret_cast<const bool *>(deviceMask.Data());
        if (variant < 0) {
            ROCmKernels::Kernels::AttentionForwardFlash(
                deviceQ.Data(), deviceK.Data(), deviceV.Data(), boolMask, deviceOutput.Data(), deviceLse.Data(), batchSize, heads, queries, keys,
                radius, qStrides, kStrides, kStrides, qStrides
            );
        } else {
            ROCmKernels::Kernels::AttentionForwardFlashTuned(
                deviceQ.Data(), deviceK.Data(), deviceV.Data(), boolMask, deviceOutput.Data(), deviceLse.Data(), batchSize, heads, queries, keys,
                radius, variant, qStrides, kStrides, kStrides, qStrides
            );
        }
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> actualOutput(qElements);
        std::vector<float> actualLse(reference.lse.size());
        deviceOutput.CopyToHost(actualOutput.data(), actualOutput.size());
        deviceLse.CopyToHost(actualLse.data(), actualLse.size());
        ROCmKernels::Tests::CheckClose(reference.output, actualOutput, 5e-4);
        ROCmKernels::Tests::CheckClose(reference.lse, actualLse, 5e-4);
    }
} // namespace

TEST_CASE("Attention forward and backward match the reference", "[attention]") {
    SECTION("32-wide heads") { RunAttentionCase<32>(false, 2, 2, 4, 6); }

    SECTION("32-wide tiled") { RunAttentionCase<32>(true, 1, 1, 40, 40); }

    SECTION("32-wide flash") { RunFlashCase(-1, 1, 2, 40, 40, -1); }

    SECTION("32-wide flash banded") { RunFlashCase(-1, 1, 2, 64, 64, 5); }

    SECTION("32-wide flash scalar broadcast") { RunFlashCase(8, 1, 1, 40, 40, -1); }

    SECTION("32-wide flash fast exp") { RunFlashCase(1, 1, 1, 40, 40, -1); }

    SECTION("64-wide heads") { RunAttentionCase<64>(false, 1, 2, 4, 6); }
}
