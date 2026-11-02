#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/AttentionGeneric.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../TestSupport.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace {
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

    struct Reference {
        std::vector<float> output;
        std::vector<float> lse;
    };

    /// Double-precision forward reference over the exact values the device reads.
    Reference ComputeReference(const std::vector<float> &q, const std::vector<float> &k, const std::vector<float> &v,
            const std::vector<std::uint8_t> &mask, const int headWidth, const int batchSize, const int heads, const int queries, const int keys,
            const int radius) {
        const double scale = 1.0 / std::sqrt(static_cast<double>(headWidth));

        Reference reference;
        reference.output.assign(static_cast<std::size_t>(batchSize) * heads * queries * headWidth, 0.0f);
        reference.lse.assign(static_cast<std::size_t>(batchSize) * heads * queries * 2, 0.0f);

        std::vector<double> scores(static_cast<std::size_t>(keys));
        std::vector<double> probabilities(static_cast<std::size_t>(keys));

        for (int batch = 0; batch < batchSize; ++batch) {
            for (int head = 0; head < heads; ++head) {
                for (int query = 0; query < queries; ++query) {
                    const std::size_t qBase =
                            (static_cast<std::size_t>(batch) * heads + head) * queries * headWidth + static_cast<std::size_t>(query) * headWidth;

                    double maximum = -INFINITY;
                    for (int key = 0; key < keys; ++key) {
                        scores[key] = -INFINITY;
                        if (!mask[static_cast<std::size_t>(batch) * keys + key]) continue;
                        if (radius >= 0 && std::abs(query - key) > radius) continue;

                        const std::size_t kBase =
                                (static_cast<std::size_t>(batch) * heads + head) * keys * headWidth + static_cast<std::size_t>(key) * headWidth;
                        double dot = 0.0;
                        for (int column = 0; column < headWidth; ++column) {
                            dot += static_cast<double>(q[qBase + column]) * static_cast<double>(k[kBase + column]);
                        }
                        scores[key] = dot * scale;
                        maximum = std::max(maximum, scores[key]);
                    }

                    double total = 0.0;
                    for (int key = 0; key < keys; ++key) {
                        probabilities[key] = maximum == -INFINITY ? 0.0 : std::exp(scores[key] - maximum);
                        total += probabilities[key];
                    }

                    const std::size_t row = (static_cast<std::size_t>(batch) * heads + head) * queries + query;
                    reference.lse[row * 2] = maximum == -INFINITY ? -INFINITY : static_cast<float>(maximum);
                    reference.lse[row * 2 + 1] = static_cast<float>(total);

                    for (int column = 0; column < headWidth; ++column) {
                        double value = 0.0;
                        for (int key = 0; key < keys; ++key) {
                            if (probabilities[key] == 0.0) continue;
                            const std::size_t kBase = (static_cast<std::size_t>(batch) * heads + head) * keys * headWidth +
                                                      static_cast<std::size_t>(key) * headWidth;
                            value += probabilities[key] * static_cast<double>(v[kBase + column]);
                        }
                        reference.output[qBase + column] = static_cast<float>(total > 0 ? value / total : 0.0);
                    }
                }
            }
        }

        return reference;
    }

    template<typename T, int HeadWidth>
    void RunCase(const int batchSize, const int heads, const int queries, const int keys, const int radius) {
        const std::size_t qElements = static_cast<std::size_t>(batchSize) * heads * queries * HeadWidth;
        const std::size_t kElements = static_cast<std::size_t>(batchSize) * heads * keys * HeadWidth;

        const std::vector<T> q = Narrow<T>(ROCmKernels::Tests::RandomValues<float>(qElements, 101, -1.0f, 1.0f));
        const std::vector<T> k = Narrow<T>(ROCmKernels::Tests::RandomValues<float>(kElements, 102, -1.0f, 1.0f));
        const std::vector<T> v = Narrow<T>(ROCmKernels::Tests::RandomValues<float>(kElements, 103, -1.0f, 1.0f));

        std::vector<std::uint8_t> mask(static_cast<std::size_t>(batchSize) * keys, 1);
        mask[3] = 0;

        const Reference reference =
                ComputeReference(Widen<T>(q), Widen<T>(k), Widen<T>(v), mask, HeadWidth, batchSize, heads, queries, keys, radius);

        ROCmKernels::DeviceBuffer<T> deviceQ(qElements);
        ROCmKernels::DeviceBuffer<T> deviceK(kElements);
        ROCmKernels::DeviceBuffer<T> deviceV(kElements);
        ROCmKernels::DeviceBuffer<std::uint8_t> deviceMask(mask.size());
        ROCmKernels::DeviceBuffer<T> deviceOutput(qElements);
        ROCmKernels::DeviceBuffer<float> deviceLse(static_cast<std::size_t>(batchSize) * heads * queries * 2);

        deviceQ.CopyFromHost(q.data(), q.size());
        deviceK.CopyFromHost(k.data(), k.size());
        deviceV.CopyFromHost(v.data(), v.size());
        deviceMask.CopyFromHost(mask.data(), mask.size());

        const ROCmKernels::Kernels::Strides qStrides{heads * queries * HeadWidth, queries * HeadWidth, HeadWidth};
        const ROCmKernels::Kernels::Strides kStrides{heads * keys * HeadWidth, keys * HeadWidth, HeadWidth};

        ROCmKernels::Kernels::AttentionGenericForward(deviceQ.Data(), deviceK.Data(), deviceV.Data(),
                reinterpret_cast<const bool *>(deviceMask.Data()), deviceOutput.Data(), deviceLse.Data(), batchSize, heads, queries, keys,
                HeadWidth, radius, qStrides, kStrides, kStrides, qStrides, ROCmKernels::Kernels::DTypeOf<T>::value);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actualOutput(qElements);
        std::vector<float> actualLse(reference.lse.size());
        deviceOutput.CopyToHost(actualOutput.data(), actualOutput.size());
        deviceLse.CopyToHost(actualLse.data(), actualLse.size());

        const double tolerance = std::is_same_v<T, __half> ? 2e-3 : 5e-4;
        ROCmKernels::Tests::CheckClose(reference.output, Widen<T>(actualOutput), tolerance);
        ROCmKernels::Tests::CheckClose(reference.lse, actualLse, tolerance);
    }
} // namespace

TEST_CASE("AttentionGeneric matches the reference", "[generic][attention]") {
    SECTION("head width 16") {
        SECTION("fp32") {
            RunCase<float, 16>(1, 2, 40, 48, -1);
            RunCase<float, 16>(2, 1, 33, 37, 3);
        }

        SECTION("fp16") {
            RunCase<__half, 16>(1, 2, 40, 48, -1);
            RunCase<__half, 16>(2, 1, 33, 37, 3);
        }
    }

    SECTION("head width 32") {
        SECTION("fp32") {
            RunCase<float, 32>(1, 2, 40, 48, -1);
            RunCase<float, 32>(2, 1, 33, 37, 3);
        }

        SECTION("fp16") {
            RunCase<__half, 32>(1, 2, 40, 48, -1);
            RunCase<__half, 32>(2, 1, 33, 37, 3);
        }
    }

    SECTION("head width 64") {
        SECTION("fp32") {
            RunCase<float, 64>(1, 2, 40, 48, -1);
            RunCase<float, 64>(2, 1, 33, 37, 3);
        }

        SECTION("fp16") {
            RunCase<__half, 64>(1, 2, 40, 48, -1);
            RunCase<__half, 64>(2, 1, 33, 37, 3);
        }
    }
}
