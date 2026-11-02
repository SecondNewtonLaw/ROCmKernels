#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/TopK.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../TestSupport.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
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

    template<typename T> void RunCase(const int rows, const int classes, const int k, const unsigned int seed) {
        const std::size_t elementCount = static_cast<std::size_t>(rows) * classes;
        const std::size_t outputCount = static_cast<std::size_t>(rows) * k;

        const std::vector<float> hostValues = ROCmKernels::Tests::RandomValues<float>(elementCount, seed, -1.0f, 1.0f);

        std::vector<T> scores(elementCount);
        std::vector<float> widened(elementCount);
        for (std::size_t index = 0; index < elementCount; ++index) {
            scores[index] = Narrow<T>(hostValues[index]);
            widened[index] = Widen<T>(scores[index]);
        }

        // Reference: sort each row descending and keep the first k. Only values are compared,
        // so ties do not need a particular index order.
        std::vector<float> reference(outputCount);
        std::vector<float> sorted(static_cast<std::size_t>(classes));
        for (int row = 0; row < rows; ++row) {
            const std::size_t rowBase = static_cast<std::size_t>(row) * classes;
            for (int column = 0; column < classes; ++column) sorted[column] = widened[rowBase + column];

            std::sort(sorted.begin(), sorted.end(), std::greater<float>());

            const std::size_t outputBase = static_cast<std::size_t>(row) * k;
            for (int slot = 0; slot < k; ++slot) reference[outputBase + slot] = sorted[slot];
        }

        ROCmKernels::DeviceBuffer<T> deviceScores(elementCount);
        ROCmKernels::DeviceBuffer<float> deviceValues(outputCount);
        ROCmKernels::DeviceBuffer<int> deviceIndices(outputCount);

        deviceScores.CopyFromHost(scores.data(), scores.size());

        ROCmKernels::Kernels::TopKForward(deviceScores.Data(), deviceValues.Data(), deviceIndices.Data(), rows, classes, k,
                ROCmKernels::Kernels::DTypeOf<T>::value);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> actualValues(outputCount);
        std::vector<int> actualIndices(outputCount);
        deviceValues.CopyToHost(actualValues.data(), actualValues.size());
        deviceIndices.CopyToHost(actualIndices.data(), actualIndices.size());

        ROCmKernels::Tests::CheckClose(reference, actualValues, 1e-6);

        for (int row = 0; row < rows; ++row) {
            const std::size_t rowBase = static_cast<std::size_t>(row) * classes;
            const std::size_t outputBase = static_cast<std::size_t>(row) * k;

            for (int slot = 0; slot < k; ++slot) {
                const std::size_t outputIndex = outputBase + slot;
                const int classIndex = actualIndices[outputIndex];

                REQUIRE(classIndex >= 0);
                REQUIRE(classIndex < classes);

                // Every returned index must resolve to the returned value.
                CHECK(widened[rowBase + classIndex] == actualValues[outputIndex]);

                if (slot + 1 < k) CHECK(actualValues[outputIndex] >= actualValues[outputIndex + 1]);
            }
        }
    }
} // namespace

TEST_CASE("TopK matches the reference", "[generic][topk]") {
    SECTION("fp32 rows 5 classes 100 k 8") { RunCase<float>(5, 100, 8, 201); }

    SECTION("fp16 rows 3 classes 5000 k 8") { RunCase<__half>(3, 5000, 8, 202); }

    SECTION("k 1") {
        RunCase<float>(7, 1000, 1, 203);
        RunCase<__half>(4, 513, 1, 204);
    }

    SECTION("k 8 with classes 8") {
        RunCase<float>(2, 8, 8, 205);
        RunCase<__half>(3, 8, 8, 206);
    }

    SECTION("k 3 spanning the block size") {
        RunCase<float>(4, 257, 3, 207);
        RunCase<__half>(4, 257, 3, 208);
    }
}
