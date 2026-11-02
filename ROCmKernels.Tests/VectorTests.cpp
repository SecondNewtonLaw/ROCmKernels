//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Kernels/Elementwise/Copy.hpp>
#include <ROCmKernels/Kernels/Elementwise/Vector.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

namespace {
    template <typename T> std::vector<T> MakeRandomValues(const std::size_t count, const unsigned int seed) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<T> distribution(static_cast<T>(-8), static_cast<T>(8));

        std::vector<T> values(count);
        for (T &value : values) {
            value = distribution(generator);
        }

        return values;
    }

    /// Verifies the first mismatch instead of the whole buffer so failures stay readable.
    template <typename T> void CheckAgainstReference(const std::vector<T> &expected, const std::vector<T> &actual) {
        const auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
        INFO("first mismatch at index " << std::distance(expected.begin(), mismatch.first));
        CHECK(mismatch.first == expected.end());
    }

    template <typename T> void CheckVectorAdd(const std::size_t count) {
        const std::vector<T> lhs = MakeRandomValues<T>(count, 1234);
        const std::vector<T> rhs = MakeRandomValues<T>(count, 9876);

        std::vector<T> expected(count);
        for (std::size_t index = 0; index < count; ++index) {
            expected[index] = lhs[index] + rhs[index];
        }

        ROCmKernels::DeviceBuffer<T> deviceLhs(count);
        ROCmKernels::DeviceBuffer<T> deviceRhs(count);
        ROCmKernels::DeviceBuffer<T> deviceOut(count);

        deviceLhs.CopyFromHost(lhs.data(), count);
        deviceRhs.CopyFromHost(rhs.data(), count);

        ROCmKernels::Kernels::VectorAdd(deviceLhs.Data(), deviceRhs.Data(), deviceOut.Data(), count);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actual(count);
        deviceOut.CopyToHost(actual.data(), count);

        CheckAgainstReference(expected, actual);
    }
} // namespace

TEST_CASE("VectorAdd matches the host reference", "[vector]") {
    SECTION("single element") { CheckVectorAdd<float>(1); }

    SECTION("odd element count") { CheckVectorAdd<float>(257); }

    SECTION("large buffer") { CheckVectorAdd<float>(1 << 20); }

    SECTION("double precision") { CheckVectorAdd<double>(4096); }
}

TEST_CASE("VectorAdd tolerates an empty input", "[vector]") {
    ROCmKernels::Kernels::VectorAdd(static_cast<const float *>(nullptr), static_cast<const float *>(nullptr), static_cast<float *>(nullptr), 0);
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());
}

TEST_CASE("VectorAdd honours an explicit stream", "[vector]") {
    constexpr std::size_t count = 1024;

    const std::vector<float> lhs = MakeRandomValues<float>(count, 42);
    const std::vector<float> rhs = MakeRandomValues<float>(count, 43);

    std::vector<float> expected(count);
    for (std::size_t index = 0; index < count; ++index) {
        expected[index] = lhs[index] + rhs[index];
    }

    ROCmKernels::DeviceBuffer<float> deviceLhs(count);
    ROCmKernels::DeviceBuffer<float> deviceRhs(count);
    ROCmKernels::DeviceBuffer<float> deviceOut(count);

    hipStream_t stream = nullptr;
    ROCMKERNELS_HIP_CHECK(hipStreamCreate(&stream));

    deviceLhs.CopyFromHost(lhs.data(), count, stream);
    deviceRhs.CopyFromHost(rhs.data(), count, stream);
    ROCmKernels::Kernels::VectorAdd(deviceLhs.Data(), deviceRhs.Data(), deviceOut.Data(), count, stream);

    std::vector<float> actual(count);
    deviceOut.CopyToHost(actual.data(), count, stream);

    ROCMKERNELS_HIP_CHECK(hipStreamSynchronize(stream));
    ROCMKERNELS_HIP_CHECK(hipStreamDestroy(stream));

    CheckAgainstReference(expected, actual);
}

TEST_CASE("Copy matches the source", "[copy][elementwise]") {
    const auto check = [](const std::size_t count) {
        const std::vector<float> source = MakeRandomValues<float>(count, 77);

        ROCmKernels::DeviceBuffer<float> deviceSource(count);
        ROCmKernels::DeviceBuffer<float> deviceDestination(count);
        deviceSource.CopyFromHost(source.data(), count);

        ROCmKernels::Kernels::Copy(deviceSource.Data(), deviceDestination.Data(), count);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> actual(count);
        deviceDestination.CopyToHost(actual.data(), count);
        CheckAgainstReference(source, actual);
    };

    SECTION("vectorized") { check(1u << 20); }

    SECTION("scalar tail") { check(1001); }
}
