//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Embedding/EmbeddingGrad.hpp>

#include <catch2/catch_test_macros.hpp>

#include "TestSupport.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
    std::vector<float> ReferenceDense(
        const std::vector<std::int64_t> &keys, const std::vector<std::int64_t> &order, const std::vector<float> &values, const int rows, const int columns
    ) {
        std::vector<float> dense(static_cast<std::size_t>(rows) * columns, 0.0f);

        for (int position = 0; position < static_cast<int>(keys.size()); ++position) {
            const std::int64_t row = keys[position];
            for (int column = 0; column < columns; ++column) {
                dense[static_cast<std::size_t>(row) * columns + column] += values[static_cast<std::size_t>(order[position]) * columns + column];
            }
        }

        return dense;
    }
} // namespace

TEST_CASE("Embedding gradient conversion matches the reference", "[embedding]") {
    constexpr int rows = 8;
    constexpr int columns = 8;
    constexpr int nnz = 3;

    const std::vector<std::int64_t> keys{0, 2, 5};
    const std::vector<std::int64_t> order{2, 0, 1};
    const std::vector<float> values = ROCmKernels::Tests::RandomValues<float>(nnz * columns, 71, -1.0f, 1.0f);

    const std::vector<float> expected = ReferenceDense(keys, order, values, rows, columns);

    ROCmKernels::DeviceBuffer<std::int64_t> deviceKeys(nnz);
    ROCmKernels::DeviceBuffer<std::int64_t> deviceOrder(nnz);
    ROCmKernels::DeviceBuffer<float> deviceValues(nnz * columns);
    ROCmKernels::DeviceBuffer<float> deviceDense(rows * columns);

    deviceKeys.CopyFromHost(keys.data(), nnz);
    deviceOrder.CopyFromHost(order.data(), nnz);
    deviceValues.CopyFromHost(values.data(), values.size());

    SECTION("scalar") {
        ROCmKernels::Kernels::EmbeddingGrad(deviceKeys.Data(), deviceOrder.Data(), deviceValues.Data(), deviceDense.Data(), nnz, rows, columns);
    }

    SECTION("vectorized") {
        ROCmKernels::Kernels::EmbeddingGradTuned(deviceKeys.Data(), deviceOrder.Data(), deviceValues.Data(), deviceDense.Data(), nnz, rows, columns, true);
    }

    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> actual(rows * columns);
    deviceDense.CopyToHost(actual.data(), actual.size());

    ROCmKernels::Tests::CheckClose(expected, actual, 1e-6);
}

TEST_CASE("Embedding gradient conversion zeroes a dense buffer with no rows", "[embedding]") {
    constexpr int rows = 4;
    constexpr int columns = 8;

    ROCmKernels::DeviceBuffer<float> deviceDense(rows * columns);
    std::vector<float> initial(rows * columns, 1.0f);
    deviceDense.CopyFromHost(initial.data(), initial.size());

    ROCmKernels::Kernels::EmbeddingGrad(nullptr, nullptr, nullptr, deviceDense.Data(), 0, rows, columns);
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> actual(rows * columns);
    deviceDense.CopyToHost(actual.data(), actual.size());

    ROCmKernels::Tests::CheckClose(std::vector<float>(rows * columns, 0.0f), actual, 0.0);
}
