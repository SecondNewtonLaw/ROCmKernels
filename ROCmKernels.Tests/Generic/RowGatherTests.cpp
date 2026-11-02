#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/RowGather.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../TestSupport.hpp"

#include <cstddef>
#include <type_traits>
#include <vector>

namespace {
    constexpr int kRows = 37;
    constexpr int kColumns = 100;
    constexpr int kTableRows = 512;

    template<typename T> T Narrow(const float value) {
        if constexpr (std::is_same_v<T, __half>) return __float2half_rn(value);
        else return value;
    }

    template<typename T> float Widen(const T value) {
        if constexpr (std::is_same_v<T, __half>) return __half2float(value);
        else return value;
    }

    template<typename T> std::vector<T> RandomTyped(const std::size_t count, const unsigned int seed) {
        const std::vector<float> values = ROCmKernels::Tests::RandomValues<float>(count, seed, -1.0f, 1.0f);

        std::vector<T> typed(count);
        for (std::size_t index = 0; index < count; ++index) typed[index] = Narrow<T>(values[index]);
        return typed;
    }

    template<typename T> std::vector<float> WidenAll(const std::vector<T> &values) {
        std::vector<float> widened(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) widened[index] = Widen<T>(values[index]);
        return widened;
    }

    /// Distinct indices because the stride is coprime with the table row count.
    std::vector<int> DistinctIndices() {
        std::vector<int> indices(kRows);
        for (int row = 0; row < kRows; ++row) indices[row] = (row * 13 + 7) % kTableRows;
        return indices;
    }

    template<typename T> void RunGetRows(const unsigned int seed) {
        const std::vector<T> table = RandomTyped<T>(static_cast<std::size_t>(kTableRows) * kColumns, seed);
        const std::vector<int> indices = DistinctIndices();

        std::vector<T> expected(static_cast<std::size_t>(kRows) * kColumns);
        for (int row = 0; row < kRows; ++row) {
            for (int column = 0; column < kColumns; ++column) {
                expected[static_cast<std::size_t>(row) * kColumns + column] = table[static_cast<std::size_t>(indices[row]) * kColumns + column];
            }
        }

        ROCmKernels::DeviceBuffer<T> deviceTable(table.size());
        ROCmKernels::DeviceBuffer<int> deviceIndices(indices.size());
        ROCmKernels::DeviceBuffer<T> deviceOutput(expected.size());

        deviceTable.CopyFromHost(table.data(), table.size());
        deviceIndices.CopyFromHost(indices.data(), indices.size());

        ROCmKernels::Kernels::GetRowsForward(
            deviceTable.Data(), deviceIndices.Data(), deviceOutput.Data(), kRows, kColumns, ROCmKernels::Kernels::DTypeOf<T>::value
        );
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actual(expected.size());
        deviceOutput.CopyToHost(actual.data(), actual.size());

        ROCmKernels::Tests::CheckClose(WidenAll(expected), WidenAll(actual), 0.0);
    }

    template<typename T> void RunSetRows(const unsigned int seed) {
        const std::vector<T> table = RandomTyped<T>(static_cast<std::size_t>(kTableRows) * kColumns, seed);
        const std::vector<T> input = RandomTyped<T>(static_cast<std::size_t>(kRows) * kColumns, seed + 1);
        const std::vector<int> indices = DistinctIndices();

        std::vector<T> expected = table;
        for (int row = 0; row < kRows; ++row) {
            for (int column = 0; column < kColumns; ++column) {
                expected[static_cast<std::size_t>(indices[row]) * kColumns + column] = input[static_cast<std::size_t>(row) * kColumns + column];
            }
        }

        ROCmKernels::DeviceBuffer<T> deviceTable(table.size());
        ROCmKernels::DeviceBuffer<T> deviceInput(input.size());
        ROCmKernels::DeviceBuffer<int> deviceIndices(indices.size());

        deviceTable.CopyFromHost(table.data(), table.size());
        deviceInput.CopyFromHost(input.data(), input.size());
        deviceIndices.CopyFromHost(indices.data(), indices.size());

        ROCmKernels::Kernels::SetRowsForward(
            deviceInput.Data(), deviceIndices.Data(), deviceTable.Data(), kRows, kColumns, ROCmKernels::Kernels::DTypeOf<T>::value
        );
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actual(expected.size());
        deviceTable.CopyToHost(actual.data(), actual.size());

        ROCmKernels::Tests::CheckClose(WidenAll(expected), WidenAll(actual), 0.0);
    }
} // namespace

TEST_CASE("Row gather copies the indexed table rows", "[generic][rowgather]") {
    SECTION("fp32") { RunGetRows<float>(301); }

    SECTION("fp16") { RunGetRows<__half>(302); }
}

TEST_CASE("Row scatter overwrites the indexed table rows", "[generic][rowgather]") {
    SECTION("fp32") { RunSetRows<float>(303); }

    SECTION("fp16") { RunSetRows<__half>(304); }
}

TEST_CASE("Row scatter adds rows that share an index", "[generic][rowgather]") {
    // The first two rows land on the same table row, so the kernel must accumulate rather than overwrite.
    std::vector<int> indices = DistinctIndices();
    indices[1] = indices[0];

    const std::vector<float> table = ROCmKernels::Tests::RandomValues<float>(static_cast<std::size_t>(kTableRows) * kColumns, 305, -1.0f, 1.0f);
    const std::vector<float> input = ROCmKernels::Tests::RandomValues<float>(static_cast<std::size_t>(kRows) * kColumns, 306, -1.0f, 1.0f);

    std::vector<float> expected = table;
    for (int row = 0; row < kRows; ++row) {
        for (int column = 0; column < kColumns; ++column) {
            expected[static_cast<std::size_t>(indices[row]) * kColumns + column] += input[static_cast<std::size_t>(row) * kColumns + column];
        }
    }

    ROCmKernels::DeviceBuffer<float> deviceTable(table.size());
    ROCmKernels::DeviceBuffer<float> deviceInput(input.size());
    ROCmKernels::DeviceBuffer<int> deviceIndices(indices.size());

    deviceTable.CopyFromHost(table.data(), table.size());
    deviceInput.CopyFromHost(input.data(), input.size());
    deviceIndices.CopyFromHost(indices.data(), indices.size());

    ROCmKernels::Kernels::AddRowsForward(
        deviceInput.Data(), deviceIndices.Data(), deviceTable.Data(), kRows, kColumns, ROCmKernels::Kernels::DType::Float32
    );
    ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> actual(expected.size());
    deviceTable.CopyToHost(actual.data(), actual.size());

    ROCmKernels::Tests::CheckClose(expected, actual, 1e-5);
}

TEST_CASE("Row gather rejects invalid arguments", "[generic][rowgather]") {
    float table = 0.0f;
    int index = 0;
    float output = 0.0f;

    CHECK(get_rows_forward(nullptr, &index, &output, kRows, kColumns, ROCmKernels::Kernels::DType::Float32, nullptr) == hipErrorInvalidValue);
    CHECK(get_rows_forward(&table, nullptr, &output, kRows, kColumns, ROCmKernels::Kernels::DType::Float32, nullptr) == hipErrorInvalidValue);
    CHECK(get_rows_forward(&table, &index, nullptr, kRows, kColumns, ROCmKernels::Kernels::DType::Float32, nullptr) == hipErrorInvalidValue);
    CHECK(get_rows_forward(&table, &index, &output, 0, kColumns, ROCmKernels::Kernels::DType::Float32, nullptr) == hipErrorInvalidValue);
    CHECK(get_rows_forward(&table, &index, &output, kRows, 0, ROCmKernels::Kernels::DType::Float32, nullptr) == hipErrorInvalidValue);

    CHECK(set_rows_forward(nullptr, &index, &table, kRows, kColumns, ROCmKernels::Kernels::DType::Float32, nullptr) == hipErrorInvalidValue);
    CHECK(set_rows_forward(&output, &index, nullptr, kRows, kColumns, ROCmKernels::Kernels::DType::Float32, nullptr) == hipErrorInvalidValue);

    CHECK(add_rows_forward(&output, &index, &table, kRows, kColumns, ROCmKernels::Kernels::DType::Float16, nullptr) == hipErrorNotSupported);
}
