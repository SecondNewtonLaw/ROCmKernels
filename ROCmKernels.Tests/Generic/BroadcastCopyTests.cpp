#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/BroadcastCopy.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../TestSupport.hpp"

#include <cstddef>
#include <type_traits>
#include <vector>

namespace {
    constexpr int kRows = 7;
    constexpr int kColumns = 100;

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

    const char *OpName(const ROCmKernels::Kernels::BinaryOp op) {
        using ROCmKernels::Kernels::BinaryOp;

        switch (op) {
            case BinaryOp::Add: return "add";
            case BinaryOp::Sub: return "sub";
            case BinaryOp::Mul: return "mul";
            case BinaryOp::Div: return "div";
        }

        return "unknown";
    }

    const char *ModeName(const ROCmKernels::Kernels::BroadcastMode mode) {
        using ROCmKernels::Kernels::BroadcastMode;

        switch (mode) {
            case BroadcastMode::None: return "none";
            case BroadcastMode::Row: return "row";
            case BroadcastMode::Column: return "column";
            case BroadcastMode::Scalar: return "scalar";
        }

        return "unknown";
    }

    double Apply(const ROCmKernels::Kernels::BinaryOp op, const double a, const double b) {
        using ROCmKernels::Kernels::BinaryOp;

        switch (op) {
            case BinaryOp::Add: return a + b;
            case BinaryOp::Sub: return a - b;
            case BinaryOp::Mul: return a * b;
            case BinaryOp::Div: return a / b;
        }

        return 0.0;
    }

    int OperandElements(const ROCmKernels::Kernels::BroadcastMode mode, const int rows, const int columns) {
        using ROCmKernels::Kernels::BroadcastMode;

        switch (mode) {
            case BroadcastMode::None: return rows * columns;
            case BroadcastMode::Row: return columns;
            case BroadcastMode::Column: return rows;
            case BroadcastMode::Scalar: return 1;
        }

        return 0;
    }

    long long OperandIndex(const ROCmKernels::Kernels::BroadcastMode mode, const int row, const int column, const int columns) {
        using ROCmKernels::Kernels::BroadcastMode;

        switch (mode) {
            case BroadcastMode::None: return static_cast<long long>(row) * columns + column;
            case BroadcastMode::Row: return column;
            case BroadcastMode::Column: return row;
            case BroadcastMode::Scalar: return 0;
        }

        return 0;
    }

    template<typename T>
    void RunBroadcast(const ROCmKernels::Kernels::BinaryOp op, const ROCmKernels::Kernels::BroadcastMode mode, const unsigned int seed) {
        using ROCmKernels::Kernels::BinaryOp;
        using ROCmKernels::Kernels::BroadcastMode;
        using ROCmKernels::Kernels::DTypeOf;

        INFO("op " << OpName(op) << ", mode " << ModeName(mode));

        const std::size_t elements = static_cast<std::size_t>(kRows) * kColumns;
        const float operandLow = op == BinaryOp::Div ? 0.5f : -2.0f;

        const std::vector<T> a = Narrow<T>(ROCmKernels::Tests::RandomValues<float>(elements, seed, -2.0f, 2.0f));
        const std::vector<T> b = Narrow<T>(ROCmKernels::Tests::RandomValues<float>(
                static_cast<std::size_t>(OperandElements(mode, kRows, kColumns)), seed + 1, operandLow, 2.0f));

        const std::vector<float> wideA = Widen<T>(a);
        const std::vector<float> wideB = Widen<T>(b);
        std::vector<float> expected(elements);
        for (int row = 0; row < kRows; ++row) {
            for (int column = 0; column < kColumns; ++column) {
                const long long index = OperandIndex(mode, row, column, kColumns);
                expected[static_cast<std::size_t>(row) * kColumns + column] =
                        static_cast<float>(Apply(op, wideA[static_cast<std::size_t>(row) * kColumns + column], wideB[static_cast<std::size_t>(index)]));
            }
        }

        ROCmKernels::DeviceBuffer<T> deviceA(elements);
        ROCmKernels::DeviceBuffer<T> deviceB(b.size());
        ROCmKernels::DeviceBuffer<T> deviceOut(elements);

        deviceA.CopyFromHost(a.data(), a.size());
        deviceB.CopyFromHost(b.data(), b.size());

        ROCmKernels::Kernels::BroadcastBinary(deviceA.Data(), deviceB.Data(), deviceOut.Data(), kRows, kColumns, op, mode, DTypeOf<T>::value);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actual(elements);
        deviceOut.CopyToHost(actual.data(), actual.size());

        const double tolerance = std::is_same_v<T, __half> ? 4e-3 : 5e-4;
        ROCmKernels::Tests::CheckClose(expected, Widen<T>(actual), tolerance);
    }

    template<typename T> void RunCopy(const unsigned int seed) {
        using ROCmKernels::Kernels::DTypeOf;

        const int sourceRowStride = kColumns + 3;
        const int destinationRowStride = kColumns + 5;
        const std::size_t sourceElements = static_cast<std::size_t>(kRows) * sourceRowStride;
        const std::size_t destinationElements = static_cast<std::size_t>(kRows) * destinationRowStride;

        const std::vector<T> source = Narrow<T>(ROCmKernels::Tests::RandomValues<float>(sourceElements, seed, -2.0f, 2.0f));
        const std::vector<T> sentinel(destinationElements, Narrow<T>(-3.0f));
        std::vector<T> expected = sentinel;
        for (int row = 0; row < kRows; ++row) {
            for (int column = 0; column < kColumns; ++column) {
                expected[static_cast<std::size_t>(row) * destinationRowStride + column] =
                        source[static_cast<std::size_t>(row) * sourceRowStride + column];
            }
        }

        ROCmKernels::DeviceBuffer<T> deviceSource(sourceElements);
        ROCmKernels::DeviceBuffer<T> deviceDestination(destinationElements);

        deviceSource.CopyFromHost(source.data(), source.size());
        deviceDestination.CopyFromHost(sentinel.data(), sentinel.size());

        ROCmKernels::Kernels::Copy2D(deviceSource.Data(), deviceDestination.Data(), kRows, kColumns, sourceRowStride, destinationRowStride,
                DTypeOf<T>::value);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actual(destinationElements);
        deviceDestination.CopyToHost(actual.data(), actual.size());

        // Padding is checked too: a stride error would overwrite the sentinel.
        ROCmKernels::Tests::CheckClose(Widen<T>(expected), Widen<T>(actual), 0.0);
    }

    template<typename T> void RunFill(const unsigned int seed) {
        using ROCmKernels::Kernels::DTypeOf;

        constexpr long long elements = 1000;
        constexpr float value = 0.75f;

        const std::vector<T> initial = Narrow<T>(ROCmKernels::Tests::RandomValues<float>(elements, seed, -2.0f, 2.0f));
        const std::vector<float> expected(elements, value);

        ROCmKernels::DeviceBuffer<T> deviceOutput(elements);
        deviceOutput.CopyFromHost(initial.data(), initial.size());

        ROCmKernels::Kernels::Fill(deviceOutput.Data(), value, elements, DTypeOf<T>::value);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actual(elements);
        deviceOutput.CopyToHost(actual.data(), actual.size());

        ROCmKernels::Tests::CheckClose(expected, Widen<T>(actual), 0.0);
    }
} // namespace

TEST_CASE("Broadcast binary matches the reference", "[generic][broadcast]") {
    using ROCmKernels::Kernels::BinaryOp;
    using ROCmKernels::Kernels::BroadcastMode;

    constexpr BinaryOp ops[]{BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div};
    constexpr BroadcastMode modes[]{BroadcastMode::None, BroadcastMode::Row, BroadcastMode::Column, BroadcastMode::Scalar};

    SECTION("fp32") {
        unsigned int seed = 101;
        for (const BinaryOp op : ops) {
            for (const BroadcastMode mode : modes) RunBroadcast<float>(op, mode, seed++);
        }
    }

    SECTION("fp16") {
        unsigned int seed = 201;
        for (const BinaryOp op : ops) {
            for (const BroadcastMode mode : modes) RunBroadcast<__half>(op, mode, seed++);
        }
    }
}

TEST_CASE("Copy2D honours row strides", "[generic][copy]") {
    SECTION("fp32") { RunCopy<float>(301); }
    SECTION("fp16") { RunCopy<__half>(302); }
}

TEST_CASE("Fill writes every element", "[generic][fill]") {
    SECTION("fp32") { RunFill<float>(401); }
    SECTION("fp16") { RunFill<__half>(402); }
}
