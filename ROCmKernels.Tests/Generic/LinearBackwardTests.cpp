#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <ROCmKernels/Kernels/Generic/LinearBackward.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../TestSupport.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace {
    double ReferenceSiLUDerivative(const double value) {
        const double sigmoid = 1.0 / (1.0 + std::exp(-value));
        return sigmoid * (1.0 + value * (1.0 - sigmoid));
    }
} // namespace

TEST_CASE("Transpose fp16 matches the reference", "[generic][linear]") {
    constexpr int rows = 96;
    constexpr int columns = 40;

    const std::vector<float> source = ROCmKernels::Tests::RandomValues<float>(rows * columns, 31, -1.0f, 1.0f);
    std::vector<__half> hostSource(source.size());
    for (std::size_t index = 0; index < source.size(); ++index) hostSource[index] = __float2half_rn(source[index]);

    ROCmKernels::DeviceBuffer<__half> deviceSource(hostSource.size());
    ROCmKernels::DeviceBuffer<__half> deviceDestination(hostSource.size());
    deviceSource.CopyFromHost(hostSource.data(), hostSource.size());

    ROCmKernels::Kernels::TransposeFp16(deviceSource.Data(), deviceDestination.Data(), rows, columns);

    std::vector<__half> actual(hostSource.size());
    deviceDestination.CopyToHost(actual.data(), actual.size());

    int mismatches = 0;
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < columns; ++column)
            if (__half2float(actual[column * rows + row]) != __half2float(hostSource[row * columns + column])) ++mismatches;
    CHECK(mismatches == 0);
}

TEST_CASE("SiLU backward and bias gradient match the reference", "[generic][linear]") {
    constexpr int rows = 128;
    constexpr int columns = 64;

    const std::vector<float> hostDh = ROCmKernels::Tests::RandomValues<float>(rows * columns, 41, -1.0f, 1.0f);
    const std::vector<float> hostZ = ROCmKernels::Tests::RandomValues<float>(rows * columns, 42, -4.0f, 4.0f);
    std::vector<__half> dh(hostDh.size()), z(hostZ.size());
    for (std::size_t index = 0; index < hostDh.size(); ++index) {
        dh[index] = __float2half_rn(hostDh[index]);
        z[index] = __float2half_rn(hostZ[index]);
    }

    ROCmKernels::DeviceBuffer<__half> deviceDh(dh.size());
    ROCmKernels::DeviceBuffer<__half> deviceZ(z.size());
    ROCmKernels::DeviceBuffer<__half> deviceDz(dh.size());
    ROCmKernels::DeviceBuffer<float> deviceDbias(columns);
    ROCmKernels::DeviceBuffer<__half> deviceH(dh.size());
    deviceDh.CopyFromHost(dh.data(), dh.size());
    deviceZ.CopyFromHost(z.data(), z.size());

    ROCmKernels::Kernels::SiluBackwardFp16(deviceDh.Data(), deviceZ.Data(), deviceDz.Data(), deviceH.Data(), deviceDbias.Data(), rows, columns);

    std::vector<__half> actualDz(dh.size());
    std::vector<float> actualDbias(columns);
    deviceDz.CopyToHost(actualDz.data(), actualDz.size());
    deviceDbias.CopyToHost(actualDbias.data(), actualDbias.size());

    std::vector<__half> actualH(dh.size());
    deviceH.CopyToHost(actualH.data(), actualH.size());
    double worst = 0.0, worstH = 0.0;
    for (std::size_t index = 0; index < dh.size(); ++index) {
        const double zValue = static_cast<double>(__half2float(z[index]));
        const double expected = static_cast<double>(__half2float(dh[index])) * ReferenceSiLUDerivative(zValue);
        worst = std::max(worst, std::abs(__half2float(actualDz[index]) - expected));
        const double silu = zValue / (1.0 + std::exp(-zValue));
        worstH = std::max(worstH, std::abs(__half2float(actualH[index]) - silu));
    }
    CHECK(worst < 5e-3);
    CHECK(worstH < 2e-3);

    double worstBias = 0.0;
    for (int column = 0; column < columns; ++column) {
        double expected = 0.0;
        for (int row = 0; row < rows; ++row)
            expected += __half2float(dh[static_cast<std::size_t>(row) * columns + column]) *
                        ReferenceSiLUDerivative(static_cast<double>(__half2float(z[static_cast<std::size_t>(row) * columns + column])));
        worstBias = std::max(worstBias, std::abs(static_cast<double>(actualDbias[column]) - expected));
    }
    CHECK(worstBias < 2e-2);
}


