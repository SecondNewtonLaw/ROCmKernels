#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <ROCmKernels/Kernels/Generic/LinearActFast.hpp>
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

TEST_CASE("TN linear with fp32 output matches the reference", "[generic][linear]") {
    constexpr int inner = 200;
    // Both tile configurations, with row and column tails.
    for (const auto [rows, columns] : {std::pair{136, 72}, std::pair{72, 200}}) {
        const std::vector<float> hostA = ROCmKernels::Tests::RandomValues<float>(inner * rows, 51, -1.0f, 1.0f);
        const std::vector<float> hostB = ROCmKernels::Tests::RandomValues<float>(inner * columns, 52, -1.0f, 1.0f);
        std::vector<__half> a(hostA.size()), b(hostB.size());
        for (std::size_t index = 0; index < a.size(); ++index) a[index] = __float2half_rn(hostA[index]);
        for (std::size_t index = 0; index < b.size(); ++index) b[index] = __float2half_rn(hostB[index]);

        ROCmKernels::DeviceBuffer<__half> deviceA(a.size());
        ROCmKernels::DeviceBuffer<__half> deviceB(b.size());
        ROCmKernels::DeviceBuffer<float> deviceOutput(static_cast<std::size_t>(rows) * columns);
        deviceA.CopyFromHost(a.data(), a.size());
        deviceB.CopyFromHost(b.data(), b.size());

        REQUIRE(linear_tn_fp32_out(deviceA.Data(), deviceB.Data(), deviceOutput.Data(), rows, columns, inner, nullptr) == hipSuccess);

        std::vector<float> actual(static_cast<std::size_t>(rows) * columns);
        deviceOutput.CopyToHost(actual.data(), actual.size());

        double worst = 0.0;
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < columns; ++column) {
                double expected = 0.0;
                for (int k = 0; k < inner; ++k)
                    expected += static_cast<double>(__half2float(a[static_cast<std::size_t>(k) * rows + row])) *
                                __half2float(b[static_cast<std::size_t>(k) * columns + column]);
                worst = std::max(worst, std::abs(actual[static_cast<std::size_t>(row) * columns + column] - expected));
            }
        CHECK(worst < 1e-3);
    }
}



TEST_CASE("SiLU backward from h and s matches the reference", "[generic][linear]") {
    constexpr int rows = 128;
    constexpr int columns = 64;

    const std::vector<float> hostDh = ROCmKernels::Tests::RandomValues<float>(rows * columns, 81, -1.0f, 1.0f);
    const std::vector<float> hostZ = ROCmKernels::Tests::RandomValues<float>(rows * columns, 82, -4.0f, 4.0f);
    std::vector<__half> dh(hostDh.size()), h(hostDh.size()), s(hostDh.size());
    for (std::size_t index = 0; index < hostDh.size(); ++index) {
        dh[index] = __float2half_rn(hostDh[index]);
        const float zValue = __half2float(__float2half_rn(hostZ[index]));
        const float sigmoid = 1.0f / (1.0f + std::exp(-zValue));
        h[index] = __float2half_rn(zValue * sigmoid);
        s[index] = __float2half_rn(sigmoid);
    }

    ROCmKernels::DeviceBuffer<__half> deviceDh(dh.size());
    ROCmKernels::DeviceBuffer<__half> deviceH(h.size());
    ROCmKernels::DeviceBuffer<__half> deviceS(s.size());
    ROCmKernels::DeviceBuffer<__half> deviceDz(dh.size());
    ROCmKernels::DeviceBuffer<float> deviceDbias(columns);
    deviceDh.CopyFromHost(dh.data(), dh.size());
    deviceH.CopyFromHost(h.data(), h.size());
    deviceS.CopyFromHost(s.data(), s.size());

    ROCmKernels::Kernels::SiluBackwardHsFp16(deviceDh.Data(), deviceH.Data(), deviceS.Data(), deviceDz.Data(), deviceDbias.Data(), rows, columns);

    std::vector<__half> actualDz(dh.size());
    std::vector<float> actualDbias(columns);
    deviceDz.CopyToHost(actualDz.data(), actualDz.size());
    deviceDbias.CopyToHost(actualDbias.data(), actualDbias.size());

    double worst = 0.0, worstBias = 0.0;
    for (std::size_t index = 0; index < dh.size(); ++index) {
        const float sigmoid = __half2float(s[index]);
        const float activation = __half2float(h[index]);
        const float expected = __half2float(dh[index]) * (sigmoid + activation * (1.0f - sigmoid));
        worst = std::max(worst, static_cast<double>(std::abs(__half2float(actualDz[index]) - expected)));
    }
    for (int column = 0; column < columns; ++column) {
        double expected = 0.0;
        for (int row = 0; row < rows; ++row)
            expected += __half2float(actualDz[static_cast<std::size_t>(row) * columns + column]);
        worstBias = std::max(worstBias, std::abs(static_cast<double>(actualDbias[column]) - expected));
    }
    CHECK(worst < 5e-3);
    CHECK(worstBias < 2e-2);
}

TEST_CASE("SiLU backward from h and s handles a tail", "[generic][linear]") {
    constexpr int rows = 127;
    constexpr int columns = 63;

    const std::vector<float> hostDh = ROCmKernels::Tests::RandomValues<float>(rows * columns, 91, -1.0f, 1.0f);
    const std::vector<float> hostZ = ROCmKernels::Tests::RandomValues<float>(rows * columns, 92, -4.0f, 4.0f);
    std::vector<__half> dh(hostDh.size()), h(hostDh.size()), s(hostDh.size());
    for (std::size_t index = 0; index < hostDh.size(); ++index) {
        dh[index] = __float2half_rn(hostDh[index]);
        const float zValue = __half2float(__float2half_rn(hostZ[index]));
        const float sigmoid = 1.0f / (1.0f + std::exp(-zValue));
        h[index] = __float2half_rn(zValue * sigmoid);
        s[index] = __float2half_rn(sigmoid);
    }

    ROCmKernels::DeviceBuffer<__half> deviceDh(dh.size());
    ROCmKernels::DeviceBuffer<__half> deviceH(h.size());
    ROCmKernels::DeviceBuffer<__half> deviceS(s.size());
    ROCmKernels::DeviceBuffer<__half> deviceDz(dh.size());
    deviceDh.CopyFromHost(dh.data(), dh.size());
    deviceH.CopyFromHost(h.data(), h.size());
    deviceS.CopyFromHost(s.data(), s.size());

    ROCmKernels::Kernels::SiluBackwardHsFp16(deviceDh.Data(), deviceH.Data(), deviceS.Data(), deviceDz.Data(), nullptr, rows, columns);

    std::vector<__half> actualDz(dh.size());
    deviceDz.CopyToHost(actualDz.data(), actualDz.size());

    double worst = 0.0;
    for (std::size_t index = 0; index < dh.size(); ++index) {
        const float sigmoid = __half2float(s[index]);
        const float activation = __half2float(h[index]);
        const float expected = __half2float(dh[index]) * (sigmoid + activation * (1.0f - sigmoid));
        worst = std::max(worst, static_cast<double>(std::abs(__half2float(actualDz[index]) - expected)));
    }
    CHECK(worst < 5e-3);
}
