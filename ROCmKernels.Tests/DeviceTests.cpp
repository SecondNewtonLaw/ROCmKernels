//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Common/Device.hpp>
#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Common/Error.hpp>

#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <vector>

TEST_CASE("The HIP runtime sees the gfx1031 target device", "[device]") {
    const int deviceCount = ROCmKernels::GetDeviceCount();
    REQUIRE(deviceCount > 0);

    const ROCmKernels::DeviceInfo info = ROCmKernels::QueryDevice(0);
    INFO("Device: " << info.name << " (" << info.gcnArchName << ")");
    REQUIRE(info.gcnArchName.rfind("gfx1031", 0) == 0);
    REQUIRE(info.computeUnitCount > 0);
    REQUIRE(info.totalGlobalMemory > 0);
    REQUIRE(info.warpSize > 0);
    REQUIRE(info.runtimeVersion > 0);
}

TEST_CASE("DeviceBuffer round-trips host data", "[device][buffer]") {
    constexpr std::size_t elementCount = 1024;

    std::vector<int> source(elementCount);
    std::iota(source.begin(), source.end(), 0);

    ROCmKernels::DeviceBuffer<int> buffer(elementCount);
    REQUIRE(buffer.Data() != nullptr);
    REQUIRE(buffer.Count() == elementCount);
    REQUIRE(buffer.ByteSize() == elementCount * sizeof(int));

    buffer.CopyFromHost(source.data(), elementCount);

    std::vector<int> destination(elementCount, -1);
    buffer.CopyToHost(destination.data(), elementCount);

    CHECK(destination == source);
}

TEST_CASE("DeviceBuffer can be moved without freeing the allocation", "[device][buffer]") {
    constexpr std::size_t elementCount = 16;

    ROCmKernels::DeviceBuffer<float> source(elementCount);
    const float *const originalData = source.Data();

    ROCmKernels::DeviceBuffer<float> destination(std::move(source));
    REQUIRE(destination.Data() == originalData);
    REQUIRE(destination.Count() == elementCount);
    REQUIRE(source.Data() == nullptr);
    REQUIRE(source.Count() == 0);
}

TEST_CASE("DeviceBuffer releases its allocation", "[device][buffer]") {
    constexpr std::size_t elementCount = 64;

    ROCmKernels::DeviceBuffer<float> buffer(elementCount);
    REQUIRE(buffer.Data() != nullptr);

    buffer.Release();
    REQUIRE(buffer.Data() == nullptr);
    REQUIRE(buffer.Count() == 0);
}
