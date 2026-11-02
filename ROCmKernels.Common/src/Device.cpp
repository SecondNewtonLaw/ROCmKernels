//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Common/Device.hpp>

#include <ROCmKernels/Common/Error.hpp>

namespace ROCmKernels {
    int GetDeviceCount() {
        int deviceCount = 0;
        ROCMKERNELS_HIP_CHECK(hipGetDeviceCount(&deviceCount));
        return deviceCount;
    }

    DeviceInfo QueryDevice(const int deviceIndex) {
        hipDeviceProp_t properties{};
        ROCMKERNELS_HIP_CHECK(hipGetDeviceProperties(&properties, deviceIndex));

        DeviceInfo info{};
        info.name = properties.name;
        info.gcnArchName = properties.gcnArchName;
        info.totalGlobalMemory = static_cast<std::uint64_t>(properties.totalGlobalMem);
        info.computeUnitCount = static_cast<std::uint32_t>(properties.multiProcessorCount);
        info.clockRateKHz = static_cast<std::uint32_t>(properties.clockRate);
        info.warpSize = static_cast<std::uint32_t>(properties.warpSize);

        ROCMKERNELS_HIP_CHECK(hipRuntimeGetVersion(&info.runtimeVersion));
        ROCMKERNELS_HIP_CHECK(hipDriverGetVersion(&info.driverVersion));

        return info;
    }

    void SynchronizeDevice(const int deviceIndex) {
        ROCMKERNELS_HIP_CHECK(hipSetDevice(deviceIndex));
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());
    }

    void ResetDevice(const int deviceIndex) {
        ROCMKERNELS_HIP_CHECK(hipSetDevice(deviceIndex));
        ROCMKERNELS_HIP_CHECK(hipDeviceReset());
    }
} // namespace ROCmKernels
