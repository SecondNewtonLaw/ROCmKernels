//
// Created by Dottik on 2/11/2026.
//

#pragma once

#include <cstdint>
#include <string>

namespace ROCmKernels {
    /// Snapshot of the properties HIP reports for one device.
    struct DeviceInfo {
        std::string name;
        std::string gcnArchName;
        std::uint64_t totalGlobalMemory = 0;
        std::uint32_t computeUnitCount = 0;
        std::uint32_t clockRateKHz = 0;
        std::uint32_t warpSize = 0;
        int runtimeVersion = 0;
        int driverVersion = 0;
    };

    /// Number of devices the HIP runtime can see. Throws when the runtime is unusable.
    [[nodiscard]] int GetDeviceCount();

    /// Properties of the device at deviceIndex. Throws when it cannot be queried.
    [[nodiscard]] DeviceInfo QueryDevice(int deviceIndex = 0);

    /// Blocks until every queued operation on the device has completed.
    void SynchronizeDevice(int deviceIndex = 0);

    /// Releases the device's allocations and resets its state.
    void ResetDevice(int deviceIndex = 0);
} // namespace ROCmKernels
