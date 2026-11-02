//
// Created by Dottik on 2/11/2026.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

namespace ROCmKernels::Benchmark {
    /// One measured workload. setup() runs once, before any timing.
    struct Case {
        std::string name;
        std::function<void()> setup;
        std::function<void(hipStream_t)> run;
        /// Traffic issued per launch, including re-reads served by cache.
        std::uint64_t bytesMoved = 0;
        /// Unique footprint per launch. Zero means bytesMoved.
        std::uint64_t bytesTouched = 0;
        std::uint64_t operations = 0;
    };

    struct Options {
        std::string filter;
        std::string csvPath;
        std::size_t samples = 20;
        std::size_t warmupIterations = 10;
        double minimumSampleSeconds = 0.01;
        bool flushCaches = true;
        int deviceIndex = 0;
    };

    void Register(Case benchmarkCase);
    [[nodiscard]] const std::vector<Case> &Cases();

    void Run(const Options &options);
} // namespace ROCmKernels::Benchmark
