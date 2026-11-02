//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/Device.hpp>
#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Kernels/Elementwise/Copy.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
    constexpr std::size_t kFlushBufferBytes = 256ull * 1024 * 1024;
    constexpr std::size_t kCacheCopyBytes = 16ull * 1024 * 1024;
    constexpr std::size_t kCacheEdgeCopyBytes = 48ull * 1024 * 1024;
    constexpr std::size_t kHbmCopyBytes = 256ull * 1024 * 1024;

    struct Statistics {
        double minimum = 0;
        double median = 0;
        double mean = 0;
        double standardDeviation = 0;
        double p95 = 0;
    };

    struct Measurement {
        Statistics statistics;
        std::size_t iterations = 0;
        double gigabytesPerSecond = 0;
        double gigaflops = 0;
    };

    Statistics ComputeStatistics(std::vector<double> samples) {
        std::sort(samples.begin(), samples.end());

        Statistics statistics{};
        statistics.minimum = samples.front();

        const std::size_t middle = samples.size() / 2;
        if (samples.size() % 2 == 0) {
            statistics.median = 0.5 * (samples[middle - 1] + samples[middle]);
        } else {
            statistics.median = samples[middle];
        }

        statistics.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());

        double sumOfSquares = 0;
        for (const double sample : samples) {
            const double difference = sample - statistics.mean;
            sumOfSquares += difference * difference;
        }
        statistics.standardDeviation = std::sqrt(sumOfSquares / static_cast<double>(samples.size()));

        const auto percentileIndex = static_cast<std::size_t>(0.95 * static_cast<double>(samples.size() - 1));
        statistics.p95 = samples[percentileIndex];

        return statistics;
    }

    class EventPair {
        hipEvent_t m_start{};
        hipEvent_t m_stop{};

      public:
        EventPair() {
            ROCMKERNELS_HIP_CHECK(hipEventCreate(&this->m_start));
            ROCMKERNELS_HIP_CHECK(hipEventCreate(&this->m_stop));
        }

        ~EventPair() {
            (void)hipEventDestroy(this->m_start);
            (void)hipEventDestroy(this->m_stop);
        }

        EventPair(const EventPair &) = delete;
        EventPair &operator=(const EventPair &) = delete;

        void Start(hipStream_t stream) { ROCMKERNELS_HIP_CHECK(hipEventRecord(this->m_start, stream)); }

        void Stop(hipStream_t stream) {
            ROCMKERNELS_HIP_CHECK(hipEventRecord(this->m_stop, stream));
            ROCMKERNELS_HIP_CHECK(hipEventSynchronize(this->m_stop));
        }

        [[nodiscard]] double ElapsedMilliseconds() const {
            float milliseconds = 0;
            ROCMKERNELS_HIP_CHECK(hipEventElapsedTime(&milliseconds, this->m_start, this->m_stop));
            return milliseconds;
        }
    };

    Measurement MeasureCase(
        const ROCmKernels::Benchmark::Case &benchmarkCase, const ROCmKernels::Benchmark::Options &options, ROCmKernels::DeviceBuffer<std::uint8_t> &flushBuffer
    ) {
        if (benchmarkCase.setup) {
            benchmarkCase.setup();
        }

        const hipStream_t stream = nullptr;
        for (std::size_t iteration = 0; iteration < options.warmupIterations; ++iteration) {
            benchmarkCase.run(stream);
        }
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        EventPair events;

        events.Start(stream);
        benchmarkCase.run(stream);
        events.Stop(stream);

        const double singleRunMilliseconds = std::max(events.ElapsedMilliseconds(), 1e-6);
        const double minimumSampleMilliseconds = options.minimumSampleSeconds * 1000.0;
        const double requestedIterations = std::ceil(minimumSampleMilliseconds / singleRunMilliseconds);
        const std::size_t iterations = static_cast<std::size_t>(std::clamp(requestedIterations, 1.0, 1'000'000.0));

        // Give the clocks time to settle before recording samples.
        constexpr double kWarmupSeconds = 0.1;
        const double requestedWarmupIterations = std::ceil(kWarmupSeconds * 1000.0 / singleRunMilliseconds);
        const std::size_t rampUpIterations = static_cast<std::size_t>(std::clamp(requestedWarmupIterations, 0.0, 100'000.0));
        for (std::size_t iteration = 0; iteration < rampUpIterations; ++iteration) {
            benchmarkCase.run(stream);
        }
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<double> samples;
        samples.reserve(options.samples);

        for (std::size_t sample = 0; sample < options.samples; ++sample) {
            if (flushBuffer.Count() > 0) {
                ROCMKERNELS_HIP_CHECK(hipMemsetAsync(flushBuffer.Data(), 0, flushBuffer.ByteSize(), stream));
            }

            events.Start(stream);
            for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
                benchmarkCase.run(stream);
            }
            events.Stop(stream);

            samples.push_back(events.ElapsedMilliseconds() / static_cast<double>(iterations));
        }

        Measurement measurement{};
        measurement.iterations = iterations;
        measurement.statistics = ComputeStatistics(samples);

        const double seconds = measurement.statistics.median / 1000.0;
        if (benchmarkCase.bytesMoved > 0) {
            measurement.gigabytesPerSecond = static_cast<double>(benchmarkCase.bytesMoved) / seconds / 1e9;
        }
        if (benchmarkCase.operations > 0) {
            measurement.gigaflops = static_cast<double>(benchmarkCase.operations) / seconds / 1e9;
        }

        return measurement;
    }

    /// Streaming copy throughput, used as the memory ceiling for the run.
    double MeasureCopyBandwidth(const std::size_t byteCount) {
        const std::size_t elementCount = byteCount / sizeof(float);
        ROCmKernels::DeviceBuffer<float> source(elementCount);
        ROCmKernels::DeviceBuffer<float> destination(elementCount);

        const hipStream_t stream = nullptr;
        for (int warmup = 0; warmup < 3; ++warmup) {
            ROCmKernels::Kernels::Copy(source.Data(), destination.Data(), elementCount, stream);
        }
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        EventPair events;
        std::vector<double> samples;
        samples.reserve(10);

        for (int sample = 0; sample < 10; ++sample) {
            events.Start(stream);
            ROCmKernels::Kernels::Copy(source.Data(), destination.Data(), elementCount, stream);
            events.Stop(stream);
            samples.push_back(events.ElapsedMilliseconds());
        }

        std::sort(samples.begin(), samples.end());
        const double seconds = samples[samples.size() / 2] / 1000.0;
        return 2.0 * static_cast<double>(byteCount) / seconds / 1e9;
    }

    void PrintHeader(const ROCmKernels::DeviceInfo &device, const std::size_t nameWidth, const double cacheBandwidth, const double edgeBandwidth,
            const double hbmBandwidth) {
        const double memoryGiB = static_cast<double>(device.totalGlobalMemory) / (1024.0 * 1024.0 * 1024.0);

        std::cout << device.name << " (" << device.gcnArchName << "), " << device.computeUnitCount << " compute units, " << std::fixed << std::setprecision(1)
                  << memoryGiB << " GiB\n";
        std::cout << "Copy bandwidth: 32 MiB set " << cacheBandwidth << " GB/s | 96 MiB set " << edgeBandwidth << " GB/s | 512 MiB set " << hbmBandwidth
                  << " GB/s\n\n";

        std::cout << std::left << std::setw(static_cast<int>(nameWidth)) << "Benchmark" << std::right << std::setw(12) << "Median ms" << std::setw(12)
                  << "Min ms" << std::setw(12) << "P95 ms" << std::setw(12) << "Unique MiB" << std::setw(14) << "Traffic MiB" << std::setw(14) << "GB/s"
                  << std::setw(14) << "GFLOP/s" << "\n";
        std::cout << std::string(nameWidth + 90, '-') << "\n";
    }

    void PrintMeasurement(const ROCmKernels::Benchmark::Case &benchmarkCase, const Measurement &measurement, const std::size_t nameWidth) {
        const std::uint64_t touchedBytes = benchmarkCase.bytesTouched != 0 ? benchmarkCase.bytesTouched : benchmarkCase.bytesMoved;
        const double uniqueMiB = static_cast<double>(touchedBytes) / (1024.0 * 1024.0);
        const double trafficMiB = static_cast<double>(benchmarkCase.bytesMoved) / (1024.0 * 1024.0);

        std::cout << std::left << std::setw(static_cast<int>(nameWidth)) << benchmarkCase.name << std::right << std::fixed << std::setprecision(4)
                  << std::setw(12) << measurement.statistics.median << std::setw(12) << measurement.statistics.minimum << std::setw(12)
                  << measurement.statistics.p95 << std::setprecision(1) << std::setw(12) << uniqueMiB << std::setw(14) << trafficMiB
                  << std::setprecision(2) << std::setw(14);

        if (measurement.gigabytesPerSecond > 0) {
            std::cout << measurement.gigabytesPerSecond;
        } else {
            std::cout << "-";
        }

        std::cout << std::setw(14);
        if (measurement.gigaflops > 0) {
            std::cout << measurement.gigaflops;
        } else {
            std::cout << "-";
        }
        std::cout << "\n";
    }

    void WriteCsvRow(std::ofstream &csv, const ROCmKernels::Benchmark::Case &benchmarkCase, const Measurement &measurement) {
        const std::uint64_t touchedBytes = benchmarkCase.bytesTouched != 0 ? benchmarkCase.bytesTouched : benchmarkCase.bytesMoved;

        csv << benchmarkCase.name << ',' << touchedBytes << ',' << benchmarkCase.bytesMoved << ',' << benchmarkCase.operations << ','
            << measurement.iterations << ',' << measurement.statistics.median << ',' << measurement.statistics.minimum << ','
            << measurement.statistics.mean << ',' << measurement.statistics.standardDeviation << ',' << measurement.statistics.p95 << ','
            << measurement.gigabytesPerSecond << ',' << measurement.gigaflops << '\n';
    }
} // namespace

namespace {
    std::vector<ROCmKernels::Benchmark::Case> &MutableCases() {
        static std::vector<ROCmKernels::Benchmark::Case> cases;
        return cases;
    }
} // namespace

namespace ROCmKernels::Benchmark {
    void Register(Case benchmarkCase) { MutableCases().push_back(std::move(benchmarkCase)); }

    const std::vector<Case> &Cases() { return MutableCases(); }

    void Run(const Options &options) {
        if (options.samples == 0) {
            throw std::invalid_argument("samples must be at least 1");
        }

        ROCMKERNELS_HIP_CHECK(hipSetDevice(options.deviceIndex));

        const DeviceInfo device = QueryDevice(options.deviceIndex);

        std::size_t nameWidth = 30;
        for (const Case &benchmarkCase : Cases()) {
            nameWidth = std::max(nameWidth, benchmarkCase.name.size());
        }

        const double cacheBandwidth = MeasureCopyBandwidth(kCacheCopyBytes);
        const double edgeBandwidth = MeasureCopyBandwidth(kCacheEdgeCopyBytes);
        const double hbmBandwidth = MeasureCopyBandwidth(kHbmCopyBytes);

        PrintHeader(device, nameWidth, cacheBandwidth, edgeBandwidth, hbmBandwidth);

        DeviceBuffer<std::uint8_t> flushBuffer;
        if (options.flushCaches) {
            flushBuffer.Resize(kFlushBufferBytes);
        }

        std::ofstream csv;
        if (!options.csvPath.empty()) {
            csv.open(options.csvPath, std::ios::app);
            if (!csv) {
                throw std::runtime_error("could not open " + options.csvPath);
            }
            if (csv.tellp() == std::streampos(0)) {
                csv << "name,unique_bytes,bytes,operations,iterations,median_ms,min_ms,mean_ms,stddev_ms,p95_ms,gb_per_s,gflop_per_s\n";
            }
        }

        bool ranAnyCase = false;
        for (const Case &benchmarkCase : Cases()) {
            if (!options.filter.empty() && benchmarkCase.name.find(options.filter) == std::string::npos) {
                continue;
            }

            const Measurement measurement = MeasureCase(benchmarkCase, options, flushBuffer);
            PrintMeasurement(benchmarkCase, measurement, nameWidth);
            if (csv.is_open()) {
                WriteCsvRow(csv, benchmarkCase, measurement);
            }
            ranAnyCase = true;
        }

        if (!ranAnyCase) {
            std::cout << "No benchmark matched.\n";
        }
    }
} // namespace ROCmKernels::Benchmark
