//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Benchmark/Harness.hpp>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
    void PrintUsage() {
        std::cout << "Usage: ROCmKernels.Benchmark [options]\n\n"
                     "  --list                List registered benchmarks\n"
                     "  --filter <text>       Run benchmarks whose name contains text\n"
                     "  --device <index>      HIP device index (default 0)\n"
                     "  --samples <count>     Samples per benchmark (default 20)\n"
                     "  --warmup <count>      Warmup launches before measuring (default 10)\n"
                     "  --min-sample-ms <ms>  Minimum measured time per sample (default 10)\n"
                     "  --no-flush            Keep caches warm between samples\n"
                     "  --csv <path>          Append results to a CSV file\n"
                     "  -h, --help            Show this help\n";
    }
} // namespace

int main(const int argc, char **argv) {
    ROCmKernels::Benchmark::Options options;
    bool listOnly = false;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument = argv[index];

            if (argument == "--list") {
                listOnly = true;
            } else if (argument == "--no-flush") {
                options.flushCaches = false;
            } else if (argument == "--help" || argument == "-h") {
                PrintUsage();
                return 0;
            } else if (
                argument == "--filter" || argument == "--csv" || argument == "--device" || argument == "--samples" || argument == "--warmup" ||
                argument == "--min-sample-ms"
            ) {
                if (index + 1 >= argc) {
                    throw std::invalid_argument("missing value for " + std::string(argument));
                }

                const std::string value = argv[++index];
                if (argument == "--filter") {
                    options.filter = value;
                } else if (argument == "--csv") {
                    options.csvPath = value;
                } else if (argument == "--device") {
                    options.deviceIndex = std::stoi(value);
                } else if (argument == "--samples") {
                    options.samples = std::stoul(value);
                } else if (argument == "--warmup") {
                    options.warmupIterations = std::stoul(value);
                } else {
                    options.minimumSampleSeconds = std::stod(value) / 1000.0;
                }
            } else {
                throw std::invalid_argument("unknown argument: " + std::string(argument));
            }
        }

        if (listOnly) {
            for (const ROCmKernels::Benchmark::Case &benchmarkCase : ROCmKernels::Benchmark::Cases()) {
                std::cout << benchmarkCase.name << '\n';
            }
            return 0;
        }

        ROCmKernels::Benchmark::Run(options);
    } catch (const std::exception &exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }

    return 0;
}
