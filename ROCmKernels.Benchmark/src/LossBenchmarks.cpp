//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Loss/CrossEntropy.hpp>

#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {
    struct State {
        State(const int rows, const int classes) : rows(rows), classes(classes) {}

        void Initialize() {
            this->logits.Resize(static_cast<std::size_t>(this->rows) * this->classes);
            this->targets.Resize(this->rows);
            this->losses.Resize(this->rows);
            this->statistics.Resize(static_cast<std::size_t>(this->rows) * 2);
            this->upstream.Resize(this->rows);
            this->gradient.Resize(static_cast<std::size_t>(this->rows) * this->classes);

            const std::vector<float> hostLogits(static_cast<std::size_t>(this->rows) * this->classes, 0.5f);
            const std::vector<std::int64_t> hostTargets(this->rows, 0);
            const std::vector<float> hostUpstream(this->rows, 0.25f);
            this->logits.CopyFromHost(hostLogits.data(), hostLogits.size());
            this->targets.CopyFromHost(hostTargets.data(), hostTargets.size());
            this->upstream.CopyFromHost(hostUpstream.data(), hostUpstream.size());

            ROCmKernels::Kernels::CrossEntropyForward(
                this->logits.Data(), this->targets.Data(), this->losses.Data(), this->statistics.Data(), this->rows, this->classes, this->classes
            );
        }

        int rows;
        int classes;
        ROCmKernels::DeviceBuffer<float> logits;
        ROCmKernels::DeviceBuffer<std::int64_t> targets;
        ROCmKernels::DeviceBuffer<float> losses;
        ROCmKernels::DeviceBuffer<float> statistics;
        ROCmKernels::DeviceBuffer<float> upstream;
        ROCmKernels::DeviceBuffer<float> gradient;
    };

    struct HalfState {
        HalfState(const int rows, const int classes) : rows(rows), classes(classes) {}

        void Initialize() {
            this->logits.Resize(static_cast<std::size_t>(this->rows) * this->classes);
            this->targets.Resize(this->rows);
            this->losses.Resize(this->rows);
            this->statistics.Resize(static_cast<std::size_t>(this->rows) * 2);
            this->upstream.Resize(this->rows);
            this->gradient.Resize(static_cast<std::size_t>(this->rows) * this->classes);

            const std::vector<float> hostLogits(static_cast<std::size_t>(this->rows) * this->classes, 0.5f);
            std::vector<__half> hostHalfLogits(hostLogits.size());
            for (std::size_t index = 0; index < hostLogits.size(); ++index) hostHalfLogits[index] = __float2half_rn(hostLogits[index]);
            const std::vector<std::int64_t> hostTargets(this->rows, 0);
            const std::vector<float> hostUpstream(this->rows, 0.25f);
            this->logits.CopyFromHost(hostHalfLogits.data(), hostHalfLogits.size());
            this->targets.CopyFromHost(hostTargets.data(), hostTargets.size());
            this->upstream.CopyFromHost(hostUpstream.data(), hostUpstream.size());

            ROCmKernels::Kernels::CrossEntropyForwardFp16(
                this->logits.Data(), this->targets.Data(), this->losses.Data(), this->statistics.Data(), this->rows, this->classes, this->classes
            );
        }

        int rows;
        int classes;
        ROCmKernels::DeviceBuffer<__half> logits;
        ROCmKernels::DeviceBuffer<std::int64_t> targets;
        ROCmKernels::DeviceBuffer<float> losses;
        ROCmKernels::DeviceBuffer<float> statistics;
        ROCmKernels::DeviceBuffer<float> upstream;
        ROCmKernels::DeviceBuffer<__half> gradient;
    };

    ROCmKernels::Benchmark::Case MakeForwardCase(const int rows, const int classes) {
        const auto state = std::make_shared<State>(rows, classes);

        return ROCmKernels::Benchmark::Case{
            .name = "CrossEntropy forward " + std::to_string(rows) + "x" + std::to_string(classes),
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::CrossEntropyForward(
                        state->logits.Data(), state->targets.Data(), state->losses.Data(), state->statistics.Data(), state->rows, state->classes,
                        state->classes, stream
                    );
                },
            .bytesMoved = static_cast<std::uint64_t>(rows) * classes * sizeof(float),
            .operations = static_cast<std::uint64_t>(rows) * classes,
        };
    }

    ROCmKernels::Benchmark::Case MakeBackwardCase(const int rows, const int classes) {
        const auto state = std::make_shared<State>(rows, classes);

        return ROCmKernels::Benchmark::Case{
            .name = "CrossEntropy backward " + std::to_string(rows) + "x" + std::to_string(classes),
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::CrossEntropyBackward(
                        state->logits.Data(), state->targets.Data(), state->statistics.Data(), state->upstream.Data(), state->gradient.Data(), state->rows,
                        state->classes, state->classes, 1, stream
                    );
                },
            .bytesMoved = static_cast<std::uint64_t>(rows) * classes * sizeof(float) * 2,
            .operations = static_cast<std::uint64_t>(rows) * classes,
        };
    }

    ROCmKernels::Benchmark::Case MakeForwardCaseFp16(const int rows, const int classes) {
        const auto state = std::make_shared<HalfState>(rows, classes);

        return ROCmKernels::Benchmark::Case{
            .name = "CrossEntropy forward fp16 " + std::to_string(rows) + "x" + std::to_string(classes),
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::CrossEntropyForwardFp16(
                        state->logits.Data(), state->targets.Data(), state->losses.Data(), state->statistics.Data(), state->rows, state->classes,
                        state->classes, stream
                    );
                },
            .bytesMoved = static_cast<std::uint64_t>(rows) * classes * sizeof(__half),
            .operations = static_cast<std::uint64_t>(rows) * classes,
        };
    }

    ROCmKernels::Benchmark::Case MakeBackwardCaseFp16(const int rows, const int classes) {
        const auto state = std::make_shared<HalfState>(rows, classes);

        return ROCmKernels::Benchmark::Case{
            .name = "CrossEntropy backward fp16 " + std::to_string(rows) + "x" + std::to_string(classes),
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::CrossEntropyBackwardFp16(
                        state->logits.Data(), state->targets.Data(), state->statistics.Data(), state->upstream.Data(), state->gradient.Data(), state->rows,
                        state->classes, state->classes, 1, stream
                    );
                },
            .bytesMoved = static_cast<std::uint64_t>(rows) * classes * sizeof(__half) * 2,
            .operations = static_cast<std::uint64_t>(rows) * classes,
        };
    }

    struct Registrar {
        Registrar() {
            ROCmKernels::Benchmark::Register(MakeForwardCase(512, 32768));
            ROCmKernels::Benchmark::Register(MakeBackwardCase(512, 32768));
            ROCmKernels::Benchmark::Register(MakeForwardCase(1200, 55765));
            ROCmKernels::Benchmark::Register(MakeBackwardCase(1200, 55765));
            ROCmKernels::Benchmark::Register(MakeForwardCaseFp16(1200, 55765));
            ROCmKernels::Benchmark::Register(MakeBackwardCaseFp16(1200, 55765));
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
