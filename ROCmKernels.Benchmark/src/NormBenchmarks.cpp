//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Norm/AddNorm.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {
    constexpr int kWidth = 384;

    struct State {
        explicit State(const int rows) : rows(rows) {}

        void Initialize() {
            this->x.Resize(static_cast<std::size_t>(this->rows) * kWidth);
            this->residual.Resize(static_cast<std::size_t>(this->rows) * kWidth);
            this->upstream.Resize(static_cast<std::size_t>(this->rows) * kWidth);
            this->residualUpstream.Resize(static_cast<std::size_t>(this->rows) * kWidth);
            this->weight.Resize(kWidth);
            this->bias.Resize(kWidth);
            this->summed.Resize(static_cast<std::size_t>(this->rows) * kWidth);
            this->normalized.Resize(static_cast<std::size_t>(this->rows) * kWidth);
            this->statistics.Resize(static_cast<std::size_t>(this->rows) * 2);
            this->gradient.Resize(static_cast<std::size_t>(this->rows) * kWidth);
            this->weightGradient.Resize(kWidth);
            this->biasGradient.Resize(kWidth);

            const std::vector<float> hostValues(static_cast<std::size_t>(this->rows) * kWidth, 0.5f);
            const std::vector<float> hostWeight(kWidth, 1.0f);
            const std::vector<float> hostBias(kWidth, 0.0f);
            this->x.CopyFromHost(hostValues.data(), hostValues.size());
            this->residual.CopyFromHost(hostValues.data(), hostValues.size());
            this->upstream.CopyFromHost(hostValues.data(), hostValues.size());
            this->residualUpstream.CopyFromHost(hostValues.data(), hostValues.size());
            this->weight.CopyFromHost(hostWeight.data(), hostWeight.size());
            this->bias.CopyFromHost(hostBias.data(), hostBias.size());

            ROCmKernels::Kernels::AddNormForward(
                this->x.Data(), this->residual.Data(), this->weight.Data(), this->bias.Data(), this->summed.Data(), this->normalized.Data(),
                this->statistics.Data(), this->rows
            );
        }

        int rows;
        ROCmKernels::DeviceBuffer<float> x;
        ROCmKernels::DeviceBuffer<float> residual;
        ROCmKernels::DeviceBuffer<float> upstream;
        ROCmKernels::DeviceBuffer<float> residualUpstream;
        ROCmKernels::DeviceBuffer<float> weight;
        ROCmKernels::DeviceBuffer<float> bias;
        ROCmKernels::DeviceBuffer<float> summed;
        ROCmKernels::DeviceBuffer<float> normalized;
        ROCmKernels::DeviceBuffer<float> statistics;
        ROCmKernels::DeviceBuffer<float> gradient;
        ROCmKernels::DeviceBuffer<float> weightGradient;
        ROCmKernels::DeviceBuffer<float> biasGradient;
    };

    ROCmKernels::Benchmark::Case MakeForwardCase(const int rows) {
        const auto state = std::make_shared<State>(rows);

        return ROCmKernels::Benchmark::Case{
            .name = "AddNorm forward " + std::to_string(rows) + "x384",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::AddNormForward(
                        state->x.Data(), state->residual.Data(), state->weight.Data(), state->bias.Data(), state->summed.Data(), state->normalized.Data(),
                        state->statistics.Data(), state->rows, 1e-5f, stream
                    );
                },
            .bytesMoved = static_cast<std::uint64_t>(rows) * kWidth * sizeof(float) * 5,
            .operations = 0,
        };
    }

    ROCmKernels::Benchmark::Case MakeBackwardCase(const int rows) {
        const auto state = std::make_shared<State>(rows);

        return ROCmKernels::Benchmark::Case{
            .name = "AddNorm backward " + std::to_string(rows) + "x384",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::AddNormBackward(
                        state->summed.Data(), state->upstream.Data(), state->residualUpstream.Data(), state->weight.Data(), state->statistics.Data(),
                        state->gradient.Data(), state->weightGradient.Data(), state->biasGradient.Data(), state->rows, stream
                    );
                },
            .bytesMoved = static_cast<std::uint64_t>(rows) * kWidth * sizeof(float) * 4,
            .operations = 0,
        };
    }

    struct Registrar {
        Registrar() {
            ROCmKernels::Benchmark::Register(MakeForwardCase(8192));
            ROCmKernels::Benchmark::Register(MakeBackwardCase(8192));
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
