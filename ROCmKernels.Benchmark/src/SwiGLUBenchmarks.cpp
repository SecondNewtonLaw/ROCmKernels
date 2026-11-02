//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Activation/SwiGLU.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {
    struct State {
        State(const int rows, const int width) : count(rows * width), width(width) {}

        void Initialize() {
            this->input.Resize(2 * this->count);
            this->output.Resize(this->count);
            this->upstream.Resize(this->count);
            this->gradient.Resize(2 * this->count);

            const std::vector<float> hostInput(2 * this->count, 0.5f);
            const std::vector<float> hostUpstream(this->count, 0.25f);
            this->input.CopyFromHost(hostInput.data(), hostInput.size());
            this->upstream.CopyFromHost(hostUpstream.data(), hostUpstream.size());
        }

        int count;
        int width;
        ROCmKernels::DeviceBuffer<float> input;
        ROCmKernels::DeviceBuffer<float> output;
        ROCmKernels::DeviceBuffer<float> upstream;
        ROCmKernels::DeviceBuffer<float> gradient;
    };

    ROCmKernels::Benchmark::Case MakeCase(const std::string &name, const int rows, const int width, const bool backward) {
        const auto state = std::make_shared<State>(rows, width);
        const std::uint64_t bytesMoved = static_cast<std::uint64_t>(state->count) * sizeof(float) * (backward ? 5 : 3);

        return ROCmKernels::Benchmark::Case{
            .name = name,
            .setup = [state] { state->Initialize(); },
            .run =
                [state, backward](hipStream_t stream) {
                    if (backward) {
                        ROCmKernels::Kernels::SwiGLUBackward(
                            state->input.Data(), state->upstream.Data(), state->gradient.Data(), state->count, state->width, stream
                        );
                    } else {
                        ROCmKernels::Kernels::SwiGLUForward(state->input.Data(), state->output.Data(), state->count, state->width, stream);
                    }
                },
            .bytesMoved = bytesMoved,
            .operations = 0,
        };
    }

    struct Registrar {
        Registrar() {
            ROCmKernels::Benchmark::Register(MakeCase("SwiGLU forward 2048x768", 2048, 768, false));
            ROCmKernels::Benchmark::Register(MakeCase("SwiGLU backward 2048x768", 2048, 768, true));
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
