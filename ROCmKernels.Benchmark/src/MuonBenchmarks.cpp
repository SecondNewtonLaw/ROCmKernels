//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Optimizer/Muon.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {
    struct State {
        State(const int m, const int n, const int count) : m(m), n(n), count(count) {}

        void Initialize(const bool prepared) {
            const std::size_t matrixElements = static_cast<std::size_t>(this->m) * this->n;
            const std::size_t innerElements = static_cast<std::size_t>(this->n) * this->n;

            this->gradient.Resize(matrixElements * this->count);
            this->buffer.Resize(matrixElements * this->count);
            this->y.Resize(matrixElements * this->count);
            this->parameter.Resize(matrixElements * this->count);
            this->sumsq.Resize(this->count);
            this->flags.Resize(this->count);
            this->gradientPointers.Resize(this->count);
            this->bufferPointers.Resize(this->count);
            this->parameterPointers.Resize(this->count);
            this->b.Resize(innerElements * this->count);
            this->gin.Resize(innerElements * this->count);
            this->gram.Resize(innerElements * this->count);
            this->out.Resize(matrixElements * this->count);
            this->a2.Resize(innerElements * this->count);
            this->identity.Resize(innerElements * this->count);
            std::vector<float> hostIdentity(innerElements * this->count, 0.0f);
            for (int matrix = 0; matrix < this->count; ++matrix)
                for (int index = 0; index < this->n; ++index)
                    hostIdentity[static_cast<std::size_t>(matrix) * innerElements + static_cast<std::size_t>(index) * this->n + index] = 1.0f;
            this->identity.CopyFromHost(hostIdentity.data(), hostIdentity.size());

            const std::vector<float> hostGradient(matrixElements * this->count, 0.5f);
            const std::vector<float> hostBuffer(matrixElements * this->count, 0.25f);
            const std::vector<float> hostParameter(matrixElements * this->count, 0.5f);
            const std::vector<float> hostInner(innerElements * this->count, 0.5f);

            this->gradient.CopyFromHost(hostGradient.data(), hostGradient.size());
            this->buffer.CopyFromHost(hostBuffer.data(), hostBuffer.size());
            this->parameter.CopyFromHost(hostParameter.data(), hostParameter.size());
            this->b.CopyFromHost(hostInner.data(), hostInner.size());
            this->gin.CopyFromHost(hostInner.data(), hostInner.size());

            std::vector<std::int64_t> hostGradientPointers(this->count);
            std::vector<std::int64_t> hostBufferPointers(this->count);
            std::vector<std::int64_t> hostParameterPointers(this->count);
            for (int matrix = 0; matrix < this->count; ++matrix) {
                hostGradientPointers[matrix] = reinterpret_cast<std::int64_t>(this->gradient.Data() + matrix * matrixElements);
                hostBufferPointers[matrix] = reinterpret_cast<std::int64_t>(this->buffer.Data() + matrix * matrixElements);
                hostParameterPointers[matrix] = reinterpret_cast<std::int64_t>(this->parameter.Data() + matrix * matrixElements);
            }
            this->gradientPointers.CopyFromHost(hostGradientPointers.data(), this->count);
            this->bufferPointers.CopyFromHost(hostBufferPointers.data(), this->count);
            this->parameterPointers.CopyFromHost(hostParameterPointers.data(), this->count);

            if (prepared) {
                ROCmKernels::Kernels::MuonPrepare(
                    this->gradientPointers.Data(), this->bufferPointers.Data(), this->y.Data(), this->flags.Data(), this->sumsq.Data(), this->count, this->m,
                    this->n, 0.9f
                );
            }
        }

        int m;
        int n;
        int count;
        ROCmKernels::DeviceBuffer<float> gradient;
        ROCmKernels::DeviceBuffer<float> buffer;
        ROCmKernels::DeviceBuffer<float> y;
        ROCmKernels::DeviceBuffer<float> parameter;
        ROCmKernels::DeviceBuffer<float> sumsq;
        ROCmKernels::DeviceBuffer<int> flags;
        ROCmKernels::DeviceBuffer<std::int64_t> gradientPointers;
        ROCmKernels::DeviceBuffer<std::int64_t> bufferPointers;
        ROCmKernels::DeviceBuffer<std::int64_t> parameterPointers;
        ROCmKernels::DeviceBuffer<float> b;
        ROCmKernels::DeviceBuffer<float> gin;
        ROCmKernels::DeviceBuffer<float> gram;
        ROCmKernels::DeviceBuffer<float> out;
        ROCmKernels::DeviceBuffer<float> a2;
        ROCmKernels::DeviceBuffer<float> identity;
    };

    ROCmKernels::Benchmark::Case MakePrepareCase(const int m, const int n, const int count) {
        const auto state = std::make_shared<State>(m, n, count);

        return ROCmKernels::Benchmark::Case{
            .name = "Muon prepare count=" + std::to_string(count) + " " + std::to_string(m) + "x" + std::to_string(n),
            .setup = [state] { state->Initialize(false); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::MuonPrepare(
                        state->gradientPointers.Data(), state->bufferPointers.Data(), state->y.Data(), state->flags.Data(), state->sumsq.Data(), state->count,
                        state->m, state->n, 0.9f, stream
                    );
                },
            .bytesMoved = static_cast<std::uint64_t>(count) * m * n * sizeof(float) * 3,
            .operations = 0,
        };
    }

    ROCmKernels::Benchmark::Case MakeGramCase(const int m, const int n, const int count) {
        const auto state = std::make_shared<State>(m, n, count);

        return ROCmKernels::Benchmark::Case{
            .name = "Muon gram count=" + std::to_string(count) + " " + std::to_string(m) + "x" + std::to_string(n),
            .setup = [state] { state->Initialize(true); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::MuonGram(
                        state->y.Data(), state->gin.Data(), state->gram.Data(), state->sumsq.Data(), state->count, state->m, state->n, 1.0f, 0.5f, 0.75f, false,
                        stream
                    );
                },
            .bytesMoved = static_cast<std::uint64_t>(count) * (m * n + static_cast<std::uint64_t>(n) * n * 2) * sizeof(float),
            .operations = static_cast<std::uint64_t>(count) * m * n * n * 2,
        };
    }

    ROCmKernels::Benchmark::Case MakeMultiplyCase(const int m, const int n, const int count) {
        const auto state = std::make_shared<State>(m, n, count);

        return ROCmKernels::Benchmark::Case{
            .name = "Muon multiply count=" + std::to_string(count) + " " + std::to_string(m) + "x" + std::to_string(n),
            .setup = [state] { state->Initialize(true); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::MuonMultiply(
                        state->y.Data(), state->b.Data(), state->out.Data(), state->sumsq.Data(), state->count, state->m, state->n, true, stream
                    );
                },
            .bytesMoved = static_cast<std::uint64_t>(count) * (static_cast<std::uint64_t>(m) * n * 2 + static_cast<std::uint64_t>(n) * n) * sizeof(float),
            .operations = static_cast<std::uint64_t>(count) * m * n * n * 2,
        };
    }

    ROCmKernels::Benchmark::Case MakeApplyCase(const int m, const int n, const int count) {
        const auto state = std::make_shared<State>(m, n, count);

        return ROCmKernels::Benchmark::Case{
            .name = "Muon apply count=" + std::to_string(count) + " " + std::to_string(m) + "x" + std::to_string(n),
            .setup = [state] { state->Initialize(true); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::MuonApply(
                        state->y.Data(), state->b.Data(), state->parameterPointers.Data(), state->flags.Data(), state->sumsq.Data(), state->count, state->m,
                        state->n, true, false, 0.1f, 0.01f, stream
                    );
                },
            .bytesMoved = static_cast<std::uint64_t>(count) * (static_cast<std::uint64_t>(m) * n * 2 + static_cast<std::uint64_t>(n) * n) * sizeof(float),
            .operations = static_cast<std::uint64_t>(count) * m * n * n * 2,
        };
    }

    // Full Newton-Schulz step as the trainer runs it: prepare, five quintic
    // iterations (gram, a@a, B = c3 I + c4 a + c5 a^2, y = y B) and the update.
    ROCmKernels::Benchmark::Case MakeStepCase(const int m, const int n, const int count) {
        const auto state = std::make_shared<State>(m, n, count);

        return ROCmKernels::Benchmark::Case{
            .name = "Muon NS-5 step count=" + std::to_string(count) + " " + std::to_string(m) + "x" + std::to_string(n),
            .setup = [state] { state->Initialize(false); },
            .run =
                [state](hipStream_t stream) {
                    const auto &s = *state;
                    ROCmKernels::Kernels::MuonPrepare(s.gradientPointers.Data(), s.bufferPointers.Data(), s.y.Data(), s.flags.Data(), s.sumsq.Data(),
                            s.count, s.m, s.n, 0.95f, stream);
                    for (int iteration = 0; iteration < 5; ++iteration) {
                        const bool scaled = iteration == 0;
                        ROCmKernels::Kernels::MuonGram(s.y.Data(), nullptr, s.gram.Data(), s.sumsq.Data(), s.count, s.m, s.n, 1.0f, 0.0f, 0.0f,
                                scaled, stream);
                        ROCmKernels::Kernels::MuonMultiply(s.gram.Data(), s.gram.Data(), s.a2.Data(), s.sumsq.Data(), s.count, s.n, s.n, false, stream);
                        ROCmKernels::Kernels::MuonMakeB(s.gram.Data(), s.a2.Data(), s.b.Data(), s.count, s.n, 3.4445f, -4.7750f, 2.0315f, stream);
                        ROCmKernels::Kernels::MuonMultiply(s.y.Data(), s.b.Data(), s.y.Data(), s.sumsq.Data(), s.count, s.m, s.n, scaled, stream);
                    }
                    ROCmKernels::Kernels::MuonApply(s.y.Data(), s.identity.Data(), s.parameterPointers.Data(), s.flags.Data(), s.sumsq.Data(),
                            s.count, s.m, s.n, false, s.m < s.n, 1e-2f, 1e-2f * 0.2f * std::sqrt(static_cast<float>(s.m > s.n ? s.m : s.n)), stream);
                },
            .bytesMoved = 0,
            .operations = static_cast<std::uint64_t>(count) * (5ull * 2ull * static_cast<std::uint64_t>(m) * n * n * 2 + static_cast<std::uint64_t>(m) * n * 3),
        };
    }

    struct Registrar {
        Registrar() {
            ROCmKernels::Benchmark::Register(MakePrepareCase(256, 256, 4));
            ROCmKernels::Benchmark::Register(MakeGramCase(256, 256, 4));
            ROCmKernels::Benchmark::Register(MakeMultiplyCase(256, 256, 4));
            ROCmKernels::Benchmark::Register(MakeApplyCase(256, 256, 4));
            ROCmKernels::Benchmark::Register(MakeStepCase(8192, 128, 12));
            ROCmKernels::Benchmark::Register(MakeStepCase(384, 128, 12));
            ROCmKernels::Benchmark::Register(MakeStepCase(128, 128, 17));
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace



