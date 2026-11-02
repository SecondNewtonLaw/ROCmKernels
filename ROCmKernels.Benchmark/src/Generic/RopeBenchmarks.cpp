#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/Rope.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {
    constexpr int kHeadDim = 128;
    constexpr int kRotaryDim = 128;
    constexpr float kTheta = 10000.0f;

    template<typename T> T Narrow(const float value) {
        if constexpr (std::is_same_v<T, __half>) return __float2half_rn(value);
        else return value;
    }

    template<typename T> struct State {
        State(const int batchSize, const int heads, const int length)
            : batchSize(batchSize), heads(heads), length(length), elements(batchSize * heads * length * kHeadDim) {}

        void Initialize() {
            this->input.Resize(this->elements);
            this->output.Resize(this->elements);
            this->positions.Resize(static_cast<std::size_t>(this->batchSize) * this->length);

            const std::vector<T> hostValues(this->elements, Narrow<T>(0.5f));
            std::vector<int> hostPositions(static_cast<std::size_t>(this->batchSize) * this->length);
            for (int batch = 0; batch < this->batchSize; ++batch) {
                for (int token = 0; token < this->length; ++token) hostPositions[batch * this->length + token] = token;
            }

            this->input.CopyFromHost(hostValues.data(), hostValues.size());
            this->positions.CopyFromHost(hostPositions.data(), hostPositions.size());
        }

        int batchSize;
        int heads;
        int length;
        int elements;
        ROCmKernels::DeviceBuffer<T> input;
        ROCmKernels::DeviceBuffer<T> output;
        ROCmKernels::DeviceBuffer<int> positions;
    };

    template<typename T>
    ROCmKernels::Benchmark::Case MakeCase(const std::string &name, const int batchSize, const int heads, const int length,
            const ROCmKernels::Kernels::RopeStyle style) {
        const auto state = std::make_shared<State<T>>(batchSize, heads, length);
        const ROCmKernels::Kernels::Strides strides{length * heads * kHeadDim, kHeadDim, heads * kHeadDim};
        constexpr auto dtype = ROCmKernels::Kernels::DTypeOf<T>::value;

        return ROCmKernels::Benchmark::Case{
            .name = name,
            .setup = [state] { state->Initialize(); },
            .run =
                [state, strides, style](hipStream_t stream) {
                    ROCmKernels::Kernels::RopeForward(state->input.Data(), state->output.Data(), state->positions.Data(), state->batchSize,
                        state->heads, state->length, kHeadDim, kRotaryDim, kTheta, style, strides, strides, dtype, stream);
                },
            .bytesMoved = static_cast<std::uint64_t>(state->elements) * sizeof(T) * 2,
            .bytesTouched = static_cast<std::uint64_t>(state->elements) * sizeof(T) * 2,
            .operations = static_cast<std::uint64_t>(state->batchSize) * state->heads * state->length * kRotaryDim * 6,
        };
    }

    struct Registrar {
        Registrar() {
            using ROCmKernels::Kernels::RopeStyle;

            ROCmKernels::Benchmark::Register(MakeCase<float>("Rope normal fp32 b2 h12 t1024 d128", 2, 12, 1024, RopeStyle::Normal));
            ROCmKernels::Benchmark::Register(MakeCase<float>("Rope neox fp32 b2 h12 t1024 d128", 2, 12, 1024, RopeStyle::NeoX));
            ROCmKernels::Benchmark::Register(MakeCase<__half>("Rope normal fp16 b2 h12 t1024 d128", 2, 12, 1024, RopeStyle::Normal));
            ROCmKernels::Benchmark::Register(MakeCase<__half>("Rope neox fp16 b2 h12 t1024 d128", 2, 12, 1024, RopeStyle::NeoX));
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
