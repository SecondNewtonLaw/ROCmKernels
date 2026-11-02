#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/LayerNormScaleAdd.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../TestSupport.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace {
    constexpr float kEpsilon = 1e-5f;

    template<typename T> T Narrow(const float value) {
        if constexpr (std::is_same_v<T, __half>) return __float2half_rn(value);
        else return value;
    }

    template<typename T> float Widen(const T value) {
        if constexpr (std::is_same_v<T, __half>) return __half2float(value);
        else return value;
    }

    template<typename T> std::vector<T> Narrow(const std::vector<float> &values) {
        std::vector<T> narrowed(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) narrowed[index] = Narrow<T>(values[index]);
        return narrowed;
    }

    template<typename T> std::vector<float> Widen(const std::vector<T> &values) {
        std::vector<float> widened(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) widened[index] = Widen<T>(values[index]);
        return widened;
    }

    struct Reference {
        std::vector<float> output;
        std::vector<float> statistics;
        std::vector<float> gradientX;
        std::vector<float> gradientZ;
        std::vector<float> gradientScale;
        std::vector<float> gradientWeight;
        std::vector<float> gradientBias;
    };

    /// Double-precision forward and backward reference over the exact values the device reads.
    Reference ComputeReference(const std::vector<float> &x, const std::vector<float> &z, const std::vector<float> &scale,
            const std::vector<float> &weight, const std::vector<float> &bias, const std::vector<float> &upstream, const int rows,
            const int columns, const bool affine) {
        Reference reference;
        const std::size_t elements = static_cast<std::size_t>(rows) * columns;
        reference.output.resize(elements);
        reference.statistics.resize(static_cast<std::size_t>(rows) * 2);

        for (int row = 0; row < rows; ++row) {
            const std::size_t base = static_cast<std::size_t>(row) * columns;

            double mean = 0.0;
            for (int column = 0; column < columns; ++column) {
                mean += static_cast<double>(x[base + column]) + static_cast<double>(scale[column]) * z[base + column];
            }
            mean /= columns;

            double variance = 0.0;
            for (int column = 0; column < columns; ++column) {
                const double value = static_cast<double>(x[base + column]) + static_cast<double>(scale[column]) * z[base + column];
                const double centered = value - mean;
                variance += centered * centered;
            }
            variance /= columns;

            const double rstd = 1.0 / std::sqrt(variance + kEpsilon);
            reference.statistics[static_cast<std::size_t>(row) * 2] = static_cast<float>(mean);
            reference.statistics[static_cast<std::size_t>(row) * 2 + 1] = static_cast<float>(rstd);

            for (int column = 0; column < columns; ++column) {
                const double value = static_cast<double>(x[base + column]) + static_cast<double>(scale[column]) * z[base + column];
                double normalized = (value - mean) * rstd;
                if (affine) normalized = normalized * weight[column] + bias[column];
                reference.output[base + column] = static_cast<float>(normalized);
            }
        }

        reference.gradientX.resize(elements);
        reference.gradientZ.resize(elements);
        std::vector<double> gradientScale(columns, 0.0);
        std::vector<double> gradientWeight(columns, 0.0);
        std::vector<double> gradientBias(columns, 0.0);

        for (int row = 0; row < rows; ++row) {
            const std::size_t base = static_cast<std::size_t>(row) * columns;
            const double mean = reference.statistics[static_cast<std::size_t>(row) * 2];
            const double rstd = reference.statistics[static_cast<std::size_t>(row) * 2 + 1];

            double plainMean = 0.0;
            double centeredMean = 0.0;
            for (int column = 0; column < columns; ++column) {
                const double value = static_cast<double>(x[base + column]) + static_cast<double>(scale[column]) * z[base + column];
                const double normalized = (value - mean) * rstd;
                const double dy = upstream[base + column];
                const double dyWeighted = affine ? dy * weight[column] : dy;
                plainMean += dyWeighted;
                centeredMean += dyWeighted * normalized;
            }
            plainMean /= columns;
            centeredMean /= columns;

            for (int column = 0; column < columns; ++column) {
                const double value = static_cast<double>(x[base + column]) + static_cast<double>(scale[column]) * z[base + column];
                const double normalized = (value - mean) * rstd;
                const double dy = upstream[base + column];
                const double dyWeighted = affine ? dy * weight[column] : dy;
                const double ds = rstd * (dyWeighted - plainMean - normalized * centeredMean);

                reference.gradientX[base + column] = static_cast<float>(ds);
                reference.gradientZ[base + column] = static_cast<float>(ds * scale[column]);
                gradientScale[column] += ds * z[base + column];
                gradientWeight[column] += dy * normalized;
                gradientBias[column] += dy;
            }
        }

        reference.gradientScale.resize(columns);
        reference.gradientWeight.resize(columns);
        reference.gradientBias.resize(columns);
        for (int column = 0; column < columns; ++column) {
            reference.gradientScale[column] = static_cast<float>(gradientScale[column]);
            reference.gradientWeight[column] = static_cast<float>(gradientWeight[column]);
            reference.gradientBias[column] = static_cast<float>(gradientBias[column]);
        }

        return reference;
    }

    template<typename T> void RunCase(const int rows, const int columns, const bool affine, const unsigned int seed) {
        const std::size_t elements = static_cast<std::size_t>(rows) * columns;

        const std::vector<float> hostX = ROCmKernels::Tests::RandomValues<float>(elements, seed, -1.0f, 1.0f);
        const std::vector<float> hostZ = ROCmKernels::Tests::RandomValues<float>(elements, seed + 1, -1.0f, 1.0f);
        const std::vector<float> hostScale = ROCmKernels::Tests::RandomValues<float>(columns, seed + 2, 0.5f, 1.5f);
        const std::vector<float> hostWeight = ROCmKernels::Tests::RandomValues<float>(columns, seed + 3, 0.5f, 1.0f);
        const std::vector<float> hostBias = ROCmKernels::Tests::RandomValues<float>(columns, seed + 4, -0.5f, 0.5f);
        const std::vector<float> hostUpstream = ROCmKernels::Tests::RandomValues<float>(elements, seed + 5, -1.0f, 1.0f);

        const std::vector<T> storageX = Narrow<T>(hostX);
        const std::vector<T> storageZ = Narrow<T>(hostZ);
        const std::vector<T> storageScale = Narrow<T>(hostScale);
        const std::vector<T> storageWeight = Narrow<T>(hostWeight);
        const std::vector<T> storageBias = Narrow<T>(hostBias);
        const std::vector<T> storageUpstream = Narrow<T>(hostUpstream);

        const Reference reference = ComputeReference(Widen<T>(storageX), Widen<T>(storageZ), Widen<T>(storageScale), Widen<T>(storageWeight),
                Widen<T>(storageBias), Widen<T>(storageUpstream), rows, columns, affine);

        ROCmKernels::DeviceBuffer<T> deviceX(elements);
        ROCmKernels::DeviceBuffer<T> deviceZ(elements);
        ROCmKernels::DeviceBuffer<T> deviceScale(columns);
        ROCmKernels::DeviceBuffer<T> deviceWeight(affine ? columns : 0);
        ROCmKernels::DeviceBuffer<T> deviceBias(affine ? columns : 0);
        ROCmKernels::DeviceBuffer<T> deviceOutput(elements);
        ROCmKernels::DeviceBuffer<float> deviceStatistics(static_cast<std::size_t>(rows) * 2);

        deviceX.CopyFromHost(storageX.data(), storageX.size());
        deviceZ.CopyFromHost(storageZ.data(), storageZ.size());
        deviceScale.CopyFromHost(storageScale.data(), storageScale.size());
        if (affine) {
            deviceWeight.CopyFromHost(storageWeight.data(), storageWeight.size());
            deviceBias.CopyFromHost(storageBias.data(), storageBias.size());
        }

        ROCmKernels::Kernels::LayerNormScaleAddForward(deviceX.Data(), deviceZ.Data(), deviceScale.Data(), deviceWeight.Data(),
                deviceBias.Data(), deviceOutput.Data(), deviceStatistics.Data(), rows, columns, kEpsilon,
                ROCmKernels::Kernels::DTypeOf<T>::value);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actualOutput(elements);
        std::vector<float> actualStatistics(static_cast<std::size_t>(rows) * 2);
        deviceOutput.CopyToHost(actualOutput.data(), actualOutput.size());
        deviceStatistics.CopyToHost(actualStatistics.data(), actualStatistics.size());

        const double tolerance = std::is_same_v<T, __half> ? 2e-3 : 5e-4;
        ROCmKernels::Tests::CheckClose(reference.output, Widen<T>(actualOutput), tolerance);
        ROCmKernels::Tests::CheckClose(reference.statistics, actualStatistics, tolerance);

        ROCmKernels::DeviceBuffer<T> deviceUpstream(elements);
        ROCmKernels::DeviceBuffer<T> deviceGradientX(elements);
        ROCmKernels::DeviceBuffer<T> deviceGradientZ(elements);
        ROCmKernels::DeviceBuffer<float> deviceGradientScale(columns);
        ROCmKernels::DeviceBuffer<float> deviceGradientWeight(affine ? columns : 0);
        ROCmKernels::DeviceBuffer<float> deviceGradientBias(affine ? columns : 0);

        deviceUpstream.CopyFromHost(storageUpstream.data(), storageUpstream.size());

        ROCmKernels::Kernels::LayerNormScaleAddBackward(deviceX.Data(), deviceZ.Data(), deviceScale.Data(), deviceWeight.Data(),
                deviceStatistics.Data(), deviceUpstream.Data(), deviceGradientX.Data(), deviceGradientZ.Data(), deviceGradientScale.Data(),
                deviceGradientWeight.Data(), deviceGradientBias.Data(), rows, columns, ROCmKernels::Kernels::DTypeOf<T>::value);
        ROCMKERNELS_HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> actualGradientX(elements);
        std::vector<T> actualGradientZ(elements);
        std::vector<float> actualGradientScale(columns);
        deviceGradientX.CopyToHost(actualGradientX.data(), actualGradientX.size());
        deviceGradientZ.CopyToHost(actualGradientZ.data(), actualGradientZ.size());
        deviceGradientScale.CopyToHost(actualGradientScale.data(), actualGradientScale.size());

        ROCmKernels::Tests::CheckClose(reference.gradientX, Widen<T>(actualGradientX), tolerance);
        ROCmKernels::Tests::CheckClose(reference.gradientZ, Widen<T>(actualGradientZ), tolerance);
        ROCmKernels::Tests::CheckClose(reference.gradientScale, actualGradientScale, tolerance);

        if (affine) {
            std::vector<float> actualGradientWeight(columns);
            std::vector<float> actualGradientBias(columns);
            deviceGradientWeight.CopyToHost(actualGradientWeight.data(), actualGradientWeight.size());
            deviceGradientBias.CopyToHost(actualGradientBias.data(), actualGradientBias.size());

            ROCmKernels::Tests::CheckClose(reference.gradientWeight, actualGradientWeight, tolerance);
            ROCmKernels::Tests::CheckClose(reference.gradientBias, actualGradientBias, tolerance);
        }
    }
} // namespace

TEST_CASE("LayerNormScaleAdd matches the reference", "[generic][norm]") {
    SECTION("fp32") {
        SECTION("affine") { RunCase<float>(7, 100, true, 101); }
        SECTION("no affine") { RunCase<float>(7, 100, false, 201); }
        SECTION("single row") { RunCase<float>(1, 100, true, 301); }
    }

    SECTION("fp16") {
        SECTION("affine") { RunCase<__half>(7, 100, true, 401); }
        SECTION("no affine") { RunCase<__half>(7, 100, false, 501); }
        SECTION("single row") { RunCase<__half>(1, 100, true, 601); }
    }
}
