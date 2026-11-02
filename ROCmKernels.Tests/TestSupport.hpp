#pragma once

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace ROCmKernels::Tests {
    template <typename T> std::vector<T> RandomValues(const std::size_t count, const unsigned int seed, const T low, const T high) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<T> distribution(low, high);

        std::vector<T> values(count);
        for (T &value : values) {
            value = distribution(generator);
        }

        return values;
    }

    /// Reports the worst absolute difference instead of failing on the first one.
    template <typename T> void CheckClose(const std::vector<T> &expected, const std::vector<T> &actual, const double tolerance) {
        REQUIRE(expected.size() == actual.size());

        double worst = 0;
        std::size_t worstIndex = 0;
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const double difference = std::abs(static_cast<double>(expected[index]) - static_cast<double>(actual[index]));
            if (difference > worst) {
                worst = difference;
                worstIndex = index;
            }
        }

        INFO("worst difference " << worst << " at index " << worstIndex << " (expected " << expected[worstIndex] << ", got " << actual[worstIndex] << ")");
        CHECK(worst <= tolerance);
    }
} // namespace ROCmKernels::Tests
