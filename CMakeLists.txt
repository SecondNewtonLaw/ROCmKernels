cmake_minimum_required(VERSION 3.31)

# A user-supplied toolchain file takes precedence over the bundled one.
if (NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/cmake/ROCmToolchain.cmake" CACHE FILEPATH "ROCm toolchain file")
    include("${CMAKE_TOOLCHAIN_FILE}")
endif ()

project(ROCmKernels LANGUAGES C CXX)

include(CTest)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_CCACHE "Use ccache as the compiler launcher when available." ON)
option(ALLOW_NON_STANDARD_MATH "Enable non-standard fast-math optimizations." OFF)
option(DEBUG_INFORMATION_EXTENSIVE "Emit extensive debug information." ON)
option(DEBUG_NO_INLINING "Disable function inlining for Debug builds." ON)
option(BUILD_BENCHMARKS "Build the kernel benchmark executable." ON)
option(BUILD_PROFILING_TOOLS "Build the OpenCL probe used by the profile-kernel skill." OFF)
option(BUILD_ISA_BENCH "Build the RDNA2 instruction microbenchmark." ON)
option(ROCMKERNELS_EMIT_ASSEMBLY "Provide a target that writes gfx1031 device assembly for every kernel." ON)

include(cmake/ROCmHip.cmake)

message(STATUS "ROCmKernels GPU target: ${ROCMKERNELS_GPU_TARGET}")

if (USE_CCACHE)
    include(cmake/ccache.cmake)
endif ()

if (NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    message(FATAL_ERROR "ROCmKernels requires Clang with the GNU front-end, as shipped with ROCm.")
endif ()

set(CXX_COMPILE_OPTIONS -Werror -Wall -Wextra -Wpedantic -funroll-loops)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND CXX_COMPILE_OPTIONS -Og)

    if (DEBUG_NO_INLINING)
        list(APPEND CXX_COMPILE_OPTIONS -fno-inline-functions)
    endif ()
endif ()

if (ALLOW_NON_STANDARD_MATH)
    list(APPEND CXX_COMPILE_OPTIONS -ffast-math)
endif ()

if (DEBUG_INFORMATION_EXTENSIVE)
    list(APPEND CXX_COMPILE_OPTIONS -g3)
endif ()

function(rocmkernels_apply_compile_options target)
    target_compile_options(${target} PRIVATE ${CXX_COMPILE_OPTIONS})
endfunction()

add_library(ROCmKernels.Common STATIC
        ROCmKernels.Common/src/Device.cpp

        ROCmKernels.Common/include/ROCmKernels/Common/Device.hpp
        ROCmKernels.Common/include/ROCmKernels/Common/DeviceBuffer.hpp
        ROCmKernels.Common/include/ROCmKernels/Common/Error.hpp
        ROCmKernels.Common/include/ROCmKernels/Common/Export.hpp
)
target_include_directories(ROCmKernels.Common PUBLIC ROCmKernels.Common/include)
target_link_libraries(ROCmKernels.Common PUBLIC ROCmKernels.Hip)
rocmkernels_apply_compile_options(ROCmKernels.Common)

set(ROCMKERNELS_KERNEL_SOURCES
        ROCmKernels.Kernels/src/Activation/SwiGLU.hip
        ROCmKernels.Kernels/src/Attention/Attention.hip
        ROCmKernels.Kernels/src/Attention/Attention64.hip
        ROCmKernels.Kernels/src/Elementwise/Copy.hip
        ROCmKernels.Kernels/src/Elementwise/Vector.hip
        ROCmKernels.Kernels/src/Embedding/EmbeddingGrad.hip
        ROCmKernels.Kernels/src/Loss/CrossEntropy.hip
        ROCmKernels.Kernels/src/Norm/AddNorm.hip
        ROCmKernels.Kernels/src/Optimizer/Muon.hip
        ROCmKernels.Kernels/src/Positional/Rotary.hip
)

set(ROCMKERNELS_KERNEL_HEADERS
        ROCmKernels.Kernels/src/Attention/AttentionTile32.h

        ROCmKernels.Kernels/include/ROCmKernels/Kernels/Strides.hpp
        ROCmKernels.Kernels/include/ROCmKernels/Kernels/DType.hpp
        ROCmKernels.Kernels/include/ROCmKernels/Kernels/Activation/SwiGLU.hpp
        ROCmKernels.Kernels/include/ROCmKernels/Kernels/Attention/Attention.hpp
        ROCmKernels.Kernels/include/ROCmKernels/Kernels/Attention/Attention64.hpp
        ROCmKernels.Kernels/include/ROCmKernels/Kernels/Elementwise/Vector.hpp
        ROCmKernels.Kernels/include/ROCmKernels/Kernels/Elementwise/Copy.hpp
        ROCmKernels.Kernels/include/ROCmKernels/Kernels/Embedding/EmbeddingGrad.hpp
        ROCmKernels.Kernels/include/ROCmKernels/Kernels/Loss/CrossEntropy.hpp
        ROCmKernels.Kernels/include/ROCmKernels/Kernels/Norm/AddNorm.hpp
        ROCmKernels.Kernels/include/ROCmKernels/Kernels/Optimizer/Muon.hpp
        ROCmKernels.Kernels/include/ROCmKernels/Kernels/Positional/Rotary.hpp
)

# New dtype-generic kernels live under Generic/ so they can be added without
# touching the imported per-domain source lists.
file(GLOB_RECURSE ROCMKERNELS_GENERIC_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/ROCmKernels.Kernels/src/Generic/*.hip")
file(GLOB_RECURSE ROCMKERNELS_GENERIC_HEADERS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/ROCmKernels.Kernels/src/Generic/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/ROCmKernels.Kernels/include/ROCmKernels/Kernels/Generic/*.hpp")
list(APPEND ROCMKERNELS_KERNEL_SOURCES ${ROCMKERNELS_GENERIC_SOURCES})
list(APPEND ROCMKERNELS_KERNEL_HEADERS ${ROCMKERNELS_GENERIC_HEADERS})

foreach (_kernel_source IN LISTS ROCMKERNELS_KERNEL_SOURCES)
    rocmkernels_enable_hip_source(${_kernel_source})
endforeach ()

add_library(ROCmKernels.Kernels STATIC
        ${ROCMKERNELS_KERNEL_SOURCES}
        ${ROCMKERNELS_KERNEL_HEADERS}
)
target_include_directories(ROCmKernels.Kernels PUBLIC ROCmKernels.Kernels/include)
target_link_libraries(ROCmKernels.Kernels PUBLIC ROCmKernels.Common)
rocmkernels_apply_compile_options(ROCmKernels.Kernels)

# Same sources as a DLL, so a training loop can call the kernels through ctypes.
option(BUILD_SHARED_KERNELS "Build ROCmKernels.Kernels.Shared for ctypes harnesses." ON)
if (BUILD_SHARED_KERNELS)
    add_library(ROCmKernels.Kernels.Shared SHARED
            ${ROCMKERNELS_KERNEL_SOURCES}
            ${ROCMKERNELS_KERNEL_HEADERS}
    )
    target_include_directories(ROCmKernels.Kernels.Shared PUBLIC ROCmKernels.Kernels/include)
    target_link_libraries(ROCmKernels.Kernels.Shared PUBLIC ROCmKernels.Common)
    rocmkernels_apply_compile_options(ROCmKernels.Kernels.Shared)
    rocmkernels_deploy_hip_runtime(ROCmKernels.Kernels.Shared)
endif ()

if (ROCMKERNELS_EMIT_ASSEMBLY)
    set(ROCMKERNELS_ASSEMBLY_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/assembly")
    set(ROCMKERNELS_ASSEMBLY_OUTPUTS "")

    foreach (_kernel_source IN LISTS ROCMKERNELS_KERNEL_SOURCES)
        get_filename_component(_kernel_source "${_kernel_source}" ABSOLUTE)
        get_filename_component(_kernel_name "${_kernel_source}" NAME_WE)
        set(_assembly_output "${ROCMKERNELS_ASSEMBLY_DIRECTORY}/${_kernel_name}.s")

        add_custom_command(OUTPUT "${_assembly_output}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${ROCMKERNELS_ASSEMBLY_DIRECTORY}"
                COMMAND "${CMAKE_CXX_COMPILER}" -x hip
                        "--rocm-device-lib-path=${ROCM_BITCODE_DIR}"
                        "--offload-arch=${ROCMKERNELS_GPU_TARGET}"
                        -S --cuda-device-only -O3 -std=c++23 -ffp-contract=off -D__HIP_PLATFORM_AMD__
                        "-I${CMAKE_CURRENT_SOURCE_DIR}/ROCmKernels.Kernels/include"
                        "-I${CMAKE_CURRENT_SOURCE_DIR}/ROCmKernels.Common/include"
                        "-isystem" "${ROCM_HIP_INCLUDE_DIR}"
                        "${_kernel_source}"
                        -o "${_assembly_output}"
                DEPENDS "${_kernel_source}" ${ROCMKERNELS_KERNEL_HEADERS}
                VERBATIM
        )
        list(APPEND ROCMKERNELS_ASSEMBLY_OUTPUTS "${_assembly_output}")
    endforeach ()

    add_custom_target(ROCmKernels.Assembly DEPENDS ${ROCMKERNELS_ASSEMBLY_OUTPUTS})

    find_program(ROCMKERNELS_POWERSHELL NAMES pwsh powershell)
    if (ROCMKERNELS_POWERSHELL)
        add_custom_target(ROCmKernels.InstructionMix
                COMMAND "${ROCMKERNELS_POWERSHELL}" -NoProfile -ExecutionPolicy Bypass -File
                        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/InstructionMix.ps1"
                        -AssemblyDirectory "${ROCMKERNELS_ASSEMBLY_DIRECTORY}"
                DEPENDS ${ROCMKERNELS_ASSEMBLY_OUTPUTS}
                VERBATIM
        )
    endif ()
endif ()

if (BUILD_TESTING)
    include(cmake/CPM.cmake)

    CPMAddPackage(
            NAME Catch2
            GITHUB_REPOSITORY catchorg/Catch2
            GIT_TAG v3.14.0
    )

    file(GLOB ROCMKERNELS_GENERIC_TESTS CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/ROCmKernels.Tests/Generic/*.cpp")

    add_executable(ROCmKernels.Tests
            ROCmKernels.Tests/ActivationTests.cpp
            ROCmKernels.Tests/AttentionTests.cpp
            ROCmKernels.Tests/DeviceTests.cpp
            ROCmKernels.Tests/EmbeddingTests.cpp
            ROCmKernels.Tests/LossTests.cpp
            ROCmKernels.Tests/NormTests.cpp
            ROCmKernels.Tests/OptimizerTests.cpp
            ROCmKernels.Tests/PositionalTests.cpp
            ROCmKernels.Tests/VectorTests.cpp
            ${ROCMKERNELS_GENERIC_TESTS}

            ROCmKernels.Tests/TestSupport.hpp
    )
    target_link_libraries(ROCmKernels.Tests PRIVATE ROCmKernels.Kernels Catch2::Catch2WithMain)
    rocmkernels_apply_compile_options(ROCmKernels.Tests)
    rocmkernels_deploy_hip_runtime(ROCmKernels.Tests)

    list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
    include(Catch)
    catch_discover_tests(ROCmKernels.Tests
            PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${ROCM_HIP_RUNTIME_BIN_DIR}"
    )
endif ()

if (BUILD_BENCHMARKS)
    file(GLOB ROCMKERNELS_GENERIC_BENCHMARKS CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/ROCmKernels.Benchmark/src/Generic/*.cpp")

    add_executable(ROCmKernels.Benchmark
            ROCmKernels.Benchmark/src/AttentionBenchmarks.cpp
            ROCmKernels.Benchmark/src/EmbeddingBenchmarks.cpp
            ROCmKernels.Benchmark/src/Harness.cpp
            ROCmKernels.Benchmark/src/LossBenchmarks.cpp
            ROCmKernels.Benchmark/src/MuonBenchmarks.cpp
            ROCmKernels.Benchmark/src/NormBenchmarks.cpp
            ROCmKernels.Benchmark/src/RotaryBenchmarks.cpp
            ROCmKernels.Benchmark/src/SwiGLUBenchmarks.cpp
            ROCmKernels.Benchmark/src/VectorBenchmarks.cpp
            ROCmKernels.Benchmark/src/main.cpp
            ${ROCMKERNELS_GENERIC_BENCHMARKS}

            ROCmKernels.Benchmark/include/ROCmKernels/Benchmark/Harness.hpp
    )
    target_include_directories(ROCmKernels.Benchmark PRIVATE ROCmKernels.Benchmark/include)
    target_link_libraries(ROCmKernels.Benchmark PRIVATE ROCmKernels.Kernels)
    rocmkernels_apply_compile_options(ROCmKernels.Benchmark)
    rocmkernels_deploy_hip_runtime(ROCmKernels.Benchmark)
endif ()

if (BUILD_PROFILING_TOOLS AND WIN32)
    add_executable(ROCmKernels.OpenClProbe tools/ProfileKernel/OpenClProbe.cpp)
    target_include_directories(ROCmKernels.OpenClProbe SYSTEM PRIVATE "${ROCM_HIP_INCLUDE_DIR}")
    rocmkernels_apply_compile_options(ROCmKernels.OpenClProbe)
endif ()

if (BUILD_ISA_BENCH)
    set(ROCMKERNELS_ISA_BENCH_SOURCE tools/IsaBench/IsaBench.hip)
    rocmkernels_enable_hip_source(${ROCMKERNELS_ISA_BENCH_SOURCE})
    add_executable(ROCmKernels.IsaBench ${ROCMKERNELS_ISA_BENCH_SOURCE})
    target_link_libraries(ROCmKernels.IsaBench PRIVATE ROCmKernels.Common)
    rocmkernels_apply_compile_options(ROCmKernels.IsaBench)
    rocmkernels_deploy_hip_runtime(ROCmKernels.IsaBench)
endif ()

