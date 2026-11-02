# HIP runtime and the flags for Clang's HIP front-end.
#
# CMake's HIP language needs the hip-lang package, which the pip ROCm wheels do
# not ship. .hip files are compiled as C++ with -x hip instead.

if (NOT ROCM_PATH)
    message(FATAL_ERROR "ROCM_PATH is not set. Configure through cmake/ROCmToolchain.cmake or pass -DROCM_PATH=<root>.")
endif ()

set(ROCM_HIP_INCLUDE_DIR "${ROCM_PATH}/include" CACHE PATH "HIP headers")
set(ROCM_HIP_LIBRARY_DIR "${ROCM_PATH}/lib" CACHE PATH "HIP runtime libraries")
set(ROCM_HIP_RUNTIME_BIN_DIR "${ROCM_PATH}/bin" CACHE PATH "HIP runtime binaries")

add_library(ROCmKernels.Hip INTERFACE)
target_include_directories(ROCmKernels.Hip SYSTEM INTERFACE "${ROCM_HIP_INCLUDE_DIR}")
target_compile_definitions(ROCmKernels.Hip INTERFACE "__HIP_PLATFORM_AMD__")
target_link_directories(ROCmKernels.Hip INTERFACE "${ROCM_HIP_LIBRARY_DIR}")
target_link_libraries(ROCmKernels.Hip INTERFACE amdhip64)

set(ROCMKERNELS_GPU_TARGET "gfx1031")

string(REGEX REPLACE "^gfx" "" _rocmkernels_isa "${ROCMKERNELS_GPU_TARGET}")
if (NOT EXISTS "${ROCM_BITCODE_DIR}/oclc_isa_version_${_rocmkernels_isa}.bc")
    message(FATAL_ERROR "ROCm at '${ROCM_PATH}' has no device library for ${ROCMKERNELS_GPU_TARGET}.")
endif ()

set(ROCMKERNELS_HIP_COMPILE_OPTIONS
        "-x" "hip"
        "--rocm-device-lib-path=${ROCM_BITCODE_DIR}"
        "--offload-arch=${ROCMKERNELS_GPU_TARGET}"
        # The imported kernels were validated without FMA contraction.
        "-ffp-contract=off"
        # CMake's dependency scanner reuses these flags without -x hip.
        "-Qunused-arguments"
)

function(rocmkernels_enable_hip_source source_file)
    set_source_files_properties("${source_file}" PROPERTIES
            LANGUAGE CXX
            COMPILE_OPTIONS "${ROCMKERNELS_HIP_COMPILE_OPTIONS}")
endfunction()

option(ROCMKERNELS_COPY_HIP_RUNTIME "Copy the HIP runtime DLLs next to built executables." ON)

# The application directory wins over PATH, so the copy keeps the binaries on
# the runtime they were linked against.
if (WIN32)
    file(GLOB _rocm_hip_runtime_dlls "${ROCM_HIP_RUNTIME_BIN_DIR}/amdhip64*.dll")
    foreach (_dll_name amd_comgr.dll rocm_kpack.dll)
        if (EXISTS "${ROCM_HIP_RUNTIME_BIN_DIR}/${_dll_name}")
            list(APPEND _rocm_hip_runtime_dlls "${ROCM_HIP_RUNTIME_BIN_DIR}/${_dll_name}")
        endif ()
    endforeach ()
    set(ROCM_HIP_RUNTIME_LIBRARIES "${_rocm_hip_runtime_dlls}" CACHE INTERNAL "HIP runtime DLLs to deploy" FORCE)
endif ()

function(rocmkernels_deploy_hip_runtime target)
    if (NOT WIN32 OR NOT ROCMKERNELS_COPY_HIP_RUNTIME)
        return()
    endif ()

    foreach (_runtime_library IN LISTS ROCM_HIP_RUNTIME_LIBRARIES)
        add_custom_command(TARGET ${target} POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_runtime_library}" "$<TARGET_FILE_DIR:${target}>"
                VERBATIM)
    endforeach ()
endfunction()
