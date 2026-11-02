# Resolves the ROCm installation used for host and HIP compilation.
#
# Search order: ROCM_PATH cache and environment, active Python environment,
# HIP_PATH, hipconfig, default HIP SDK directories.

set(ROCM_PATH "" CACHE PATH "ROCm installation used for host and HIP compilation.")

function(_rocmkernels_try_rocm candidate_root out_root out_clang_dir out_bitcode_dir)
    set(_root "")
    set(_clang_dir "")
    set(_bitcode_dir "")
    set(_has_runtime_library FALSE)

    if (candidate_root)
        file(TO_CMAKE_PATH "${candidate_root}" _candidate)
        if (IS_DIRECTORY "${_candidate}")
            if (CMAKE_HOST_WIN32)
                set(_compiler_suffix ".exe")
            else ()
                set(_compiler_suffix "")
            endif ()

            # pip wheels keep Clang under lib/llvm, the HIP SDK keeps it in bin/.
            if (EXISTS "${_candidate}/lib/llvm/bin/clang++${_compiler_suffix}")
                set(_clang_dir "${_candidate}/lib/llvm/bin")
                set(_bitcode_dir "${_candidate}/lib/llvm/amdgcn/bitcode")
            elseif (EXISTS "${_candidate}/bin/clang++${_compiler_suffix}")
                set(_clang_dir "${_candidate}/bin")
                set(_bitcode_dir "${_candidate}/amdgcn/bitcode")
            endif ()

            foreach (_library_name amdhip64.lib libamdhip64.so amdhip64.so)
                if (EXISTS "${_candidate}/lib/${_library_name}")
                    set(_has_runtime_library TRUE)
                endif ()
            endforeach ()

            if (_clang_dir AND EXISTS "${_candidate}/include/hip/hip_runtime.h" AND _has_runtime_library)
                set(_root "${_candidate}")
            endif ()
        endif ()
    endif ()

    set(${out_root} "${_root}" PARENT_SCOPE)
    set(${out_clang_dir} "${_clang_dir}" PARENT_SCOPE)
    set(${out_bitcode_dir} "${_bitcode_dir}" PARENT_SCOPE)
endfunction()

# Reconfigurations and try_compile() sub-builds already have the compilers.
if (CMAKE_C_COMPILER AND CMAKE_CXX_COMPILER)
    return()
endif ()

set(_rocm_candidates "")

if (ROCM_PATH)
    list(APPEND _rocm_candidates "${ROCM_PATH}")
endif ()

foreach (_environment_variable ROCM_PATH VIRTUAL_ENV CONDA_PREFIX HIP_PATH)
    if (NOT DEFINED ENV{${_environment_variable}})
        continue()
    endif ()

    file(TO_CMAKE_PATH "$ENV{${_environment_variable}}" _environment_path)
    if (_environment_path STREQUAL "")
        continue()
    endif ()

    if (_environment_variable MATCHES "VIRTUAL_ENV|CONDA_PREFIX")
        file(GLOB _pip_candidates
                "${_environment_path}/Lib/site-packages/_rocm_sdk_core"
                "${_environment_path}/lib/python*/site-packages/_rocm_sdk_core")
        list(APPEND _rocm_candidates ${_pip_candidates})
    else ()
        list(APPEND _rocm_candidates "${_environment_path}")
    endif ()
endforeach ()

find_program(ROCM_HIPCONFIG_EXECUTABLE NAMES hipconfig)
if (ROCM_HIPCONFIG_EXECUTABLE)
    execute_process(
            COMMAND "${ROCM_HIPCONFIG_EXECUTABLE}" --path
            OUTPUT_VARIABLE _rocm_hipconfig_path
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _rocm_hipconfig_result
            ERROR_QUIET
    )
    if (_rocm_hipconfig_result EQUAL 0 AND _rocm_hipconfig_path)
        list(APPEND _rocm_candidates "${_rocm_hipconfig_path}")
    endif ()
endif ()

if (CMAKE_HOST_WIN32)
    file(GLOB _rocm_default_candidates "C:/Program Files/AMD/ROCm/*")
else ()
    file(GLOB _rocm_default_candidates "/opt/rocm-*")
    list(APPEND _rocm_default_candidates "/opt/rocm" "/usr/local")
endif ()
list(SORT _rocm_default_candidates COMPARE NATURAL ORDER DESCENDING)
list(APPEND _rocm_candidates ${_rocm_default_candidates})

list(REMOVE_DUPLICATES _rocm_candidates)

set(_rocm_root "")
set(_rocm_clang_dir "")
set(_rocm_bitcode_dir "")
set(_rocm_attempted_candidates "")

foreach (_candidate IN LISTS _rocm_candidates)
    _rocmkernels_try_rocm("${_candidate}" _candidate_root _candidate_clang_dir _candidate_bitcode_dir)
    if (NOT _rocm_attempted_candidates STREQUAL "")
        string(APPEND _rocm_attempted_candidates "\n    ")
    endif ()
    string(APPEND _rocm_attempted_candidates "${_candidate}")
    if (_candidate_root)
        set(_rocm_root "${_candidate_root}")
        set(_rocm_clang_dir "${_candidate_clang_dir}")
        set(_rocm_bitcode_dir "${_candidate_bitcode_dir}")
        break()
    endif ()
endforeach ()

if (NOT _rocm_root)
    message(FATAL_ERROR
            "Could not find a usable ROCm installation.\n"
            "Checked:\n    ${_rocm_attempted_candidates}\n"
            "Configure with -DROCM_PATH=<root> or set the ROCM_PATH environment variable.")
endif ()

if (CMAKE_HOST_WIN32)
    set(_rocm_compiler_suffix ".exe")
else ()
    set(_rocm_compiler_suffix "")
endif ()

# Host code and device code must come from the same Clang.
if (NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "${_rocm_clang_dir}/clang${_rocm_compiler_suffix}" CACHE FILEPATH "C compiler")
endif ()
if (NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${_rocm_clang_dir}/clang++${_rocm_compiler_suffix}" CACHE FILEPATH "C++ compiler")
endif ()

set(ROCM_PATH "${_rocm_root}" CACHE PATH "ROCm installation used for host and HIP compilation." FORCE)
set(ROCM_BITCODE_DIR "${_rocm_bitcode_dir}" CACHE INTERNAL "ROCm device bitcode directory")

file(READ "${_rocm_root}/include/hip/hip_version.h" _rocm_hip_version_header)
set(_rocm_hip_version "unknown")
if (_rocm_hip_version_header MATCHES "#define HIP_VERSION_MAJOR[ \t]+([0-9]+)")
    set(_rocm_hip_version_major "${CMAKE_MATCH_1}")
endif ()
if (_rocm_hip_version_header MATCHES "#define HIP_VERSION_MINOR[ \t]+([0-9]+)")
    set(_rocm_hip_version_minor "${CMAKE_MATCH_1}")
endif ()
if (_rocm_hip_version_header MATCHES "#define HIP_VERSION_PATCH[ \t]+([0-9]+)")
    set(_rocm_hip_version_patch "${CMAKE_MATCH_1}")
endif ()
if (_rocm_hip_version_major AND _rocm_hip_version_minor AND _rocm_hip_version_patch)
    set(_rocm_hip_version "${_rocm_hip_version_major}.${_rocm_hip_version_minor}.${_rocm_hip_version_patch}")
endif ()
set(ROCM_HIP_VERSION "${_rocm_hip_version}" CACHE INTERNAL "ROCm HIP version")

message(STATUS "ROCm ${ROCM_HIP_VERSION} found at ${ROCM_PATH}")
message(STATUS "ROCm compiler directory: ${_rocm_clang_dir}")
