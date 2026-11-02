#pragma once

// The kernels keep a C ABI so the same sources can still be built as a shared
// library and loaded from Python. In a static build the attribute is harmless.
#if defined(__HIP_DEVICE_COMPILE__)
#define ROCMKERNELS_EXPORT
#elif defined(_WIN32)
#define ROCMKERNELS_EXPORT __declspec(dllexport)
#else
#define ROCMKERNELS_EXPORT __attribute__((visibility("default")))
#endif
