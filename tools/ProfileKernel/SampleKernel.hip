//
// Sample kernel used by the profile-kernel skill.
// It intentionally takes plain pointers and scalars so it can be dispatched
// through the OpenCL runtime without an argument translation shim.
//

#include <hip/hip_runtime.h>

extern "C" __global__ void add_arrays(const float *a, const float *b, float *out, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = a[index] + b[index];
    }
}
