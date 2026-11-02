//
// Loads a HIP-produced hsaco through the OpenCL runtime and dispatches it.
//
// Two argument layouts are supported: the four-argument vector-add sample and the
// attention profiling shims, whose pointers are plain kernel arguments because the
// OpenCL runtime cannot patch cl_mem handles inside a by-value struct.
//

#define CL_TARGET_OPENCL_VERSION 220

#include <CL/cl.h>
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define LOAD_OPENCL(name)                                                                                                                                          \
    pfnCl##name = reinterpret_cast<decltype(&::cl##name)>(GetProcAddress(openClModule, "cl" #name));                                                               \
    if (!pfnCl##name) {                                                                                                                                            \
        std::printf("missing OpenCL entry point: cl" #name "\n");                                                                                                  \
        return 1;                                                                                                                                                  \
    }

namespace {
    HMODULE openClModule = nullptr;
    decltype(&::clGetPlatformIDs) pfnClGetPlatformIDs = nullptr;
    decltype(&::clGetDeviceIDs) pfnClGetDeviceIDs = nullptr;
    decltype(&::clCreateContext) pfnClCreateContext = nullptr;
    decltype(&::clCreateCommandQueueWithProperties) pfnClCreateCommandQueueWithProperties = nullptr;
    decltype(&::clCreateProgramWithBinary) pfnClCreateProgramWithBinary = nullptr;
    decltype(&::clBuildProgram) pfnClBuildProgram = nullptr;
    decltype(&::clGetProgramInfo) pfnClGetProgramInfo = nullptr;
    decltype(&::clGetProgramBuildInfo) pfnClGetProgramBuildInfo = nullptr;
    decltype(&::clCreateKernel) pfnClCreateKernel = nullptr;
    decltype(&::clCreateBuffer) pfnClCreateBuffer = nullptr;
    decltype(&::clSetKernelArg) pfnClSetKernelArg = nullptr;
    decltype(&::clEnqueueNDRangeKernel) pfnClEnqueueNDRangeKernel = nullptr;
    decltype(&::clEnqueueReadBuffer) pfnClEnqueueReadBuffer = nullptr;
    decltype(&::clFinish) pfnClFinish = nullptr;

    struct Strides {
        int batch;
        int head;
        int token;
    };

    struct AttentionShape {
        int batch = 2;
        int heads = 12;
        int queries = 1024;
        int keys = 1024;
        int radius = -1;
        int warps = 16;
        int blockRows = 64;
    };

    std::vector<char> ReadFile(const std::string &path) {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream) return {};
        const std::streamsize size = stream.tellg();
        stream.seekg(0, std::ios::beg);
        std::vector<char> data(static_cast<std::size_t>(size));
        stream.read(data.data(), size);
        return data;
    }

    void PrintUsage() {
        std::printf(
                "usage: OpenClProbe --hsaco <file> [--kernel <name>] [--iterations <n>] [--wait-file <path>] [--list]\n"
                "                   [--batch <n>] [--heads <n>] [--queries <n>] [--keys <n>] [--radius <n>]\n"
                "                   [--warps <n>] [--block-rows <n>]\n"
                "\n"
                "  --list        print every kernel in the code object and exit\n"
                "  --iterations  dispatch the kernel this many times (default 1)\n"
                "  --wait-file   block before the first dispatch until the file exists\n"
                "  --ready-file  create this file once the OpenCL context is ready\n"
                "  --warps       attention work-group warps (default 16)\n"
                "  --block-rows  attention query rows per work-group (default 64)\n");
    }

    int BindAttention(cl_context context, cl_command_queue queue, cl_kernel kernel, const AttentionShape &shape, const int iterations) {
        const std::size_t qElements = static_cast<std::size_t>(shape.batch) * shape.heads * shape.queries * 32;
        const std::size_t kElements = static_cast<std::size_t>(shape.batch) * shape.heads * shape.keys * 32;
        const std::size_t maskElements = static_cast<std::size_t>(shape.batch) * shape.keys;
        const std::size_t lseElements = static_cast<std::size_t>(shape.batch) * shape.heads * shape.queries * 2;

        std::vector<float> hostQ(qElements, 0.5f);
        std::vector<float> hostK(kElements, 0.25f);
        std::vector<unsigned char> hostMask(maskElements, 1);
        std::vector<float> hostOutput(qElements, 0.0f);

        cl_int error = CL_SUCCESS;
        cl_mem q = pfnClCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, qElements * sizeof(float), hostQ.data(), &error);
        cl_mem k = pfnClCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kElements * sizeof(float), hostK.data(), &error);
        cl_mem v = pfnClCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kElements * sizeof(float), hostK.data(), &error);
        cl_mem mask = pfnClCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, maskElements, hostMask.data(), &error);
        cl_mem output = pfnClCreateBuffer(context, CL_MEM_WRITE_ONLY, qElements * sizeof(float), nullptr, &error);
        cl_mem lse = pfnClCreateBuffer(context, CL_MEM_WRITE_ONLY, lseElements * sizeof(float), nullptr, &error);

        const Strides qs{shape.heads * shape.queries * 32, shape.queries * 32, 32};
        const Strides ks{shape.heads * shape.keys * 32, shape.keys * 32, 32};
        const Strides os = qs;

        int argument = 0;
        pfnClSetKernelArg(kernel, argument++, sizeof(cl_mem), &q);
        pfnClSetKernelArg(kernel, argument++, sizeof(cl_mem), &k);
        pfnClSetKernelArg(kernel, argument++, sizeof(cl_mem), &v);
        pfnClSetKernelArg(kernel, argument++, sizeof(cl_mem), &mask);
        pfnClSetKernelArg(kernel, argument++, sizeof(cl_mem), &output);
        pfnClSetKernelArg(kernel, argument++, sizeof(cl_mem), &lse);
        pfnClSetKernelArg(kernel, argument++, sizeof(int), &shape.heads);
        pfnClSetKernelArg(kernel, argument++, sizeof(int), &shape.queries);
        pfnClSetKernelArg(kernel, argument++, sizeof(int), &shape.keys);
        pfnClSetKernelArg(kernel, argument++, sizeof(int), &shape.radius);
        pfnClSetKernelArg(kernel, argument++, sizeof(Strides), &qs);
        pfnClSetKernelArg(kernel, argument++, sizeof(Strides), &ks);
        pfnClSetKernelArg(kernel, argument++, sizeof(Strides), &ks);
        pfnClSetKernelArg(kernel, argument++, sizeof(Strides), &os);

        const std::size_t local = static_cast<std::size_t>(shape.warps) * 32;
        const std::size_t blocks = (static_cast<std::size_t>(shape.queries) + shape.blockRows - 1) / shape.blockRows;
        // Keep the HIP grid shape: x is query blocks, y is heads, z is batch. A flat
        // global size would leave head and batch at zero in the kernel.
        const std::size_t global[3] = {blocks * local, static_cast<std::size_t>(shape.heads), static_cast<std::size_t>(shape.batch)};
        const std::size_t localSize[3] = {local, 1, 1};

        for (int iteration = 0; iteration < iterations; ++iteration) {
            error = pfnClEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, localSize, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                std::printf("clEnqueueNDRangeKernel: %d\n", error);
                return 1;
            }
            if ((iteration % 1000) == 0) {
                pfnClFinish(queue);
                std::printf("%d\n", iteration);
                std::fflush(stdout);
            }
        }
        pfnClFinish(queue);

        error = pfnClEnqueueReadBuffer(queue, output, CL_TRUE, 0, qElements * sizeof(float), hostOutput.data(), 0, nullptr, nullptr);
        if (error != CL_SUCCESS) {
            std::printf("clEnqueueReadBuffer: %d\n", error);
            return 1;
        }
        std::printf("out[0]=%f out[last]=%f\n", hostOutput[0], hostOutput[qElements - 1]);
        return hostOutput[0] > 0.249f && hostOutput[0] < 0.251f ? 0 : 2;
    }
    struct LinearShape {
        int rows = 8192;
        int columns = 8192;
        int inner = 128;
    };

    int BindLinear(cl_context context, cl_command_queue queue, cl_kernel kernel, const LinearShape &shape, const int iterations) {
        const std::size_t inputElements = static_cast<std::size_t>(shape.rows) * shape.inner;
        const std::size_t weightElements = static_cast<std::size_t>(shape.inner) * shape.columns;
        const std::size_t outputElements = static_cast<std::size_t>(shape.rows) * shape.columns;

        const std::vector<unsigned short> hostInput(inputElements, 0x3800);   // 0.5 in fp16
        const std::vector<unsigned short> hostWeight(weightElements, 0x3400); // 0.25 in fp16
        const std::vector<unsigned short> hostBias(shape.columns, 0x3000);    // 0.125 in fp16
        std::vector<unsigned short> hostOutput(1, 0);

        cl_int error = CL_SUCCESS;
        cl_mem input = pfnClCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputElements * 2,
                const_cast<unsigned short *>(hostInput.data()), &error);
        cl_mem weight = pfnClCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weightElements * 2,
                const_cast<unsigned short *>(hostWeight.data()), &error);
        cl_mem bias = pfnClCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, shape.columns * 2,
                const_cast<unsigned short *>(hostBias.data()), &error);
        cl_mem output = pfnClCreateBuffer(context, CL_MEM_WRITE_ONLY, outputElements * 2, nullptr, &error);
        if (error != CL_SUCCESS) {
            std::printf("linear clCreateBuffer: %d\n", error);
            return 1;
        }

        int argument = 0;
        pfnClSetKernelArg(kernel, argument++, sizeof(cl_mem), &input);
        pfnClSetKernelArg(kernel, argument++, sizeof(cl_mem), &weight);
        pfnClSetKernelArg(kernel, argument++, sizeof(cl_mem), &bias);
        pfnClSetKernelArg(kernel, argument++, sizeof(cl_mem), &output);
        pfnClSetKernelArg(kernel, argument++, sizeof(int), &shape.rows);
        pfnClSetKernelArg(kernel, argument++, sizeof(int), &shape.columns);
        pfnClSetKernelArg(kernel, argument++, sizeof(int), &shape.inner);

        const std::size_t blockX = 16;
        const std::size_t blockY = 16;
        const std::size_t gridX = (static_cast<std::size_t>(shape.columns) + 63) / 64;
        const std::size_t gridY = (static_cast<std::size_t>(shape.rows) + 63) / 64;
        const std::size_t global[2] = {gridX * blockX, gridY * blockY};
        const std::size_t localSize[2] = {blockX, blockY};

        for (int iteration = 0; iteration < iterations; ++iteration) {
            error = pfnClEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, localSize, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                std::printf("clEnqueueNDRangeKernel: %d\n", error);
                return 1;
            }
            if ((iteration % 200) == 0) {
                pfnClFinish(queue);
                std::printf("%d\n", iteration);
                std::fflush(stdout);
            }
        }
        pfnClFinish(queue);

        error = pfnClEnqueueReadBuffer(queue, output, CL_TRUE, 0, 2, hostOutput.data(), 0, nullptr, nullptr);
        if (error != CL_SUCCESS) {
            std::printf("clEnqueueReadBuffer: %d\n", error);
            return 1;
        }
        std::printf("out[0]=0x%04x\n", hostOutput[0]);
        return 0;
    }
} // namespace

int main(int argc, char **argv) {
    std::string hsacoPath;
    std::string kernelName = "add_arrays";
    std::string waitFile;
    std::string readyFile;
    AttentionShape attentionShape;
    LinearShape linearShape;
    bool linear = false;
    int iterations = 1;
    bool listOnly = false;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&]() -> std::string { return index + 1 < argc ? argv[++index] : std::string(); };
        if (argument == "--hsaco") hsacoPath = next();
        else if (argument == "--kernel") kernelName = next();
        else if (argument == "--iterations") iterations = std::stoi(next());
        else if (argument == "--wait-file") waitFile = next();
        else if (argument == "--ready-file") readyFile = next();
        else if (argument == "--list") listOnly = true;
        else if (argument == "--batch") attentionShape.batch = std::stoi(next());
        else if (argument == "--heads") attentionShape.heads = std::stoi(next());
        else if (argument == "--queries") attentionShape.queries = std::stoi(next());
        else if (argument == "--keys") attentionShape.keys = std::stoi(next());
        else if (argument == "--radius") attentionShape.radius = std::stoi(next());
        else if (argument == "--warps") attentionShape.warps = std::stoi(next());
        else if (argument == "--block-rows") attentionShape.blockRows = std::stoi(next());
        else if (argument == "--linear") {
            linear = true;
            const std::string shape = next();
            if (shape == "down") {
                linearShape.rows = 8192;
                linearShape.columns = 128;
                linearShape.inner = 8192;
            } else {
                linearShape.rows = 8192;
                linearShape.columns = 8192;
                linearShape.inner = 128;
            }
        }
        else {
            PrintUsage();
            return 1;
        }
    }

    if (hsacoPath.empty()) {
        PrintUsage();
        return 1;
    }

    std::vector<char> binary = ReadFile(hsacoPath);
    if (binary.empty()) {
        std::printf("cannot read %s\n", hsacoPath.c_str());
        return 1;
    }

    openClModule = LoadLibraryA("OpenCL.dll");
    if (!openClModule) {
        std::printf("OpenCL.dll not found\n");
        return 1;
    }

    LOAD_OPENCL(GetPlatformIDs)
    LOAD_OPENCL(GetDeviceIDs)
    LOAD_OPENCL(CreateContext)
    LOAD_OPENCL(CreateCommandQueueWithProperties)
    LOAD_OPENCL(CreateProgramWithBinary)
    LOAD_OPENCL(BuildProgram)
    LOAD_OPENCL(GetProgramInfo)
    LOAD_OPENCL(GetProgramBuildInfo)
    LOAD_OPENCL(CreateKernel)
    LOAD_OPENCL(CreateBuffer)
    LOAD_OPENCL(SetKernelArg)
    LOAD_OPENCL(EnqueueNDRangeKernel)
    LOAD_OPENCL(EnqueueReadBuffer)
    LOAD_OPENCL(Finish)

    cl_platform_id platform = nullptr;
    cl_uint platformCount = 0;
    if (pfnClGetPlatformIDs(1, &platform, &platformCount) != CL_SUCCESS || platformCount == 0) {
        std::printf("no OpenCL platform\n");
        return 1;
    }

    cl_device_id device = nullptr;
    if (pfnClGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS) {
        std::printf("no OpenCL GPU\n");
        return 1;
    }

    cl_int error = CL_SUCCESS;
    cl_context context = pfnClCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    cl_command_queue queue = pfnClCreateCommandQueueWithProperties(context, device, nullptr, &error);

    const std::size_t binarySize = binary.size();
    const unsigned char *binaryData = reinterpret_cast<const unsigned char *>(binary.data());
    cl_program program = pfnClCreateProgramWithBinary(context, 1, &device, &binarySize, &binaryData, nullptr, &error);
    if (error != CL_SUCCESS) {
        std::printf("clCreateProgramWithBinary: %d\n", error);
        return 1;
    }

    error = pfnClBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        std::size_t logSize = 0;
        pfnClGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize + 1, 0);
        pfnClGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::printf("clBuildProgram: %d\n%s\n", error, log.data());
        return 1;
    }

    if (listOnly) {
        std::size_t namesSize = 0;
        pfnClGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, nullptr, &namesSize);
        std::vector<char> names(namesSize + 1, 0);
        pfnClGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, namesSize, names.data(), nullptr);
        std::printf("%s\n", names.data());
        return 0;
    }

    cl_kernel kernel = pfnClCreateKernel(program, kernelName.c_str(), &error);
    if (error != CL_SUCCESS) {
        std::printf("pfnClCreateKernel(%s): %d\n", kernelName.c_str(), error);
        return 1;
    }

    if (!readyFile.empty()) {
        std::ofstream(readyFile).close();
    }

    // The launcher creates its context before this point, so the driver has
    // already seen the process. Wait for the capture request before issuing the
    // first dispatch, otherwise RGP may attach after the work is queued.
    if (!waitFile.empty()) {
        const ULONGLONG deadline = GetTickCount64() + 60000;
        while (GetFileAttributesA(waitFile.c_str()) == INVALID_FILE_ATTRIBUTES) {
            if (GetTickCount64() > deadline) {
                std::printf("timed out waiting for %s\n", waitFile.c_str());
                return 1;
            }
            Sleep(10);
        }
    }

    if (linear) return BindLinear(context, queue, kernel, linearShape, iterations);
    const bool attention = kernelName.rfind("attention_", 0) == 0;
    if (attention) return BindAttention(context, queue, kernel, attentionShape, iterations);

    constexpr int count = 1 << 20;
    const std::vector<float> hostA(count, 1.0f);
    const std::vector<float> hostB(count, 2.0f);
    std::vector<float> hostOut(count, 0.0f);

    cl_mem bufferA = pfnClCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(float),
            const_cast<float *>(hostA.data()), &error);
    cl_mem bufferB = pfnClCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(float),
            const_cast<float *>(hostB.data()), &error);
    cl_mem bufferOut = pfnClCreateBuffer(context, CL_MEM_WRITE_ONLY, count * sizeof(float), nullptr, &error);
    int countArgument = count;

    pfnClSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    pfnClSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    pfnClSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferOut);
    pfnClSetKernelArg(kernel, 3, sizeof(int), &countArgument);

    const std::size_t global = count;
    const std::size_t local = 256;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        error = pfnClEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        if (error != CL_SUCCESS) {
            std::printf("clEnqueueNDRangeKernel: %d\n", error);
            return 1;
        }
        if ((iteration % 100000) == 0) {
            pfnClFinish(queue);
            std::printf("%d\n", iteration);
            std::fflush(stdout);
        }
    }
    pfnClFinish(queue);

    pfnClEnqueueReadBuffer(queue, bufferOut, CL_TRUE, 0, count * sizeof(float), hostOut.data(), 0, nullptr, nullptr);
    std::printf("out[0]=%f out[last]=%f\n", hostOut[0], hostOut[count - 1]);
    return (hostOut[0] == 3.0f && hostOut[count - 1] == 3.0f) ? 0 : 2;
}



