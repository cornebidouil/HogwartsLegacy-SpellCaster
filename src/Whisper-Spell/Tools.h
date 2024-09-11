#ifndef DEF_TOOLS
#define DEF_TOOLS

#include <iostream>
#include <windows.h>
#include <string>
#include <functional>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <comdef.h>
#include <Wbemidl.h>
#pragma comment(lib, "wbemuuid.lib")

struct DeviceInfo {
    int deviceId;
    std::string name;
    int computeMajor;
    int computeMinor;
    bool hipblasCompatible;
    std::string arch;
};

enum class GPUVendor {
    NVIDIA_DEFAULT,
    NVIDIA_CUDA,
    AMD_DEFAULT,
    AMD_ROCM,
    DEFAULT_GPU,
    CPU
};

std::size_t hashFile(const std::string& filepath);
bool hasNvidiaGPU();
GPUVendor detectGPU();
std::string getAMDGPUArch();
std::vector<DeviceInfo> checkHipBlasCompatibility();
std::string GetActiveWindowTitle();
std::string trimTrailingSpaces(const std::string& str);

#endif