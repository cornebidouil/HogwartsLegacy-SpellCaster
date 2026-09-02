#ifndef DEF_TOOLS
#define DEF_TOOLS

// Console color codes
#define ORANGE "\033[38;5;208m"
#define RESET "\033[0m"
#define BOLD "\033[1m"

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
#include <ViGEm/Client.h>
#include <intrin.h>

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
bool isWindowOpen(const std::string& windowName);
HWND findWindow(const std::string& windowName);
bool isWindowOpenIgnoreCase(const std::string& windowName);
void handleViGEmError(VIGEM_ERROR error);
std::wstring utf8ToWstring(const std::string& str);
std::wstring stringToWString(const std::string& str);  // Alias for utf8ToWstring
bool isOpenVINOCompatible();
bool fileExists(const std::string& filepath);
bool fileExists_permissions(const std::string& filepath);
std::string wstringToString(const std::wstring& wstr);
bool launchDetachedProcess(const std::wstring& exePath, const std::wstring& windowTitle = L"");

#endif