#include "Tools.h"

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>


// Structure to pass data to the enum callback
struct WindowSearchData {
    std::string targetWindowName;
    bool found;
    HWND foundHandle;  // Optional: store the handle if found
};

int getGPUPriority(GPUVendor vendor) {
    switch (vendor) {
    case GPUVendor::NVIDIA_CUDA: return 6;
    case GPUVendor::AMD_ROCM: return 5;
    case GPUVendor::NVIDIA_DEFAULT: return 4;
    case GPUVendor::AMD_DEFAULT: return 3;
    case GPUVendor::DEFAULT_GPU: return 2;
    case GPUVendor::CPU: return 1;
    default: return 0;
    }
}


std::size_t hashFile(const std::string& filepath) {
	std::ifstream file(filepath, std::ifstream::binary);
	if (!file) {
        return 0;
	}

	std::string file_content((std::istreambuf_iterator<char>(file)),
		std::istreambuf_iterator<char>());

	std::hash<std::string> hasher;
	return hasher(file_content);
}


bool hasNvidiaGPU() {
    HRESULT hres;
    hres = CoInitializeEx(0, COINIT_MULTITHREADED);
    if (FAILED(hres)) return false;

    hres = CoInitializeSecurity(NULL, -1, NULL, NULL, RPC_C_AUTHN_LEVEL_DEFAULT, RPC_C_IMP_LEVEL_IMPERSONATE, NULL, EOAC_NONE, NULL);
    if (FAILED(hres)) {
        CoUninitialize();
        return false;
    }

    IWbemLocator* pLoc = NULL;
    hres = CoCreateInstance(CLSID_WbemLocator, 0, CLSCTX_INPROC_SERVER, IID_IWbemLocator, (LPVOID*)&pLoc);
    if (FAILED(hres)) {
        CoUninitialize();
        return false;
    }

    IWbemServices* pSvc = NULL;
    hres = pLoc->ConnectServer(_bstr_t(L"ROOT\\CIMV2"), NULL, NULL, 0, NULL, 0, 0, &pSvc);
    if (FAILED(hres)) {
        pLoc->Release();
        CoUninitialize();
        return false;
    }

    hres = CoSetProxyBlanket(pSvc, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, NULL, RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE, NULL, EOAC_NONE);
    if (FAILED(hres)) {
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        return false;
    }

    IEnumWbemClassObject* pEnumerator = NULL;
    hres = pSvc->ExecQuery(bstr_t("WQL"), bstr_t("SELECT * FROM Win32_VideoController WHERE AdapterCompatibility='NVIDIA'"), WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY, NULL, &pEnumerator);
    if (FAILED(hres)) {
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        return false;
    }

    IWbemClassObject* pclsObj = NULL;
    ULONG uReturn = 0;
    bool hasNvidia = false;

    while (pEnumerator) {
        HRESULT hr = pEnumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn);
        if (0 == uReturn) break;
        hasNvidia = true;
        pclsObj->Release();
    }

    pSvc->Release();
    pLoc->Release();
    pEnumerator->Release();
    CoUninitialize();

    return hasNvidia;
}

GPUVendor detectGPU() {
    std::cout << "Detected GPUs :" << std::endl;
    HRESULT hres;
    hres = CoInitializeEx(0, COINIT_MULTITHREADED);
    if (FAILED(hres)) return GPUVendor::CPU;

    hres = CoInitializeSecurity(NULL, -1, NULL, NULL, RPC_C_AUTHN_LEVEL_DEFAULT, RPC_C_IMP_LEVEL_IMPERSONATE, NULL, EOAC_NONE, NULL);
    if (FAILED(hres)) {
        CoUninitialize();
        return GPUVendor::CPU;
    }

    IWbemLocator* pLoc = NULL;
    hres = CoCreateInstance(CLSID_WbemLocator, 0, CLSCTX_INPROC_SERVER, IID_IWbemLocator, (LPVOID*)&pLoc);
    if (FAILED(hres)) {
        CoUninitialize();
        return GPUVendor::CPU;
    }

    IWbemServices* pSvc = NULL;
    hres = pLoc->ConnectServer(_bstr_t(L"ROOT\\CIMV2"), NULL, NULL, 0, NULL, 0, 0, &pSvc);
    if (FAILED(hres)) {
        pLoc->Release();
        CoUninitialize();
        return GPUVendor::CPU;
    }

    hres = CoSetProxyBlanket(pSvc, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, NULL, RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE, NULL, EOAC_NONE);
    if (FAILED(hres)) {
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        return GPUVendor::CPU;
    }

    IEnumWbemClassObject* pEnumerator = NULL;
    hres = pSvc->ExecQuery(bstr_t("WQL"), bstr_t("SELECT * FROM Win32_VideoController"), WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY, NULL, &pEnumerator);
    if (FAILED(hres)) {
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        return GPUVendor::CPU;
    }

    IWbemClassObject* pclsObj = NULL;
    ULONG uReturn = 0;
    
    std::vector<GPUVendor> detectedGPUs = { GPUVendor::CPU };

    while (pEnumerator) {
        HRESULT hr = pEnumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn);
        if (0 == uReturn) break;

        VARIANT vtProp;
        VariantInit(&vtProp);
        hr = pclsObj->Get(L"Name", 0, &vtProp, 0, 0);
        if (SUCCEEDED(hr)) {
            std::wstring gpuName = vtProp.bstrVal;
            std::wcout << "\tGPU : " << gpuName << std::endl;
            if (gpuName.find(L"NVIDIA") != std::wstring::npos) {
                
                detectedGPUs.push_back(GPUVendor::NVIDIA_CUDA);

            } else if (gpuName.find(L"AMD") != std::wstring::npos || gpuName.find(L"ATI") != std::wstring::npos) {

                std::vector<DeviceInfo> devices = checkHipBlasCompatibility();
                if (!devices.empty()) {
                    detectedGPUs.push_back(GPUVendor::AMD_ROCM);
                } else {
                    detectedGPUs.push_back(GPUVendor::AMD_DEFAULT);
                }
            } else {
                detectedGPUs.push_back(GPUVendor::DEFAULT_GPU);
            }
            VariantClear(&vtProp);
        }
        pclsObj->Release();
    }

    std::cout << std::endl;

    pSvc->Release();
    pLoc->Release();
    pEnumerator->Release();
    CoUninitialize();

    auto bestGPU = std::max_element(detectedGPUs.begin(), detectedGPUs.end(),
    [](GPUVendor a, GPUVendor b) {
        return getGPUPriority(a) < getGPUPriority(b);
    });

    return *bestGPU;
    
}

std::string getAMDGPUArch() {
    HRESULT hres;
    hres = CoInitializeEx(0, COINIT_MULTITHREADED);
    if (FAILED(hres)) return "Unknown";

    IWbemLocator* pLoc = NULL;
    hres = CoCreateInstance(CLSID_WbemLocator, 0, CLSCTX_INPROC_SERVER, IID_IWbemLocator, (LPVOID*)&pLoc);
    if (FAILED(hres)) {
        CoUninitialize();
        return "Unknown";
    }

    IWbemServices* pSvc = NULL;
    hres = pLoc->ConnectServer(_bstr_t(L"ROOT\\CIMV2"), NULL, NULL, 0, NULL, 0, 0, &pSvc);
    if (FAILED(hres)) {
        pLoc->Release();
        CoUninitialize();
        return "Unknown";
    }

    hres = CoSetProxyBlanket(pSvc, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, NULL, RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE, NULL, EOAC_NONE);
    if (FAILED(hres)) {
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        return "Unknown";
    }

    IEnumWbemClassObject* pEnumerator = NULL;
    hres = pSvc->ExecQuery(bstr_t("WQL"), bstr_t("SELECT * FROM Win32_VideoController WHERE AdapterCompatibility='Advanced Micro Devices, Inc.' OR AdapterCompatibility='AMD'"), WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY, NULL, &pEnumerator);
    if (FAILED(hres)) {
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        return "Unknown";
    }

    IWbemClassObject* pclsObj = NULL;
    ULONG uReturn = 0;
    std::string arch = "Unknown";

    while (pEnumerator) {
        HRESULT hr = pEnumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn);
        if (0 == uReturn) break;

        VARIANT vtProp;
        VariantInit(&vtProp);
        hr = pclsObj->Get(L"VideoProcessor", 0, &vtProp, 0, 0); //Name
        if (SUCCEEDED(hr)) {
            std::wstring processor = vtProp.bstrVal;
            std::wcout << L"VideoProcessor: " << processor << std::endl;
            // Extract architecture information from the processor
            size_t pos = processor.find(L"gfx");
            if (pos != std::wstring::npos) {
                arch = std::string(processor.begin() + pos, processor.begin() + pos + 6);
            }
            VariantClear(&vtProp);
        }
        pclsObj->Release();
    }

    pSvc->Release();
    pLoc->Release();
    pEnumerator->Release();
    CoUninitialize();

    return arch;
}


std::vector<DeviceInfo> checkHipBlasCompatibility() {
    std::vector<DeviceInfo> devices;
    int deviceCount;
    hipError_t hipStatus = hipGetDeviceCount(&deviceCount);

    if (hipStatus != hipSuccess) {
        std::cerr << "Failed to get device count. Error: " << hipGetErrorString(hipStatus) << std::endl;
        return devices;
    }

    for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t deviceProp;
        hipStatus = hipGetDeviceProperties(&deviceProp, i);

        if (hipStatus != hipSuccess) {
            std::cerr << "Failed to get device properties for device " << i << ". Error: " << hipGetErrorString(hipStatus) << std::endl;
            continue;
        }

        DeviceInfo info;
        info.deviceId = i;
        info.name = deviceProp.name;
        info.computeMajor = deviceProp.major;
        info.computeMinor = deviceProp.minor;

        // Check hipBLAS compatibility
        hipblasHandle_t handle;
        hipblasStatus_t blasStatus = hipblasCreate(&handle);

        info.hipblasCompatible = (blasStatus == HIPBLAS_STATUS_SUCCESS);

        if (info.hipblasCompatible) {
            hipblasDestroy(handle);
        }

        info.arch ="gfx"+std::string(deviceProp.gcnArchName);

        devices.push_back(info);
    }

    return devices;
}

std::string GetActiveWindowTitle() {
    char windowTitle[256];
    HWND hwnd = GetForegroundWindow(); // Get handle of the active window
    if (hwnd != NULL) {
        if (GetWindowTextA(hwnd, windowTitle, sizeof(windowTitle)) > 0) {
            return std::string(windowTitle);
        }
    }
    return "";
}

std::string trimTrailingSpaces(const std::string& str) {
    size_t end = str.find_last_not_of(' ');
    if (end != std::string::npos) {
        return str.substr(0, end + 1);
    }
    else {
        return "";  // If the string is all spaces, return an empty string
    }
}

// Callback function for EnumWindows
BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam) {
    WindowSearchData* data = reinterpret_cast<WindowSearchData*>(lParam);

    // Skip invisible windows
    if (!IsWindowVisible(hwnd)) {
        return TRUE; // Continue enumeration
    }

    // Get window title
    char windowTitle[256];
    int length = GetWindowTextA(hwnd, windowTitle, sizeof(windowTitle));

    if (length > 0) {
        std::string title(windowTitle);
        std::string cleanTitle = trimTrailingSpaces(title);

        // Compare with target window name (case-sensitive)
        if (cleanTitle == data->targetWindowName) {
            data->found = true;
            data->foundHandle = hwnd;
            return FALSE; // Stop enumeration
        }
    }

    return TRUE; // Continue enumeration
}

// Main function to check if window exists
bool isWindowOpen(const std::string& windowName) {
    WindowSearchData data;
    data.targetWindowName = windowName;
    data.found = false;
    data.foundHandle = nullptr;

    EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(&data));

    return data.found;
}

// Alternative version that also returns the window handle
HWND findWindow(const std::string& windowName) {
    WindowSearchData data;
    data.targetWindowName = windowName;
    data.found = false;
    data.foundHandle = nullptr;

    EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(&data));

    return data.foundHandle; // Returns nullptr if not found
}

// Case-insensitive version
bool isWindowOpenIgnoreCase(const std::string& windowName) {
    WindowSearchData data;
    data.targetWindowName = windowName;
    data.found = false;
    data.foundHandle = nullptr;

    // Convert target to lowercase for comparison
    std::string lowerTarget = windowName;
    std::transform(lowerTarget.begin(), lowerTarget.end(), lowerTarget.begin(),
        [](unsigned char c) { return std::tolower(c); });
    data.targetWindowName = lowerTarget;

    // Modified callback for case-insensitive comparison
    EnumWindows([](HWND hwnd, LPARAM lParam) -> BOOL {
        WindowSearchData* data = reinterpret_cast<WindowSearchData*>(lParam);

        if (!IsWindowVisible(hwnd)) {
            return TRUE;
        }

        char windowTitle[256];
        int length = GetWindowTextA(hwnd, windowTitle, sizeof(windowTitle));

        if (length > 0) {
            std::string title(windowTitle);
            std::string cleanTitle = trimTrailingSpaces(title);

            // Convert to lowercase for comparison
            std::transform(cleanTitle.begin(), cleanTitle.end(), cleanTitle.begin(),
                [](unsigned char c) { return std::tolower(c); });

            if (cleanTitle == data->targetWindowName) {
                data->found = true;
                data->foundHandle = hwnd;
                return FALSE;
            }
        }

        return TRUE;
        }, reinterpret_cast<LPARAM>(&data));

    return data.found;
}

void handleViGEmError(VIGEM_ERROR error) {
    switch(error) {
        case VIGEM_ERROR_BUS_NOT_FOUND:
            std::cerr << "The ViGEm bus was not found. Make sure ViGEmBus is installed correctly." << std::endl << "You can download the latest release here : https://github.com/nefarius/ViGEmBus/releases or use ViGEmBus_1.22.0_x64_x86_arm64.exe in the mod's folder." << std::endl;
            break;
        case VIGEM_ERROR_NO_FREE_SLOT:
            std::cerr << "All device slots are occupied. No new device can be spawned." << std::endl;
            break;
        case VIGEM_ERROR_INVALID_TARGET:
            std::cerr << "The target device is invalid." << std::endl;
            break;
        case VIGEM_ERROR_REMOVAL_FAILED:
            std::cerr << "Failed to remove the device." << std::endl;
            break;
        case VIGEM_ERROR_ALREADY_CONNECTED:
            std::cerr << "The device is already connected." << std::endl;
            break;
        case VIGEM_ERROR_TARGET_UNINITIALIZED:
            std::cerr << "The target device is not initialized." << std::endl;
            break;
        case VIGEM_ERROR_TARGET_NOT_PLUGGED_IN:
            std::cerr << "The target device is not plugged in." << std::endl;
            break;
        case VIGEM_ERROR_BUS_VERSION_MISMATCH:
            std::cerr << "Incompatible driver version. Please update ViGEmBus." << std::endl;
            break;
        case VIGEM_ERROR_BUS_ACCESS_FAILED:
            std::cerr << "Failed to open a handle to the bus driver. Check your permissions." << std::endl;
            break;
        case VIGEM_ERROR_CALLBACK_ALREADY_REGISTERED:
            std::cerr << "The callback is already registered." << std::endl;
            break;
        case VIGEM_ERROR_CALLBACK_NOT_FOUND:
            std::cerr << "The specified callback was not found." << std::endl;
            break;
        case VIGEM_ERROR_BUS_ALREADY_CONNECTED:
            std::cerr << "The bus is already connected." << std::endl;
            break;
        case VIGEM_ERROR_BUS_INVALID_HANDLE:
            std::cerr << "The bus handle is invalid." << std::endl;
            break;
        case VIGEM_ERROR_XUSB_USERINDEX_OUT_OF_RANGE:
            std::cerr << "The XUSB user index is out of range." << std::endl;
            break;
        case VIGEM_ERROR_INVALID_PARAMETER:
            std::cerr << "An invalid parameter was provided." << std::endl;
            break;
        case VIGEM_ERROR_NOT_SUPPORTED:
            std::cerr << "The API is not supported by the driver." << std::endl;
            break;
        case VIGEM_ERROR_WINAPI:
            std::cerr << "An unexpected Win32 API error occurred. Check GetLastError() for details." << std::endl;
            break;
        case VIGEM_ERROR_TIMED_OUT:
            std::cerr << "The operation timed out." << std::endl;
            break;
        default:
            std::cerr << "An unknown error occurred." << std::endl;
    }
}


std::wstring utf8ToWstring(const std::string& str) {
    if (str.empty()) {
        return std::wstring();
    }

    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstr[0], size_needed);
    return wstr;
}

std::wstring stringToWString(const std::string& str) {
    return utf8ToWstring(str);
}


bool isOpenVINOCompatible() {
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    
    // Check for SSE4.1 support (required by OpenVINO)
    bool hasSSE41 = (cpuInfo[2] & (1 << 19)) != 0;
    
    // Check for AVX2 support (recommended for better performance)
    __cpuid(cpuInfo, 7);
    bool hasAVX2 = (cpuInfo[1] & (1 << 5)) != 0;
    
    std::cout << "CPU Compatibility:" << std::endl;
    std::cout << "SSE4.1: " << (hasSSE41 ? "Yes" : "No") << std::endl;
    std::cout << "AVX2: " << (hasAVX2 ? "Yes" : "No") << std::endl;
    
    return hasSSE41; // Base requirement for OpenVINO
}


bool fileExists(const std::string& filepath) {
    std::ifstream file(filepath);
    return file.good();
}

bool fileExists_permissions(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        // Print the specific error
        std::cerr << "Failed to open file: " << filepath << std::endl;
        
        // Use strerror_s instead of strerror
        char errMsg[256];
        strerror_s(errMsg, sizeof(errMsg), errno);
        std::cerr << "Error: " << errMsg << std::endl;

        return false;
    }
    return true;
}

std::string wstringToString(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();

    // Determine required buffer size
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), NULL, 0, NULL, NULL);

    // Create the output string of required size
    std::string strTo(size_needed, 0);

    // Perform the actual conversion
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);

    return strTo;
}


bool launchDetachedProcess(const std::wstring& exePath, const std::wstring& windowTitle) {
    // If window title is provided, try to find existing window first
    if (!windowTitle.empty()) {
        HWND hwnd = FindWindowW(NULL, windowTitle.c_str());
        if (hwnd) {
            // Window found, bring it to front
            if (IsIconic(hwnd)) {
                ShowWindow(hwnd, SW_RESTORE);
            }
            SetForegroundWindow(hwnd);
            return true;
        }
    }

    STARTUPINFO si;
    PROCESS_INFORMATION pi;

    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // Create the process
    if (CreateProcessW(
        NULL,                           // Application name
        (LPWSTR)exePath.c_str(),        // Command line
        NULL,                           // Process handle not inheritable
        NULL,                           // Thread handle not inheritable
        FALSE,                          // Set handle inheritance
        CREATE_NEW_CONSOLE,             // Creation flags - detached
        NULL,                           // Use parent's environment block
        NULL,                           // Use parent's starting directory 
        &si,                           // Pointer to STARTUPINFO structure
        &pi                            // Pointer to PROCESS_INFORMATION structure
    )) {
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        return true;
    }
    return false;
}
