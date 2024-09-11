#include "Tools.h"

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>


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
		throw std::runtime_error("Cannot open file");
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
    
    //GPUVendor result = GPUVendor::CPU;
    std::vector<GPUVendor> detectedGPUs = { GPUVendor::CPU };

    while (pEnumerator) {
        HRESULT hr = pEnumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn);
        if (0 == uReturn) break;

        VARIANT vtProp;
        VariantInit(&vtProp);
        //hr = pclsObj->Get(L"AdapterCompatibility", 0, &vtProp, 0, 0);
        hr = pclsObj->Get(L"Name", 0, &vtProp, 0, 0);
        if (SUCCEEDED(hr)) {
            std::wstring gpuName = vtProp.bstrVal;
            std::wcout << "\tGPU : " << gpuName << std::endl;
            //std::wstring adapterCompatibility = vtProp.bstrVal;
            //if (adapterCompatibility == L"NVIDIA") {
            if (gpuName.find(L"NVIDIA") != std::wstring::npos) {
                /*result = GPUVendor::CUDA;
                VariantClear(&vtProp);*/
                
                //result = GPUVendor::NVIDIA_CUDA;
                detectedGPUs.push_back(GPUVendor::NVIDIA_CUDA);

                /*if (isCudaCompatible())
                    detectedGPUs.push_back(GPUVendor::NVIDIA_CUDA);
                else
                    detectedGPUs.push_back(GPUVendor::NVIDIA_DEFAULT);*/
                // Check CUDA compatibility
                /*int cudaDeviceCount = 0;
                cudaError_t cudaStatus = cudaGetDeviceCount(&cudaDeviceCount);
                if (cudaStatus == cudaSuccess && cudaDeviceCount > 0) {
                    result = GPUVendor::NVIDIA_CUDA;
                } else {
                    result = GPUVendor::NVIDIA_DEFAULT;
                }*/
                
            //} else if (adapterCompatibility == L"Advanced Micro Devices, Inc." || adapterCompatibility == L"AMD") {
            } else if (gpuName.find(L"AMD") != std::wstring::npos || gpuName.find(L"ATI") != std::wstring::npos) {
                /*result = GPUVendor::HIP;
                VariantClear(&vtProp);*/

                std::vector<DeviceInfo> devices = checkHipBlasCompatibility();
                if (!devices.empty()) {
                    //result = GPUVendor::AMD_ROCM;
                    detectedGPUs.push_back(GPUVendor::AMD_ROCM);
                } else {
                    //result = GPUVendor::AMD_DEFAULT;
                    detectedGPUs.push_back(GPUVendor::AMD_DEFAULT);
                }
            } else {
                //result = GPUVendor::DEFAULT_GPU;
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
    
    //return result;
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

        //hipDeviceAttribute_t::hipDeviceAttributeCudaCompatibleEnd 
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