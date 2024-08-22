#include "Tools.h"

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