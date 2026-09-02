#pragma once

#include <initguid.h>
#include <iostream>
#include <vector>
#include <string>
#include <SetupAPI.h>
#include <Windows.h>
#include <dinput.h>
#include <hidclass.h>
#include <initguid.h>
#include <iomanip>

extern "C" {
#include <hidsdi.h>
}

DEFINE_GUID(GUID_DEVCLASS_HIDCLASS, 0x745a17a0L, 0x74d3, 0x11d0, 0xb6, 0xfe, 0x00, 0xa0, 0xc9, 0x0f, 0x57, 0xda);

#define MAX_DEVICE_ID_LEN 256

#define MAGENTA "\033[45m"
#define RESET   "\033[0m"


inline std::vector<DIDEVICEINSTANCE> enumerateControllers(LPDIRECTINPUT8 di) {
    std::vector<DIDEVICEINSTANCE> controllers;

    di->EnumDevices(DI8DEVCLASS_GAMECTRL,
        [](LPCDIDEVICEINSTANCE deviceInstance, LPVOID pvRef) -> BOOL {
            auto controllers = static_cast<std::vector<DIDEVICEINSTANCE>*>(pvRef);
            controllers->push_back(*deviceInstance);
            std::cout << "\tController [" << controllers->size() - 1 << "]:" << std::endl;
            std::wcout << L"\t  Name: " << deviceInstance->tszProductName << std::endl;
            std::cout << "\t  VID: 0x" << std::hex << LOWORD(deviceInstance->guidProduct.Data1) << std::endl;
            std::cout << "\t  PID: 0x" << std::hex << HIWORD(deviceInstance->guidProduct.Data1) << std::dec << std::endl;
            std::cout << std::endl;
            return DIENUM_CONTINUE;
        },
        &controllers, DIEDFL_ATTACHEDONLY);

    return controllers;
}

inline bool isMatchingDevice(HDEVINFO deviceInfoSet, const SP_DEVINFO_DATA& deviceInfoData, WORD vid, WORD pid) {
    std::vector<wchar_t> hardwareId(1024);
    if (SetupDiGetDeviceRegistryProperty(deviceInfoSet,
        const_cast<PSP_DEVINFO_DATA>(&deviceInfoData),
        SPDRP_HARDWAREID,
        nullptr,
        reinterpret_cast<PBYTE>(hardwareId.data()),
        hardwareId.size() * sizeof(wchar_t),
        nullptr)) {

        std::wstring hwId(hardwareId.data());

        // Create VID/PID pattern to match
        wchar_t pattern[32];
        swprintf_s(pattern, L"VID_%04X&PID_%04X", vid, pid);

        return hwId.find(pattern) != std::wstring::npos;
    }
    return false;
}

inline std::wstring getDeviceInstancePath(HDEVINFO deviceInfoSet, const SP_DEVINFO_DATA& deviceInfoData) {
    std::vector<wchar_t> instanceId(1024);
    if (SetupDiGetDeviceInstanceId(deviceInfoSet,
        const_cast<PSP_DEVINFO_DATA>(&deviceInfoData),
        instanceId.data(),
        instanceId.size(),
        nullptr)) {
        return std::wstring(instanceId.data());
    }
    return L"";
}

inline void enumerateDeviceClass(const GUID& deviceClass, WORD vid, WORD pid, std::vector<std::wstring>& paths) {
    HDEVINFO deviceInfoSet = SetupDiGetClassDevs(&deviceClass, nullptr, nullptr,
        DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);

    if (deviceInfoSet == INVALID_HANDLE_VALUE) {
        // Try without DIGCF_DEVICEINTERFACE for some classes
        deviceInfoSet = SetupDiGetClassDevs(&deviceClass, nullptr, nullptr, DIGCF_PRESENT);
        if (deviceInfoSet == INVALID_HANDLE_VALUE) return;
    }

    SP_DEVICE_INTERFACE_DATA deviceInterfaceData = {};
    deviceInterfaceData.cbSize = sizeof(SP_DEVICE_INTERFACE_DATA);

    for (DWORD deviceIndex = 0;
        SetupDiEnumDeviceInterfaces(deviceInfoSet, nullptr, &deviceClass,
            deviceIndex, &deviceInterfaceData);
        deviceIndex++) {

        SP_DEVINFO_DATA deviceInfoData = {};
        deviceInfoData.cbSize = sizeof(SP_DEVINFO_DATA);

        // Get device info data
        DWORD requiredSize = 0;
        SetupDiGetDeviceInterfaceDetail(deviceInfoSet, &deviceInterfaceData, nullptr, 0, &requiredSize, &deviceInfoData);

        if (requiredSize > 0) {
            std::vector<BYTE> buffer(requiredSize);
            PSP_DEVICE_INTERFACE_DETAIL_DATA detailData =
                reinterpret_cast<PSP_DEVICE_INTERFACE_DETAIL_DATA>(buffer.data());
            detailData->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);

            if (SetupDiGetDeviceInterfaceDetail(deviceInfoSet, &deviceInterfaceData,
                detailData, requiredSize, nullptr, &deviceInfoData)) {

                if (isMatchingDevice(deviceInfoSet, deviceInfoData, vid, pid)) {
                    std::wstring path = getDeviceInstancePath(deviceInfoSet, deviceInfoData);
                    if (!path.empty() && std::find(paths.begin(), paths.end(), path) == paths.end()) {
                        paths.push_back(path);
                    }
                }
            }
        }
    }

    SetupDiDestroyDeviceInfoList(deviceInfoSet);
}

inline void enumerateBySetupClass(WORD vid, WORD pid, std::vector<std::wstring>& paths) {
    // Enumerate all devices and check their hardware IDs
    HDEVINFO deviceInfoSet = SetupDiGetClassDevs(nullptr, nullptr, nullptr,
        DIGCF_PRESENT | DIGCF_ALLCLASSES);

    if (deviceInfoSet == INVALID_HANDLE_VALUE) return;

    SP_DEVINFO_DATA deviceInfoData = {};
    deviceInfoData.cbSize = sizeof(SP_DEVINFO_DATA);

    for (DWORD deviceIndex = 0;
        SetupDiEnumDeviceInfo(deviceInfoSet, deviceIndex, &deviceInfoData);
        deviceIndex++) {

        if (isMatchingDevice(deviceInfoSet, deviceInfoData, vid, pid)) {
            std::wstring path = getDeviceInstancePath(deviceInfoSet, deviceInfoData);
            if (!path.empty() && std::find(paths.begin(), paths.end(), path) == paths.end()) {
                paths.push_back(path);
            }
        }
    }

    SetupDiDestroyDeviceInfoList(deviceInfoSet);
}

inline std::wstring findMatchingDeviceIds(WORD vid, WORD pid) {
    HDEVINFO deviceInfoSet = SetupDiGetClassDevs(
        &GUID_DEVINTERFACE_HID,
        nullptr,
        nullptr,
        DIGCF_PRESENT | DIGCF_DEVICEINTERFACE
    );

    SP_DEVINFO_DATA deviceInfoData = { sizeof(SP_DEVINFO_DATA) };
    DWORD deviceIndex = 0;

    std::wstringstream vidStr, pidStr;
    vidStr << std::uppercase << std::hex << std::setfill(L'0') << std::setw(4) << vid;
    pidStr << std::uppercase << std::hex << std::setfill(L'0') << std::setw(4) << pid;

    while (SetupDiEnumDeviceInfo(deviceInfoSet, deviceIndex++, &deviceInfoData)) {
        TCHAR deviceInstanceID[MAX_DEVICE_ID_LEN];
        if (SetupDiGetDeviceInstanceId(deviceInfoSet, &deviceInfoData, deviceInstanceID, MAX_DEVICE_ID_LEN, nullptr)) {
            std::wstring instanceId = deviceInstanceID;
            if (instanceId.find(L"VID_" + vidStr.str()) != std::wstring::npos &&
                instanceId.find(L"PID_" + pidStr.str()) != std::wstring::npos) {
                return instanceId;
            }
        }
    }

    SetupDiDestroyDeviceInfoList(deviceInfoSet);
    return L"";
}

inline std::vector<std::wstring> getAllDeviceInstancePaths(const DIDEVICEINSTANCE& deviceInstance) {
    std::vector<std::wstring> paths;
    WORD vid = LOWORD(deviceInstance.guidProduct.Data1);
    WORD pid = HIWORD(deviceInstance.guidProduct.Data1);

    // Get HID GUID using proper HID API
    GUID hidGuid;
    HidD_GetHidGuid(&hidGuid);

    // Define all possible device interface classes for gamepad access
    std::vector<GUID> deviceClasses = {
        hidGuid,                         // Standard HID interface (from HidD_GetHidGuid)
        GUID_DEVCLASS_HIDCLASS,          // HID device class
        {0x25dbce51, 0x6c8f, 0x4a72, {0x8a,0x6d,0xb5,0x4c,0x2b,0x4f,0xc8,0x35}}, // GUID_DEVINTERFACE_USB_DEVICE
        {0xa5dcbf10, 0x6530, 0x11d2, {0x90,0x1f,0x00,0xc0,0x4f,0xb9,0x51,0xed}}, // GUID_DEVINTERFACE_USB_HUB
        {0x4d36e97d, 0xe325, 0x11ce, {0xbf,0xc1,0x08,0x00,0x2b,0xe1,0x03,0x18}}   // GUID_DEVCLASS_SYSTEM (for Xbox drivers)
    };

    for (const auto& deviceClass : deviceClasses) {
        enumerateDeviceClass(deviceClass, vid, pid, paths);
    }

    // Also enumerate by device setup class for comprehensive coverage
    enumerateBySetupClass(vid, pid, paths);

    return paths;
}


inline bool checkAdministratorPrivileges() {
    BOOL isAdmin = FALSE;
    PSID adminGroup = nullptr;

    SID_IDENTIFIER_AUTHORITY ntAuthority = SECURITY_NT_AUTHORITY;
    if (AllocateAndInitializeSid(&ntAuthority, 2, SECURITY_BUILTIN_DOMAIN_RID,
        DOMAIN_ALIAS_RID_ADMINS, 0, 0, 0, 0, 0, 0, &adminGroup)) {
        CheckTokenMembership(nullptr, adminGroup, &isAdmin);
        FreeSid(adminGroup);
    }

    bool isAdministrator = (isAdmin == TRUE);
    return isAdministrator;
}


inline void CheckControllerInputs(LPDIRECTINPUTDEVICE8 device, const Config& cfg) {
    if (!device) return;
    
    // Prepare device
    device->Acquire();
    
    std::cout << "=== CONTROLLER INPUT CHECKER ===" << std::endl << std::endl;
    std::cout << "Press buttons to see their DirectInput IDs and XInput mappings." << std::endl;
    std::cout << "Press ESC key on keyboard to exit." << std::endl << std::endl;
    
    DIJOYSTATE2 currentState;
    DIJOYSTATE2 previousState;
    
    // Initialize previous state
    ZeroMemory(&previousState, sizeof(DIJOYSTATE2));
    
    // Map XInput button names for display
    std::map<WORD, std::string> buttonNames = {
        {XUSB_GAMEPAD_A, "A"},
        {XUSB_GAMEPAD_B, "B"},
        {XUSB_GAMEPAD_X, "X"},
        {XUSB_GAMEPAD_Y, "Y"},
        {XUSB_GAMEPAD_LEFT_SHOULDER, "LEFT_SHOULDER"},
        {XUSB_GAMEPAD_RIGHT_SHOULDER, "RIGHT_SHOULDER"},
        {XUSB_GAMEPAD_BACK, "BACK"},
        {XUSB_GAMEPAD_START, "START"},
        {XUSB_GAMEPAD_LEFT_THUMB, "LEFT_THUMB"},
        {XUSB_GAMEPAD_RIGHT_THUMB, "RIGHT_THUMB"},
        {XUSB_GAMEPAD_GUIDE, "GUIDE"},
        {XUSB_GAMEPAD_DPAD_UP, "DPAD_UP"},
        {XUSB_GAMEPAD_DPAD_DOWN, "DPAD_DOWN"},
        {XUSB_GAMEPAD_DPAD_LEFT, "DPAD_LEFT"},
        {XUSB_GAMEPAD_DPAD_RIGHT, "DPAD_RIGHT"}
    };
    
    bool running = true;
    while (running) {
        // Check if ESC is pressed on keyboard to exit
        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
            running = false;
            break;
        }
        
        device->Poll();
        if (SUCCEEDED(device->GetDeviceState(sizeof(DIJOYSTATE2), &currentState))) {
            
            // Check for button presses
            for (int i = 0; i < 128; i++) {
                if (currentState.rgbButtons[i] && !previousState.rgbButtons[i]) {
                    std::cout << "Button id: " << i;
                    
                    // Check if this button is mapped in active profile

                    auto it = cfg.gamepad.buttonMapping.find(i);
                    if (it != cfg.gamepad.buttonMapping.end()) {
                        std::cout << " (Mapped to XInput: ";
                        switch(it->second) {
                            case XUSB_GAMEPAD_X: std::cout << "X"; break;
                            case XUSB_GAMEPAD_A: std::cout << "A"; break;
                            case XUSB_GAMEPAD_B: std::cout << "B"; break;
                            case XUSB_GAMEPAD_Y: std::cout << "Y"; break;
                            case XUSB_GAMEPAD_LEFT_SHOULDER: std::cout << "LB"; break;
                            case XUSB_GAMEPAD_RIGHT_SHOULDER: std::cout << "RB"; break;
                            case XUSB_GAMEPAD_BACK: std::cout << "Back"; break;
                            case XUSB_GAMEPAD_START: std::cout << "Start"; break;
                            case XUSB_GAMEPAD_LEFT_THUMB: std::cout << "LS"; break;
                            case XUSB_GAMEPAD_RIGHT_THUMB: std::cout << "RS"; break;
                            case XUSB_GAMEPAD_GUIDE: std::cout << "Guide"; break;
                        }
                        std::cout << ")";
                    }
                    std::cout << " pressed" << std::endl;
                }
            }
            
            // Check for D-pad changes
            if (currentState.rgdwPOV[0] != previousState.rgdwPOV[0]) {
                if (currentState.rgdwPOV[0] == 0xFFFFFFFF) {
                    std::cout << "D-pad released" << std::endl;
                } else {
                    std::cout << "D-pad position: " << currentState.rgdwPOV[0] / 100 << " degrees" << std::endl;
                    
                    // Show mapped XInput buttons for this D-pad position
                    std::string mapping;
                    const DWORD pov = currentState.rgdwPOV[0];
                    if (pov <= 2250 || pov >= 34650) mapping += "DPAD_UP ";
                    if (pov >= 2250 && pov <= 11250) mapping += "DPAD_RIGHT ";
                    if (pov >= 11250 && pov <= 20250) mapping += "DPAD_DOWN ";
                    if (pov >= 20250 && pov <= 34650) mapping += "DPAD_LEFT";
                    
                    if (!mapping.empty()) {
                        std::cout << "  (Mapped to: " << mapping << ")" << std::endl;
                    }
                }
            }
            
            // Show axis changes
            const std::pair<std::string, LONG*> axes[] = {
                {"Left Stick X (lX)", &currentState.lX},
                {"Left Stick Y (lY)", &currentState.lY},
                {"Right Stick X (lZ)", &currentState.lZ},
                {"Right Stick Y (lRz)", &currentState.lRz},
                {"Left Trigger (lRx)", &currentState.lRx},
                {"Right Trigger (lRy)", &currentState.lRy}
            };
            
            const LONG* prevAxes[] = {
                &previousState.lX, &previousState.lY, &previousState.lZ,
                &previousState.lRz, &previousState.lRx, &previousState.lRy
            };
            
            const int threshold = 5000; // Show changes larger than this
            for (int i = 0; i < 6; i++) {
                if (abs(*axes[i].second - *prevAxes[i]) > threshold) {
                    std::cout << axes[i].first << ": " << *axes[i].second << std::endl;
                }
            }
            
            // Update previous state
            previousState = currentState;
        }
        
        Sleep(16); // Small delay to prevent console flooding
    }
    
    device->Unacquire();
    std::cout << "Input checker exited." << std::endl;
}

// Add this method to the InputRedirector class
inline void ShowInputChecker(LPDIRECTINPUT8 di, const DIDEVICEINSTANCE& deviceInstance, const Config& cfg) {
    std::cout << std::endl;
    std::cout << "Do you want to check controller inputs? (1 = Yes, 0 = No): ";
    int choice;
    std::cin >> choice;
    std::cout << std::endl;
    
    if (choice == 1) {
        ShellExecute(NULL, L"open", L"notepad.exe", L"config.ini", NULL, SW_SHOW);

        LPDIRECTINPUTDEVICE8 device;
        di->CreateDevice(deviceInstance.guidInstance, &device, nullptr);
        device->SetDataFormat(&c_dfDIJoystick2);
        device->SetCooperativeLevel(GetConsoleWindow(), DISCL_BACKGROUND | DISCL_NONEXCLUSIVE);
        
        CheckControllerInputs(device, cfg);
        
        device->Unacquire();
        device->Release();

        std::cout << std::endl << std::endl << MAGENTA << "Please save the 'config.ini' file in notepad before pressing ENTER." << RESET << std::endl;
        system("PAUSE");
    }
}
