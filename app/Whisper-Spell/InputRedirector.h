#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <dinput.h>
#include <Xinput.h>
#include <queue>
#include <mutex>
#include <thread>
#include <chrono>
#include <optional>
#include <ViGEm/Client.h>

struct XINPUT_CAPABILITIES_EX {
    XINPUT_CAPABILITIES Capabilities;
    WORD vendorId;
    WORD productId;
    WORD revisionId;
    DWORD a4; // unknown
};

typedef DWORD(__stdcall* _XInputGetCapabilitiesEx)(DWORD a1, DWORD dwUserIndex, DWORD dwFlags, XINPUT_CAPABILITIES_EX* pCapabilities);

enum class InputType {
    DirectInput,
    XInput,
    VR
};

struct TimedReport {
    XUSB_REPORT report;
    std::chrono::milliseconds duration;
    std::chrono::steady_clock::time_point startTime;
    bool activated;
    
    TimedReport(XUSB_REPORT& report, std::chrono::milliseconds d)
        : report(report), duration(d), activated(false) {}

    void activate() {
        if (!activated) {
            startTime = std::chrono::steady_clock::now();
            activated = true;
        }
    }
    
    bool isExpired() const {
        if (!activated) return false;
        return (std::chrono::steady_clock::now() - startTime) >= duration;
    }
};

class InputRedirector {
private:
    const Config& cfg;
    std::queue<TimedReport> inputQueue;
    std::mutex queueMutex;
    std::atomic<bool> running{ false };
    std::thread redirectThread;
    std::optional<TimedReport> currentEvent;
    InputType inputType;

    LPDIRECTINPUT8 di;
    DIDEVICEINSTANCE deviceInstance;
    PVIGEM_CLIENT client;
    PVIGEM_TARGET gamepad;

    XUSB_REPORT convertDInputToXInput(const DIJOYSTATE2& joyState) {
        XUSB_REPORT report;
        XUSB_REPORT_INIT(&report);

        // Analog sticks - DirectInput range is 0 to 65535, XInput range is -32768 to 32767
        report.sThumbLX = static_cast<SHORT>((joyState.lX - 32768) * 32768 / 32768);
        report.sThumbLY = static_cast<SHORT>((32767 - joyState.lY) * 32768 / 32768);
        report.sThumbRX = static_cast<SHORT>((joyState.lZ - 32768) * 32768 / 32768);
        report.sThumbRY = static_cast<SHORT>((32767 - joyState.lRz) * 32768 / 32768);

        // Triggers - DirectInput range is 0 to 65535, XInput range is 0 to 255
        report.bLeftTrigger = static_cast<BYTE>((joyState.lRx >> 8) & 0xFF);
        report.bRightTrigger = static_cast<BYTE>((joyState.lRy >> 8) & 0xFF);

        // Buttons mapping using Config::gamepad
        for (int i = 0; i < 128; i++) {
            if (joyState.rgbButtons[i]) {
                auto it = cfg.gamepad.buttonMapping.find(i);
                if (it != cfg.gamepad.buttonMapping.end()) {
                    report.wButtons |= it->second;
                }
            }
        }

        // D-pad handling
        switch (joyState.rgdwPOV[0]) {
            case 0: report.wButtons |= XUSB_GAMEPAD_DPAD_UP; break;
            case 4500: report.wButtons |= (XUSB_GAMEPAD_DPAD_UP | XUSB_GAMEPAD_DPAD_RIGHT); break;
            case 9000: report.wButtons |= XUSB_GAMEPAD_DPAD_RIGHT; break;
            case 13500: report.wButtons |= (XUSB_GAMEPAD_DPAD_RIGHT | XUSB_GAMEPAD_DPAD_DOWN); break;
            case 18000: report.wButtons |= XUSB_GAMEPAD_DPAD_DOWN; break;
            case 22500: report.wButtons |= (XUSB_GAMEPAD_DPAD_DOWN | XUSB_GAMEPAD_DPAD_LEFT); break;
            case 27000: report.wButtons |= XUSB_GAMEPAD_DPAD_LEFT; break;
            case 31500: report.wButtons |= (XUSB_GAMEPAD_DPAD_LEFT | XUSB_GAMEPAD_DPAD_UP); break;
        }
        // if (joyState.rgdwPOV[0] == 0xFFFFFFFF) then nothing about the D-pad is specified

        return report;
    }

    DS4_REPORT convertXInputToDS4(const XUSB_REPORT& xInput) {
        DS4_REPORT ds4;
        DS4_REPORT_INIT(&ds4);

        // Buttons mapping
        if (xInput.wButtons & XUSB_GAMEPAD_A) ds4.wButtons |= DS4_BUTTON_CROSS;
        if (xInput.wButtons & XUSB_GAMEPAD_B) ds4.wButtons |= DS4_BUTTON_CIRCLE;
        if (xInput.wButtons & XUSB_GAMEPAD_X) ds4.wButtons |= DS4_BUTTON_SQUARE;
        if (xInput.wButtons & XUSB_GAMEPAD_Y) ds4.wButtons |= DS4_BUTTON_TRIANGLE;
        if (xInput.wButtons & XUSB_GAMEPAD_LEFT_SHOULDER) ds4.wButtons |= DS4_BUTTON_SHOULDER_LEFT;
        if (xInput.wButtons & XUSB_GAMEPAD_RIGHT_SHOULDER) ds4.wButtons |= DS4_BUTTON_SHOULDER_RIGHT;
        if (xInput.wButtons & XUSB_GAMEPAD_BACK) ds4.wButtons |= DS4_BUTTON_SHARE;
        if (xInput.wButtons & XUSB_GAMEPAD_START) ds4.wButtons |= DS4_BUTTON_OPTIONS;
        if (xInput.wButtons & XUSB_GAMEPAD_LEFT_THUMB) ds4.wButtons |= DS4_BUTTON_THUMB_LEFT;
        if (xInput.wButtons & XUSB_GAMEPAD_RIGHT_THUMB) ds4.wButtons |= DS4_BUTTON_THUMB_RIGHT;

        // D-pad
        if (xInput.wButtons & XUSB_GAMEPAD_DPAD_UP) ds4.wButtons |= DS4_BUTTON_DPAD_NORTH;
        if (xInput.wButtons & XUSB_GAMEPAD_DPAD_RIGHT) ds4.wButtons |= DS4_BUTTON_DPAD_EAST;
        if (xInput.wButtons & XUSB_GAMEPAD_DPAD_DOWN) ds4.wButtons |= DS4_BUTTON_DPAD_SOUTH;
        if (xInput.wButtons & XUSB_GAMEPAD_DPAD_LEFT) ds4.wButtons |= DS4_BUTTON_DPAD_WEST;

        // Analog sticks
        ds4.bThumbLX = static_cast<BYTE>((xInput.sThumbLX + 32768) >> 8);
        ds4.bThumbLY = static_cast<BYTE>((xInput.sThumbLY + 32768) >> 8);
        ds4.bThumbRX = static_cast<BYTE>((xInput.sThumbRX + 32768) >> 8);
        ds4.bThumbRY = static_cast<BYTE>((xInput.sThumbRY + 32768) >> 8);

        // Triggers
        ds4.bTriggerL = xInput.bLeftTrigger;
        ds4.bTriggerR = xInput.bRightTrigger;

        return ds4;
    }

    DS4_REPORT convertDInputToDS4(const DIJOYSTATE2& joyState) {
        DS4_REPORT ds4;
        DS4_REPORT_INIT(&ds4);

        // Analog sticks
        ds4.bThumbLX = static_cast<BYTE>(joyState.lX * 255 / 65535);
        ds4.bThumbLY = static_cast<BYTE>((65535 - joyState.lY) * 255 / 65535);
        ds4.bThumbRX = static_cast<BYTE>(joyState.lRx * 255 / 65535);
        ds4.bThumbRY = static_cast<BYTE>((65535 - joyState.lRy) * 255 / 65535);

        // Triggers
        ds4.bTriggerL = static_cast<BYTE>(joyState.lZ * 255 / 65535);
        ds4.bTriggerR = static_cast<BYTE>(joyState.lRz * 255 / 65535);

        // Buttons mapping
        if (joyState.rgbButtons[0]) ds4.wButtons |= DS4_BUTTON_SQUARE;
        if (joyState.rgbButtons[1]) ds4.wButtons |= DS4_BUTTON_CROSS;
        if (joyState.rgbButtons[2]) ds4.wButtons |= DS4_BUTTON_CIRCLE;
        if (joyState.rgbButtons[3]) ds4.wButtons |= DS4_BUTTON_TRIANGLE;
        if (joyState.rgbButtons[4]) ds4.wButtons |= DS4_BUTTON_SHOULDER_LEFT;
        if (joyState.rgbButtons[5]) ds4.wButtons |= DS4_BUTTON_SHOULDER_RIGHT;
        if (joyState.rgbButtons[10]) ds4.wButtons |= DS4_BUTTON_THUMB_LEFT;
        if (joyState.rgbButtons[11]) ds4.wButtons |= DS4_BUTTON_THUMB_RIGHT;
        if (joyState.rgbButtons[8]) ds4.wButtons |= DS4_BUTTON_SHARE;
        if (joyState.rgbButtons[9]) ds4.wButtons |= DS4_BUTTON_OPTIONS;

        // D-pad handling
        switch (joyState.rgdwPOV[0]) {
            case 0: ds4.wButtons |= DS4_BUTTON_DPAD_NORTH; break;
            case 4500: ds4.wButtons |= (DS4_BUTTON_DPAD_NORTH | DS4_BUTTON_DPAD_EAST); break;
            case 9000: ds4.wButtons |= DS4_BUTTON_DPAD_EAST; break;
            case 13500: ds4.wButtons |= (DS4_BUTTON_DPAD_EAST | DS4_BUTTON_DPAD_SOUTH); break;
            case 18000: ds4.wButtons |= DS4_BUTTON_DPAD_SOUTH; break;
            case 22500: ds4.wButtons |= (DS4_BUTTON_DPAD_SOUTH | DS4_BUTTON_DPAD_WEST); break;
            case 27000: ds4.wButtons |= DS4_BUTTON_DPAD_WEST; break;
            case 31500: ds4.wButtons |= (DS4_BUTTON_DPAD_WEST | DS4_BUTTON_DPAD_NORTH); break;
        }

        return ds4;
    }

    XUSB_REPORT combineReports(const XUSB_REPORT& controllerState, const XUSB_REPORT& externalEvent) {
        XUSB_REPORT finalReport = controllerState;
        
        finalReport.wButtons = controllerState.wButtons | externalEvent.wButtons;
        finalReport.bLeftTrigger = max(controllerState.bLeftTrigger, externalEvent.bLeftTrigger);
        finalReport.bRightTrigger = max(controllerState.bRightTrigger, externalEvent.bRightTrigger);
        finalReport.sThumbLX = externalEvent.sThumbLX != 0 ? externalEvent.sThumbLX : controllerState.sThumbLX;
        finalReport.sThumbLY = externalEvent.sThumbLY != 0 ? externalEvent.sThumbLY : controllerState.sThumbLY;
        finalReport.sThumbRX = externalEvent.sThumbRX != 0 ? externalEvent.sThumbRX : controllerState.sThumbRX;
        finalReport.sThumbRY = externalEvent.sThumbRY != 0 ? externalEvent.sThumbRY : controllerState.sThumbRY;
        
        return finalReport;
    }

    void redirectLoop() {
        if (inputType == InputType::XInput) {
            redirectXInputLoop();
        } else if (inputType == InputType::DirectInput) {
            redirectDirectInputLoop();
        } else {
            std::cerr << "ERROR: Not supported redirecting type ..." << std::endl;
#ifdef _WIN32
            system("PAUSE");
#endif // !_WIN32
            exit(-2);
        }
    }

    void redirectDirectInputLoop() {
        LPDIRECTINPUTDEVICE8 device;
        di->CreateDevice(deviceInstance.guidInstance, &device, nullptr);
        device->SetDataFormat(&c_dfDIJoystick2);
        device->SetCooperativeLevel(GetConsoleWindow(), DISCL_BACKGROUND | DISCL_NONEXCLUSIVE);

        DIJOYSTATE2 joyState;
        XUSB_REPORT controllerReport;
        XUSB_REPORT finalReport;

        while (running) {
            HRESULT hr = device->Poll();
            if (FAILED(hr)) {
                hr = device->Acquire();
                while (hr == DIERR_INPUTLOST) {
                    hr = device->Acquire();
                }
                continue;
            }

            if (SUCCEEDED(device->GetDeviceState(sizeof(DIJOYSTATE2), &joyState))) {
                XUSB_REPORT_INIT(&finalReport);

                controllerReport = convertDInputToXInput(joyState);

                bool hasExternalEvent = false;
                {
                    std::lock_guard<std::mutex> lock(queueMutex);

                    if (!currentEvent || currentEvent->isExpired()) {
                        if (!inputQueue.empty()) {
                            currentEvent = inputQueue.front();
                            inputQueue.pop();
                            currentEvent->activate();
                            hasExternalEvent = true;
                        } else {
                            currentEvent.reset();
                        }
                    } else {
                        hasExternalEvent = true;
                    }
                }

                finalReport = hasExternalEvent ? 
                    combineReports(controllerReport, currentEvent.value().report) : 
                    controllerReport;

                // Update virtual controller
                auto error = vigem_target_x360_update(client, gamepad, finalReport);
                if (!VIGEM_SUCCESS(error)) {
                    std::cout << "Error updating virtual controller: " << std::hex << error << "\n";
                }
            }
            Sleep(4);
        }
        device->Unacquire();
        device->Release();
    }


    void redirectXInputLoop() {
        XINPUT_STATE state;
        XUSB_REPORT controllerReport;
        XUSB_REPORT finalReport;
        DWORD userIndex = 0;

        // Find the correct XInput user index for this controller

        userIndex = findXInputSlot(deviceInstance);


        while(running) {
            if(XInputGetState(userIndex, &state) == ERROR_SUCCESS) {
                XUSB_REPORT_INIT(&finalReport);

                // Convert XInput state to XUSB report
                controllerReport.wButtons = state.Gamepad.wButtons;
                controllerReport.bLeftTrigger = state.Gamepad.bLeftTrigger;
                controllerReport.bRightTrigger = state.Gamepad.bRightTrigger;
                controllerReport.sThumbLX = state.Gamepad.sThumbLX;
                controllerReport.sThumbLY = state.Gamepad.sThumbLY;
                controllerReport.sThumbRX = state.Gamepad.sThumbRX;
                controllerReport.sThumbRY = state.Gamepad.sThumbRY;

                bool hasExternalEvent = false;
                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    if(!currentEvent || currentEvent->isExpired()) {
                        if(!inputQueue.empty()) {
                            currentEvent = inputQueue.front();
                            inputQueue.pop();
                            currentEvent->activate();
                            hasExternalEvent = true;
                        } else {
                            currentEvent.reset();
                        }
                    } else {
                        hasExternalEvent = true;
                    }
                }

                finalReport = hasExternalEvent ? 
                    combineReports(controllerReport, currentEvent.value().report) : 
                    controllerReport;


                auto error = vigem_target_x360_update(client, gamepad, finalReport);
                if (!VIGEM_SUCCESS(error)) {
                    std::cout << "Error updating virtual controller: " << std::hex << error << "\n";
                }
            }
            Sleep(4);
        }
    }

public:
    InputRedirector(const Config& config)
        : cfg(config)
        , inputType(InputType::VR) {
        
        // Only create virtual controller without redirection
        client = vigem_alloc();
        const auto retval = vigem_connect(client);
        if (!VIGEM_SUCCESS(retval)) {
            std::cout << "ViGEm Bus connection failed" << std::endl; 
            handleViGEmError(retval);
#ifdef _WIN32
            system("PAUSE");
#endif // !_WIN32
            exit(-2);
        }
        gamepad = vigem_target_x360_alloc();

        vigem_target_set_vid(gamepad, 0x1209);
        vigem_target_set_pid(gamepad, 0x0001);

        const auto pir = vigem_target_add(client, gamepad);
        if (!VIGEM_SUCCESS(pir)) {
            std::cout << "Failed to add virtual controller" << std::endl;
            handleViGEmError(pir);
#ifdef _WIN32
            system("PAUSE");
#endif // !_WIN32
            exit(-2);
        }
    }

    InputRedirector(LPDIRECTINPUT8 di, const DIDEVICEINSTANCE& device, const Config& config)
        : di(di)
        , deviceInstance(device)
        , inputType(detectInputType(device))
        , cfg(config) {

        client = vigem_alloc();
        const auto retval = vigem_connect(client);
        if (!VIGEM_SUCCESS(retval)) {
            std::cout << "ViGEm Bus connection failed" << std::endl;
            handleViGEmError(retval);
#ifdef _WIN32
            system("PAUSE");
#endif // !_WIN32
            exit(-2);
        }
        gamepad = vigem_target_x360_alloc();

        vigem_target_set_vid(gamepad, 0x1209);
        vigem_target_set_pid(gamepad, 0x0001);

        const auto pir = vigem_target_add(client, gamepad);
        if (!VIGEM_SUCCESS(pir)) {
            std::cout << "Failed to add virtual controller" << std::endl;
            handleViGEmError(pir);
#ifdef _WIN32
            system("PAUSE");
#endif // !_WIN32
            exit(-2);
        }

    }

    ~InputRedirector() {
        stop();
        vigem_target_remove(client, gamepad);
        vigem_target_free(gamepad);
        vigem_free(client);
    }

    void start() {
        running = true;
        redirectThread = std::thread(&InputRedirector::redirectLoop, this);
    }

    void stop() {
        running = false;
        if (redirectThread.joinable()) {
            redirectThread.join();
        }
    }

    void queueSingleButton(WORD button, int duration_ms = 50) {
        std::lock_guard<std::mutex> lock(queueMutex);
        
        XUSB_REPORT releaseReport;
        XUSB_REPORT_INIT(&releaseReport);

        releaseReport.wButtons |= button;

        inputQueue.push(TimedReport(releaseReport, std::chrono::milliseconds(duration_ms == 0 ? 50 : duration_ms)));
    }
    
    void queueButtonCombination(const std::vector<WORD>& buttons, int duration_ms = 50) {
        std::lock_guard<std::mutex> lock(queueMutex);

        XUSB_REPORT comboReport;
        XUSB_REPORT_INIT(&comboReport);
        
        for (size_t i = 0; i < buttons.size(); i++) {
            comboReport.wButtons |= buttons[i];  // Accumulate button presses
            inputQueue.push(TimedReport(comboReport, std::chrono::milliseconds(i < buttons.size() - 1 ? duration_ms : 50)));
        }
    }
    
    void queuePrincipalSpell(const std::vector<WORD>& buttons, int duration_ms = 50) {
        std::lock_guard<std::mutex> lock(queueMutex);
        
        // Trigger press event
        XUSB_REPORT triggerReport;
        XUSB_REPORT_INIT(&triggerReport);
        triggerReport.bRightTrigger = 255;  // Max trigger press
        inputQueue.push(TimedReport(triggerReport, std::chrono::milliseconds(50)));
        
        // Queue each button press
        for (const auto& button : buttons) {
            XUSB_REPORT spellReport;
            XUSB_REPORT_INIT(&spellReport);
            spellReport.bRightTrigger = 255;
            spellReport.wButtons |= button;
            inputQueue.push(TimedReport(spellReport, std::chrono::milliseconds(duration_ms)));
        }
    }
    
    void queueSimultaneousButtons(const std::vector<WORD>& buttons, int duration_ms = 50) {
        std::lock_guard<std::mutex> lock(queueMutex);

        XUSB_REPORT simReport;
        XUSB_REPORT_INIT(&simReport);

        for (const auto& button : buttons) {
            simReport.wButtons |= button;
        }
        inputQueue.push(TimedReport(simReport, std::chrono::milliseconds(duration_ms)));
    }
    
    
    static InputType detectInputType(const DIDEVICEINSTANCE& device) {
        // Extract VID/PID from DirectInput device
        WORD deviceVid = LOWORD(device.guidProduct.Data1);
        WORD devicePid = HIWORD(device.guidProduct.Data1);

        // Method 1: Try undocumented XInputGetCapabilitiesEx (if available)
        HMODULE moduleHandle = LoadLibrary(TEXT("XInput1_4.dll"));
        if (moduleHandle) {
            typedef DWORD(__stdcall* _XInputGetCapabilitiesEx)(DWORD, DWORD, DWORD, XINPUT_CAPABILITIES_EX*);
            _XInputGetCapabilitiesEx XInputGetCapabilitiesEx =
                (_XInputGetCapabilitiesEx)GetProcAddress(moduleHandle, (LPCSTR)108);

            if (XInputGetCapabilitiesEx) {
                for (DWORD i = 0; i < XUSER_MAX_COUNT; i++) {
                    XINPUT_CAPABILITIES_EX capsEx;
                    if (XInputGetCapabilitiesEx(1, i, 0, &capsEx) == ERROR_SUCCESS) {
                        if (capsEx.vendorId == deviceVid && capsEx.productId == devicePid) {
                            FreeLibrary(moduleHandle);
                            return InputType::XInput;
                        }
                    }
                }
            }
            FreeLibrary(moduleHandle);
        }

        // Method 2: Fallback - Check if any XInput slot responds
        // This works for controllers that emulate Xbox controllers perfectly
        for (DWORD i = 0; i < XUSER_MAX_COUNT; i++) {
            XINPUT_STATE state;
            if (XInputGetState(i, &state) == ERROR_SUCCESS) {
                // Controller is connected via XInput
                // For a more thorough match, you could store the controller's
                // unique identifier and cross-reference later

                // Simple heuristic: if we found an active XInput controller
                // and this DirectInput device has a gaming-related name, 
                // it's likely the same device in XInput mode
                std::wstring deviceName = device.tszProductName;
                if (deviceName.find(L"XBOX") != std::wstring::npos ||
                    deviceName.find(L"Xbox") != std::wstring::npos) { // Add known controller names
                    return InputType::XInput;
                }
            }
        }

        return InputType::DirectInput;
    }

    static DWORD findXInputSlot(const DIDEVICEINSTANCE& device) {
        WORD deviceVid = LOWORD(device.guidProduct.Data1);
        WORD devicePid = HIWORD(device.guidProduct.Data1);

        // Try exact VID/PID matching first
        HMODULE moduleHandle = LoadLibrary(TEXT("XInput1_4.dll"));
        if (moduleHandle) {
            _XInputGetCapabilitiesEx XInputGetCapabilitiesEx =
                (_XInputGetCapabilitiesEx)GetProcAddress(moduleHandle, (LPCSTR)108);

            if (XInputGetCapabilitiesEx) {
                for (DWORD i = 0; i < XUSER_MAX_COUNT; i++) {
                    XINPUT_CAPABILITIES_EX capsEx;
                    if (XInputGetCapabilitiesEx(1, i, 0, &capsEx) == ERROR_SUCCESS) {
                        if (capsEx.vendorId == deviceVid && capsEx.productId == devicePid) {
                            FreeLibrary(moduleHandle);
                            return i;
                        }
                    }
                }
            }
            FreeLibrary(moduleHandle);
        }

        // Fallback: return first available XInput slot
        for (DWORD i = 0; i < XUSER_MAX_COUNT; i++) {
            XINPUT_STATE state;
            if (XInputGetState(i, &state) == ERROR_SUCCESS) {
                return i;
            }
        }

        return XUSER_MAX_COUNT; // Not found
    }
};
