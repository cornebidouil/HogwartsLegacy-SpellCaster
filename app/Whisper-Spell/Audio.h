#pragma once

#include <mutex>
#include <vector>
#include <soxr.h>
#include <portaudio.h>

#include "Config.h"
#include "InteractiveMenu.h"

// Audio Resampler Class Integration
class AudioResampler {
private:
    static constexpr int TARGET_SAMPLE_RATE = 16000;
    static constexpr int MAX_CHANNELS = 2;

    soxr_t soxrResampler;
    std::vector<float> resampleBuffer;
    std::vector<float> tempMonoBuffer;
    std::mutex resamplerMutex;
    Config::AudioConfig* config;

public:
    AudioResampler(Config::AudioConfig* cfg) :
        soxrResampler(nullptr),
        config(cfg) {}

    ~AudioResampler() {
        cleanup();
    }

    bool initialize(int inputSampleRate, int inputChannels) {
        if (inputSampleRate == TARGET_SAMPLE_RATE) {
            // No resampling needed
            return true;
        }

        soxr_error_t error;
        soxr_io_spec_t ioSpec = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);

        soxr_quality_spec_t qualitySpec;
        switch (config->resamplingQuality) {
        case 0: qualitySpec = soxr_quality_spec(SOXR_QQ, 0); break;
        case 2: qualitySpec = soxr_quality_spec(SOXR_HQ, 0); break;
        default: qualitySpec = soxr_quality_spec(SOXR_MQ, 0); break;
        }

        soxrResampler = soxr_create(inputSampleRate, TARGET_SAMPLE_RATE, 1,
            &error, &ioSpec, &qualitySpec, nullptr);

        if (!soxrResampler) {
            std::cerr << "SoXR initialization error: " << soxr_strerror(error) << std::endl;
            return false;
        }

        // Allocate buffers
        size_t maxOutputFrames = (config->framesPerBuffer * TARGET_SAMPLE_RATE) / inputSampleRate + 64;
        resampleBuffer.resize(maxOutputFrames);
        tempMonoBuffer.resize(config->framesPerBuffer);

        std::cout << "SoXR resampler initialized: " << inputSampleRate << "Hz -> " << TARGET_SAMPLE_RATE << "Hz" << std::endl;
        return true;
    }

    size_t processAudio(const float* input, size_t inputFrames, int inputChannels, std::vector<float>& output) {
        std::lock_guard<std::mutex> lock(resamplerMutex);

        if (!soxrResampler) {
            // Direct conversion (no resampling needed)
            output.resize(inputFrames);
            if (inputChannels == 1) {
                std::copy(input, input + inputFrames, output.begin());
            }
            else {
                // Convert to mono by taking left channel
                for (size_t i = 0; i < inputFrames; i++) {
                    output[i] = input[i * inputChannels];
                }
            }
            return inputFrames;
        }

        // Convert to mono if needed
        const float* monoInput;
        if (inputChannels == 1) {
            monoInput = input;
        }
        else {
            // Convert multi-channel to mono
            for (size_t i = 0; i < inputFrames; i++) {
                tempMonoBuffer[i] = input[i * inputChannels]; // Take left channel
            }
            monoInput = tempMonoBuffer.data();
        }

        // Perform resampling
        size_t inputDone, outputDone;
        soxr_error_t error = soxr_process(soxrResampler,
            monoInput, inputFrames, &inputDone,
            resampleBuffer.data(), resampleBuffer.size(), &outputDone);

        if (error) {
            std::cerr << "SoXR processing error: " << soxr_strerror(error) << std::endl;
            return 0;
        }

        // Copy to output buffer
        output.resize(outputDone);
        std::copy(resampleBuffer.begin(), resampleBuffer.begin() + outputDone, output.begin());

        return outputDone;
    }

    void cleanup() {
        std::lock_guard<std::mutex> lock(resamplerMutex);
        if (soxrResampler) {
            soxr_delete(soxrResampler);
            soxrResampler = nullptr;
        }
    }
};




// Device Selection Functions - Forward Declarations
bool testDeviceCanOpen(PaDeviceIndex device);
bool testSampleRate(PaDeviceIndex device, int sampleRate);

// Device List Hashing & Validation Functions
int computeDeviceListHash() {
    int numDevices = Pa_GetDeviceCount();
    if (numDevices < 0) return 0;

    std::hash<std::string> hasher;
    size_t hashValue = 0;

    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo && deviceInfo->maxInputChannels > 0) {
            std::string deviceName = deviceInfo->name;
            const PaHostApiInfo* hostApiInfo = Pa_GetHostApiInfo(deviceInfo->hostApi);
            std::string apiName = hostApiInfo ? hostApiInfo->name : "";

            // Skip virtual devices (same filtering as selectInputDevice)
            if (apiName.find("ASIO") != std::string::npos ||
                deviceName.find("Sound Mapper") != std::string::npos ||
                deviceName.find("Primary Sound") != std::string::npos) {
                continue;
            }

            // FNV-1a style hash combination
            hashValue ^= hasher(deviceName + "|" + apiName) + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
        }
    }

    return static_cast<int>(hashValue);
}

struct DeviceValidationResult {
    bool isValid;
    PaDeviceIndex deviceIndex;
    std::string failureReason;
};

DeviceValidationResult validateSavedDevice(const std::string& savedName, const std::string& savedAPI) {
    DeviceValidationResult result = {false, paNoDevice, ""};

    int numDevices = Pa_GetDeviceCount();
    if (numDevices < 0) {
        result.failureReason = "Failed to enumerate devices";
        return result;
    }

    // Find device matching (name, API) pair
    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo->maxInputChannels > 0) {
            std::string deviceName = deviceInfo->name;
            const PaHostApiInfo* hostApiInfo = Pa_GetHostApiInfo(deviceInfo->hostApi);
            std::string apiName = hostApiInfo->name;

            if (deviceName == savedName && apiName == savedAPI) {
                // Found match - test if it can be opened
                if (testDeviceCanOpen(i)) {
                    result.isValid = true;
                    result.deviceIndex = i;
                    return result;
                } else {
                    result.failureReason = "Device exists but cannot be opened";
                    return result;
                }
            }
        }
    }

    result.failureReason = "Device not found (hardware removed or drivers changed)";
    return result;
}

// Device Selection Functions
bool testDeviceCanOpen(PaDeviceIndex device) {
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(device);
    const PaHostApiInfo* hostApiInfo = Pa_GetHostApiInfo(deviceInfo->hostApi);

    std::vector<int> testRates = {
        (int)deviceInfo->defaultSampleRate,
        44100, 48000, 16000, 22050
    };

    for (int sampleRate : testRates) {
        PaStreamParameters inputParams = {};
        inputParams.device = device;
        inputParams.channelCount = 1;
        inputParams.sampleFormat = paFloat32;
        inputParams.suggestedLatency = deviceInfo->defaultLowInputLatency;

        PaStream* testStream = nullptr;
        PaError err = Pa_OpenStream(&testStream, &inputParams, nullptr,
            sampleRate, 512, paClipOff, nullptr, nullptr);

        bool canOpen = (err == paNoError);
        if (testStream) {
            Pa_CloseStream(testStream);
        }

        if (canOpen) {
            return true;
        }
    }

    return false;
}

bool testSampleRate(PaDeviceIndex device, int sampleRate) {
    PaStreamParameters inputParams = {};
    inputParams.device = device;
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = Pa_GetDeviceInfo(device)->defaultLowInputLatency;

    PaError err = Pa_IsFormatSupported(&inputParams, nullptr, sampleRate);
    return err == paFormatIsSupported;
}

PaDeviceIndex selectInputDevice(Config::AudioConfig& config, bool forceSelection = false) {
    // === SAVED DEVICE VALIDATION ===
    if (!forceSelection && !config.selectedDeviceName.empty()) {
        std::cout << "\n=== Checking Saved Audio Device ===" << std::endl;
        std::cout << "Saved: " << config.selectedDeviceName << " (" << config.selectedDeviceAPI << ")" << std::endl;

        DeviceValidationResult validation = validateSavedDevice(config.selectedDeviceName, config.selectedDeviceAPI);

        if (validation.isValid) {
            std::cout << "✅ Device available and working" << std::endl;

            const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(validation.deviceIndex);

            // Test and select sample rate (same logic as original)
            std::vector<int> testRates = { 16000, 22050, 44100, 48000, 96000 };
            std::vector<int> supportedRates;

            for (int rate : testRates) {
                if (testSampleRate(validation.deviceIndex, rate)) {
                    supportedRates.push_back(rate);
                }
            }

            if (!supportedRates.empty()) {
                // Priority: 16000Hz first, then others
                std::vector<int> priorityOrder = {16000, 22050, 44100, 48000, 96000};
                int selectedRate = supportedRates[0];

                for (int priorityRate : priorityOrder) {
                    for (int supportedRate : supportedRates) {
                        if (supportedRate == priorityRate) {
                            selectedRate = supportedRate;
                            goto saved_rate_found;
                        }
                    }
                }
                saved_rate_found:
                config.inputSampleRate = selectedRate;
            } else {
                config.inputSampleRate = (int)deviceInfo->defaultSampleRate;
            }

            config.inputChannels = (std::min)(2, deviceInfo->maxInputChannels);

            std::cout << "Restored configuration:" << std::endl;
            std::cout << "  Sample rate: " << config.inputSampleRate << "Hz" << std::endl;
            std::cout << "  Channels: " << config.inputChannels << std::endl;

            return validation.deviceIndex;
        } else {
            std::cout << "⚠️  Device invalid: " << validation.failureReason << std::endl;
            std::cout << "Proceeding with device selection...\n" << std::endl;
        }
    }

    // Fall through to normal device selection UI...
    int numDevices = Pa_GetDeviceCount();
    if (numDevices < 0) {
        std::cerr << "Error getting device count" << std::endl;
        return paNoDevice;
    }

    std::cout << "\n=== Audio Device Selection ===" << std::endl;
    std::cout << "Testing devices for compatibility..." << std::endl;

    struct DeviceEntry {
        PaDeviceIndex index;
        std::string name;
        std::string apiName;
        int apiPriority;
        const PaDeviceInfo* info;
    };

    std::vector<DeviceEntry> validDevices;

    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo->maxInputChannels > 0) {
            std::string deviceName = deviceInfo->name;
            const PaHostApiInfo* hostApiInfo = Pa_GetHostApiInfo(deviceInfo->hostApi);
            std::string apiName = hostApiInfo->name;

            // Skip ASIO and obvious virtual devices
            if (apiName.find("ASIO") != std::string::npos ||
                deviceName.find("Sound Mapper") != std::string::npos ||
                deviceName.find("Primary Sound") != std::string::npos) {
                continue;
            }

            if (!testDeviceCanOpen(i)) {
                continue;
            }

            // API priority
            int apiPriority = 0;
            if (apiName.find("WASAPI") != std::string::npos) apiPriority = 3;
            else if (apiName.find("DirectSound") != std::string::npos) apiPriority = 2;
            else if (apiName.find("MME") != std::string::npos) apiPriority = 1;

            DeviceEntry entry;
            entry.index = i;
            entry.name = deviceName;
            entry.apiName = apiName;
            entry.apiPriority = apiPriority;
            entry.info = deviceInfo;

            validDevices.push_back(entry);
        }
    }

    if (validDevices.empty()) {
        std::cerr << "No suitable input devices found!" << std::endl;
        return paNoDevice;
    }

    // Display devices
    std::map<std::string, DeviceEntry> bestDevices;
    for (const auto& device : validDevices) {
        std::string key = device.name;
        auto existing = bestDevices.find(key);
        if (existing == bestDevices.end() || device.apiPriority > existing->second.apiPriority) {
            bestDevices[key] = device;
        }
    }

    std::vector<DeviceEntry> deviceList;
    for (const auto& pair : bestDevices) {
        deviceList.push_back(pair.second);
    }

    // Build interactive menu items
    std::vector<InteractiveMenu::MenuItem> deviceMenuItems;
    for (const auto& device : deviceList) {
        std::string title = device.name;
        std::string desc = std::to_string(device.info->maxInputChannels) + " channels, " +
                          std::to_string((int)device.info->defaultSampleRate) + "Hz | " +
                          device.apiName;
        deviceMenuItems.push_back(InteractiveMenu::MenuItem(title, desc));
    }

    InteractiveMenu deviceMenu("Select Audio Input Device:", deviceMenuItems, 0);
    int selection = deviceMenu.show();

    const auto& selectedDevice = deviceList[selection];

    // Test sample rates
    std::vector<int> testRates = { 16000, 22050, 44100, 48000, 96000 };
    std::vector<int> supportedRates;

    std::cout << "\nTesting sample rates for selected device..." << std::endl;
    for (int rate : testRates) {
        if (testSampleRate(selectedDevice.index, rate)) {
            supportedRates.push_back(rate);
            std::cout << "  " << rate << "Hz - Supported" << std::endl;
        }
    }

    if (!supportedRates.empty()) {
        if (config.audio_debug) {
            // DEBUG MODE: Interactive sample rate selection
            std::cout << "\n=== Debug Mode: Manual Sample Rate Selection ===" << std::endl;

            std::vector<InteractiveMenu::MenuItem> rateMenuItems;
            int defaultIndex = 0;
            for (size_t i = 0; i < supportedRates.size(); i++) {
                std::string title = std::to_string(supportedRates[i]) + "Hz";
                std::string desc = (supportedRates[i] == 16000) ?
                                  "Optimal - No resampling required" :
                                  "Requires resampling from " + std::to_string(supportedRates[i]) + "Hz to 16000Hz";
                rateMenuItems.push_back(InteractiveMenu::MenuItem(title, desc));

                if (supportedRates[i] == 16000) {
                    defaultIndex = i;
                }
            }

            InteractiveMenu rateMenu("Select Sample Rate:", rateMenuItems, defaultIndex);
            int rateSelection = rateMenu.show();
            config.inputSampleRate = supportedRates[rateSelection];
        } else {
            // PRODUCTION MODE: Automatic optimal sample rate selection
            std::cout << "\n🚀 Production mode: Auto-selecting optimal sample rate..." << std::endl;
            
            // Priority order: 16000Hz (no resampling) > others by performance
            std::vector<int> priorityOrder = {16000, 22050, 44100, 48000, 96000};
            
            int selectedRate = supportedRates[0]; // Fallback to first supported
            
            // Find highest priority supported rate
            for (int priorityRate : priorityOrder) {
                for (int supportedRate : supportedRates) {
                    if (supportedRate == priorityRate) {
                        selectedRate = supportedRate;
                        goto rate_found; // Break out of nested loops
                    }
                }
            }
            rate_found:
            
            config.inputSampleRate = selectedRate;
            
            std::cout << "   Selected: " << selectedRate << "Hz";
            if (selectedRate == 16000) {
                std::cout << " ✅ (Optimal - No resampling needed)";
            } else {
                std::cout << " 🔄 (Will resample to 16000Hz)";
            }
            std::cout << std::endl;
        }
    }
    else {
        config.inputSampleRate = (int)selectedDevice.info->defaultSampleRate;
    }

    config.inputChannels = (std::min)(2, selectedDevice.info->maxInputChannels);

    // Save selected device info for future sessions
    config.selectedDeviceName = selectedDevice.name;
    config.selectedDeviceAPI = selectedDevice.apiName;

    std::cout << "\n✅ Selected Configuration:" << std::endl;
    std::cout << "Mode: " << (config.audio_debug ? "🔧 Debug" : "🚀 Production") << std::endl;
    std::cout << "Device: " << selectedDevice.name << std::endl;
    std::cout << "API: " << selectedDevice.apiName << std::endl;
    std::cout << "Input: " << config.inputSampleRate << "Hz, " << config.inputChannels << " channels" << std::endl;
    std::cout << "Output: " << config.sampleRate << "Hz, 1 channel" << std::endl;

    return selectedDevice.index;
}