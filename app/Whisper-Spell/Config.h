#pragma once

#include <iostream>
#include <string>
#include <memory>
#include "inih/INIReader.h"
#include "Tools.h"
#include "SecretStore.h"

class Config {
private:
    std::unique_ptr<INIReader> reader;

public:
    ~Config() {
        SecretStore::wipe(crowdsourcing.password);
    }

    struct AudioConfig {
        int sampleRate = 16000;    // Target sample rate for processing
        int framesPerBuffer = 512;      // Buffer size for audio processing
        int selectedDevice = 0;

        // Resampling parameters
        int inputSampleRate = 44100;    // Actual input device sample rate
        int inputChannels = 1;        // Number of input channels
        bool enableResampling = true;     // Enable automatic resampling
        int resamplingQuality = 1;        // 0=Quick, 1=Medium, 2=High

        // Preference persistence
        std::string selectedDeviceName;   // Saved device name for session restore
        std::string selectedDeviceAPI;    // Saved device API (WASAPI, DirectSound, etc.)
        int deviceListHash = 0;           // Hash to detect device list changes

        // Debug flag for manual sample rate selection (hardcoded)
        static constexpr bool audio_debug = false;  // Set to true for debug mode
    } audio;

    struct ModelConfig {
        enum class TranscriptionEngine {
            WHISPER,
            MOONSHINE
        };
        TranscriptionEngine engine = TranscriptionEngine::WHISPER;  // Default to Whisper

        // Whisper model paths
        std::string whisperModel = "models/whisper/ggml-model.bin";

        // Moonshine model paths
        std::string moonshineModel = "models/moonshine/";

        // Common settings
        std::string keybindingFile = "keybinding.txt";
        std::string gameBindingPath;
    } models;

    struct TranscriptionConfig {
        // Whisper-specific settings
        float whisperConfidenceThreshold = 0.65f;

        // Moonshine-specific settings
        float moonshineDurationThreshold = 0.3f;
        float moonshineVadThreshold = 0.5f;
    } transcription;

    struct GamepadConfig {
        std::unordered_map<int, WORD> buttonMapping;
    } gamepad;

    struct VRConfig {
        bool enabled = true;
        std::string pipeName = "UEVRSpellCaster";
        uint32_t connectionTimeout = 5000;     // 5 seconds
        uint32_t reconnectionInterval = 2000;  // 2 seconds
        bool debugLogging = false;             // For development/troubleshooting
    } vr;

    struct SpellTransmitterConfig {
        bool enabled = true;                                     // Use direct spell transmission mode
        std::string sharedMemoryName = "SpellCasterSharedMemory"; // Must match UE4SS mod
        uint32_t connectionTimeoutMs = 2000;                     // Connection timeout in ms
        bool debugLogging = false;                               // Enable verbose logging
    } spellTransmitter;

    struct GameConfig {
        std::string gamePath;                                    // Path to game's Win64 folder (auto-detected if empty)
        bool autoInstallUE4SS = true;                            // Auto-install/update UE4SS and SpellCaster mod
    } game;

    struct CrowdsourcingConfig {
        bool enabled = false;                                    // Enable crowdsourcing feature
        bool consentGiven = false;                               // User consent to contribute voice data
        bool firstRunComplete = false;                           // Has user seen consent dialog

        // Authentication
        std::string authType = "anonymous";                      // "account" or "anonymous"
        std::string username;                                    // For account authentication
        std::string password;                                    // For account authentication
        std::string uuid;                                        // For anonymous authentication

        // Optional metadata
        std::string nationality;                                 // User's country
        std::string gender;                                      // "male", "female", "other", "prefer_not_to_say"

        // Server configuration
        std::string serverHost = "hogwartslegacyspellcaster.xyz"; // API server hostname
        bool autoSync = true;                                    // Automatically sync on startup
        bool debugLogging = false;                               // Enable verbose logging

        // Storage paths (auto-detected if empty)
        std::string pendingFolder;                               // Recordings waiting to sync
        std::string syncedFolder;                                // Successfully synced recordings
    } crowdsourcing;

    struct PreferencesConfig {
        bool firstRunComplete = false;                           // Has user completed initial setup
        bool autoLaunchGame = false;                             // Auto-launch game before listening
    } preferences;

    bool load(const std::string& filename = "config.ini") {
        if (!fileExists(filename)) {
            setDefaultGameBindingPath();
            setDefaultGamepadMapping();
            save(filename);
        }

        reader = std::make_unique<INIReader>(filename);
        if (reader->ParseError() < 0) return false;

        // Load model selection
        std::string engineStr = reader->Get("MODELS", "engine", "whisper");
        if (engineStr == "whisper") {
            models.engine = ModelConfig::TranscriptionEngine::WHISPER;
        } else if (engineStr == "moonshine") {
            models.engine = ModelConfig::TranscriptionEngine::MOONSHINE;
        } else {
            std::cerr << "Unknown transcription engine: " << engineStr << ", defaulting to whisper" << std::endl;
            models.engine = ModelConfig::TranscriptionEngine::WHISPER;
        }

        // Load model paths
        models.whisperModel = reader->Get("MODELS", "whisper_model", "models/whisper/ggml-model.bin");
        models.moonshineModel = reader->Get("MODELS", "moonshine_model", "models/moonshine/");

        // Load transcription settings
        transcription.whisperConfidenceThreshold = reader->GetReal("TRANSCRIPTION", "whisper_confidence_threshold", 0.65f);
        transcription.moonshineDurationThreshold = reader->GetReal("TRANSCRIPTION", "moonshine_duration_threshold", 0.3f);
        transcription.moonshineVadThreshold = reader->GetReal("TRANSCRIPTION", "moonshine_vad_threshold", 0.5f);

        // Load settings
        models.gameBindingPath = reader->Get("PATH", "game_binding_path", "");

        // Load gamepad button mappings
        gamepad.buttonMapping.clear();
        gamepad.buttonMapping[reader->GetInteger("DIRECT_INPUT_CONTROLLER", "XBOX_X", 0)] = XUSB_GAMEPAD_X;
        gamepad.buttonMapping[reader->GetInteger("DIRECT_INPUT_CONTROLLER", "XBOX_A", 1)] = XUSB_GAMEPAD_A;
        gamepad.buttonMapping[reader->GetInteger("DIRECT_INPUT_CONTROLLER", "XBOX_B", 2)] = XUSB_GAMEPAD_B;
        gamepad.buttonMapping[reader->GetInteger("DIRECT_INPUT_CONTROLLER", "XBOX_Y", 3)] = XUSB_GAMEPAD_Y;
        gamepad.buttonMapping[reader->GetInteger("DIRECT_INPUT_CONTROLLER", "XBOX_LB", 4)] = XUSB_GAMEPAD_LEFT_SHOULDER;
        gamepad.buttonMapping[reader->GetInteger("DIRECT_INPUT_CONTROLLER", "XBOX_RB", 5)] = XUSB_GAMEPAD_RIGHT_SHOULDER;
        gamepad.buttonMapping[reader->GetInteger("DIRECT_INPUT_CONTROLLER", "XBOX_BACK", 8)] = XUSB_GAMEPAD_BACK;
        gamepad.buttonMapping[reader->GetInteger("DIRECT_INPUT_CONTROLLER", "XBOX_START", 9)] = XUSB_GAMEPAD_START;
        gamepad.buttonMapping[reader->GetInteger("DIRECT_INPUT_CONTROLLER", "XBOX_LS", 10)] = XUSB_GAMEPAD_LEFT_THUMB;
        gamepad.buttonMapping[reader->GetInteger("DIRECT_INPUT_CONTROLLER", "XBOX_RS", 11)] = XUSB_GAMEPAD_RIGHT_THUMB;
        gamepad.buttonMapping[reader->GetInteger("DIRECT_INPUT_CONTROLLER", "XBOX_GUIDE", 12)] = XUSB_GAMEPAD_GUIDE;
        
        // Load VR settings
        vr.debugLogging         = reader->GetBoolean("VR", "debug_logging", false);

        // Load SpellTransmitter settings
        spellTransmitter.enabled = reader->GetBoolean("SPELL_TRANSMITTER", "enabled", true);
        spellTransmitter.debugLogging = reader->GetBoolean("SPELL_TRANSMITTER", "debug_logging", false);

        // Load Game settings
        game.gamePath = reader->Get("GAME", "game_path", "");
        game.autoInstallUE4SS = reader->GetBoolean("GAME", "auto_install_ue4ss", true);

        // Load Crowdsourcing settings
        crowdsourcing.enabled = reader->GetBoolean("CROWDSOURCING", "enabled", false);
        crowdsourcing.consentGiven = reader->GetBoolean("CROWDSOURCING", "consent_given", false);
        crowdsourcing.firstRunComplete = reader->GetBoolean("CROWDSOURCING", "first_run_complete", false);
        crowdsourcing.authType = reader->Get("CROWDSOURCING", "auth_type", "anonymous");
        crowdsourcing.username = reader->Get("CROWDSOURCING", "username", "");

        // The account password is kept encrypted for the current Windows user
        // (password_protected, see SecretStore.h). Files written by earlier
        // versions may still carry it in clear text under "password": read it
        // once and rewrite the file in the protected form.
        // The protected value is a few hundred characters long, hence the
        // INI_MAX_LINE=4096 definition in the project settings.
        bool rewriteProtectedPassword = false;
        SecretStore::wipe(crowdsourcing.password);
        {
            std::string protectedPassword = reader->Get("CROWDSOURCING", "password_protected", "");
            std::string legacyPassword = reader->Get("CROWDSOURCING", "password", "");
            if (!protectedPassword.empty()) {
                bool decrypted = SecretStore::unprotect(protectedPassword, crowdsourcing.password);
                if (!decrypted && crowdsourcing.enabled && crowdsourcing.consentGiven) {
                    std::cerr << "[Crowdsourcing] The stored account password could not be decrypted "
                                 "(it is tied to the Windows account that saved it). "
                                 "You will be asked for it again." << std::endl;
                }
            } else if (!legacyPassword.empty()) {
                crowdsourcing.password = legacyPassword;
                rewriteProtectedPassword = true;
            }
            SecretStore::wipe(legacyPassword);
        }

        crowdsourcing.uuid = reader->Get("CROWDSOURCING", "uuid", "");
        crowdsourcing.nationality = reader->Get("CROWDSOURCING", "nationality", "");
        crowdsourcing.gender = reader->Get("CROWDSOURCING", "gender", "");
        crowdsourcing.serverHost = reader->Get("CROWDSOURCING", "server_host", "hogwartslegacyspellcaster.xyz");
        crowdsourcing.autoSync = reader->GetBoolean("CROWDSOURCING", "auto_sync", true);
        crowdsourcing.debugLogging = reader->GetBoolean("CROWDSOURCING", "debug_logging", false);
        crowdsourcing.pendingFolder = reader->Get("CROWDSOURCING", "pending_folder", "");
        crowdsourcing.syncedFolder = reader->Get("CROWDSOURCING", "synced_folder", "");

        // Set default crowdsourcing paths if empty
        if (crowdsourcing.pendingFolder.empty() || crowdsourcing.syncedFolder.empty()) {
            setDefaultCrowdsourcingPaths();
        }

        // Load audio preferences
        audio.selectedDeviceName = reader->Get("AUDIO", "selected_device_name", "");
        audio.selectedDeviceAPI = reader->Get("AUDIO", "selected_device_api", "");
        audio.deviceListHash = reader->GetInteger("AUDIO", "device_list_hash", 0);

        // Load preferences
        preferences.firstRunComplete = reader->GetBoolean("PREFERENCES", "first_run_complete", false);
        preferences.autoLaunchGame = reader->GetBoolean("PREFERENCES", "auto_launch_game", false);

        if (rewriteProtectedPassword) {
            // Replace the clear-text password on disk with its protected form.
            save(filename);
        }

        return true;
    }

    bool save(const std::string& filename = "config.ini") {
        std::ofstream configFile(filename);
        if (!configFile.is_open()) {
            std::cerr << "Failed to open config file: "<< filename <<" for writing" << std::endl;
#ifdef _WIN32
            system("PAUSE");
#endif
            return false;
        }

        configFile << "[MODELS]\n"
                   << "; Choose transcription engine: whisper or moonshine\n"
                   << "engine=" << (models.engine == ModelConfig::TranscriptionEngine::WHISPER ? "whisper" : "moonshine") << "\n\n"
                   << "; Whisper model path (used when engine=whisper)\n"
                   << "whisper_model=" << models.whisperModel << "\n\n"
                   << "; Moonshine model path (used when engine=moonshine)\n"
                   << "moonshine_model=" << models.moonshineModel << "\n\n";

        configFile << "[TRANSCRIPTION]\n"
                   << "; Whisper confidence threshold (0.0-1.0, higher = stricter)\n"
                   << "whisper_confidence_threshold=" << transcription.whisperConfidenceThreshold << "\n\n"
                   << "; Moonshine duration threshold (seconds, minimum speech duration)\n"
                   << "moonshine_duration_threshold=" << transcription.moonshineDurationThreshold << "\n\n"
                   << "; TEN VAD threshold (0.0-1.0, used by both models)\n"
                   << "moonshine_vad_threshold=" << transcription.moonshineVadThreshold << "\n\n";

        configFile << "[PATH]\n"
                   << "game_binding_path=" << models.gameBindingPath << "\n";

        
        const std::vector<std::pair<WORD, std::string>> buttonOrder = {
            {XUSB_GAMEPAD_X, "XBOX_X"},
            {XUSB_GAMEPAD_A, "XBOX_A"},
            {XUSB_GAMEPAD_B, "XBOX_B"},
            {XUSB_GAMEPAD_Y, "XBOX_Y"},
            {XUSB_GAMEPAD_LEFT_SHOULDER, "XBOX_LB"},
            {XUSB_GAMEPAD_RIGHT_SHOULDER, "XBOX_RB"},
            {XUSB_GAMEPAD_BACK, "XBOX_BACK"},
            {XUSB_GAMEPAD_START, "XBOX_START"},
            {XUSB_GAMEPAD_LEFT_THUMB, "XBOX_LS"},
            {XUSB_GAMEPAD_RIGHT_THUMB, "XBOX_RS"},
            {XUSB_GAMEPAD_GUIDE, "XBOX_GUIDE"}
        };
        
        configFile << "\n[DIRECT_INPUT_CONTROLLER]\n";
        for (const auto& [xinput, buttonName] : buttonOrder) {
            for (const auto& [dinput, mapping] : gamepad.buttonMapping) {
                if (mapping == xinput) {
                    configFile << buttonName << "=" << dinput << "\n";
                    break;
                }
            }
        }

        configFile << "\n[AUDIO]\n"
                   << "; Saved audio device preferences\n"
                   << "selected_device_name=" << audio.selectedDeviceName << "\n"
                   << "selected_device_api=" << audio.selectedDeviceAPI << "\n"
                   << "device_list_hash=" << audio.deviceListHash << "\n";

        configFile << "\n[PREFERENCES]\n"
                   << "; First-run configuration status\n"
                   << "first_run_complete=" << (preferences.firstRunComplete ? "true" : "false") << "\n"
                   << "; Auto-launch game before listening\n"
                   << "auto_launch_game=" << (preferences.autoLaunchGame ? "true" : "false") << "\n";

        configFile << "\n[VR]\n"
                   << "debug_logging=" << (vr.debugLogging ? "true" : "false") << "\n";

        configFile << "\n[SPELL_TRANSMITTER]\n"
                   << "enabled=" << (spellTransmitter.enabled ? "true" : "false") << "\n"
                   << "debug_logging=" << (spellTransmitter.debugLogging ? "true" : "false") << "\n";

        configFile << "\n[GAME]\n"
                   << "; Path to Hogwarts Legacy Win64 folder (leave empty for auto-detection)\n"
                   << "game_path=" << game.gamePath << "\n"
                   << "; Auto-install/update UE4SS and SpellCaster mod on startup\n"
                   << "auto_install_ue4ss=" << (game.autoInstallUE4SS ? "true" : "false") << "\n";

        configFile << "\n[CROWDSOURCING]\n"
                   << "; Enable voice data crowdsourcing feature\n"
                   << "enabled=" << (crowdsourcing.enabled ? "true" : "false") << "\n"
                   << "consent_given=" << (crowdsourcing.consentGiven ? "true" : "false") << "\n"
                   << "first_run_complete=" << (crowdsourcing.firstRunComplete ? "true" : "false") << "\n"
                   << "\n; Authentication (choose 'account' or 'anonymous')\n"
                   << "auth_type=" << crowdsourcing.authType << "\n";

        if (crowdsourcing.authType == "account") {
            configFile << "username=" << crowdsourcing.username << "\n";

            // Never written in clear text: DPAPI-encrypted for the current Windows user.
            std::string protectedPassword = SecretStore::protect(crowdsourcing.password);
            if (!crowdsourcing.password.empty() && protectedPassword.empty()) {
                std::cerr << "[Crowdsourcing] Could not protect the account password; it was not saved." << std::endl;
            }
            configFile << "; Password encrypted for the current Windows user (delete this line to be asked again)\n"
                       << "password_protected=" << protectedPassword << "\n";
        } else {
            configFile << "uuid=" << crowdsourcing.uuid << "\n";
        }

        configFile << "\n; Optional metadata\n"
                   << "nationality=" << crowdsourcing.nationality << "\n"
                   << "gender=" << crowdsourcing.gender << "\n"
                   << "\n; Server configuration\n"
                   << "server_host=" << crowdsourcing.serverHost << "\n"
                   << "auto_sync=" << (crowdsourcing.autoSync ? "true" : "false") << "\n"
                   << "debug_logging=" << (crowdsourcing.debugLogging ? "true" : "false") << "\n"
                   << "\n; Storage paths (auto-detected if empty)\n"
                   << "pending_folder=" << crowdsourcing.pendingFolder << "\n"
                   << "synced_folder=" << crowdsourcing.syncedFolder << "\n";

        configFile.close();

        return true;
    }

    void setDefaultGameBindingPath() {
        char* userProfile;
        size_t profile_size;

        // Get the value of the "USERPROFILE" environment variable
        if (_dupenv_s(&userProfile, &profile_size, "USERPROFILE") != 0 || userProfile == nullptr) {
            std::cerr << "Failed to get USERPROFILE environment variable" << std::endl;
#ifdef _WIN32
            system("PAUSE");
#endif
            exit(-1);
        }

        models.gameBindingPath = std::string(userProfile) + "\\AppData\\Local\\Hogwarts Legacy\\Saved\\Config\\WindowsNoEditor\\Input.ini";

        if (userProfile != nullptr) {
            free(userProfile);
        }

    }

    void setDefaultGamepadMapping() {
        gamepad.buttonMapping = {
            {0, XUSB_GAMEPAD_X},
            {1, XUSB_GAMEPAD_A},
            {2, XUSB_GAMEPAD_B},
            {3, XUSB_GAMEPAD_Y},
            {4, XUSB_GAMEPAD_LEFT_SHOULDER},
            {5, XUSB_GAMEPAD_RIGHT_SHOULDER},
            {8, XUSB_GAMEPAD_BACK},
            {9, XUSB_GAMEPAD_START},
            {10, XUSB_GAMEPAD_LEFT_THUMB},
            {11, XUSB_GAMEPAD_RIGHT_THUMB},
            {12, XUSB_GAMEPAD_GUIDE}
        };
    }

    void setDefaultCrowdsourcingPaths() {
        char* appData;
        size_t appdata_size;

        // Get the value of the "APPDATA" environment variable
        if (_dupenv_s(&appData, &appdata_size, "APPDATA") != 0 || appData == nullptr) {
            // Fallback to LOCALAPPDATA if APPDATA not available
            if (_dupenv_s(&appData, &appdata_size, "LOCALAPPDATA") != 0 || appData == nullptr) {
                std::cerr << "Failed to get APPDATA environment variable" << std::endl;
                return;
            }
        }

        std::string baseFolder = std::string(appData) + "\\SpellCaster";
        crowdsourcing.pendingFolder = baseFolder + "\\pending";
        crowdsourcing.syncedFolder = baseFolder + "\\synced";

        if (appData != nullptr) {
            free(appData);
        }
    }
};
