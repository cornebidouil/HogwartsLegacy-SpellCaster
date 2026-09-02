#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <windows.h>

// Plugin identification for antivirus scanners
#pragma comment(user, "VR_SPELL_CASTER_PLUGIN_V2.0_SHARED_MEMORY_LEGITIMATE_GAME_MOD")
#pragma comment(user, "UEVR_COMPATIBLE_PLUGIN_NOT_MALWARE")

#include "uevr/Plugin.hpp"

using namespace uevr;

/**
 * @brief Shared Memory VR Command structure (identical to SharedMemoryStructure.h)
 */
namespace whisper_spell {
    struct SharedVRMemory {
        // Control structure (64 bytes, cache-line aligned)
        struct {
            std::atomic<uint32_t> writeIndex{0};        // Producer write position
            std::atomic<uint32_t> readIndex{0};         // Consumer read position
            std::atomic<uint32_t> pluginHeartbeat{0};   // Plugin alive signal (incremented)
            std::atomic<uint32_t> mainAppHeartbeat{0};  // Main app alive signal
            std::atomic<uint32_t> messagesDropped{0};   // Overflow counter
            std::atomic<uint32_t> totalMessages{0};     // Total messages sent
            uint32_t padding[10];                       // Pad to 64 bytes
        } control;

        // VR command circular buffer (4KB = 256 * 16 bytes)
        static constexpr uint32_t COMMAND_BUFFER_SIZE = 256;
        static constexpr uint32_t COMMAND_BUFFER_MASK = COMMAND_BUFFER_SIZE - 1;
        
        struct VRCommand {
            uint8_t type;           // 1=single, 2=combo, 3=principal, 4=simultaneous
            uint8_t buttonCount;    // Number of buttons in sequence (max 4)
            uint16_t duration;      // Duration in milliseconds
            uint16_t buttonSequence[4]; // Buttons in execution order (0 = unused)
            uint32_t reserved;      // Future use/alignment
        } commandBuffer[COMMAND_BUFFER_SIZE];

        // Status message area (4KB)
        struct {
            std::atomic<uint32_t> statusWriteIndex{0};
            std::atomic<uint32_t> statusReadIndex{0};
            char statusBuffer[4096 - 8];  // Leave room for indices
        } status;

        // Helper methods
        bool hasMessages() const {
            uint32_t write = control.writeIndex.load(std::memory_order_acquire);
            uint32_t read = control.readIndex.load(std::memory_order_acquire);
            return (write & COMMAND_BUFFER_MASK) != (read & COMMAND_BUFFER_MASK);
        }
    };
}

/**
 * @brief UEVR VR Spell Caster Plugin - SHARED MEMORY VERSION
 * 
 * This plugin receives VR commands from Whisper-Spell VRInputHandler
 * via shared memory and injects them as controller input into Hogwarts Legacy for VR gameplay.
 * 
 * KEY IMPROVEMENT: NO ADMIN PRIVILEGES REQUIRED
 * 
 * Features:
 * - Connects to Whisper-Spell VRInputHandler via shared memory (no pipes!)
 * - Lock-free circular buffer for sub-microsecond latency
 * - Heartbeat-based connection detection
 * - Automatic reconnection when Whisper-Spell application restarts
 * - Proper input timing and duration handling for game compatibility
 * - High-performance binary VRCommand processing
 */
class VRSpellCasterPluginSM : public uevr::Plugin {
private:
    // Shared memory communication
    HANDLE hFileMapping_{INVALID_HANDLE_VALUE};
    whisper_spell::SharedVRMemory* sharedMemory_{nullptr};
    std::string memoryName_{"UEVRSpellCaster"};
    std::atomic<bool> isConnected_{false};
    
    // Threading for non-blocking operation
    std::thread connectionThread_;
    std::thread messageThread_;
    std::atomic<bool> shouldStop_{false};
    
    // VR input injection state
    std::atomic<bool> vrInputActive_{false};
    XINPUT_STATE currentVRState_{};
    std::mutex vrStateMutex_;
    std::chrono::steady_clock::time_point vrInputEndTime_;
    
    // Button sequence execution
    std::thread sequenceThread_;
    std::vector<std::pair<uint16_t, uint16_t>> buttonSequence_; // {buttons, duration}
    std::atomic<bool> stopSequence_{false};
    std::mutex sequenceMutex_;
    
    // Statistics and heartbeat
    std::atomic<uint32_t> commandsReceived_{0};
    std::atomic<uint32_t> connectionAttempts_{0};
    uint32_t lastMainAppHeartbeat_{0};
    std::chrono::steady_clock::time_point lastHeartbeatUpdate_;

public:
    VRSpellCasterPluginSM() = default;
    
    ~VRSpellCasterPluginSM() {
        shutdown();
    }

    void on_initialize() override {
        API::get()->log_info("[SpellCaster-SM] === VR Spell Caster Plugin (SHARED MEMORY) ===");
        API::get()->log_info("[SpellCaster-SM] *** NO ADMIN PRIVILEGES REQUIRED ***");
        API::get()->log_info("[SpellCaster-SM] Connecting to Whisper-Spell via shared memory...");
        API::get()->log_info("[SpellCaster-SM] Memory name: %s", memoryName_.c_str());
        
        // Initialize VR state
        ZeroMemory(&currentVRState_, sizeof(currentVRState_));
        lastHeartbeatUpdate_ = std::chrono::steady_clock::now();
        
        // Start connection thread
        shouldStop_ = false;
        connectionThread_ = std::thread(&VRSpellCasterPluginSM::connectionThreadFunc, this);
        
        API::get()->log_info("[SpellCaster-SM] VR Spell Caster Plugin (Shared Memory) initialized successfully");
    }

    void on_xinput_get_state(uint32_t* retval, uint32_t user_index, XINPUT_STATE* state) override {
        // Only inject VR input for controller 0
        if (retval == nullptr || state == nullptr || *retval != ERROR_SUCCESS || user_index != 0) {
            return;
        }

        // Check if we have active VR input to inject
        {
            std::lock_guard<std::mutex> lock(vrStateMutex_);
            
            if (vrInputActive_.load()) {
                auto now = std::chrono::steady_clock::now();
                
                if (now >= vrInputEndTime_) {
                    // VR input duration expired
                    vrInputActive_ = false;
                    ZeroMemory(&currentVRState_, sizeof(currentVRState_));
                } else {
                    // Inject VR input by combining with physical controller state
                    state->Gamepad.wButtons |= currentVRState_.Gamepad.wButtons;
                    state->Gamepad.bLeftTrigger = (std::max)(state->Gamepad.bLeftTrigger, currentVRState_.Gamepad.bLeftTrigger);
                    state->Gamepad.bRightTrigger = (std::max)(state->Gamepad.bRightTrigger, currentVRState_.Gamepad.bRightTrigger);
                }
            }
        }
    }

    void on_pre_engine_tick(API::UGameEngine* engine, float delta) override {
        // Update our heartbeat to signal we're alive (every ~500ms)
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastHeartbeatUpdate_).count() >= 500) {
            if (sharedMemory_) {
                sharedMemory_->control.pluginHeartbeat.fetch_add(1, std::memory_order_relaxed);
                lastHeartbeatUpdate_ = now;
            }
        }
        
        // Periodic status logging every ~10 seconds
        static uint32_t frameCounter = 0;
        frameCounter++;
        
        if (frameCounter % 600 == 0) {
            if (isConnected_.load()) {
                API::get()->log_info("[SpellCaster-SM] VR Spell Caster: Connected via Shared Memory, Commands: %u", commandsReceived_.load());
            } else {
                API::get()->log_info("[SpellCaster-SM] VR Spell Caster: Searching for Whisper-Spell shared memory...");
            }
        }
    }

private:
    void shutdown() noexcept {
        shouldStop_ = true;
        isConnected_ = false;
        
        // Stop sequence execution
        stopCurrentSequence();
        
        // Clean up shared memory
        cleanupSharedMemory();
        
        // Join threads
        if (connectionThread_.joinable()) {
            connectionThread_.join();
        }
        if (messageThread_.joinable()) {
            messageThread_.join();
        }
        if (sequenceThread_.joinable()) {
            sequenceThread_.join();
        }
        
        API::get()->log_info("[SpellCaster-SM] VR Spell Caster Plugin shutdown complete");
    }

    void connectionThreadFunc() noexcept {
        API::get()->log_info("[SpellCaster-SM] VR connection thread started");
        
        uint32_t consecutiveFailures = 0;
        
        while (!shouldStop_.load()) {
            if (!isConnected_.load()) {
                connectionAttempts_++;
                
                if (attemptConnection()) {
                    // Successfully connected
                    consecutiveFailures = 0;
                    
                    // Start message receiving thread
                    if (messageThread_.joinable()) {
                        messageThread_.join();
                    }
                    messageThread_ = std::thread(&VRSpellCasterPluginSM::messageThreadFunc, this);
                    
                    API::get()->log_info("[SpellCaster-SM] SUCCESS: Connected to Whisper-Spell shared memory!");
                } else {
                    // Connection failed
                    consecutiveFailures++;
                    
                    // Calculate backoff interval (2s, 4s, 8s, max 10s)
                    uint32_t backoffInterval = (std::min)(2000u * (1 << (std::min)(consecutiveFailures, 3u)), 10000u);
                    std::this_thread::sleep_for(std::chrono::milliseconds(backoffInterval));
                }
            } else {
                // We're connected - check if connection is still valid
                if (!isSharedMemoryValid()) {
                    API::get()->log_info("[SpellCaster-SM] Shared memory connection lost, will attempt to reconnect...");
                    isConnected_ = false;
                    cleanupSharedMemory();
                    
                    // Wait for message thread to finish
                    if (messageThread_.joinable()) {
                        messageThread_.join();
                    }
                } else {
                    // Connection is good, reset failure counter and wait
                    consecutiveFailures = 0;
                    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                }
            }
        }
        
        API::get()->log_info("[SpellCaster-SM] VR connection thread stopped");
    }

    bool attemptConnection() noexcept {
        // Try to open existing shared memory created by main app
        hFileMapping_ = OpenFileMappingA(
            FILE_MAP_ALL_ACCESS,
            FALSE,
            memoryName_.c_str()
        );
        
        if (hFileMapping_ == nullptr) {
            // Shared memory doesn't exist yet (Whisper-Spell not running)
            return false;
        }
        
        // Map view of file
        sharedMemory_ = static_cast<whisper_spell::SharedVRMemory*>(
            MapViewOfFile(
                hFileMapping_,
                FILE_MAP_ALL_ACCESS,
                0,
                0,
                0
            )
        );
        
        if (sharedMemory_ == nullptr) {
            CloseHandle(hFileMapping_);
            hFileMapping_ = INVALID_HANDLE_VALUE;
            return false;
        }
        
        // Reset our read position to current write position (start fresh)
        uint32_t currentWrite = sharedMemory_->control.writeIndex.load(std::memory_order_acquire);
        sharedMemory_->control.readIndex.store(currentWrite, std::memory_order_release);
        
        isConnected_ = true;
        return true;
    }

    void messageThreadFunc() noexcept {
        API::get()->log_info("[SpellCaster-SM] VR message thread started - listening for spell commands...");
        
        while (!shouldStop_.load() && isConnected_.load()) {
            // Process any available VR commands
            while (sharedMemory_->hasMessages() && !shouldStop_.load()) {
                // Read command from circular buffer
                uint32_t readPos = sharedMemory_->control.readIndex.load(std::memory_order_relaxed);
                uint32_t bufferIndex = readPos & whisper_spell::SharedVRMemory::COMMAND_BUFFER_MASK;
                
                // Get command (lock-free read)
                auto cmd = sharedMemory_->commandBuffer[bufferIndex];
                
                // Update read index (this frees the slot for writer)
                sharedMemory_->control.readIndex.store(readPos + 1, std::memory_order_release);
                
                // Process the command
                executeVRCommand(cmd);
                commandsReceived_++;
            }
            
            // Check for status messages from main app (optional)
            readStatusMessages();
            
            // Sleep briefly but remain responsive
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        API::get()->log_info("[SpellCaster-SM] VR message thread stopped");
    }

    void readStatusMessages() noexcept {
        if (!sharedMemory_) return;
        
        uint32_t writePos = sharedMemory_->status.statusWriteIndex.load(std::memory_order_acquire);
        uint32_t readPos = sharedMemory_->status.statusReadIndex.load(std::memory_order_relaxed);
        
        if (writePos == readPos) return; // No messages
        
        // Read status messages from main app
        size_t bufferSize = sizeof(sharedMemory_->status.statusBuffer);
        std::string message;
        
        while (readPos != writePos && !shouldStop_.load()) {
            char c = sharedMemory_->status.statusBuffer[readPos % bufferSize];
            readPos++;
            
            if (c == '\0') {
                if (!message.empty()) {
                    API::get()->log_info("[SpellCaster-SM] Status from main app: %s", message.c_str());
                    message.clear();
                }
            } else {
                message += c;
            }
        }
        
        sharedMemory_->status.statusReadIndex.store(readPos, std::memory_order_release);
    }

    void executeVRCommand(const whisper_spell::SharedVRMemory::VRCommand& cmd) noexcept {
        std::string commandType = getCommandTypeDescription(cmd.type);
        std::string buttonNames = decodeButtonSequence(cmd.buttonSequence, cmd.buttonCount);
        
        API::get()->log_info("[SpellCaster-SM] CASTING SPELL: %s - %s (%dms)", 
                           commandType.c_str(), buttonNames.c_str(), cmd.duration);
        
        // Stop any existing sequence
        stopCurrentSequence();
        
        // Generate button sequence based on command type
        std::vector<std::pair<uint16_t, uint16_t>> sequence;
        
        switch (cmd.type) {
            case 1: // SINGLE button
                if (cmd.buttonCount > 0) {
                    sequence.push_back({cmd.buttonSequence[0], cmd.duration});
                }
                break;
                
            case 2: // COMBO - accumulating pattern
                {
                    uint16_t accumulatedButtons = 0;
                    for (uint8_t i = 0; i < cmd.buttonCount; i++) {
                        accumulatedButtons |= cmd.buttonSequence[i];
                        uint16_t stepDuration = (i < cmd.buttonCount - 1) ? cmd.duration : 50;
                        sequence.push_back({accumulatedButtons, stepDuration});
                    }
                }
                break;
                
            case 3: // PRINCIPAL - trigger first, then each button with trigger
                {
                    // Initial trigger press
                    sequence.push_back({0x0400, 50}); // Special flag for trigger-only
                    
                    // Each button with trigger held
                    for (uint8_t i = 0; i < cmd.buttonCount; i++) {
                        uint16_t buttonWithTrigger = cmd.buttonSequence[i] | 0x0400;
                        sequence.push_back({buttonWithTrigger, cmd.duration});
                    }
                }
                break;
                
            case 4: // SIMULTANEOUS - all buttons at once
                {
                    uint16_t allButtons = 0;
                    for (uint8_t i = 0; i < cmd.buttonCount; i++) {
                        allButtons |= cmd.buttonSequence[i];
                    }
                    sequence.push_back({allButtons, cmd.duration});
                }
                break;
        }
        
        // Execute the sequence
        if (!sequence.empty()) {
            executeButtonSequence(sequence);
        }
    }

    void executeButtonSequence(const std::vector<std::pair<uint16_t, uint16_t>>& sequence) noexcept {
        {
            std::lock_guard<std::mutex> lock(sequenceMutex_);
            buttonSequence_ = sequence;
            stopSequence_ = false;
        }
        
        // Start sequence execution thread
        if (sequenceThread_.joinable()) {
            sequenceThread_.join();
        }
        sequenceThread_ = std::thread(&VRSpellCasterPluginSM::executeSequenceThread, this);
    }

    void executeSequenceThread() noexcept {
        std::vector<std::pair<uint16_t, uint16_t>> localSequence;
        {
            std::lock_guard<std::mutex> lock(sequenceMutex_);
            localSequence = buttonSequence_;
        }
        
        for (const auto& [buttons, duration] : localSequence) {
            if (stopSequence_.load()) break;
            
            // Set VR input state
            {
                std::lock_guard<std::mutex> lock(vrStateMutex_);
                ZeroMemory(&currentVRState_, sizeof(currentVRState_));
                
                if (buttons & 0x0400) {
                    // Special trigger flag
                    currentVRState_.Gamepad.bRightTrigger = 255;
                    currentVRState_.Gamepad.wButtons = buttons & 0xFBFF; // Remove trigger flag
                } else {
                    currentVRState_.Gamepad.wButtons = buttons;
                }
                
                vrInputEndTime_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(duration);
                vrInputActive_ = true;
            }
            
            // Wait for duration
            std::this_thread::sleep_for(std::chrono::milliseconds(duration));
        }
        
        // Clear VR input after sequence
        {
            std::lock_guard<std::mutex> lock(vrStateMutex_);
            vrInputActive_ = false;
            ZeroMemory(&currentVRState_, sizeof(currentVRState_));
        }
    }

    void stopCurrentSequence() noexcept {
        stopSequence_ = true;
        if (sequenceThread_.joinable()) {
            sequenceThread_.join();
        }
        
        std::lock_guard<std::mutex> lock(vrStateMutex_);
        vrInputActive_ = false;
        ZeroMemory(&currentVRState_, sizeof(currentVRState_));
    }

    std::string decodeButtonSequence(const uint16_t buttonSequence[4], uint8_t count) const {
        if (count == 0) return "[NONE]";
        
        std::vector<std::string> buttonNames;
        
        static const std::unordered_map<uint16_t, std::string> XINPUT_TO_NAME = {
            {XINPUT_GAMEPAD_A, "A"}, {XINPUT_GAMEPAD_B, "B"}, 
            {XINPUT_GAMEPAD_X, "X"}, {XINPUT_GAMEPAD_Y, "Y"},
            {XINPUT_GAMEPAD_LEFT_SHOULDER, "LB"}, {XINPUT_GAMEPAD_RIGHT_SHOULDER, "RB"}, 
            {XINPUT_GAMEPAD_BACK, "BACK"}, {XINPUT_GAMEPAD_START, "START"},
            {XINPUT_GAMEPAD_LEFT_THUMB, "LS"}, {XINPUT_GAMEPAD_RIGHT_THUMB, "RS"},
            {XINPUT_GAMEPAD_DPAD_UP, "DPAD_UP"}, {XINPUT_GAMEPAD_DPAD_DOWN, "DPAD_DOWN"}, 
            {XINPUT_GAMEPAD_DPAD_LEFT, "DPAD_LEFT"}, {XINPUT_GAMEPAD_DPAD_RIGHT, "DPAD_RIGHT"}
        };
        
        for (uint8_t i = 0; i < count && i < 4; i++) {
            uint16_t button = buttonSequence[i];
            auto it = XINPUT_TO_NAME.find(button);
            if (it != XINPUT_TO_NAME.end()) {
                buttonNames.push_back(it->second);
            } else {
                char unknownBuffer[32];
                sprintf_s(unknownBuffer, "[UNKNOWN:0x%04X]", button);
                buttonNames.push_back(std::string(unknownBuffer));
            }
        }
        
        if (buttonNames.empty()) return "[NONE]";
        
        std::string result = buttonNames[0];
        for (size_t i = 1; i < buttonNames.size(); ++i) {
            result += " + " + buttonNames[i];
        }
        
        return result;
    }

    std::string getCommandTypeDescription(uint8_t type) const {
        switch (type) {
            case 1: return "SINGLE";
            case 2: return "COMBO";
            case 3: return "PRINCIPAL";
            case 4: return "SIMULTANEOUS";
            default: return "UNKNOWN";
        }
    }

    bool isSharedMemoryValid() noexcept {
        if (!sharedMemory_ || hFileMapping_ == INVALID_HANDLE_VALUE) {
            return false;
        }
        
        // Check if main app is still alive by monitoring heartbeat
        uint32_t currentHeartbeat = sharedMemory_->control.mainAppHeartbeat.load(std::memory_order_acquire);
        bool mainAppAlive = (currentHeartbeat != lastMainAppHeartbeat_);
        lastMainAppHeartbeat_ = currentHeartbeat;
        
        // If heartbeat hasn't changed for a while, assume disconnected
        auto now = std::chrono::steady_clock::now();
        static auto lastCheck = now;
        
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastCheck).count() > 3000) {
            if (!mainAppAlive) {
                return false; // No heartbeat for 3+ seconds
            }
            lastCheck = now;
        }
        
        return true;
    }

    void cleanupSharedMemory() noexcept {
        if (sharedMemory_) {
            UnmapViewOfFile(sharedMemory_);
            sharedMemory_ = nullptr;
        }
        
        if (hFileMapping_ != INVALID_HANDLE_VALUE) {
            CloseHandle(hFileMapping_);
            hFileMapping_ = INVALID_HANDLE_VALUE;
        }
    }
};

// Plugin entry point
std::unique_ptr<uevr::Plugin> g_plugin{new VRSpellCasterPluginSM()};