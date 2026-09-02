#pragma once

#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <Windows.h>
#include <iostream>
#include "VRSharedMemoryServer.h"
#include "SharedMemoryStructure.h"
#include "Config.h"

/**
 * @brief Alias for SharedMemory VRCommand for compatibility
 */
using SharedVRCommand = whisper_spell::SharedVRMemory::VRCommand;

/**
 * @brief VR Input Handler for Whisper-Spell (Shared Memory Version)
 * 
 * Provides the same interface as InputRedirector but sends commands
 * via shared memory to UEVR plugin instead of virtual gamepad simulation.
 * 
 * Key Improvements over Named Pipe version:
 * - NO ADMIN PRIVILEGES REQUIRED
 * - Lower latency (sub-microsecond for VR commands)
 * - Lock-free circular buffer for high performance
 * - Better memory efficiency
 * - Robust heartbeat-based connection detection
 * 
 * Features:
 * - Fire-and-forget messaging (drops when UEVR disconnected)  
 * - High-performance binary protocol (16-byte messages)
 * - Exact same interface as InputRedirector queue methods
 * - Atomic connection state tracking
 * - Zero-copy message transmission via shared memory
 */
class VRInputHandler {
public:
    /**
     * @brief Constructor with configuration reference
     */
    explicit VRInputHandler(const Config& cfg);

    /**
     * @brief Destructor ensures clean shutdown
     */
    ~VRInputHandler();

    // Disable copy semantics
    VRInputHandler(const VRInputHandler&) = delete;
    VRInputHandler& operator=(const VRInputHandler&) = delete;

    // Enable move semantics
    VRInputHandler(VRInputHandler&& other) noexcept;
    VRInputHandler& operator=(VRInputHandler&& other) noexcept;

    /**
     * @brief Initialize VR input handler and start shared memory server
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * @brief Shutdown VR input handler and stop shared memory server
     */
    void shutdown();

    /**
     * @brief Check if VR handler is ready and connected
     * @return true if UEVR plugin is connected
     */
    bool isReady() const { return isConnected_.load(); }

    // Same interface as InputRedirector for seamless integration
    void queueSingleButton(WORD button, int duration_ms = 50);
    void queueButtonCombination(const std::vector<WORD>& buttons, int duration_ms = 50);
    void queuePrincipalSpell(const std::vector<WORD>& buttons, int duration_ms = 50);
    void queueSimultaneousButtons(const std::vector<WORD>& buttons, int duration_ms = 50);

    /**
     * @brief Get current statistics
     */
    whisper_spell::PipeStatistics getStatistics() const;

private:
    /**
     * @brief Fill VRCommand button sequence from WORD vector
     */
    void fillButtonSequence(SharedVRCommand& cmd, const std::vector<WORD>& buttons) const;

    /**
     * @brief Convert single XUSB button to XInput value
     */
    uint16_t convertButton(WORD button) const;

    /**
     * @brief Send VR command via shared memory (fire-and-forget)
     */
    bool sendCommand(const SharedVRCommand& cmd);

    /**
     * @brief Handle incoming messages from UEVR plugin
     */
    void handleIncomingMessage(const std::string& message);

    /**
     * @brief Handle connection state changes
     */
    void handleConnectionChange(bool connected);

    // Configuration and state
    const Config& cfg_;
    std::atomic<bool> isConnected_{false};
    std::atomic<bool> isInitialized_{false};

    // Shared memory server for communication with UEVR plugin
    std::unique_ptr<whisper_spell::VRSharedMemoryServer> sharedMemoryServer_;

    // Lookup table for XUSB to XInput conversion
    static const std::unordered_map<WORD, WORD> XUSB_TO_XINPUT;
};