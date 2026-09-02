#pragma once

#include <cstdint>
#include <atomic>
#include <Windows.h>

namespace whisper_spell {

/**
 * @brief Shared memory layout for VR communication
 * 
 * Lock-free circular buffer design optimized for:
 * - High-frequency VR command transmission (16-byte VRCommand)
 * - Occasional text status messages (up to 1KB)
 * - Single producer (main app), single consumer (UEVR plugin)
 * - No admin privileges required
 */
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
    // Power of 2 size for efficient modulo operations
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

    // Total size: ~12KB (well under typical page size limits)
    
    /**
     * @brief Check if command buffer has space for new message
     */
    bool hasSpace() const {
        uint32_t write = control.writeIndex.load(std::memory_order_acquire);
        uint32_t read = control.readIndex.load(std::memory_order_acquire);
        return ((write + 1) & COMMAND_BUFFER_MASK) != (read & COMMAND_BUFFER_MASK);
    }
    
    /**
     * @brief Check if command buffer has messages to read
     */
    bool hasMessages() const {
        uint32_t write = control.writeIndex.load(std::memory_order_acquire);
        uint32_t read = control.readIndex.load(std::memory_order_acquire);
        return (write & COMMAND_BUFFER_MASK) != (read & COMMAND_BUFFER_MASK);
    }
    
    /**
     * @brief Get number of messages in buffer
     */
    uint32_t getMessageCount() const {
        uint32_t write = control.writeIndex.load(std::memory_order_acquire);
        uint32_t read = control.readIndex.load(std::memory_order_acquire);
        return (write - read) & COMMAND_BUFFER_MASK;
    }
    
    /**
     * @brief Check if plugin is connected (based on heartbeat)
     */
    bool isPluginConnected(uint32_t timeoutMs = 2000) const {
        // Simple heartbeat-based detection
        // In practice, plugin should increment heartbeat every ~500ms
        static uint32_t lastHeartbeat = 0;
        static DWORD lastCheckTime = 0;
        
        DWORD currentTime = GetTickCount();
        uint32_t currentHeartbeat = control.pluginHeartbeat.load(std::memory_order_acquire);
        
        if (currentTime - lastCheckTime > timeoutMs) {
            bool connected = (currentHeartbeat != lastHeartbeat);
            lastHeartbeat = currentHeartbeat;
            lastCheckTime = currentTime;
            return connected;
        }
        
        return true; // Assume connected within timeout window
    }
};

} // namespace whisper_spell