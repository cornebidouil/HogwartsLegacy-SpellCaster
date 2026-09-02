#pragma once

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <Windows.h>
#include <atomic>
#include <cstdint>
#include <string>
#include <memory>

// ============================================================================
// Shared Memory Structure for Spell Commands
// Must stay byte-identical to namespace SharedMemory in ue4ss-mod/SpellCasterMod/dllmain.cpp
// (layout and rules: docs/ipc-protocol.md)
// ============================================================================

namespace SpellCaster {
    // Spell command structure (32 bytes per command)
    struct SpellCommand {
        char spellName[28];     // Spell name (null-terminated, e.g., "Lumos", "Incendio")
        uint32_t flags;         // Reserved for future use (priority, etc.)
    };

    // Shared memory layout
    struct SharedSpellMemory {
        // Control structure (64 bytes, cache-line aligned)
        struct {
            std::atomic<uint32_t> writeIndex{0};        // Producer write position
            std::atomic<uint32_t> readIndex{0};         // Consumer read position
            std::atomic<uint32_t> producerHeartbeat{0}; // Producer alive signal
            std::atomic<uint32_t> consumerHeartbeat{0}; // Consumer alive signal (UE4SS mod)
            std::atomic<uint32_t> messagesDropped{0};   // Overflow counter
            std::atomic<uint32_t> totalMessages{0};     // Total messages sent
            uint32_t padding[10];                       // Pad to 64 bytes
        } control;

        // Spell command circular buffer (256 commands * 32 bytes = 8KB)
        static constexpr uint32_t COMMAND_BUFFER_SIZE = 256;
        static constexpr uint32_t COMMAND_BUFFER_MASK = COMMAND_BUFFER_SIZE - 1;

        SpellCommand commandBuffer[COMMAND_BUFFER_SIZE];

        // Status message area for feedback (4KB)
        struct {
            std::atomic<uint32_t> statusWriteIndex{0};
            std::atomic<uint32_t> statusReadIndex{0};
            char statusBuffer[4096 - 8];
        } status;

        // Helper: check if there's space in the buffer
        bool hasSpace() const {
            uint32_t write = control.writeIndex.load(std::memory_order_acquire);
            uint32_t read = control.readIndex.load(std::memory_order_acquire);
            return ((write + 1) & COMMAND_BUFFER_MASK) != (read & COMMAND_BUFFER_MASK);
        }
    };

    static constexpr const char* SHARED_MEMORY_NAME = "SpellCasterSharedMemory";
}

// ============================================================================
// SpellTransmitter Statistics
// ============================================================================

struct SpellTransmitterStats {
    uint32_t spellsSent;
    uint32_t spellsDropped;
    uint32_t reconnectCount;
    bool isConnected;
};

// ============================================================================
// SpellTransmitter Class
// ============================================================================

class SpellTransmitter {
public:
    SpellTransmitter();
    ~SpellTransmitter();

    // Disable copy
    SpellTransmitter(const SpellTransmitter&) = delete;
    SpellTransmitter& operator=(const SpellTransmitter&) = delete;

    // Enable move
    SpellTransmitter(SpellTransmitter&& other) noexcept;
    SpellTransmitter& operator=(SpellTransmitter&& other) noexcept;

    /**
     * @brief Initialize shared memory connection
     * @return true if successful
     */
    bool initialize();

    /**
     * @brief Shutdown and cleanup
     */
    void shutdown();

    /**
     * @brief Send spell name to UE4SS mod
     * @param spellName The spell name (e.g., "Lumos", "Incendio")
     * @return true if sent successfully, false if dropped
     */
    bool sendSpell(const std::string& spellName);

    /**
     * @brief Check if UE4SS mod is connected
     */
    bool isConnected() const { return isConnected_.load(); }

    /**
     * @brief Check if transmitter is initialized
     */
    bool isReady() const { return isInitialized_.load(); }

    /**
     * @brief Get statistics
     */
    SpellTransmitterStats getStatistics() const;

    /**
     * @brief Update heartbeat and check connection state
     * Should be called periodically (e.g., in main loop)
     */
    void update();

    /**
     * @brief Set debug logging enabled/disabled
     */
    void setDebugLogging(bool enabled) { debugLogging_ = enabled; }

private:
    bool createSharedMemory();
    void cleanupSharedMemory();
    bool writeSpellCommand(const std::string& normalizedSpell);
    void updateConnectionState();

    /**
     * @brief Normalize spell name for transmission
     * - Remove whitespace, punctuation
     * - Remove Whisper artifacts
     * - Capitalize words (e.g., "avada kedavra" -> "AvadaKedavra")
     */
    std::string normalizeSpellName(const std::string& rawSpell);

    // Shared memory handles
    HANDLE hFileMapping_{INVALID_HANDLE_VALUE};
    SpellCaster::SharedSpellMemory* sharedMemory_{nullptr};

    // Connection state
    std::atomic<bool> isConnected_{false};
    std::atomic<bool> isInitialized_{false};
    DWORD lastConnectionCheck_{0};
    uint32_t lastConsumerHeartbeat_{0};

    // Statistics
    std::atomic<uint32_t> spellsSent_{0};
    std::atomic<uint32_t> spellsDropped_{0};
    std::atomic<uint32_t> reconnectCount_{0};

    // Debug logging
    bool debugLogging_{false};

    // Constants
    static constexpr DWORD CONNECTION_CHECK_INTERVAL_MS = 500;
    static constexpr DWORD CONSUMER_TIMEOUT_MS = 2000;
};
