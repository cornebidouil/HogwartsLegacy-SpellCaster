#pragma once

#include "VRPipeInterface.h"
#include "SharedMemoryStructure.h"
#include <windows.h>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <string>

namespace whisper_spell {

/**
 * @brief Shared memory VR server for UEVR plugin communication
 * 
 * Drop-in replacement for VRPipeServer using shared memory instead of named pipes.
 * Maintains exact same interface and behavior but requires no admin privileges.
 * 
 * Features:
 * - Lock-free circular buffer for high-performance VR commands
 * - Fire-and-forget messaging (drops messages when UEVR not connected)
 * - Heartbeat-based connection detection
 * - Thread-safe operations using atomic operations
 * - Exception-safe RAII design
 * - Zero admin privileges required
 */
class VRSharedMemoryServer final : public IVRPipeServer {
public:
    /**
     * @brief Construct VR server with default configuration
     */
    VRSharedMemoryServer() noexcept;

    /**
     * @brief Destructor ensures clean shutdown
     */
    ~VRSharedMemoryServer() noexcept override;

    // Disable copy semantics
    VRSharedMemoryServer(const VRSharedMemoryServer&) = delete;
    VRSharedMemoryServer& operator=(const VRSharedMemoryServer&) = delete;

    // Enable move semantics
    VRSharedMemoryServer(VRSharedMemoryServer&& other) noexcept;
    VRSharedMemoryServer& operator=(VRSharedMemoryServer&& other) noexcept;

    // IVRPipeServer interface implementation
    bool start(
        const std::string& memoryName,
        MessageHandler messageHandler,
        ConnectionStateHandler connectionHandler = nullptr
    ) override;

    void stop() override;

    bool sendMessage(const std::string& message) override;

    bool isRunning() const override { return isRunning_.load(); }

    PipeStatistics getStatistics() const override;

private:
    /**
     * @brief Main VR server thread function
     */
    void sharedMemoryServerThread() noexcept;

    /**
     * @brief Create or open shared memory mapping
     */
    bool createSharedMemory(const std::string& memoryName) noexcept;

    /**
     * @brief Clean up shared memory resources
     */
    void cleanupSharedMemory() noexcept;

    /**
     * @brief Write VR command to shared buffer (lock-free)
     */
    bool writeCommand(const SharedVRMemory::VRCommand& cmd) noexcept;

    /**
     * @brief Write status message to shared buffer
     */
    bool writeStatusMessage(const std::string& message) noexcept;

    /**
     * @brief Read status messages from plugin (if any)
     */
    void readStatusMessages() noexcept;

    /**
     * @brief Check and update connection state based on heartbeat
     */
    void updateConnectionState() noexcept;

    /**
     * @brief Convert string message to VRCommand (for backward compatibility)
     */
    bool parseMessageToCommand(const std::string& message, SharedVRMemory::VRCommand& cmd) noexcept;

    // Thread management
    std::thread serverThread_;
    std::atomic<bool> shouldStop_{false};
    std::atomic<bool> isRunning_{false};
    
    // Shutdown event for clean termination
    HANDLE shutdownEvent_{INVALID_HANDLE_VALUE};

    // Shared memory state
    HANDLE hFileMapping_{INVALID_HANDLE_VALUE};
    SharedVRMemory* sharedMemory_{nullptr};
    std::string memoryName_;

    // Connection state
    std::atomic<bool> pluginConnected_{false};
    DWORD lastConnectionCheck_{0};
    uint32_t lastPluginHeartbeat_{0};

    // Callbacks (thread-safe access)
    mutable std::mutex callbackMutex_;
    MessageHandler messageHandler_;
    ConnectionStateHandler connectionHandler_;

    // Statistics (atomic for thread safety)
    std::atomic<std::uint32_t> messagesSent_{0};
    std::atomic<std::uint32_t> messagesDropped_{0};
    std::atomic<std::uint32_t> messagesReceived_{0};
    std::atomic<std::uint32_t> reconnectCount_{0};

    // Constants
    static constexpr DWORD CONNECTION_CHECK_INTERVAL_MS = 500;
    static constexpr DWORD PLUGIN_TIMEOUT_MS = 2000;
    static constexpr DWORD SHARED_MEMORY_SIZE = sizeof(SharedVRMemory);
};

} // namespace whisper_spell