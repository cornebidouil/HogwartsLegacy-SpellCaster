#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <string>
#include <functional>
#include <cstdint>

namespace whisper_spell {

/**
 * @brief Statistics structure for monitoring VR pipe performance
 */
struct PipeStatistics {
    std::uint32_t messagesSent{0};
    std::uint32_t messagesDropped{0};
    std::uint32_t messagesReceived{0};
    std::uint32_t reconnectCount{0};
    bool isConnected{false};
};

/**
 * @brief Callback type for handling received messages from UEVR plugin
 * @param message The received message content
 */
using MessageHandler = std::function<void(const std::string& message)>;

/**
 * @brief Callback type for handling UEVR plugin connection state changes
 * @param connected true when UEVR plugin connects, false when disconnects
 */
using ConnectionStateHandler = std::function<void(bool connected)>;

/**
 * @brief Interface for VR pipe server functionality
 * 
 * This interface defines the contract for a fire-and-forget pipe server
 * specifically designed for UEVR plugin communication. Messages are dropped
 * when no UEVR plugin is connected, ensuring non-blocking operation.
 */
class IVRPipeServer {
public:
    virtual ~IVRPipeServer() = default;

    /**
     * @brief Start the VR pipe server for UEVR communication
     * @param pipeName Name of the pipe (without \\\\.\\pipe\\ prefix)
     * @param messageHandler Callback for handling received messages from UEVR
     * @param connectionHandler Optional callback for UEVR connection state changes
     * @return true if server started successfully, false otherwise
     */
    virtual bool start(
        const std::string& pipeName,
        MessageHandler messageHandler,
        ConnectionStateHandler connectionHandler = nullptr
    ) = 0;

    /**
     * @brief Stop the VR pipe server
     */
    virtual void stop() = 0;

    /**
     * @brief Send a VR command (fire-and-forget)
     * @param message Binary VR command to send to UEVR plugin
     * @return true if message was sent, false if dropped (no UEVR connected)
     */
    virtual bool sendMessage(const std::string& message) = 0;

    /**
     * @brief Check if VR pipe server is running
     * @return true if server is active
     */
    virtual bool isRunning() const = 0;

    /**
     * @brief Get current VR pipe statistics
     * @return Current pipe statistics for monitoring
     */
    virtual PipeStatistics getStatistics() const = 0;
};

} // namespace whisper_spell