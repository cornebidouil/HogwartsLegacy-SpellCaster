#include "VRSharedMemoryServer.h"
#include <iostream>
#include <chrono>
#include <sstream>

namespace whisper_spell {

VRSharedMemoryServer::VRSharedMemoryServer() noexcept {
    // Create manual reset event for shutdown signaling
    shutdownEvent_ = CreateEvent(nullptr, TRUE, FALSE, nullptr);
    if (shutdownEvent_ == nullptr) {
        std::cerr << "[VR-SM] Failed to create shutdown event" << std::endl;
    }
}

VRSharedMemoryServer::~VRSharedMemoryServer() noexcept {
    stop();
    
    // Clean up shutdown event
    if (shutdownEvent_ != INVALID_HANDLE_VALUE) {
        CloseHandle(shutdownEvent_);
        shutdownEvent_ = INVALID_HANDLE_VALUE;
    }
}

VRSharedMemoryServer::VRSharedMemoryServer(VRSharedMemoryServer&& other) noexcept {
    std::lock_guard<std::mutex> lock(other.callbackMutex_);
    
    serverThread_ = std::move(other.serverThread_);
    shouldStop_ = other.shouldStop_.load();
    isRunning_ = other.isRunning_.load();
    hFileMapping_ = other.hFileMapping_;
    sharedMemory_ = other.sharedMemory_;
    memoryName_ = std::move(other.memoryName_);
    pluginConnected_ = other.pluginConnected_.load();
    lastConnectionCheck_ = other.lastConnectionCheck_;
    lastPluginHeartbeat_ = other.lastPluginHeartbeat_;
    messageHandler_ = std::move(other.messageHandler_);
    connectionHandler_ = std::move(other.connectionHandler_);
    messagesSent_ = other.messagesSent_.load();
    messagesDropped_ = other.messagesDropped_.load();
    messagesReceived_ = other.messagesReceived_.load();
    reconnectCount_ = other.reconnectCount_.load();
    shutdownEvent_ = other.shutdownEvent_;
    
    // Reset other's state
    other.hFileMapping_ = INVALID_HANDLE_VALUE;
    other.sharedMemory_ = nullptr;
    other.shouldStop_ = true;
    other.isRunning_ = false;
    other.pluginConnected_ = false;
    other.shutdownEvent_ = INVALID_HANDLE_VALUE;
}

VRSharedMemoryServer& VRSharedMemoryServer::operator=(VRSharedMemoryServer&& other) noexcept {
    if (this != &other) {
        stop(); // Clean up current state
        
        std::lock_guard<std::mutex> lock1(callbackMutex_);
        std::lock_guard<std::mutex> lock2(other.callbackMutex_);
        
        serverThread_ = std::move(other.serverThread_);
        shouldStop_ = other.shouldStop_.load();
        isRunning_ = other.isRunning_.load();
        hFileMapping_ = other.hFileMapping_;
        sharedMemory_ = other.sharedMemory_;
        memoryName_ = std::move(other.memoryName_);
        pluginConnected_ = other.pluginConnected_.load();
        lastConnectionCheck_ = other.lastConnectionCheck_;
        lastPluginHeartbeat_ = other.lastPluginHeartbeat_;
        messageHandler_ = std::move(other.messageHandler_);
        connectionHandler_ = std::move(other.connectionHandler_);
        messagesSent_ = other.messagesSent_.load();
        messagesDropped_ = other.messagesDropped_.load();
        messagesReceived_ = other.messagesReceived_.load();
        reconnectCount_ = other.reconnectCount_.load();
        shutdownEvent_ = other.shutdownEvent_;
        
        // Reset other's state
        other.hFileMapping_ = INVALID_HANDLE_VALUE;
        other.sharedMemory_ = nullptr;
        other.shouldStop_ = true;
        other.isRunning_ = false;
        other.pluginConnected_ = false;
        other.shutdownEvent_ = INVALID_HANDLE_VALUE;
    }
    return *this;
}

bool VRSharedMemoryServer::start(
    const std::string& memoryName,
    MessageHandler messageHandler,
    ConnectionStateHandler connectionHandler
) {
    if (isRunning_.load()) {
        return false; // Already running
    }

    {
        std::lock_guard<std::mutex> lock(callbackMutex_);
        memoryName_ = memoryName;
        messageHandler_ = std::move(messageHandler);
        connectionHandler_ = std::move(connectionHandler);
    }

    // Create shared memory
    if (!createSharedMemory(memoryName)) {
        std::cerr << "[VR-SM] Failed to create shared memory: " << memoryName << std::endl;
        return false;
    }

    shouldStop_ = false;
    serverThread_ = std::thread(&VRSharedMemoryServer::sharedMemoryServerThread, this);
    
    // Wait for server to start
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    return isRunning_.load();
}

void VRSharedMemoryServer::stop() {
    shouldStop_ = true;
    
    // Signal shutdown event
    if (shutdownEvent_ != INVALID_HANDLE_VALUE) {
        SetEvent(shutdownEvent_);
    }
    
    if (serverThread_.joinable()) {
        std::cout << "[VR-SM] Shutting down shared memory server thread..." << std::endl;
        
        // Join with timeout
        const int JOIN_TIMEOUT_MS = 3000;
        auto joinStart = std::chrono::steady_clock::now();
        
        bool joined = false;
        while (!joined) {
            auto elapsed = std::chrono::steady_clock::now() - joinStart;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >= JOIN_TIMEOUT_MS) {
                std::cout << "[VR-SM] Thread join timeout, detaching thread" << std::endl;
                serverThread_.detach();
                joined = true;
                break;
            }
            
            if (serverThread_.joinable()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                if (!isRunning_.load()) {
                    serverThread_.join();
                    std::cout << "[VR-SM] Shared memory server thread stopped" << std::endl;
                    joined = true;
                }
            } else {
                joined = true;
            }
        }
    }
    
    cleanupSharedMemory();
    isRunning_ = false;
    pluginConnected_ = false;
}

bool VRSharedMemoryServer::sendMessage(const std::string& message) {
    if (!pluginConnected_.load() || !sharedMemory_) {
        messagesDropped_++;
        return false;
    }

    // Try to parse as VRCommand first (binary protocol)
    if (message.size() == sizeof(SharedVRMemory::VRCommand)) {
        const SharedVRMemory::VRCommand* cmd = 
            reinterpret_cast<const SharedVRMemory::VRCommand*>(message.data());
        
        if (writeCommand(*cmd)) {
            messagesSent_++;
            return true;
        }
    } else {
        // Handle as status message
        if (writeStatusMessage(message)) {
            messagesSent_++;
            return true;
        }
    }
    
    messagesDropped_++;
    return false;
}

PipeStatistics VRSharedMemoryServer::getStatistics() const {
    return PipeStatistics{
        messagesSent_.load(),
        messagesDropped_.load(),
        messagesReceived_.load(),
        reconnectCount_.load(),
        pluginConnected_.load()
    };
}

void VRSharedMemoryServer::sharedMemoryServerThread() noexcept {
    isRunning_ = true;
    
    std::cout << "[VR-SM] Starting shared memory server: " << memoryName_ << std::endl;
    
    while (!shouldStop_.load()) {
        try {
            // Update our heartbeat to signal we're alive
            if (sharedMemory_) {
                sharedMemory_->control.mainAppHeartbeat.fetch_add(1, std::memory_order_relaxed);
            }
            
            // Check connection state
            updateConnectionState();
            
            // Read any incoming status messages from plugin
            readStatusMessages();
            
            // Sleep but remain responsive to shutdown
            for (int i = 0; i < 10 && !shouldStop_.load(); i++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            
        } catch (const std::exception& e) {
            std::cerr << "[VR-SM] Exception in server thread: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } catch (...) {
            std::cerr << "[VR-SM] Unknown exception in server thread" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
    std::cout << "[VR-SM] Shared memory server shutting down" << std::endl;
    isRunning_ = false;
}

bool VRSharedMemoryServer::createSharedMemory(const std::string& memoryName) noexcept {
    // Create file mapping object
    hFileMapping_ = CreateFileMappingA(
        INVALID_HANDLE_VALUE,    // Use paging file
        nullptr,                 // Default security
        PAGE_READWRITE,          // Read/write access
        0,                       // High-order DWORD of size
        SHARED_MEMORY_SIZE,      // Low-order DWORD of size
        memoryName.c_str()       // Name of mapping object
    );

    if (hFileMapping_ == nullptr) {
        DWORD error = GetLastError();
        std::cerr << "[VR-SM] CreateFileMapping failed. Error: " << error << std::endl;
        return false;
    }

    // Map view of file
    sharedMemory_ = static_cast<SharedVRMemory*>(
        MapViewOfFile(
            hFileMapping_,       // Handle to map object
            FILE_MAP_ALL_ACCESS, // Read/write permission
            0,                   // High-order DWORD of offset
            0,                   // Low-order DWORD of offset
            SHARED_MEMORY_SIZE   // Number of bytes to map
        )
    );

    if (sharedMemory_ == nullptr) {
        DWORD error = GetLastError();
        std::cerr << "[VR-SM] MapViewOfFile failed. Error: " << error << std::endl;
        CloseHandle(hFileMapping_);
        hFileMapping_ = INVALID_HANDLE_VALUE;
        return false;
    }

    // Initialize shared memory if we're the first to create it
    if (GetLastError() != ERROR_ALREADY_EXISTS) {
        std::cout << "[VR-SM] Initializing new shared memory region" << std::endl;
        new (sharedMemory_) SharedVRMemory(); // Placement new for proper initialization
    } else {
        std::cout << "[VR-SM] Connected to existing shared memory region" << std::endl;
    }

    std::cout << "[VR-SM] Shared memory created successfully: " << memoryName << std::endl;
    return true;
}

void VRSharedMemoryServer::cleanupSharedMemory() noexcept {
    if (sharedMemory_) {
        UnmapViewOfFile(sharedMemory_);
        sharedMemory_ = nullptr;
    }
    
    if (hFileMapping_ != INVALID_HANDLE_VALUE) {
        CloseHandle(hFileMapping_);
        hFileMapping_ = INVALID_HANDLE_VALUE;
    }
}

bool VRSharedMemoryServer::writeCommand(const SharedVRMemory::VRCommand& cmd) noexcept {
    if (!sharedMemory_ || !sharedMemory_->hasSpace()) {
        return false;
    }

    // Get current write position
    uint32_t writePos = sharedMemory_->control.writeIndex.load(std::memory_order_relaxed);
    uint32_t bufferIndex = writePos & SharedVRMemory::COMMAND_BUFFER_MASK;
    
    // Write command
    sharedMemory_->commandBuffer[bufferIndex] = cmd;
    
    // Update write index (this makes the message visible to reader)
    sharedMemory_->control.writeIndex.store(writePos + 1, std::memory_order_release);
    sharedMemory_->control.totalMessages.fetch_add(1, std::memory_order_relaxed);
    
    return true;
}

bool VRSharedMemoryServer::writeStatusMessage(const std::string& message) noexcept {
    if (!sharedMemory_) {
        return false;
    }

    // Simple status message writing (not as performance critical)
    uint32_t writePos = sharedMemory_->status.statusWriteIndex.load(std::memory_order_relaxed);
    uint32_t readPos = sharedMemory_->status.statusReadIndex.load(std::memory_order_acquire);
    
    size_t bufferSize = sizeof(sharedMemory_->status.statusBuffer);
    size_t availableSpace = bufferSize - ((writePos - readPos) % bufferSize);
    
    if (message.length() + 1 > availableSpace) {
        return false; // Not enough space
    }
    
    // Write message
    for (char c : message) {
        sharedMemory_->status.statusBuffer[writePos % bufferSize] = c;
        writePos++;
    }
    sharedMemory_->status.statusBuffer[writePos % bufferSize] = '\0';
    writePos++;
    
    sharedMemory_->status.statusWriteIndex.store(writePos, std::memory_order_release);
    return true;
}

void VRSharedMemoryServer::readStatusMessages() noexcept {
    if (!sharedMemory_) {
        return;
    }

    uint32_t writePos = sharedMemory_->status.statusWriteIndex.load(std::memory_order_acquire);
    uint32_t readPos = sharedMemory_->status.statusReadIndex.load(std::memory_order_relaxed);
    
    if (writePos == readPos) {
        return; // No messages
    }
    
    // Read messages from plugin
    size_t bufferSize = sizeof(sharedMemory_->status.statusBuffer);
    std::string message;
    
    while (readPos != writePos) {
        char c = sharedMemory_->status.statusBuffer[readPos % bufferSize];
        readPos++;
        
        if (c == '\0') {
            if (!message.empty()) {
                messagesReceived_++;
                
                // Call message handler
                {
                    std::lock_guard<std::mutex> lock(callbackMutex_);
                    if (messageHandler_) {
                        try {
                            messageHandler_(message);
                        } catch (...) {
                            std::cerr << "[VR-SM] Exception in message handler" << std::endl;
                        }
                    }
                }
                message.clear();
            }
        } else {
            message += c;
        }
    }
    
    sharedMemory_->status.statusReadIndex.store(readPos, std::memory_order_release);
}

void VRSharedMemoryServer::updateConnectionState() noexcept {
    if (!sharedMemory_) {
        return;
    }

    DWORD currentTime = GetTickCount();
    
    // Check connection state periodically
    if (currentTime - lastConnectionCheck_ >= CONNECTION_CHECK_INTERVAL_MS) {
        uint32_t currentHeartbeat = sharedMemory_->control.pluginHeartbeat.load(std::memory_order_acquire);
        bool wasConnected = pluginConnected_.load();
        
        // Plugin is connected if heartbeat changed recently
        bool isConnected = (currentHeartbeat != lastPluginHeartbeat_) ||
                          (currentTime - lastConnectionCheck_ < PLUGIN_TIMEOUT_MS);
        
        if (isConnected != wasConnected) {
            pluginConnected_.store(isConnected);
            
            if (isConnected) {
                reconnectCount_++;
                std::cout << "[VR-SM] Plugin connected (#" << reconnectCount_.load() << ")" << std::endl;
            } else {
                std::cout << "[VR-SM] Plugin disconnected" << std::endl;
            }
            
            // Notify connection handler
            {
                std::lock_guard<std::mutex> lock(callbackMutex_);
                if (connectionHandler_) {
                    try {
                        connectionHandler_(isConnected);
                    } catch (...) {
                        std::cerr << "[VR-SM] Exception in connection handler" << std::endl;
                    }
                }
            }
        }
        
        lastPluginHeartbeat_ = currentHeartbeat;
        lastConnectionCheck_ = currentTime;
    }
}

bool VRSharedMemoryServer::parseMessageToCommand(const std::string& message, SharedVRMemory::VRCommand& cmd) noexcept {
    // Simple parsing for backward compatibility
    // In practice, the VRInputHandler sends binary VRCommand directly
    
    std::istringstream iss(message);
    std::string token;
    
    // Reset command
    memset(&cmd, 0, sizeof(cmd));
    
    // Parse basic format: "type:1 buttons:A duration:50"
    while (std::getline(iss, token, ' ')) {
        size_t colonPos = token.find(':');
        if (colonPos == std::string::npos) continue;
        
        std::string key = token.substr(0, colonPos);
        std::string value = token.substr(colonPos + 1);
        
        if (key == "type") {
            cmd.type = static_cast<uint8_t>(std::stoi(value));
        } else if (key == "duration") {
            cmd.duration = static_cast<uint16_t>(std::stoi(value));
        }
        // Add more parsing as needed
    }
    
    return cmd.type > 0 && cmd.type <= 4;
}

} // namespace whisper_spell
