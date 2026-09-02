#include "SpellTransmitter.h"
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cstring>

// ============================================================================
// Constructor / Destructor
// ============================================================================

SpellTransmitter::SpellTransmitter() {
    // Default initialization handled by member initializers
}

SpellTransmitter::~SpellTransmitter() {
    shutdown();
}

SpellTransmitter::SpellTransmitter(SpellTransmitter&& other) noexcept
    : hFileMapping_(other.hFileMapping_)
    , sharedMemory_(other.sharedMemory_)
    , isConnected_(other.isConnected_.load())
    , isInitialized_(other.isInitialized_.load())
    , lastConnectionCheck_(other.lastConnectionCheck_)
    , lastConsumerHeartbeat_(other.lastConsumerHeartbeat_)
    , spellsSent_(other.spellsSent_.load())
    , spellsDropped_(other.spellsDropped_.load())
    , reconnectCount_(other.reconnectCount_.load())
    , debugLogging_(other.debugLogging_)
{
    other.hFileMapping_ = INVALID_HANDLE_VALUE;
    other.sharedMemory_ = nullptr;
    other.isConnected_.store(false);
    other.isInitialized_.store(false);
}

SpellTransmitter& SpellTransmitter::operator=(SpellTransmitter&& other) noexcept {
    if (this != &other) {
        shutdown();

        hFileMapping_ = other.hFileMapping_;
        sharedMemory_ = other.sharedMemory_;
        isConnected_.store(other.isConnected_.load());
        isInitialized_.store(other.isInitialized_.load());
        lastConnectionCheck_ = other.lastConnectionCheck_;
        lastConsumerHeartbeat_ = other.lastConsumerHeartbeat_;
        spellsSent_.store(other.spellsSent_.load());
        spellsDropped_.store(other.spellsDropped_.load());
        reconnectCount_.store(other.reconnectCount_.load());
        debugLogging_ = other.debugLogging_;

        other.hFileMapping_ = INVALID_HANDLE_VALUE;
        other.sharedMemory_ = nullptr;
        other.isConnected_.store(false);
        other.isInitialized_.store(false);
    }
    return *this;
}

// ============================================================================
// Initialization / Shutdown
// ============================================================================

bool SpellTransmitter::initialize() {
    if (isInitialized_.load()) {
        return true;
    }

    if (!createSharedMemory()) {
        return false;
    }

    isInitialized_.store(true);
    lastConnectionCheck_ = GetTickCount();

    std::cout << "[SpellTx] Initialized - listening on shared memory: "
              << SpellCaster::SHARED_MEMORY_NAME << std::endl;

    return true;
}

void SpellTransmitter::shutdown() {
    if (!isInitialized_.load()) {
        return;
    }

    std::cout << "[SpellTx] Shutting down..." << std::endl;

    cleanupSharedMemory();

    isConnected_.store(false);
    isInitialized_.store(false);

    std::cout << "[SpellTx] Shutdown complete. Stats: sent=" << spellsSent_.load()
              << ", dropped=" << spellsDropped_.load() << std::endl;
}

bool SpellTransmitter::createSharedMemory() {
    // Try to open existing shared memory (created by UE4SS mod)
    hFileMapping_ = OpenFileMappingA(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        SpellCaster::SHARED_MEMORY_NAME
    );

    if (hFileMapping_ == nullptr || hFileMapping_ == INVALID_HANDLE_VALUE) {
        // Shared memory doesn't exist - create it ourselves
        hFileMapping_ = CreateFileMappingA(
            INVALID_HANDLE_VALUE,
            nullptr,
            PAGE_READWRITE,
            0,
            sizeof(SpellCaster::SharedSpellMemory),
            SpellCaster::SHARED_MEMORY_NAME
        );

        if (hFileMapping_ == nullptr || hFileMapping_ == INVALID_HANDLE_VALUE) {
            std::cerr << "[SpellTx] ERROR: Failed to create shared memory!" << std::endl;
            return false;
        }

        std::cout << "[SpellTx] Created shared memory: " << SpellCaster::SHARED_MEMORY_NAME << std::endl;
    } else {
        std::cout << "[SpellTx] Opened existing shared memory: " << SpellCaster::SHARED_MEMORY_NAME << std::endl;
    }

    // Map the shared memory
    sharedMemory_ = static_cast<SpellCaster::SharedSpellMemory*>(
        MapViewOfFile(
            hFileMapping_,
            FILE_MAP_ALL_ACCESS,
            0,
            0,
            sizeof(SpellCaster::SharedSpellMemory)
        )
    );

    if (sharedMemory_ == nullptr) {
        std::cerr << "[SpellTx] ERROR: Failed to map shared memory!" << std::endl;
        CloseHandle(hFileMapping_);
        hFileMapping_ = INVALID_HANDLE_VALUE;
        return false;
    }

    // Initialize the memory if we created it (use placement new for atomics)
    DWORD lastError = GetLastError();
    if (lastError != ERROR_ALREADY_EXISTS) {
        // Zero out the memory first
        memset(sharedMemory_, 0, sizeof(SpellCaster::SharedSpellMemory));
        // Properly initialize atomics
        new (&sharedMemory_->control.writeIndex) std::atomic<uint32_t>(0);
        new (&sharedMemory_->control.readIndex) std::atomic<uint32_t>(0);
        new (&sharedMemory_->control.producerHeartbeat) std::atomic<uint32_t>(0);
        new (&sharedMemory_->control.consumerHeartbeat) std::atomic<uint32_t>(0);
        new (&sharedMemory_->control.messagesDropped) std::atomic<uint32_t>(0);
        new (&sharedMemory_->control.totalMessages) std::atomic<uint32_t>(0);
        new (&sharedMemory_->status.statusWriteIndex) std::atomic<uint32_t>(0);
        new (&sharedMemory_->status.statusReadIndex) std::atomic<uint32_t>(0);
    }

    return true;
}

void SpellTransmitter::cleanupSharedMemory() {
    if (sharedMemory_) {
        UnmapViewOfFile(sharedMemory_);
        sharedMemory_ = nullptr;
    }

    if (hFileMapping_ != INVALID_HANDLE_VALUE) {
        CloseHandle(hFileMapping_);
        hFileMapping_ = INVALID_HANDLE_VALUE;
    }
}

// ============================================================================
// Spell Transmission
// ============================================================================

bool SpellTransmitter::sendSpell(const std::string& spellName) {
    if (!isInitialized_.load()) {
        if (debugLogging_) {
            std::cout << "[SpellTx] Not initialized, dropping: " << spellName << std::endl;
        }
        spellsDropped_++;
        return false;
    }

    // Normalize the spell name
    std::string normalizedSpell = normalizeSpellName(spellName);

    if (normalizedSpell.empty()) {
        if (debugLogging_) {
            std::cout << "[SpellTx] Empty spell after normalization, dropping" << std::endl;
        }
        return false;
    }

    // Log the transmission attempt
    std::cout << "[SpellTx] \"" << spellName << "\" -> \"" << normalizedSpell << "\"";

    // Fire-and-forget: drop if not connected (but still try to write)
    if (!isConnected_.load()) {
        std::cout << " [QUEUED - UE4SS not detected yet]" << std::endl;
    } else {
        std::cout << std::endl;
    }

    // Write to shared memory
    return writeSpellCommand(normalizedSpell);
}

bool SpellTransmitter::writeSpellCommand(const std::string& normalizedSpell) {
    if (!sharedMemory_) {
        spellsDropped_++;
        return false;
    }

    // Check buffer space
    if (!sharedMemory_->hasSpace()) {
        if (debugLogging_) {
            std::cout << "[SpellTx] Buffer full, dropping: " << normalizedSpell << std::endl;
        }
        sharedMemory_->control.messagesDropped.fetch_add(1, std::memory_order_relaxed);
        spellsDropped_++;
        return false;
    }

    // Get current write position
    uint32_t writePos = sharedMemory_->control.writeIndex.load(std::memory_order_relaxed);
    uint32_t bufferIndex = writePos & SpellCaster::SharedSpellMemory::COMMAND_BUFFER_MASK;

    // Write spell command
    SpellCaster::SpellCommand& cmd = sharedMemory_->commandBuffer[bufferIndex];

    // Copy spell name (truncate if too long)
    size_t copyLen = (std::min)(normalizedSpell.length(), size_t(27));
    memcpy(cmd.spellName, normalizedSpell.c_str(), copyLen);
    cmd.spellName[copyLen] = '\0';
    cmd.flags = 0;

    // Publish (makes visible to consumer via release)
    sharedMemory_->control.writeIndex.store(writePos + 1, std::memory_order_release);
    sharedMemory_->control.totalMessages.fetch_add(1, std::memory_order_relaxed);

    spellsSent_++;
    return true;
}

// ============================================================================
// Spell Name Normalization
// ============================================================================

std::string SpellTransmitter::normalizeSpellName(const std::string& rawSpell) {
    std::string result;
    result.reserve(rawSpell.length());

    bool capitalizeNext = true;
    bool inBracket = false;

    for (size_t i = 0; i < rawSpell.length(); i++) {
        char c = rawSpell[i];

        // Skip Whisper artifacts in brackets like [BLANK_AUDIO]
        if (c == '[') {
            inBracket = true;
            continue;
        }
        if (c == ']') {
            inBracket = false;
            continue;
        }
        if (inBracket) {
            continue;
        }

        // Skip punctuation and special characters
        if (c == '.' || c == ',' || c == '!' || c == '?' ||
            c == '"' || c == '\'' || c == ':' || c == ';' ||
            c == '(' || c == ')' || c == '-' || c == '_') {
            continue;
        }

        // Handle whitespace - skip but mark next char for capitalization
        if (std::isspace(static_cast<unsigned char>(c))) {
            capitalizeNext = true;
            continue;
        }

        // Skip non-alphabetic characters
        if (!std::isalpha(static_cast<unsigned char>(c))) {
            continue;
        }

        // Capitalize first letter of each word
        if (capitalizeNext) {
            result += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            capitalizeNext = false;
        } else {
            result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }

    return result;
}

// ============================================================================
// Connection Management
// ============================================================================

void SpellTransmitter::update() {
    if (!isInitialized_.load() || !sharedMemory_) {
        return;
    }

    DWORD currentTime = GetTickCount();

    // Update our heartbeat
    sharedMemory_->control.producerHeartbeat.fetch_add(1, std::memory_order_relaxed);

    // Check connection state periodically
    if (currentTime - lastConnectionCheck_ >= CONNECTION_CHECK_INTERVAL_MS) {
        updateConnectionState();
        lastConnectionCheck_ = currentTime;
    }
}

void SpellTransmitter::updateConnectionState() {
    if (!sharedMemory_) {
        isConnected_.store(false);
        return;
    }

    uint32_t currentConsumerHeartbeat = sharedMemory_->control.consumerHeartbeat.load(std::memory_order_acquire);

    bool wasConnected = isConnected_.load();
    bool nowConnected = (currentConsumerHeartbeat != lastConsumerHeartbeat_);

    if (nowConnected && !wasConnected) {
        std::cout << "[SpellTx] UE4SS mod connected!" << std::endl;
        reconnectCount_++;
    } else if (!nowConnected && wasConnected) {
        std::cout << "[SpellTx] UE4SS mod disconnected - spells will be queued" << std::endl;
    }

    isConnected_.store(nowConnected);
    lastConsumerHeartbeat_ = currentConsumerHeartbeat;
}

// ============================================================================
// Statistics
// ============================================================================

SpellTransmitterStats SpellTransmitter::getStatistics() const {
    SpellTransmitterStats stats;
    stats.spellsSent = spellsSent_.load();
    stats.spellsDropped = spellsDropped_.load();
    stats.reconnectCount = reconnectCount_.load();
    stats.isConnected = isConnected_.load();
    return stats;
}
