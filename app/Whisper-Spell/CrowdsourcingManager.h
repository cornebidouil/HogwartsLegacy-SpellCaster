#pragma once

#include "Config.h"
#include "CrowdsourcingStorage.h"
#include "CrowdsourcingClient.h"
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <string>
#include <vector>
#include <map>

class CrowdsourcingManager {
public:
    /**
     * Construct crowdsourcing manager
     * @param config Reference to application config
     * @param deviceName Audio input device name
     */
    CrowdsourcingManager(Config& config, const std::string& deviceName);
    ~CrowdsourcingManager();

    /**
     * Initialize storage and client
     * Creates directories, validates configuration
     * @return true if successful, false on error
     */
    bool initialize();

    /**
     * Shutdown background threads and cleanup
     */
    void shutdown();

    /**
     * Check if first-run dialog needs to be shown
     * Shows dialog if necessary and updates config
     * @return true if consent was given (or already given), false if declined
     */
    bool checkFirstRun();

    /**
     * Save a recording to pending folder (thread-safe)
     * Called from ThreadPool after successful transcription
     * @param spellName Name of the spell being cast
     * @param audioSamples Float audio samples (-1.0 to 1.0)
     * @param sampleRate Sample rate in Hz
     */
    void saveRecording(const std::string& spellName,
                       const std::vector<float>& audioSamples,
                       int sampleRate);

    /**
     * Save a recording to pending folder (raw pointer version)
     */
    void saveRecording(const std::string& spellName,
                       const float* audioSamples,
                       size_t numSamples,
                       int sampleRate);

    /**
     * Start background sync thread
     * Asynchronously syncs pending recordings with server
     */
    void syncAsync();

    /**
     * Get statistics about recordings
     */
    struct Stats {
        int pendingCount = 0;
        int syncedCount = 0;
        int lastSyncAccepted = 0;
        int lastSyncRejected = 0;
        std::string lastSyncError;
    };
    Stats getStats() const;

private:
    Config& config_;
    std::string deviceName_;
    bool debugLogging_;
    std::unique_ptr<CrowdsourcingStorage> storage_;
    std::unique_ptr<CrowdsourcingClient> client_;
    std::mutex managerMutex_;
    std::atomic<bool> isSyncing_;
    std::atomic<bool> shouldStop_;
    std::unique_ptr<std::thread> syncThread_;

    // Last sync stats
    mutable std::mutex statsMutex_;
    int lastSyncAccepted_;
    int lastSyncRejected_;
    std::string lastSyncError_;

    // Upload limits (formula -> current/max counts)
    mutable std::mutex limitsMutex_;
    std::map<std::string, CrowdsourcingClient::FormulaLimit> formulaLimits_;

    /**
     * Check if we should save this recording based on server limits
     * @param spellName Name of the spell
     * @return true if under limit, false if at/over limit or unknown formula
     */
    bool shouldSaveRecording(const std::string& spellName) const;

    /**
     * Update formula limits from sync response
     * @param limits Map of formula limits from server
     */
    void updateLimits(const std::map<std::string, CrowdsourcingClient::FormulaLimit>& limits);

    /**
     * Increment the current count for a formula (after saving locally)
     * @param spellName Name of the spell
     */
    void incrementLocalCount(const std::string& spellName);

    /**
     * Perform synchronous sync operation
     * Called from background thread
     */
    void performSync();

    /**
     * Background thread function
     */
    void syncThreadFunc();
};
