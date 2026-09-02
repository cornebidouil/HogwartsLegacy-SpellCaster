#include "CrowdsourcingManager.h"
#include "FirstRunDialog.h"
#include "CrowdsourcingUtils.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cctype>

// Helper macros for conditional debug logging
#define LOG_CROWDSOURCING(msg) if (debugLogging_) { std::cout << msg << std::endl; }
#define LOG_CROWDSOURCING_ERROR(msg) if (debugLogging_) { std::cerr << msg << std::endl; }

CrowdsourcingManager::CrowdsourcingManager(Config& config, const std::string& deviceName)
    : config_(config)
    , deviceName_(deviceName)
    , debugLogging_(config.crowdsourcing.debugLogging)
    , isSyncing_(false)
    , shouldStop_(false)
    , lastSyncAccepted_(0)
    , lastSyncRejected_(0) {
}

CrowdsourcingManager::~CrowdsourcingManager() {
    shutdown();
}

bool CrowdsourcingManager::initialize() {
    std::lock_guard<std::mutex> lock(managerMutex_);

    // Create storage
    storage_ = std::make_unique<CrowdsourcingStorage>();
    if (!storage_->initialize(config_.crowdsourcing.pendingFolder,
                               config_.crowdsourcing.syncedFolder)) {
        LOG_CROWDSOURCING_ERROR("\033[1;31m[Crowdsourcing] Failed to initialize storage\033[0m");
        return false;
    }

    // Create client
    client_ = std::make_unique<CrowdsourcingClient>(config_.crowdsourcing.serverHost);

    // Enable debug logging if configured
    client_->setDebugLogging(debugLogging_);

    LOG_CROWDSOURCING("\033[1;32m[Crowdsourcing] Initialized successfully\033[0m");
    LOG_CROWDSOURCING("\033[1;36m[Crowdsourcing] Upload limits will be fetched on first sync\033[0m");

    return true;
}

void CrowdsourcingManager::shutdown() {
    shouldStop_ = true;

    // Wait for sync thread to finish
    if (syncThread_ && syncThread_->joinable()) {
        syncThread_->join();
    }

    std::lock_guard<std::mutex> lock(managerMutex_);
    storage_.reset();
    client_.reset();
}

bool CrowdsourcingManager::checkFirstRun() {
    if (config_.crowdsourcing.firstRunComplete) {
        return config_.crowdsourcing.consentGiven;
    }

    // Show first-run dialog
    FirstRunDialog::UserConsent consent;
    if (!FirstRunDialog::show(consent)) {
        // User declined
        config_.crowdsourcing.enabled = false;
        config_.crowdsourcing.consentGiven = false;
        config_.crowdsourcing.firstRunComplete = true;
        return false;
    }

    // Update config with user choices
    config_.crowdsourcing.enabled = true;           // Enable the feature
    config_.crowdsourcing.consentGiven = true;
    config_.crowdsourcing.firstRunComplete = true;

    if (consent.hasAccount) {
        config_.crowdsourcing.authType = "account";
        config_.crowdsourcing.username = consent.username;
        config_.crowdsourcing.password = consent.password;
    } else {
        config_.crowdsourcing.authType = "anonymous";
        config_.crowdsourcing.uuid = CrowdsourcingUtils::generateUUID();
    }

    config_.crowdsourcing.nationality = consent.nationality;
    config_.crowdsourcing.gender = consent.gender;

    return true;
}

void CrowdsourcingManager::saveRecording(const std::string& spellName,
                                          const std::vector<float>& audioSamples,
                                          int sampleRate) {
    if (!config_.crowdsourcing.enabled || !config_.crowdsourcing.consentGiven) {
        return;
    }

    if (!storage_) {
        return;
    }

    // Validate and normalize spell name
    if (spellName.empty()) {
        return; // Empty spell name
    }

    // Check if spell name is only dots (noise pattern: "........")
    bool isOnlyDots = true;
    for (char c : spellName) {
        if (c != '.') {
            isOnlyDots = false;
            break;
        }
    }

    std::string normalizedSpellName = spellName;
    if (isOnlyDots) {
        // Skip noise samples entirely
        return;
    }

    // Reject very short spell names (likely transcription errors)
    if (normalizedSpellName.length() < 3) {
        return;
    }

    // Check if server needs this recording (based on upload limits)
    if (!shouldSaveRecording(normalizedSpellName)) {
        // Spell at or over limit - skip silently
        return;
    }

    // Save recording (thread-safe)
    if (storage_->saveRecording(normalizedSpellName, audioSamples, sampleRate)) {
        // Get current quota for logging
        std::string quotaInfo;
        {
            std::lock_guard<std::mutex> lock(limitsMutex_);
            std::string lowerSpell = normalizedSpellName;
            std::transform(lowerSpell.begin(), lowerSpell.end(), lowerSpell.begin(), ::tolower);

            for (const auto& pair : formulaLimits_) {
                std::string lowerFormula = pair.first;
                std::transform(lowerFormula.begin(), lowerFormula.end(), lowerFormula.begin(), ::tolower);

                if (lowerFormula == lowerSpell) {
                    quotaInfo = " [" + std::to_string(pair.second.current) + "/" +
                                std::to_string(pair.second.max) + "]";
                    break;
                }
            }
        }

        if (debugLogging_) {
            std::cout << "\033[1;36m[Crowdsourcing] Saved: " << normalizedSpellName << quotaInfo << "\033[0m" << std::endl;
        }

        // Increment local count so subsequent saves in this session see updated limit
        incrementLocalCount(normalizedSpellName);
    } else {
        if (debugLogging_) {
            std::cerr << "\033[1;33m[Crowdsourcing] Failed to save recording: "
                      << spellName << "\033[0m" << std::endl;
        }
    }
}

void CrowdsourcingManager::saveRecording(const std::string& spellName,
                                          const float* audioSamples,
                                          size_t numSamples,
                                          int sampleRate) {
    if (!config_.crowdsourcing.enabled || !config_.crowdsourcing.consentGiven) {
        return;
    }

    if (!storage_) {
        return;
    }

    // Validate and normalize spell name
    if (spellName.empty()) {
        return; // Empty spell name
    }

    // Check if spell name is only dots (noise pattern: "........")
    bool isOnlyDots = true;
    for (char c : spellName) {
        if (c != '.') {
            isOnlyDots = false;
            break;
        }
    }

    std::string normalizedSpellName = spellName;
    if (isOnlyDots) {
        // Skip noise samples entirely
        return;
    }

    // Reject very short spell names (likely transcription errors)
    if (normalizedSpellName.length() < 3) {
        return;
    }

    // Check if server needs this recording (based on upload limits)
    if (!shouldSaveRecording(normalizedSpellName)) {
        // Spell at or over limit - skip silently
        return;
    }

    // Save recording (thread-safe)
    if (storage_->saveRecording(normalizedSpellName, audioSamples, numSamples, sampleRate)) {
        // Get current quota for logging
        std::string quotaInfo;
        {
            std::lock_guard<std::mutex> lock(limitsMutex_);
            std::string lowerSpell = normalizedSpellName;
            std::transform(lowerSpell.begin(), lowerSpell.end(), lowerSpell.begin(), ::tolower);

            for (const auto& pair : formulaLimits_) {
                std::string lowerFormula = pair.first;
                std::transform(lowerFormula.begin(), lowerFormula.end(), lowerFormula.begin(), ::tolower);

                if (lowerFormula == lowerSpell) {
                    quotaInfo = " [" + std::to_string(pair.second.current) + "/" +
                                std::to_string(pair.second.max) + "]";
                    break;
                }
            }
        }

        if (debugLogging_) {
            std::cout << "\033[1;36m[Crowdsourcing] Saved: " << normalizedSpellName << quotaInfo << "\033[0m" << std::endl;
        }

        // Increment local count so subsequent saves in this session see updated limit
        incrementLocalCount(normalizedSpellName);
    } else {
        if (debugLogging_) {
            std::cerr << "\033[1;33m[Crowdsourcing] Failed to save recording: "
                      << spellName << "\033[0m" << std::endl;
        }
    }
}

void CrowdsourcingManager::syncAsync() {
    if (isSyncing_) {
        return; // Already syncing
    }

    if (!config_.crowdsourcing.enabled || !config_.crowdsourcing.consentGiven) {
        return;
    }

    // Account mode without a password (skipped at the prompt): nothing to authenticate with
    if (config_.crowdsourcing.authType == "account" && config_.crowdsourcing.password.empty()) {
        LOG_CROWDSOURCING("\033[1;33m[Crowdsourcing] No account password available, sync skipped\033[0m");
        return;
    }

    // Start background sync thread
    isSyncing_ = true;
    syncThread_ = std::make_unique<std::thread>(&CrowdsourcingManager::syncThreadFunc, this);
}

void CrowdsourcingManager::syncThreadFunc() {
    try {
        performSync();
    } catch (const std::exception& e) {
        if (debugLogging_) {
            std::cerr << "\033[1;31m[Crowdsourcing] Sync thread exception: "
                      << e.what() << "\033[0m" << std::endl;
        }
    }

    isSyncing_ = false;
}

void CrowdsourcingManager::performSync() {
    if (!storage_ || !client_) {
        return;
    }

    LOG_CROWDSOURCING("\033[1;36m[Crowdsourcing] Starting sync...\033[0m");

    // Load pending recordings
    auto pendingRecordings = storage_->loadPendingRecordings();

    if (pendingRecordings.empty()) {
        LOG_CROWDSOURCING("\033[1;36m[Crowdsourcing] No pending recordings\033[0m");
    } else {
        if (debugLogging_) {
            std::cout << "\033[1;36m[Crowdsourcing] Found " << pendingRecordings.size()
                      << " pending recordings\033[0m" << std::endl;
        }
    }

    // Build sync request (even if no recordings, we need to get/refresh limits)
    CrowdsourcingClient::SyncRequest syncReq;
    syncReq.authType = config_.crowdsourcing.authType;
    syncReq.username = config_.crowdsourcing.username;
    syncReq.password = config_.crowdsourcing.password;
    syncReq.uuid = config_.crowdsourcing.uuid;
    syncReq.nationality = config_.crowdsourcing.nationality;
    syncReq.gender = config_.crowdsourcing.gender;
    syncReq.deviceName = deviceName_;

    // Build list of recordings to offer (if any)
    int noiseCount = 0;
    int validCount = 0;
    if (!pendingRecordings.empty()) {
        LOG_CROWDSOURCING("\033[1;34m[Crowdsourcing] Offering spells to server:\033[0m");
        for (const auto& rec : pendingRecordings) {
            CrowdsourcingClient::RecordingInfo info;
            info.formula = rec.metadata.formula;
            info.hash = rec.metadata.hash;
            info.size = rec.metadata.fileSize;
            syncReq.recordings.push_back(info);

            if (rec.metadata.formula == "Noise") {
                noiseCount++;
            } else {
                validCount++;
                if (debugLogging_) {
                    std::cout << "\033[1;34m  - " << rec.metadata.formula
                              << " (" << (rec.metadata.fileSize / 1024) << " KB)\033[0m" << std::endl;
                }
            }
        }
        if (noiseCount > 0 && debugLogging_) {
            std::cout << "\033[0;90m  - " << noiseCount << " noise samples (will be cleaned up)\033[0m" << std::endl;
        }
        if (debugLogging_) {
            std::cout << std::endl;
        }
    }

    // Call sync endpoint
    CrowdsourcingClient::SyncResponse syncResp;
    if (!client_->sync(syncReq, syncResp)) {
        std::lock_guard<std::mutex> lock(statsMutex_);
        lastSyncError_ = syncResp.error;
        if (syncResp.error == "Invalid credentials") {
            // Always shown: the user has to act on it (wrong or changed password)
            std::cerr << "\033[1;31m[Crowdsourcing] The server rejected the credentials of '"
                      << config_.crowdsourcing.username
                      << "'. Delete the password_protected line in config.ini and restart to enter the password again.\033[0m"
                      << std::endl;
        } else if (debugLogging_) {
            std::cerr << "\033[1;31m[Crowdsourcing] Sync failed: "
                      << syncResp.error << "\033[0m" << std::endl;
        }
        return;
    }

    // Update limits from sync response
    if (!syncResp.limits.empty()) {
        bool isFirstSync = formulaLimits_.empty();
        updateLimits(syncResp.limits);

        if (isFirstSync && debugLogging_) {
            // First sync - show summary
            std::cout << "\033[1;32m[Crowdsourcing] Upload limits loaded: " << syncResp.limits.size() << " formulas\033[0m" << std::endl;

            int atLimit = 0;
            int nearLimit = 0;
            for (const auto& pair : syncResp.limits) {
                if (pair.second.current >= pair.second.max) atLimit++;
                else if (pair.second.current >= pair.second.max - 1) nearLimit++;
            }
            std::cout << "\033[1;36m[Crowdsourcing] Summary: " << atLimit << " at limit, "
                      << nearLimit << " near limit\033[0m" << std::endl;
        }
    }

    if (syncResp.wanted.empty()) {
        if (debugLogging_) {
            if (validCount > 0) {
                LOG_CROWDSOURCING("\033[1;36m[Crowdsourcing] Server doesn't need these recordings (may have enough already)\033[0m");
            } else if (noiseCount > 0) {
                LOG_CROWDSOURCING("\033[1;36m[Crowdsourcing] Server rejected all recordings (only noise samples were offered)\033[0m");
            } else {
                LOG_CROWDSOURCING("\033[1;36m[Crowdsourcing] Server doesn't need any recordings\033[0m");
            }
        }

        // Cleanup ALL pending recordings after sync attempt
        int deletedCount = storage_->cleanupAllPending();
        if (deletedCount > 0 && debugLogging_) {
            std::cout << "\033[1;36m[Crowdsourcing] Cleaned up " << deletedCount
                      << " pending recordings\033[0m" << std::endl;
        }

        return;
    }

    if (debugLogging_) {
        std::cout << "\033[1;36m[Crowdsourcing] Server wants " << syncResp.wanted.size()
                  << " recordings\033[0m" << std::endl;
    }

    // Filter to wanted recordings only and log which spells are requested
    std::vector<CrowdsourcingStorage::PendingRecording> toUpload;
    LOG_CROWDSOURCING("\033[1;35m[Crowdsourcing] Requested spells:\033[0m");
    for (const auto& rec : pendingRecordings) {
        if (std::find(syncResp.wanted.begin(), syncResp.wanted.end(), rec.metadata.hash)
                != syncResp.wanted.end()) {
            toUpload.push_back(rec);
            if (debugLogging_) {
                std::cout << "\033[1;35m  - " << rec.metadata.formula
                          << " (hash: " << rec.metadata.hash.substr(0, 8) << "...)\033[0m" << std::endl;
            }
        }
    }

    if (toUpload.empty()) {
        LOG_CROWDSOURCING("\033[1;36m[Crowdsourcing] No matching recordings to upload\033[0m");

        // Cleanup all pending recordings (server didn't want any of them)
        int deletedCount = storage_->cleanupAllPending();
        if (deletedCount > 0 && debugLogging_) {
            std::cout << "\033[1;36m[Crowdsourcing] Cleaned up " << deletedCount
                      << " pending recordings\033[0m" << std::endl;
        }

        return;
    }

    // Upload batch
    CrowdsourcingClient::BatchUploadResult uploadResult;
    if (!client_->uploadBatch(syncResp.token, toUpload, deviceName_, uploadResult)) {
        std::lock_guard<std::mutex> lock(statsMutex_);
        lastSyncError_ = uploadResult.error;
        if (debugLogging_) {
            std::cerr << "\033[1;31m[Crowdsourcing] Upload failed: "
                      << uploadResult.error << "\033[0m" << std::endl;
        }
        return;
    }

    // Update stats
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        lastSyncAccepted_ = uploadResult.accepted;
        lastSyncRejected_ = uploadResult.rejected;
        lastSyncError_.clear();
    }

    if (debugLogging_) {
        std::cout << "\033[1;32m[Crowdsourcing] Upload complete: "
                  << uploadResult.accepted << " accepted, "
                  << uploadResult.rejected << " rejected\033[0m" << std::endl;

        // Log detailed results
        for (const auto& result : uploadResult.results) {
            // Find the spell name for this hash
            std::string spellName = "unknown";
            for (const auto& rec : toUpload) {
                if (rec.metadata.hash == result.hash) {
                    spellName = rec.metadata.formula;
                    break;
                }
            }

            if (result.status == "accepted") {
                std::cout << "\033[1;32m  ✓ Accepted: " << spellName
                          << " (" << result.hash.substr(0, 8) << "...)\033[0m" << std::endl;
            } else if (result.status == "rejected") {
                std::cout << "\033[1;33m  ✗ Rejected: " << spellName
                          << " (" << result.hash.substr(0, 8) << "...)"
                          << " - " << result.reason << "\033[0m" << std::endl;
            }
        }
    }

    // Delete accepted recordings (no need to keep after successful upload)
    if (uploadResult.accepted > 0) {
        std::vector<std::string> acceptedHashes;
        for (const auto& result : uploadResult.results) {
            if (result.status == "accepted") {
                acceptedHashes.push_back(result.hash);
            }
        }

        int deletedCount = storage_->deleteRecordings(acceptedHashes);
        if (debugLogging_) {
            std::cout << "\033[1;32m[Crowdsourcing] Deleted " << deletedCount
                      << " uploaded recordings\033[0m" << std::endl;
        }
    }

    // Update limits from upload response (optimization: no need for 3rd API call)
    if (!uploadResult.limits.empty()) {
        updateLimits(uploadResult.limits);
        if (debugLogging_) {
            std::cout << "\033[1;32m[Crowdsourcing] Updated limits after upload:\033[0m" << std::endl;

            // Show only formulas that changed or are near/at limit
            int shown = 0;
            for (const auto& pair : uploadResult.limits) {
                if (shown < 10 && (pair.second.current >= pair.second.max - 1 || pair.second.current > 0)) {
                    std::cout << "\033[1;32m  - " << pair.first << ": "
                              << pair.second.current << "/" << pair.second.max << "\033[0m" << std::endl;
                    shown++;
                }
            }
            if (shown == 0) {
                std::cout << "\033[1;32m  (All formulas still available)\033[0m" << std::endl;
            }
        }
    }

    // Cleanup ALL remaining pending recordings after sync
    // (includes unwanted recordings and rejected recordings)
    int pendingDeleted = storage_->cleanupAllPending();
    if (pendingDeleted > 0 && debugLogging_) {
        std::cout << "\033[1;36m[Crowdsourcing] Cleaned up " << pendingDeleted
                  << " remaining pending recordings\033[0m" << std::endl;
    }
}

CrowdsourcingManager::Stats CrowdsourcingManager::getStats() const {
    Stats stats;

    if (storage_) {
        auto storageStats = storage_->getStats();
        stats.pendingCount = storageStats.pendingCount;
        stats.syncedCount = storageStats.syncedCount;
    }

    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats.lastSyncAccepted = lastSyncAccepted_;
        stats.lastSyncRejected = lastSyncRejected_;
        stats.lastSyncError = lastSyncError_;
    }

    return stats;
}

bool CrowdsourcingManager::shouldSaveRecording(const std::string& spellName) const {
    std::lock_guard<std::mutex> lock(limitsMutex_);

    // If we don't have limits yet, save everything (will get limits on first sync)
    if (formulaLimits_.empty()) {
        if (debugLogging_) {
            std::cout << "\033[1;33m[Crowdsourcing] DEBUG: No limits loaded yet, accepting '"
                      << spellName << "'\033[0m" << std::endl;
        }
        return true;
    }

    // Case-insensitive search for formula
    std::string lowerSpell = spellName;
    std::transform(lowerSpell.begin(), lowerSpell.end(), lowerSpell.begin(), ::tolower);

    for (const auto& pair : formulaLimits_) {
        std::string lowerFormula = pair.first;
        std::transform(lowerFormula.begin(), lowerFormula.end(), lowerFormula.begin(), ::tolower);

        if (lowerFormula == lowerSpell) {
            // Found it - check if under limit
            bool shouldSave = pair.second.current < pair.second.max;
            if (debugLogging_) {
                std::cout << "\033[1;33m[Crowdsourcing] DEBUG: '" << spellName << "' matched '"
                          << pair.first << "' (" << pair.second.current << "/" << pair.second.max
                          << ") -> " << (shouldSave ? "SAVE" : "SKIP") << "\033[0m" << std::endl;
            }
            return shouldSave;
        }
    }

    // Unknown formula - don't save
    if (debugLogging_) {
        std::cout << "\033[1;33m[Crowdsourcing] DEBUG: '" << spellName
                  << "' not found in limits map (" << formulaLimits_.size()
                  << " entries), skipping\033[0m" << std::endl;
    }
    return false;
}

void CrowdsourcingManager::updateLimits(const std::map<std::string, CrowdsourcingClient::FormulaLimit>& limits) {
    std::lock_guard<std::mutex> lock(limitsMutex_);
    formulaLimits_ = limits;
}

void CrowdsourcingManager::incrementLocalCount(const std::string& spellName) {
    std::lock_guard<std::mutex> lock(limitsMutex_);

    // Find the formula (case-insensitive search)
    for (auto& pair : formulaLimits_) {
        // Compare lowercase versions
        std::string lowerFormula = pair.first;
        std::string lowerSpell = spellName;
        std::transform(lowerFormula.begin(), lowerFormula.end(), lowerFormula.begin(), ::tolower);
        std::transform(lowerSpell.begin(), lowerSpell.end(), lowerSpell.begin(), ::tolower);

        if (lowerFormula == lowerSpell) {
            int oldCount = pair.second.current;
            pair.second.current++;
            if (debugLogging_) {
                std::cout << "\033[1;33m[Crowdsourcing] DEBUG: Incremented '" << pair.first
                          << "' from " << oldCount << "/" << pair.second.max
                          << " to " << pair.second.current << "/" << pair.second.max << "\033[0m" << std::endl;
            }
            return;
        }
    }

    // Spell not found in limits map
    if (debugLogging_) {
        std::cout << "\033[1;33m[Crowdsourcing] DEBUG: Could not increment '" << spellName
                  << "' - not found in limits map\033[0m" << std::endl;
    }
}
