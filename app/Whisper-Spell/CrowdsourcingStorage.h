#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <cstdint>

class CrowdsourcingStorage {
public:
    struct RecordingMetadata {
        std::string hash;
        std::string formula;
        std::string timestamp;
        std::string filename;
        size_t fileSize;
    };

    struct PendingRecording {
        RecordingMetadata metadata;
        std::vector<uint8_t> wavData;
    };

    CrowdsourcingStorage() = default;
    ~CrowdsourcingStorage() = default;

    /**
     * Initialize storage with directory paths
     * Creates directories if they don't exist
     * @param pendingFolder Path to pending recordings folder
     * @param syncedFolder Path to synced recordings folder
     * @return true if successful, false on error
     */
    bool initialize(const std::string& pendingFolder, const std::string& syncedFolder);

    /**
     * Save a recording to pending folder (thread-safe)
     * Calculates MD5 hash, writes WAV file, updates manifest
     * @param spellName Name of the spell being cast
     * @param audioSamples Float audio samples (-1.0 to 1.0)
     * @param sampleRate Sample rate in Hz
     * @return true if successful, false on error
     */
    bool saveRecording(const std::string& spellName,
                       const std::vector<float>& audioSamples,
                       int sampleRate);

    /**
     * Save a recording to pending folder (raw pointer version)
     * @param spellName Name of the spell being cast
     * @param audioSamples Pointer to float audio samples
     * @param numSamples Number of samples
     * @param sampleRate Sample rate in Hz
     * @return true if successful, false on error
     */
    bool saveRecording(const std::string& spellName,
                       const float* audioSamples,
                       size_t numSamples,
                       int sampleRate);

    /**
     * Load all pending recordings from disk (thread-safe)
     * @return Vector of pending recordings with metadata and WAV data
     */
    std::vector<PendingRecording> loadPendingRecordings();

    /**
     * Delete specific recordings from pending folder (after successful upload)
     * @param hashes List of MD5 hashes to delete
     * @return Number of successfully deleted files
     */
    int deleteRecordings(const std::vector<std::string>& hashes);

    /**
     * Mark recordings as synced (move from pending to synced folder)
     * @param hashes List of MD5 hashes to mark as synced
     * @return Number of successfully moved files
     * @deprecated Use deleteRecordings instead - no need to keep uploaded files
     */
    int markAsSynced(const std::vector<std::string>& hashes);

    /**
     * Cleanup old synced files (to prevent disk space buildup)
     * @param daysToKeep Keep files newer than this many days (default: 7)
     * @return Number of deleted files
     * @deprecated Not used anymore since we delete uploaded files immediately
     */
    int cleanupSynced(int daysToKeep = 7);

    /**
     * Delete noise samples from pending folder
     * Removes recordings with formula "Noise" from manifest and disk
     * @return Number of deleted noise samples
     */
    int cleanupNoiseSamples();

    /**
     * Delete all pending recordings (full cleanup after sync attempt)
     * Removes all recordings from pending folder and clears manifest
     * @return Number of deleted recordings
     */
    int cleanupAllPending();

    /**
     * Get statistics about stored recordings
     */
    struct Stats {
        int pendingCount = 0;
        int syncedCount = 0;
        size_t pendingTotalBytes = 0;
        size_t syncedTotalBytes = 0;
    };
    Stats getStats() const;

private:
    std::string pendingPath_;
    std::string syncedPath_;
    std::string manifestPath_;
    mutable std::mutex storageMutex_;

    /**
     * Update manifest.json in pending folder
     * Must be called with storageMutex_ locked
     */
    bool updateManifest();

    /**
     * Load manifest.json from pending folder
     * @param outRecordings Vector to receive recording metadata
     * @return true if successful, false if manifest doesn't exist or is invalid
     */
    bool loadManifest(std::vector<RecordingMetadata>& outRecordings);

    /**
     * Count files in a directory
     */
    int countFiles(const std::string& directory, const std::string& extension = ".wav") const;

    /**
     * Calculate total size of files in directory
     */
    size_t calculateDirectorySize(const std::string& directory, const std::string& extension = ".wav") const;
};
