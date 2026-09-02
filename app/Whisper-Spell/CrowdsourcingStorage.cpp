#include "CrowdsourcingStorage.h"
#include "CrowdsourcingUtils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <windows.h>
#include <algorithm>

// Simple JSON parsing/building (manual, no dependencies)
namespace {
    std::string escapeJSON(const std::string& str) {
        std::string result;
        for (char c : str) {
            switch (c) {
                case '"':  result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default:   result += c; break;
            }
        }
        return result;
    }

    std::string unescapeJSON(const std::string& str) {
        std::string result;
        bool escape = false;
        for (char c : str) {
            if (escape) {
                switch (c) {
                    case '"':  result += '"'; break;
                    case '\\': result += '\\'; break;
                    case 'n':  result += '\n'; break;
                    case 'r':  result += '\r'; break;
                    case 't':  result += '\t'; break;
                    default:   result += c; break;
                }
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else {
                result += c;
            }
        }
        return result;
    }

    std::string extractJSONValue(const std::string& json, const std::string& key) {
        std::string searchKey = "\"" + key + "\"";
        size_t pos = json.find(searchKey);
        if (pos == std::string::npos) return "";

        pos = json.find(":", pos);
        if (pos == std::string::npos) return "";

        pos = json.find("\"", pos);
        if (pos == std::string::npos) return "";

        size_t end = json.find("\"", pos + 1);
        if (end == std::string::npos) return "";

        return unescapeJSON(json.substr(pos + 1, end - pos - 1));
    }

    size_t extractJSONNumber(const std::string& json, const std::string& key) {
        std::string searchKey = "\"" + key + "\"";
        size_t pos = json.find(searchKey);
        if (pos == std::string::npos) return 0;

        pos = json.find(":", pos);
        if (pos == std::string::npos) return 0;

        // Skip whitespace
        while (pos < json.length() && (json[pos] == ':' || json[pos] == ' ' || json[pos] == '\t')) {
            pos++;
        }

        std::string numStr;
        while (pos < json.length() && (isdigit(json[pos]) || json[pos] == '.')) {
            numStr += json[pos++];
        }

        if (numStr.empty()) return 0;
        return (size_t)std::stoull(numStr);
    }
}

bool CrowdsourcingStorage::initialize(const std::string& pendingFolder, const std::string& syncedFolder) {
    std::lock_guard<std::mutex> lock(storageMutex_);

    pendingPath_ = pendingFolder;
    syncedPath_ = syncedFolder;
    manifestPath_ = pendingPath_ + "\\manifest.json";

    // Create directories
    if (!CrowdsourcingUtils::createDirectoryRecursive(pendingPath_)) {
        std::cerr << "Failed to create pending folder: " << pendingPath_ << std::endl;
        return false;
    }

    if (!CrowdsourcingUtils::createDirectoryRecursive(syncedPath_)) {
        std::cerr << "Failed to create synced folder: " << syncedPath_ << std::endl;
        return false;
    }

    // Create manifest if it doesn't exist
    std::ifstream manifestCheck(manifestPath_);
    if (!manifestCheck.good()) {
        std::ofstream manifest(manifestPath_);
        if (manifest.is_open()) {
            manifest << "{\"recordings\":[]}";
            manifest.close();
        }
    }

    return true;
}

bool CrowdsourcingStorage::saveRecording(const std::string& spellName,
                                          const std::vector<float>& audioSamples,
                                          int sampleRate) {
    return saveRecording(spellName, audioSamples.data(), audioSamples.size(), sampleRate);
}

bool CrowdsourcingStorage::saveRecording(const std::string& spellName,
                                          const float* audioSamples,
                                          size_t numSamples,
                                          int sampleRate) {
    std::lock_guard<std::mutex> lock(storageMutex_);

    try {
        // Convert float samples to WAV binary data (for hash calculation)
        std::vector<uint8_t> wavData;
        {
            // Temporary WAV file in memory
            std::ostringstream wavStream(std::ios::binary);

            // Convert float samples to 16-bit PCM
            std::vector<int16_t> pcmSamples(numSamples);
            for (size_t i = 0; i < numSamples; i++) {
                float sample = (std::max)(-1.0f, (std::min)(1.0f, audioSamples[i]));
                pcmSamples[i] = (int16_t)(sample * 32767.0f);
            }

            // Calculate sizes
            uint32_t dataSize = (uint32_t)(numSamples * sizeof(int16_t));
            uint32_t fileSize = 36 + dataSize;
            uint16_t bitsPerSample = 16;
            uint16_t numChannels = 1;
            uint16_t blockAlign = (uint16_t)(numChannels * bitsPerSample / 8);
            uint32_t byteRate = sampleRate * blockAlign;

            // Build WAV in memory for hash calculation
            wavData.resize(44 + dataSize);
            size_t offset = 0;

            memcpy(&wavData[offset], "RIFF", 4); offset += 4;
            memcpy(&wavData[offset], &fileSize, 4); offset += 4;
            memcpy(&wavData[offset], "WAVE", 4); offset += 4;
            memcpy(&wavData[offset], "fmt ", 4); offset += 4;
            uint32_t fmtSize = 16;
            uint16_t audioFormat = 1;
            memcpy(&wavData[offset], &fmtSize, 4); offset += 4;
            memcpy(&wavData[offset], &audioFormat, 2); offset += 2;
            memcpy(&wavData[offset], &numChannels, 2); offset += 2;
            memcpy(&wavData[offset], &sampleRate, 4); offset += 4;
            memcpy(&wavData[offset], &byteRate, 4); offset += 4;
            memcpy(&wavData[offset], &blockAlign, 2); offset += 2;
            memcpy(&wavData[offset], &bitsPerSample, 2); offset += 2;
            memcpy(&wavData[offset], "data", 4); offset += 4;
            memcpy(&wavData[offset], &dataSize, 4); offset += 4;
            memcpy(&wavData[offset], pcmSamples.data(), dataSize);
        }

        // Calculate MD5 hash
        std::string hash = CrowdsourcingUtils::calculateMD5(wavData);
        if (hash.empty()) {
            std::cerr << "Failed to calculate MD5 hash" << std::endl;
            return false;
        }

        // Create filename
        std::string filename = hash + ".wav";
        std::string filepath = pendingPath_ + "\\" + filename;

        // Check if file already exists (duplicate recording)
        std::ifstream existingFile(filepath);
        if (existingFile.good()) {
            // Recording already exists, skip
            return true;
        }

        // Write WAV file
        if (!CrowdsourcingUtils::writeWAVFile(filepath, audioSamples, numSamples, sampleRate, 1)) {
            std::cerr << "Failed to write WAV file: " << filepath << std::endl;
            return false;
        }

        // Update manifest
        std::vector<RecordingMetadata> recordings;
        loadManifest(recordings);

        // Check if this hash already exists in manifest
        bool exists = false;
        for (const auto& rec : recordings) {
            if (rec.hash == hash) {
                exists = true;
                break;
            }
        }

        if (!exists) {
            RecordingMetadata metadata;
            metadata.hash = hash;
            metadata.formula = spellName;
            metadata.timestamp = CrowdsourcingUtils::getCurrentTimestamp();
            metadata.filename = filename;
            metadata.fileSize = wavData.size();

            recordings.push_back(metadata);

            // Write manifest
            std::ofstream manifest(manifestPath_);
            if (!manifest.is_open()) {
                std::cerr << "Failed to open manifest for writing" << std::endl;
                return false;
            }

            manifest << "{\"recordings\":[";
            for (size_t i = 0; i < recordings.size(); i++) {
                if (i > 0) manifest << ",";
                manifest << "\n  {"
                         << "\"hash\":\"" << escapeJSON(recordings[i].hash) << "\","
                         << "\"formula\":\"" << escapeJSON(recordings[i].formula) << "\","
                         << "\"timestamp\":\"" << escapeJSON(recordings[i].timestamp) << "\","
                         << "\"filename\":\"" << escapeJSON(recordings[i].filename) << "\","
                         << "\"fileSize\":" << recordings[i].fileSize
                         << "}";
            }
            manifest << "\n]}";
            manifest.close();
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception in saveRecording: " << e.what() << std::endl;
        return false;
    }
}

std::vector<CrowdsourcingStorage::PendingRecording> CrowdsourcingStorage::loadPendingRecordings() {
    std::lock_guard<std::mutex> lock(storageMutex_);

    std::vector<PendingRecording> result;
    std::vector<RecordingMetadata> recordings;

    if (!loadManifest(recordings)) {
        return result;
    }

    for (const auto& metadata : recordings) {
        PendingRecording pending;
        pending.metadata = metadata;

        std::string filepath = pendingPath_ + "\\" + metadata.filename;
        if (CrowdsourcingUtils::readWAVFile(filepath, pending.wavData)) {
            result.push_back(pending);
        } else {
            std::cerr << "Failed to read WAV file: " << filepath << std::endl;
        }
    }

    return result;
}

bool CrowdsourcingStorage::loadManifest(std::vector<RecordingMetadata>& outRecordings) {
    outRecordings.clear();

    std::ifstream manifest(manifestPath_);
    if (!manifest.is_open()) {
        return false;
    }

    std::string jsonStr((std::istreambuf_iterator<char>(manifest)),
                        std::istreambuf_iterator<char>());
    manifest.close();

    // Simple JSON parsing (look for recording objects)
    size_t pos = jsonStr.find("\"recordings\"");
    if (pos == std::string::npos) return false;

    pos = jsonStr.find("[", pos);
    if (pos == std::string::npos) return false;

    size_t endPos = jsonStr.find("]", pos);
    if (endPos == std::string::npos) return false;

    // Extract each recording object
    pos = jsonStr.find("{", pos);
    while (pos != std::string::npos && pos < endPos) {
        size_t objEnd = jsonStr.find("}", pos);
        if (objEnd == std::string::npos || objEnd > endPos) break;

        std::string objStr = jsonStr.substr(pos, objEnd - pos + 1);

        RecordingMetadata metadata;
        metadata.hash = extractJSONValue(objStr, "hash");
        metadata.formula = extractJSONValue(objStr, "formula");
        metadata.timestamp = extractJSONValue(objStr, "timestamp");
        metadata.filename = extractJSONValue(objStr, "filename");
        metadata.fileSize = extractJSONNumber(objStr, "fileSize");

        if (!metadata.hash.empty()) {
            outRecordings.push_back(metadata);
        }

        pos = jsonStr.find("{", objEnd);
    }

    return true;
}

int CrowdsourcingStorage::deleteRecordings(const std::vector<std::string>& hashes) {
    std::lock_guard<std::mutex> lock(storageMutex_);

    int deletedCount = 0;
    std::vector<RecordingMetadata> recordings;

    if (!loadManifest(recordings)) {
        return 0;
    }

    // Delete files from pending folder
    for (const std::string& hash : hashes) {
        std::string filename = hash + ".wav";
        std::string filePath = pendingPath_ + "\\" + filename;

        if (DeleteFileA(filePath.c_str()) || GetLastError() == ERROR_FILE_NOT_FOUND) {
            deletedCount++;

            // Remove from manifest
            recordings.erase(
                std::remove_if(recordings.begin(), recordings.end(),
                    [&hash](const RecordingMetadata& r) { return r.hash == hash; }),
                recordings.end()
            );
        }
    }

    // Update manifest
    if (deletedCount > 0) {
        std::ofstream manifest(manifestPath_);
        if (manifest.is_open()) {
            manifest << "{\"recordings\":[";
            for (size_t i = 0; i < recordings.size(); i++) {
                if (i > 0) manifest << ",";
                manifest << "\n  {"
                         << "\"hash\":\"" << escapeJSON(recordings[i].hash) << "\","
                         << "\"formula\":\"" << escapeJSON(recordings[i].formula) << "\","
                         << "\"timestamp\":\"" << escapeJSON(recordings[i].timestamp) << "\","
                         << "\"filename\":\"" << escapeJSON(recordings[i].filename) << "\","
                         << "\"fileSize\":" << recordings[i].fileSize
                         << "}";
            }
            manifest << "\n]}";
            manifest.close();
        }
    }

    return deletedCount;
}

int CrowdsourcingStorage::markAsSynced(const std::vector<std::string>& hashes) {
    std::lock_guard<std::mutex> lock(storageMutex_);

    int movedCount = 0;
    std::vector<RecordingMetadata> recordings;

    if (!loadManifest(recordings)) {
        return 0;
    }

    // Move files from pending to synced
    for (const std::string& hash : hashes) {
        std::string filename = hash + ".wav";
        std::string srcPath = pendingPath_ + "\\" + filename;
        std::string dstPath = syncedPath_ + "\\" + filename;

        if (MoveFileA(srcPath.c_str(), dstPath.c_str()) || GetLastError() == ERROR_ALREADY_EXISTS) {
            movedCount++;

            // Remove from manifest
            recordings.erase(
                std::remove_if(recordings.begin(), recordings.end(),
                    [&hash](const RecordingMetadata& r) { return r.hash == hash; }),
                recordings.end()
            );
        }
    }

    // Update manifest
    if (movedCount > 0) {
        std::ofstream manifest(manifestPath_);
        if (manifest.is_open()) {
            manifest << "{\"recordings\":[";
            for (size_t i = 0; i < recordings.size(); i++) {
                if (i > 0) manifest << ",";
                manifest << "\n  {"
                         << "\"hash\":\"" << escapeJSON(recordings[i].hash) << "\","
                         << "\"formula\":\"" << escapeJSON(recordings[i].formula) << "\","
                         << "\"timestamp\":\"" << escapeJSON(recordings[i].timestamp) << "\","
                         << "\"filename\":\"" << escapeJSON(recordings[i].filename) << "\","
                         << "\"fileSize\":" << recordings[i].fileSize
                         << "}";
            }
            manifest << "\n]}";
            manifest.close();
        }
    }

    return movedCount;
}

int CrowdsourcingStorage::cleanupSynced(int daysToKeep) {
    std::lock_guard<std::mutex> lock(storageMutex_);

    int deletedCount = 0;
    time_t now = time(NULL);
    time_t threshold = now - (daysToKeep * 24 * 60 * 60);

    WIN32_FIND_DATAA findData;
    std::string searchPath = syncedPath_ + "\\*.wav";
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findData);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            std::string filepath = syncedPath_ + "\\" + findData.cFileName;

            // Get file modification time
            FILETIME ft = findData.ftLastWriteTime;
            ULARGE_INTEGER uli;
            uli.LowPart = ft.dwLowDateTime;
            uli.HighPart = ft.dwHighDateTime;

            // Convert to time_t (FILETIME is 100-nanosecond intervals since Jan 1, 1601)
            time_t fileTime = (time_t)((uli.QuadPart / 10000000ULL) - 11644473600ULL);

            if (fileTime < threshold) {
                if (DeleteFileA(filepath.c_str())) {
                    deletedCount++;
                }
            }
        } while (FindNextFileA(hFind, &findData));

        FindClose(hFind);
    }

    return deletedCount;
}

int CrowdsourcingStorage::cleanupNoiseSamples() {
    std::lock_guard<std::mutex> lock(storageMutex_);

    int deletedCount = 0;
    std::vector<RecordingMetadata> recordings;

    if (!loadManifest(recordings)) {
        return 0;
    }

    // Find and delete noise samples
    std::vector<RecordingMetadata> validRecordings;
    for (const auto& rec : recordings) {
        if (rec.formula == "Noise") {
            // Delete the WAV file
            std::string filepath = pendingPath_ + "\\" + rec.filename;
            if (DeleteFileA(filepath.c_str())) {
                deletedCount++;
            }
        } else {
            // Keep non-noise recordings
            validRecordings.push_back(rec);
        }
    }

    // Update manifest if any noise samples were deleted
    if (deletedCount > 0) {
        std::ofstream manifest(manifestPath_);
        if (manifest.is_open()) {
            manifest << "{\"recordings\":[";
            for (size_t i = 0; i < validRecordings.size(); i++) {
                if (i > 0) manifest << ",";
                manifest << "\n  {"
                         << "\"hash\":\"" << escapeJSON(validRecordings[i].hash) << "\","
                         << "\"formula\":\"" << escapeJSON(validRecordings[i].formula) << "\","
                         << "\"timestamp\":\"" << escapeJSON(validRecordings[i].timestamp) << "\","
                         << "\"filename\":\"" << escapeJSON(validRecordings[i].filename) << "\","
                         << "\"fileSize\":" << validRecordings[i].fileSize
                         << "}";
            }
            manifest << "\n]}";
            manifest.close();
        }
    }

    return deletedCount;
}

int CrowdsourcingStorage::cleanupAllPending() {
    std::lock_guard<std::mutex> lock(storageMutex_);

    int deletedCount = 0;
    std::vector<RecordingMetadata> recordings;

    if (!loadManifest(recordings)) {
        return 0;
    }

    // Delete all pending recordings
    for (const auto& rec : recordings) {
        std::string filepath = pendingPath_ + "\\" + rec.filename;
        if (DeleteFileA(filepath.c_str())) {
            deletedCount++;
        }
    }

    // Clear manifest (empty recordings list)
    if (deletedCount > 0) {
        std::ofstream manifest(manifestPath_);
        if (manifest.is_open()) {
            manifest << "{\"recordings\":[]}";
            manifest.close();
        }
    }

    return deletedCount;
}

CrowdsourcingStorage::Stats CrowdsourcingStorage::getStats() const {
    std::lock_guard<std::mutex> lock(storageMutex_);

    Stats stats;
    stats.pendingCount = countFiles(pendingPath_);
    stats.syncedCount = countFiles(syncedPath_);
    stats.pendingTotalBytes = calculateDirectorySize(pendingPath_);
    stats.syncedTotalBytes = calculateDirectorySize(syncedPath_);

    return stats;
}

int CrowdsourcingStorage::countFiles(const std::string& directory, const std::string& extension) const {
    int count = 0;
    WIN32_FIND_DATAA findData;
    std::string searchPath = directory + "\\*" + extension;
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findData);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                count++;
            }
        } while (FindNextFileA(hFind, &findData));

        FindClose(hFind);
    }

    return count;
}

size_t CrowdsourcingStorage::calculateDirectorySize(const std::string& directory, const std::string& extension) const {
    size_t totalSize = 0;
    WIN32_FIND_DATAA findData;
    std::string searchPath = directory + "\\*" + extension;
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findData);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                ULARGE_INTEGER fileSize;
                fileSize.LowPart = findData.nFileSizeLow;
                fileSize.HighPart = findData.nFileSizeHigh;
                totalSize += fileSize.QuadPart;
            }
        } while (FindNextFileA(hFind, &findData));

        FindClose(hFind);
    }

    return totalSize;
}
