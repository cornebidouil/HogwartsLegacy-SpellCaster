#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace CrowdsourcingUtils {
    /**
     * Calculate MD5 hash of binary data using Windows CryptoAPI
     * @param data Binary data to hash
     * @return MD5 hash as lowercase hex string (32 characters)
     */
    std::string calculateMD5(const std::vector<uint8_t>& data);

    /**
     * Calculate MD5 hash of binary data (raw pointer version)
     * @param data Pointer to binary data
     * @param size Size of data in bytes
     * @return MD5 hash as lowercase hex string (32 characters)
     */
    std::string calculateMD5(const uint8_t* data, size_t size);

    /**
     * Generate a UUID using Windows RPC API
     * @return UUID string in format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (36 characters)
     */
    std::string generateUUID();

    /**
     * Write audio samples to WAV file (PCM format)
     * @param filename Output WAV file path
     * @param samples Audio samples (float values, -1.0 to 1.0)
     * @param sampleRate Sample rate in Hz (e.g., 16000)
     * @param numChannels Number of channels (1=mono, 2=stereo)
     * @return true if successful, false on error
     */
    bool writeWAVFile(const std::string& filename,
                      const std::vector<float>& samples,
                      int sampleRate,
                      int numChannels = 1);

    /**
     * Write audio samples to WAV file (raw pointer version)
     * @param filename Output WAV file path
     * @param samples Pointer to audio samples (float values, -1.0 to 1.0)
     * @param numSamples Number of samples
     * @param sampleRate Sample rate in Hz (e.g., 16000)
     * @param numChannels Number of channels (1=mono, 2=stereo)
     * @return true if successful, false on error
     */
    bool writeWAVFile(const std::string& filename,
                      const float* samples,
                      size_t numSamples,
                      int sampleRate,
                      int numChannels = 1);

    /**
     * Read WAV file into binary data
     * @param filename Input WAV file path
     * @param outData Output vector to receive binary data
     * @return true if successful, false on error
     */
    bool readWAVFile(const std::string& filename, std::vector<uint8_t>& outData);

    /**
     * Create directory if it doesn't exist (creates parent directories as needed)
     * @param path Directory path to create
     * @return true if successful or already exists, false on error
     */
    bool createDirectoryRecursive(const std::string& path);

    /**
     * Get current timestamp in ISO 8601 format (UTC)
     * @return Timestamp string (e.g., "2024-01-15T10:30:00Z")
     */
    std::string getCurrentTimestamp();

} // namespace CrowdsourcingUtils
