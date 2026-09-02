#include "CrowdsourcingUtils.h"
#include <windows.h>
#include <wincrypt.h>
#include <rpc.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <sys/stat.h>
#include <direct.h>
#include <ctime>

#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "rpcrt4.lib")

namespace CrowdsourcingUtils {

    std::string calculateMD5(const std::vector<uint8_t>& data) {
        return calculateMD5(data.data(), data.size());
    }

    std::string calculateMD5(const uint8_t* data, size_t size) {
        HCRYPTPROV hProv = 0;
        HCRYPTHASH hHash = 0;
        BYTE hash[16];
        DWORD hashLen = 16;

        // Acquire cryptographic provider context
        if (!CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
            std::cerr << "CryptAcquireContext failed: " << GetLastError() << std::endl;
            return "";
        }

        // Create hash object
        if (!CryptCreateHash(hProv, CALG_MD5, 0, 0, &hHash)) {
            std::cerr << "CryptCreateHash failed: " << GetLastError() << std::endl;
            CryptReleaseContext(hProv, 0);
            return "";
        }

        // Hash the data
        if (!CryptHashData(hHash, data, (DWORD)size, 0)) {
            std::cerr << "CryptHashData failed: " << GetLastError() << std::endl;
            CryptDestroyHash(hHash);
            CryptReleaseContext(hProv, 0);
            return "";
        }

        // Get hash value
        if (!CryptGetHashParam(hHash, HP_HASHVAL, hash, &hashLen, 0)) {
            std::cerr << "CryptGetHashParam failed: " << GetLastError() << std::endl;
            CryptDestroyHash(hHash);
            CryptReleaseContext(hProv, 0);
            return "";
        }

        // Convert to hex string
        std::stringstream ss;
        for (int i = 0; i < 16; i++) {
            ss << std::hex << std::setfill('0') << std::setw(2) << (int)hash[i];
        }

        // Cleanup
        CryptDestroyHash(hHash);
        CryptReleaseContext(hProv, 0);

        return ss.str();
    }

    std::string generateUUID() {
        UUID uuid;
        RPC_STATUS status = UuidCreate(&uuid);

        if (status != RPC_S_OK && status != RPC_S_UUID_LOCAL_ONLY) {
            std::cerr << "UuidCreate failed: " << status << std::endl;
            return "";
        }

        RPC_CSTR str;
        status = UuidToStringA(&uuid, &str);

        if (status != RPC_S_OK) {
            std::cerr << "UuidToStringA failed: " << status << std::endl;
            return "";
        }

        std::string result((char*)str);
        RpcStringFreeA(&str);

        return result;
    }

    bool writeWAVFile(const std::string& filename,
                      const std::vector<float>& samples,
                      int sampleRate,
                      int numChannels) {
        return writeWAVFile(filename, samples.data(), samples.size(), sampleRate, numChannels);
    }

    bool writeWAVFile(const std::string& filename,
                      const float* samples,
                      size_t numSamples,
                      int sampleRate,
                      int numChannels) {
        if (!samples || numSamples == 0) {
            std::cerr << "Invalid samples data" << std::endl;
            return false;
        }

        // Open file for binary writing
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return false;
        }

        // Convert float samples to 16-bit PCM
        std::vector<int16_t> pcmSamples(numSamples);
        for (size_t i = 0; i < numSamples; i++) {
            // Clamp to [-1.0, 1.0] and scale to 16-bit range
            float sample = (std::max)(-1.0f, (std::min)(1.0f, samples[i]));
            pcmSamples[i] = (int16_t)(sample * 32767.0f);
        }

        // Calculate sizes
        uint32_t dataSize = (uint32_t)(numSamples * sizeof(int16_t));
        uint32_t fileSize = 36 + dataSize;
        uint16_t bitsPerSample = 16;
        uint16_t blockAlign = (uint16_t)(numChannels * bitsPerSample / 8);
        uint32_t byteRate = sampleRate * blockAlign;

        // Write RIFF header
        file.write("RIFF", 4);
        file.write((char*)&fileSize, 4);
        file.write("WAVE", 4);

        // Write fmt chunk
        file.write("fmt ", 4);
        uint32_t fmtSize = 16;
        uint16_t audioFormat = 1; // PCM
        file.write((char*)&fmtSize, 4);
        file.write((char*)&audioFormat, 2);
        file.write((char*)&numChannels, 2);
        file.write((char*)&sampleRate, 4);
        file.write((char*)&byteRate, 4);
        file.write((char*)&blockAlign, 2);
        file.write((char*)&bitsPerSample, 2);

        // Write data chunk
        file.write("data", 4);
        file.write((char*)&dataSize, 4);
        file.write((char*)pcmSamples.data(), dataSize);

        file.close();

        if (file.fail()) {
            std::cerr << "Error writing WAV file: " << filename << std::endl;
            return false;
        }

        return true;
    }

    bool readWAVFile(const std::string& filename, std::vector<uint8_t>& outData) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for reading: " << filename << std::endl;
            return false;
        }

        // Get file size
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Read entire file
        outData.resize(size);
        if (!file.read((char*)outData.data(), size)) {
            std::cerr << "Failed to read file: " << filename << std::endl;
            return false;
        }

        file.close();
        return true;
    }

    bool createDirectoryRecursive(const std::string& path) {
        if (path.empty()) {
            return false;
        }

        // Check if directory already exists
        struct stat info;
        if (stat(path.c_str(), &info) == 0) {
            if (info.st_mode & S_IFDIR) {
                return true; // Directory exists
            } else {
                std::cerr << "Path exists but is not a directory: " << path << std::endl;
                return false; // Path exists but is not a directory
            }
        }

        // Find parent directory
        size_t pos = path.find_last_of("\\/");
        if (pos != std::string::npos) {
            std::string parent = path.substr(0, pos);
            if (!createDirectoryRecursive(parent)) {
                return false;
            }
        }

        // Create this directory
        if (_mkdir(path.c_str()) != 0) {
            if (errno != EEXIST) {
                std::cerr << "Failed to create directory: " << path << " (errno: " << errno << ")" << std::endl;
                return false;
            }
        }

        return true;
    }

    std::string getCurrentTimestamp() {
        // Get current time
        time_t now = time(NULL);
        struct tm timeinfo;

        // Convert to UTC
        if (gmtime_s(&timeinfo, &now) != 0) {
            std::cerr << "Failed to get UTC time" << std::endl;
            return "";
        }

        // Format as ISO 8601
        char buffer[32];
        strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &timeinfo);

        return std::string(buffer);
    }

} // namespace CrowdsourcingUtils
