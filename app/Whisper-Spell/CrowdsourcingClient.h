#pragma once

#include "CrowdsourcingStorage.h"
#include <string>
#include <vector>
#include <map>

class CrowdsourcingClient {
public:
    struct RecordingInfo {
        std::string formula;
        std::string hash;
        size_t size;
    };

    struct SyncRequest {
        std::string authType;           // "account" or "anonymous"
        std::string username;           // For account auth
        std::string password;           // For account auth
        std::string uuid;               // For anonymous auth
        std::string nationality;        // Optional
        std::string gender;             // Optional
        std::string deviceName;         // Audio device name
        std::vector<RecordingInfo> recordings;
    };

    struct FormulaLimit {
        int current;
        int max;
    };

    struct SyncResponse {
        bool success;
        std::string error;
        std::string token;
        int tokenExpires;
        std::vector<std::string> wanted;
        std::map<std::string, FormulaLimit> limits;
    };

    struct FileResult {
        std::string hash;
        std::string status;  // "accepted" or "rejected"
        std::string reason;  // Only present if rejected
    };

    struct BatchUploadResult {
        bool success;
        std::string error;
        int accepted;
        int rejected;
        std::vector<FileResult> results;
        std::map<std::string, FormulaLimit> limits;
    };

    /**
     * Construct client with server hostname
     * @param serverHost Server hostname (e.g., "hogwartslegacyspellcaster.xyz")
     */
    explicit CrowdsourcingClient(const std::string& serverHost);
    ~CrowdsourcingClient();

    /**
     * Call /api/app_sync.php to authenticate and get list of wanted recordings
     * @param request Sync request with auth and recording metadata
     * @param outResponse Output response with token and wanted list
     * @return true if successful, false on error (check outResponse.error)
     */
    bool sync(const SyncRequest& request, SyncResponse& outResponse);

    /**
     * Call /api/app_upload_batch.php to upload multiple recordings in ZIP archive
     * @param token Authentication token from sync response
     * @param recordings List of recordings to upload
     * @param deviceInfo Audio device information
     * @param outResult Output result with acceptance status per file
     * @return true if successful, false on error (check outResult.error)
     */
    bool uploadBatch(const std::string& token,
                     const std::vector<CrowdsourcingStorage::PendingRecording>& recordings,
                     const std::string& deviceInfo,
                     BatchUploadResult& outResult);

    /**
     * Enable/disable debug logging for HTTP requests
     */
    void setDebugLogging(bool enabled) { debugLogging_ = enabled; }

private:
    std::string serverHost_;
    bool debugLogging_;

    /**
     * HTTP POST with JSON body
     * @param path URL path (e.g., "/api/app_sync.php")
     * @param jsonBody JSON request body
     * @param outResponse Output response body
     * @param outStatusCode Output HTTP status code
     * @return true if successful, false on network error
     */
    bool httpPostJson(const std::string& path,
                      const std::string& jsonBody,
                      std::string& outResponse,
                      int& outStatusCode);

    /**
     * HTTP POST with multipart/form-data
     * @param path URL path
     * @param token Authentication token
     * @param archiveData ZIP archive binary data
     * @param outResponse Output response body
     * @param outStatusCode Output HTTP status code
     * @return true if successful, false on network error
     */
    bool httpPostMultipart(const std::string& path,
                           const std::string& token,
                           const std::vector<uint8_t>& archiveData,
                           std::string& outResponse,
                           int& outStatusCode);

    /**
     * Build ZIP archive with manifest.json and WAV files
     * @param recordings List of recordings to include
     * @param deviceInfo Audio device information
     * @return ZIP archive binary data (empty on error)
     */
    std::vector<uint8_t> buildArchive(
        const std::vector<CrowdsourcingStorage::PendingRecording>& recordings,
        const std::string& deviceInfo);

    /**
     * Build JSON request body for sync endpoint
     */
    std::string buildSyncRequestJSON(const SyncRequest& request);

    /**
     * Parse JSON response from sync endpoint
     */
    bool parseSyncResponse(const std::string& json, SyncResponse& outResponse);

    /**
     * Parse JSON response from upload batch endpoint
     */
    bool parseBatchUploadResponse(const std::string& json, BatchUploadResult& outResult);

    /**
     * Escape string for JSON
     */
    static std::string escapeJSON(const std::string& str);

    /**
     * Extract JSON string value by key
     */
    static std::string extractJSONValue(const std::string& json, const std::string& key);

    /**
     * Extract JSON number value by key
     */
    static int extractJSONNumber(const std::string& json, const std::string& key);

    /**
     * Extract JSON boolean value by key
     */
    static bool extractJSONBoolean(const std::string& json, const std::string& key);
};
