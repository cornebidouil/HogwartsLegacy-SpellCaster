#include "CrowdsourcingClient.h"
#include "miniz/miniz.h"
#include <windows.h>
#include <winhttp.h>
#include <iostream>
#include <sstream>
#include <algorithm>

#pragma comment(lib, "winhttp.lib")

// Helper functions for JSON parsing
namespace {
    std::string escapeJSONInternal(const std::string& str) {
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

    std::string extractValue(const std::string& json, const std::string& key) {
        // Search for "key": pattern to avoid matching values
        std::string searchPattern = "\"" + key + "\":";
        size_t pos = json.find(searchPattern);
        if (pos == std::string::npos) return "";

        // Position is now at the opening quote, advance past the pattern
        pos += searchPattern.length();

        // Skip whitespace
        while (pos < json.length() && (json[pos] == ':' || json[pos] == ' ' || json[pos] == '\t')) {
            pos++;
        }

        if (pos >= json.length()) return "";

        if (json[pos] == '"') {
            // String value
            pos++;
            size_t end = pos;
            while (end < json.length() && json[end] != '"') {
                if (json[end] == '\\') end++; // Skip escaped character
                end++;
            }
            return json.substr(pos, end - pos);
        } else {
            // Number or boolean
            size_t end = pos;
            while (end < json.length() && json[end] != ',' && json[end] != '}' && json[end] != ']') {
                end++;
            }
            std::string value = json.substr(pos, end - pos);
            // Trim whitespace
            value.erase(0, value.find_first_not_of(" \t\r\n"));
            value.erase(value.find_last_not_of(" \t\r\n") + 1);
            return value;
        }
    }

    int extractNumberInternal(const std::string& json, const std::string& key) {
        std::string value = extractValue(json, key);
        if (value.empty()) return 0;
        try {
            return std::stoi(value);
        } catch (...) {
            return 0;
        }
    }

    bool extractBooleanInternal(const std::string& json, const std::string& key) {
        std::string value = extractValue(json, key);
        return (value == "true");
    }

    std::vector<std::string> extractStringArray(const std::string& json, const std::string& key) {
        std::vector<std::string> result;
        std::string searchKey = "\"" + key + "\"";
        size_t pos = json.find(searchKey);
        if (pos == std::string::npos) return result;

        pos = json.find("[", pos);
        if (pos == std::string::npos) return result;

        size_t end = json.find("]", pos);
        if (end == std::string::npos) return result;

        std::string arrayContent = json.substr(pos + 1, end - pos - 1);

        // Extract each string
        size_t strPos = 0;
        while ((strPos = arrayContent.find("\"", strPos)) != std::string::npos) {
            strPos++;
            size_t strEnd = arrayContent.find("\"", strPos);
            if (strEnd == std::string::npos) break;

            result.push_back(arrayContent.substr(strPos, strEnd - strPos));
            strPos = strEnd + 1;
        }

        return result;
    }
}

CrowdsourcingClient::CrowdsourcingClient(const std::string& serverHost)
    : serverHost_(serverHost), debugLogging_(false) {
}

CrowdsourcingClient::~CrowdsourcingClient() {
}

bool CrowdsourcingClient::sync(const SyncRequest& request, SyncResponse& outResponse) {
    std::string jsonBody = buildSyncRequestJSON(request);

    if (debugLogging_) {
        std::cout << "\033[1;34m[HTTP] POST https://" << serverHost_ << "/api/app_sync.php\033[0m" << std::endl;
    }

    std::string response;
    int statusCode = 0;

    if (!httpPostJson("/api/app_sync.php", jsonBody, response, statusCode)) {
        outResponse.success = false;
        outResponse.error = "Network error - failed to connect";
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] Network error: Failed to connect to server\033[0m" << std::endl;
        }
        return false;
    }

    if (debugLogging_) {
        std::cout << "\033[1;34m[HTTP] Response status: " << statusCode << "\033[0m" << std::endl;
    }

    if (statusCode != 200) {
        outResponse.success = false;
        outResponse.error = extractValue(response, "error");
        if (outResponse.error.empty()) {
            outResponse.error = "HTTP " + std::to_string(statusCode);
        }
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] Error: " << outResponse.error << "\033[0m" << std::endl;
        }
        return false;
    }

    return parseSyncResponse(response, outResponse);
}

bool CrowdsourcingClient::uploadBatch(const std::string& token,
                                       const std::vector<CrowdsourcingStorage::PendingRecording>& recordings,
                                       const std::string& deviceInfo,
                                       BatchUploadResult& outResult) {
    // Build ZIP archive
    std::vector<uint8_t> archiveData = buildArchive(recordings, deviceInfo);
    if (archiveData.empty()) {
        outResult.success = false;
        outResult.error = "Failed to build ZIP archive";
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] Error: Failed to build ZIP archive\033[0m" << std::endl;
        }
        return false;
    }

    if (debugLogging_) {
        std::cout << "\033[1;34m[HTTP] POST https://" << serverHost_ << "/api/app_upload_batch.php\033[0m" << std::endl;
        std::cout << "\033[1;34m[HTTP] Uploading batch: " << recordings.size()
                  << " recordings, " << (archiveData.size() / 1024) << " KB\033[0m" << std::endl;
    }

    std::string response;
    int statusCode = 0;

    if (!httpPostMultipart("/api/app_upload_batch.php", token, archiveData, response, statusCode)) {
        outResult.success = false;
        outResult.error = "Network error - failed to connect";
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] Network error: Failed to connect to server\033[0m" << std::endl;
        }
        return false;
    }

    if (debugLogging_) {
        std::cout << "\033[1;34m[HTTP] Response status: " << statusCode << "\033[0m" << std::endl;
    }

    if (statusCode != 200) {
        outResult.success = false;
        outResult.error = extractValue(response, "error");
        if (outResult.error.empty()) {
            outResult.error = "HTTP " + std::to_string(statusCode);
        }
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] Error: " << outResult.error << "\033[0m" << std::endl;
        }
        return false;
    }

    return parseBatchUploadResponse(response, outResult);
}

bool CrowdsourcingClient::httpPostJson(const std::string& path,
                                        const std::string& jsonBody,
                                        std::string& outResponse,
                                        int& outStatusCode) {
    std::wstring wHost(serverHost_.begin(), serverHost_.end());
    std::wstring wPath(path.begin(), path.end());

    HINTERNET hSession = WinHttpOpen(L"SpellCaster/1.0",
        WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS, 0);

    if (!hSession) {
        DWORD error = GetLastError();
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] WinHttpOpen failed: Error " << error << "\033[0m" << std::endl;
        }
        return false;
    }

    HINTERNET hConnect = WinHttpConnect(hSession, wHost.c_str(),
        INTERNET_DEFAULT_HTTPS_PORT, 0);

    if (!hConnect) {
        DWORD error = GetLastError();
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] WinHttpConnect failed: Error " << error << "\033[0m" << std::endl;
        }
        WinHttpCloseHandle(hSession);
        return false;
    }

    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"POST", wPath.c_str(),
        NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
        WINHTTP_FLAG_SECURE);

    if (!hRequest) {
        DWORD error = GetLastError();
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] WinHttpOpenRequest failed: Error " << error << "\033[0m" << std::endl;
        }
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    const wchar_t* headers = L"Content-Type: application/json\r\n";

    BOOL result = WinHttpSendRequest(hRequest, headers, -1,
        (LPVOID)jsonBody.c_str(), (DWORD)jsonBody.size(),
        (DWORD)jsonBody.size(), 0);

    if (!result) {
        DWORD error = GetLastError();
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] WinHttpSendRequest failed: Error " << error << "\033[0m" << std::endl;
        }
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    result = WinHttpReceiveResponse(hRequest, NULL);
    if (!result) {
        DWORD error = GetLastError();
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] WinHttpReceiveResponse failed: Error " << error << "\033[0m" << std::endl;
        }
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    // Get status code
    DWORD statusCodeDword = 0;
    DWORD statusCodeSize = sizeof(statusCodeDword);
    WinHttpQueryHeaders(hRequest,
        WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
        WINHTTP_HEADER_NAME_BY_INDEX,
        &statusCodeDword, &statusCodeSize, WINHTTP_NO_HEADER_INDEX);
    outStatusCode = (int)statusCodeDword;

    // Read response body
    outResponse.clear();
    DWORD bytesRead = 0;
    char buffer[4096];

    while (WinHttpReadData(hRequest, buffer, sizeof(buffer), &bytesRead)) {
        if (bytesRead == 0) break;
        outResponse.append(buffer, bytesRead);
    }

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);

    return true;
}

bool CrowdsourcingClient::httpPostMultipart(const std::string& path,
                                             const std::string& token,
                                             const std::vector<uint8_t>& archiveData,
                                             std::string& outResponse,
                                             int& outStatusCode) {
    std::wstring wHost(serverHost_.begin(), serverHost_.end());
    std::wstring wPath(path.begin(), path.end());
    std::string boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";

    // Build multipart body
    std::vector<uint8_t> body;

    // Token field
    std::string tokenPart = "--" + boundary + "\r\n";
    tokenPart += "Content-Disposition: form-data; name=\"token\"\r\n\r\n";
    tokenPart += token + "\r\n";
    body.insert(body.end(), tokenPart.begin(), tokenPart.end());

    // Archive field
    std::string archivePart = "--" + boundary + "\r\n";
    archivePart += "Content-Disposition: form-data; name=\"archive\"; filename=\"upload.zip\"\r\n";
    archivePart += "Content-Type: application/zip\r\n\r\n";
    body.insert(body.end(), archivePart.begin(), archivePart.end());
    body.insert(body.end(), archiveData.begin(), archiveData.end());

    std::string endPart = "\r\n--" + boundary + "--\r\n";
    body.insert(body.end(), endPart.begin(), endPart.end());

    // Send request
    HINTERNET hSession = WinHttpOpen(L"SpellCaster/1.0",
        WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS, 0);

    if (!hSession) {
        DWORD error = GetLastError();
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] WinHttpOpen failed: Error " << error << "\033[0m" << std::endl;
        }
        return false;
    }

    HINTERNET hConnect = WinHttpConnect(hSession, wHost.c_str(),
        INTERNET_DEFAULT_HTTPS_PORT, 0);

    if (!hConnect) {
        DWORD error = GetLastError();
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] WinHttpConnect failed: Error " << error << "\033[0m" << std::endl;
        }
        WinHttpCloseHandle(hSession);
        return false;
    }

    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"POST", wPath.c_str(),
        NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
        WINHTTP_FLAG_SECURE);

    if (!hRequest) {
        DWORD error = GetLastError();
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] WinHttpOpenRequest failed: Error " << error << "\033[0m" << std::endl;
        }
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    std::wstring headers = L"Content-Type: multipart/form-data; boundary=" +
        std::wstring(boundary.begin(), boundary.end()) + L"\r\n";

    BOOL result = WinHttpSendRequest(hRequest, headers.c_str(), -1,
        body.data(), (DWORD)body.size(), (DWORD)body.size(), 0);

    if (!result) {
        DWORD error = GetLastError();
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] WinHttpSendRequest failed: Error " << error << "\033[0m" << std::endl;
        }
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    result = WinHttpReceiveResponse(hRequest, NULL);
    if (!result) {
        DWORD error = GetLastError();
        if (debugLogging_) {
            std::cerr << "\033[1;31m[HTTP] WinHttpReceiveResponse failed: Error " << error << "\033[0m" << std::endl;
        }
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    // Get status code
    DWORD statusCodeDword = 0;
    DWORD statusCodeSize = sizeof(statusCodeDword);
    WinHttpQueryHeaders(hRequest,
        WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
        WINHTTP_HEADER_NAME_BY_INDEX,
        &statusCodeDword, &statusCodeSize, WINHTTP_NO_HEADER_INDEX);
    outStatusCode = (int)statusCodeDword;

    // Read response body
    outResponse.clear();
    DWORD bytesRead = 0;
    char buffer[4096];

    while (WinHttpReadData(hRequest, buffer, sizeof(buffer), &bytesRead)) {
        if (bytesRead == 0) break;
        outResponse.append(buffer, bytesRead);
    }

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);

    return true;
}

std::vector<uint8_t> CrowdsourcingClient::buildArchive(
    const std::vector<CrowdsourcingStorage::PendingRecording>& recordings,
    const std::string& deviceInfo) {

    mz_zip_archive zip = {};

    if (!mz_zip_writer_init_heap(&zip, 0, 0)) {
        std::cerr << "[Crowdsourcing] Failed to init ZIP writer" << std::endl;
        return {};
    }

    // Build manifest JSON
    std::string manifest = "{";
    manifest += "\"device_info\":\"" + escapeJSON(deviceInfo) + "\",";
    manifest += "\"files\":[";

    for (size_t i = 0; i < recordings.size(); i++) {
        if (i > 0) manifest += ",";

        std::string filename = recordings[i].metadata.hash + ".wav";

        manifest += "{";
        manifest += "\"filename\":\"" + escapeJSON(filename) + "\",";
        manifest += "\"formula\":\"" + escapeJSON(recordings[i].metadata.formula) + "\",";
        manifest += "\"hash\":\"" + escapeJSON(recordings[i].metadata.hash) + "\"";
        manifest += "}";
    }

    manifest += "]}";

    // Add manifest.json to archive
    if (!mz_zip_writer_add_mem(&zip, "manifest.json",
            manifest.c_str(), manifest.size(), MZ_BEST_COMPRESSION)) {
        std::cerr << "[Crowdsourcing] Failed to add manifest to ZIP" << std::endl;
        mz_zip_writer_end(&zip);
        return {};
    }

    // Add each WAV file
    for (const auto& rec : recordings) {
        std::string filename = rec.metadata.hash + ".wav";

        if (!mz_zip_writer_add_mem(&zip, filename.c_str(),
                rec.wavData.data(), rec.wavData.size(), MZ_BEST_COMPRESSION)) {
            std::cerr << "[Crowdsourcing] Failed to add WAV to ZIP: " << filename << std::endl;
            mz_zip_writer_end(&zip);
            return {};
        }
    }

    // Finalize archive
    void* archiveData = nullptr;
    size_t archiveSize = 0;

    if (!mz_zip_writer_finalize_heap_archive(&zip, &archiveData, &archiveSize)) {
        std::cerr << "[Crowdsourcing] Failed to finalize ZIP archive" << std::endl;
        mz_zip_writer_end(&zip);
        return {};
    }

    std::vector<uint8_t> result(
        static_cast<uint8_t*>(archiveData),
        static_cast<uint8_t*>(archiveData) + archiveSize
    );

    mz_free(archiveData);
    mz_zip_writer_end(&zip);

    return result;
}

std::string CrowdsourcingClient::buildSyncRequestJSON(const SyncRequest& request) {
    std::string json = "{";

    // Auth section
    json += "\"auth\":{";
    json += "\"type\":\"" + request.authType + "\"";

    if (request.authType == "account") {
        json += ",\"username\":\"" + escapeJSON(request.username) + "\"";
        json += ",\"password\":\"" + escapeJSON(request.password) + "\"";
    } else {
        json += ",\"uuid\":\"" + escapeJSON(request.uuid) + "\"";
        if (!request.nationality.empty()) {
            json += ",\"nationality\":\"" + escapeJSON(request.nationality) + "\"";
        }
        if (!request.gender.empty()) {
            json += ",\"gender\":\"" + escapeJSON(request.gender) + "\"";
        }
    }
    json += "},";

    // Device section
    json += "\"device\":{";
    json += "\"name\":\"" + escapeJSON(request.deviceName) + "\"";
    json += "},";

    // Recordings array
    json += "\"recordings\":[";
    for (size_t i = 0; i < request.recordings.size(); i++) {
        if (i > 0) json += ",";
        json += "{";
        json += "\"formula\":\"" + escapeJSON(request.recordings[i].formula) + "\",";
        json += "\"hash\":\"" + escapeJSON(request.recordings[i].hash) + "\",";
        json += "\"size\":" + std::to_string(request.recordings[i].size);
        json += "}";
    }
    json += "]";

    json += "}";

    return json;
}

bool CrowdsourcingClient::parseSyncResponse(const std::string& json, SyncResponse& outResponse) {
    outResponse.success = extractBooleanInternal(json, "success");
    if (!outResponse.success) {
        outResponse.error = extractValue(json, "error");
        return false;
    }

    outResponse.token = extractValue(json, "token");
    outResponse.tokenExpires = extractNumberInternal(json, "token_expires");
    outResponse.wanted = extractStringArray(json, "wanted");

    // Parse limits object: "limits": { "Accio": {"current": 0, "max": 4}, ... }
    size_t limitsPos = json.find("\"limits\":");
    if (limitsPos != std::string::npos) {
        // Find opening brace of limits object
        size_t limitsStart = json.find("{", limitsPos + 9);
        if (limitsStart != std::string::npos) {
            // Find matching closing brace (need to handle nested braces)
            int braceCount = 1;
            size_t limitsEnd = limitsStart + 1;
            while (limitsEnd < json.length() && braceCount > 0) {
                if (json[limitsEnd] == '{') braceCount++;
                else if (json[limitsEnd] == '}') braceCount--;
                limitsEnd++;
            }

            if (braceCount == 0) {
                std::string limitsContent = json.substr(limitsStart + 1, limitsEnd - limitsStart - 2);

                // Parse each formula: "FormulaName":{"current":N,"max":M}
                size_t pos = 0;
                while (pos < limitsContent.length()) {
                    // Find formula name
                    size_t nameStart = limitsContent.find("\"", pos);
                    if (nameStart == std::string::npos) break;
                    nameStart++;

                    size_t nameEnd = limitsContent.find("\"", nameStart);
                    if (nameEnd == std::string::npos) break;

                    std::string formulaName = limitsContent.substr(nameStart, nameEnd - nameStart);

                    // Find the object for this formula
                    size_t objStart = limitsContent.find("{", nameEnd);
                    if (objStart == std::string::npos) break;

                    size_t objEnd = limitsContent.find("}", objStart);
                    if (objEnd == std::string::npos) break;

                    std::string objContent = limitsContent.substr(objStart, objEnd - objStart + 1);

                    // Extract current and max
                    int current = extractNumberInternal(objContent, "current");
                    int max = extractNumberInternal(objContent, "max");

                    outResponse.limits[formulaName] = {current, max};

                    pos = objEnd + 1;
                }
            }
        }
    }

    return true;
}

bool CrowdsourcingClient::parseBatchUploadResponse(const std::string& json, BatchUploadResult& outResult) {
    outResult.success = extractBooleanInternal(json, "success");
    if (!outResult.success) {
        outResult.error = extractValue(json, "error");
        return false;
    }

    outResult.accepted = extractNumberInternal(json, "accepted");
    outResult.rejected = extractNumberInternal(json, "rejected");

    // Parse results array
    size_t pos = json.find("\"results\"");
    if (pos != std::string::npos) {
        pos = json.find("[", pos);
        if (pos != std::string::npos) {
            size_t end = json.find("]", pos);
            if (end != std::string::npos) {
                std::string arrayContent = json.substr(pos, end - pos + 1);

                // Extract each result object
                size_t objPos = 0;
                while ((objPos = arrayContent.find("{", objPos)) != std::string::npos) {
                    size_t objEnd = arrayContent.find("}", objPos);
                    if (objEnd == std::string::npos) break;

                    std::string objStr = arrayContent.substr(objPos, objEnd - objPos + 1);

                    FileResult result;
                    result.hash = extractValue(objStr, "hash");
                    result.status = extractValue(objStr, "status");
                    result.reason = extractValue(objStr, "reason");

                    if (!result.hash.empty()) {
                        outResult.results.push_back(result);
                    }

                    objPos = objEnd + 1;
                }
            }
        }
    }

    // Parse limits object (same as in parseSyncResponse)
    size_t limitsPos = json.find("\"limits\":");
    if (limitsPos != std::string::npos) {
        // Find opening brace of limits object
        size_t limitsStart = json.find("{", limitsPos + 9);
        if (limitsStart != std::string::npos) {
            // Find matching closing brace (handle nested braces)
            int braceCount = 1;
            size_t limitsEnd = limitsStart + 1;
            while (limitsEnd < json.length() && braceCount > 0) {
                if (json[limitsEnd] == '{') braceCount++;
                else if (json[limitsEnd] == '}') braceCount--;
                limitsEnd++;
            }

            if (braceCount == 0) {
                std::string limitsContent = json.substr(limitsStart + 1, limitsEnd - limitsStart - 2);

                // Parse each formula: "FormulaName":{"current":N,"max":M}
                size_t formulaPos = 0;
                while (formulaPos < limitsContent.length()) {
                    // Find formula name
                    size_t nameStart = limitsContent.find("\"", formulaPos);
                    if (nameStart == std::string::npos) break;
                    nameStart++;

                    size_t nameEnd = limitsContent.find("\"", nameStart);
                    if (nameEnd == std::string::npos) break;

                    std::string formulaName = limitsContent.substr(nameStart, nameEnd - nameStart);

                    // Find the object for this formula
                    size_t objStart = limitsContent.find("{", nameEnd);
                    if (objStart == std::string::npos) break;

                    size_t objEnd = limitsContent.find("}", objStart);
                    if (objEnd == std::string::npos) break;

                    std::string objContent = limitsContent.substr(objStart, objEnd - objStart + 1);

                    // Extract current and max
                    int current = extractNumberInternal(objContent, "current");
                    int max = extractNumberInternal(objContent, "max");

                    outResult.limits[formulaName] = {current, max};

                    formulaPos = objEnd + 1;
                }
            }
        }
    }

    return true;
}

std::string CrowdsourcingClient::escapeJSON(const std::string& str) {
    return escapeJSONInternal(str);
}

std::string CrowdsourcingClient::extractJSONValue(const std::string& json, const std::string& key) {
    return extractValue(json, key);
}

int CrowdsourcingClient::extractJSONNumber(const std::string& json, const std::string& key) {
    return extractNumberInternal(json, key);
}

bool CrowdsourcingClient::extractJSONBoolean(const std::string& json, const std::string& key) {
    return extractBooleanInternal(json, key);
}
