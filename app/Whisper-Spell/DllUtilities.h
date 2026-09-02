// DllUtilities.h - Minimal header-only version
#pragma once

#include <string>
#include <filesystem>
#include <windows.h>
#include <shlobj.h>
#include <vector>

#pragma comment(lib, "version.lib")
#pragma comment(lib, "shell32.lib")

namespace DllUtils {

    // Check if a folder exists
    inline bool folderExists(const std::string& folderPath) {
        return std::filesystem::exists(folderPath) && std::filesystem::is_directory(folderPath);
    }

    // Get %APPDATA%/Roaming/ directory
    inline std::string getRoamingAppDataPath() {
        wchar_t* path = nullptr;
        HRESULT hr = SHGetKnownFolderPath(FOLDERID_RoamingAppData, 0, nullptr, &path);

        if (SUCCEEDED(hr) && path) {
            int size = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, nullptr, nullptr);
            std::string result(size - 1, '\0');
            WideCharToMultiByte(CP_UTF8, 0, path, -1, result.data(), size, nullptr, nullptr);
            CoTaskMemFree(path);
            return result;
        }

        CoTaskMemFree(path);
        return "";
    }

    // Check if a file exists in a directory
    inline bool fileExistsInDirectory(const std::string& directory, const std::string& filename) {
        std::filesystem::path filePath = std::filesystem::path(directory) / filename;
        return std::filesystem::exists(filePath) && std::filesystem::is_regular_file(filePath);
    }

    // Get file version (returns "0.0.0.0" if no version info)
    inline std::string getFileVersion(const std::string& filePath) {
        // Convert to wide string
        int wideSize = MultiByteToWideChar(CP_UTF8, 0, filePath.c_str(), -1, nullptr, 0);
        if (wideSize == 0) return "0.0.0.0";

        std::wstring wideFilePath(wideSize - 1, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, filePath.c_str(), -1, wideFilePath.data(), wideSize);

        // Get version info size
        DWORD versionInfoSize = GetFileVersionInfoSizeW(wideFilePath.c_str(), nullptr);
        if (versionInfoSize == 0) return "0.0.0.0";

        // Get version info
        std::vector<BYTE> versionInfo(versionInfoSize);
        if (!GetFileVersionInfoW(wideFilePath.c_str(), 0, versionInfoSize, versionInfo.data())) {
            return "0.0.0.0";
        }

        // Get fixed version info
        VS_FIXEDFILEINFO* fileInfo = nullptr;
        UINT len = 0;

        if (VerQueryValueW(versionInfo.data(), L"\\", (LPVOID*)&fileInfo, &len)) {
            if (fileInfo && len >= sizeof(VS_FIXEDFILEINFO)) {
                WORD major = HIWORD(fileInfo->dwFileVersionMS);
                WORD minor = LOWORD(fileInfo->dwFileVersionMS);
                WORD build = HIWORD(fileInfo->dwFileVersionLS);
                WORD revision = LOWORD(fileInfo->dwFileVersionLS);

                return std::to_string(major) + "." +
                    std::to_string(minor) + "." +
                    std::to_string(build) + "." +
                    std::to_string(revision);
            }
        }

        return "0.0.0.0";
    }

    // Bonus: Create directory if it doesn't exist
    inline bool createDirectory(const std::string& dirPath) {
        try {
            if (folderExists(dirPath)) return true;
            return std::filesystem::create_directories(dirPath);
        }
        catch (...) {
            return false;
        }
    }

    // Bonus: Get full file path
    inline std::string getFilePath(const std::string& directory, const std::string& filename) {
        return (std::filesystem::path(directory) / filename).string();
    }

    // Copy file from source to destination (OVERWRITES existing files by default)
    inline bool copyFile(const std::string& sourcePath, const std::string& destinationPath) {
        try {
            if (!std::filesystem::exists(sourcePath)) return false;

            // Create destination directory if needed
            std::filesystem::path destPath(destinationPath);
            std::string destDir = destPath.parent_path().string();
            if (!destDir.empty()) createDirectory(destDir);

            // Always overwrite existing files
            std::filesystem::copy_file(sourcePath, destinationPath,
                std::filesystem::copy_options::overwrite_existing);
            return true;
        }
        catch (...) {
            return false;
        }
    }

    // Copy file from directory to directory (same filename) - OVERWRITES existing
    inline bool copyFileToDirectory(const std::string& sourceDir, const std::string& destDir,
        const std::string& filename) {
        return copyFile(getFilePath(sourceDir, filename), getFilePath(destDir, filename));
    }

} // namespace DllUtils

/*
Example Usage:

#include "DllUtilities.h"

int main() {
    using namespace DllUtils;

    // Get user's roaming folder
    std::string roaming = getRoamingAppDataPath();
    std::cout << "Roaming: " << roaming << std::endl;

    // Create app folder
    std::string appDir = roaming + "\\MyApp";
    if (createDirectory(appDir)) {
        std::cout << "App directory ready" << std::endl;
    }

    // Check for DLL
    std::string dllName = "myapp.dll";
    if (fileExistsInDirectory(appDir, dllName)) {
        std::string dllPath = getFilePath(appDir, dllName);
        std::string version = getFileVersion(dllPath);
        std::cout << "DLL version: " << version << std::endl;
    } else {
        std::cout << "DLL not found" << std::endl;
    }

    return 0;
}
*/