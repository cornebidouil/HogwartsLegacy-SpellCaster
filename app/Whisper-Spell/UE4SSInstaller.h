#pragma once

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <Windows.h>
#include <string>
#include <vector>
#include <functional>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================================
// UE4SS Installation Status
// ============================================================================

enum class InstallStatus {
    NOT_INSTALLED,          // UE4SS not present at all
    UE4SS_ONLY,             // UE4SS installed but no SpellCaster mod
    OUTDATED,               // SpellCaster mod outdated (needs update)
    UP_TO_DATE,             // Everything current
    GAME_NOT_FOUND,         // Game installation not found
    GAME_RUNNING,           // Game is running (files locked)
    ERROR_OCCURRED          // Error during check
};

struct InstallCheckResult {
    InstallStatus status;
    std::string message;
    bool ue4ssInstalled;
    bool spellCasterInstalled;
    bool ue4ssNeedsUpdate;
    bool spellCasterNeedsUpdate;
    std::string gamePath;
};

struct InstallResult {
    bool success;
    std::string message;
    std::vector<std::string> filesInstalled;
    std::vector<std::string> filesUpdated;
    std::vector<std::string> errors;
};

// ============================================================================
// UE4SS Installer Class
// ============================================================================

class UE4SSInstaller {
public:
    UE4SSInstaller();
    ~UE4SSInstaller() = default;

    // Disable copy
    UE4SSInstaller(const UE4SSInstaller&) = delete;
    UE4SSInstaller& operator=(const UE4SSInstaller&) = delete;

    /**
     * @brief Set the source directory containing UE4SS Plugin files
     * @param path Path to "UE4SS Plugin" folder (typically next to exe)
     */
    void setSourcePath(const std::string& path);

    /**
     * @brief Set/override the game installation path
     * @param path Path to game's Win64 folder
     */
    void setGamePath(const std::string& path);

    /**
     * @brief Check current installation status
     * @return InstallCheckResult with detailed status
     */
    InstallCheckResult checkInstallation();

    /**
     * @brief Perform installation/update based on current status
     * @param forceReinstall Force reinstall even if up-to-date
     * @return InstallResult with success/failure and details
     */
    InstallResult install(bool forceReinstall = false);

    /**
     * @brief Get the detected game path
     */
    std::string getGamePath() const { return gamePath_; }

    /**
     * @brief Check if game is currently running
     */
    bool isGameRunning() const;

    /**
     * @brief Set callback for installation progress
     */
    void setProgressCallback(std::function<void(const std::string&)> callback);

private:
    // Path detection
    bool detectGamePath();
    std::string findSteamGamePath();
    std::string findEpicGamePath();
    std::string searchCommonPaths();
    bool validateGamePath(const std::string& path);

    // Version/hash comparison
    size_t computeFileHash(const std::string& filepath);
    bool fileNeedsUpdate(const std::string& source, const std::string& target);

    // Installation operations
    bool copyFile(const std::string& source, const std::string& target);
    bool copyDirectory(const std::string& source, const std::string& target);
    bool ensureModsEntry(const std::string& modsPath);
    bool backupFile(const std::string& filepath);

    // Helpers
    void log(const std::string& message);
    std::string getRegistryString(HKEY hKey, const std::string& subKey, const std::string& valueName);

    // Member variables
    std::string sourcePath_;                    // Path to UE4SS Plugin source files
    std::string gamePath_;                      // Detected/configured game path
    std::function<void(const std::string&)> progressCallback_;

    // File hashes for version comparison
    struct FileVersionInfo {
        std::string relativePath;
        size_t sourceHash;
        size_t targetHash;
        bool needsUpdate;
    };
    std::vector<FileVersionInfo> versionInfo_;

    // Constants
    static constexpr const char* GAME_EXE_NAME = "HogwartsLegacy.exe";
    static constexpr const char* UE4SS_DLL = "UE4SS.dll";
    static constexpr const char* DWMAPI_DLL = "dwmapi.dll";
    static constexpr const char* SPELLCASTER_DLL = "Mods/SpellCaster/dlls/main.dll";
    static constexpr const char* MODS_TXT = "Mods/mods.txt";
    static constexpr const char* UE4SS_SETTINGS = "UE4SS-settings.ini";

    // Steam paths to search
    static const std::vector<std::string> STEAM_LIBRARY_PATHS;
    static const std::vector<std::string> EPIC_LIBRARY_PATHS;
    static constexpr const char* STEAM_GAME_SUBPATH = "steamapps/common/Hogwarts Legacy/Phoenix/Binaries/Win64";
    static constexpr const char* EPIC_GAME_SUBPATH = "Hogwarts Legacy/Phoenix/Binaries/Win64";
};
