#include "UE4SSInstaller.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <TlHelp32.h>
#include <ShlObj.h>

// ============================================================================
// Static Constants
// ============================================================================

const std::vector<std::string> UE4SSInstaller::STEAM_LIBRARY_PATHS = {
    "C:/Program Files (x86)/Steam/steamapps/common/Hogwarts Legacy",
    "C:/Program Files/Steam/steamapps/common/Hogwarts Legacy",
    "D:/Steam/steamapps/common/Hogwarts Legacy",
    "D:/SteamLibrary/steamapps/common/Hogwarts Legacy",
    "E:/Steam/steamapps/common/Hogwarts Legacy",
    "E:/SteamLibrary/steamapps/common/Hogwarts Legacy",
    "F:/Steam/steamapps/common/Hogwarts Legacy",
    "F:/SteamLibrary/steamapps/common/Hogwarts Legacy"
};

const std::vector<std::string> UE4SSInstaller::EPIC_LIBRARY_PATHS = {
    "C:/Program Files/Epic Games/HogwartsLegacy",
    "C:/Program Files (x86)/Epic Games/HogwartsLegacy",
    "D:/Epic Games/HogwartsLegacy",
    "E:/Epic Games/HogwartsLegacy",
    "F:/Epic Games/HogwartsLegacy"
};

// ============================================================================
// Constructor
// ============================================================================

UE4SSInstaller::UE4SSInstaller() {
    // Default source path is relative to executable
    char exePath[MAX_PATH];
    GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    std::string exeDir = fs::path(exePath).parent_path().string();
    sourcePath_ = exeDir + "/UE4SS Plugin";
}

// ============================================================================
// Configuration
// ============================================================================

void UE4SSInstaller::setSourcePath(const std::string& path) {
    sourcePath_ = path;
}

void UE4SSInstaller::setGamePath(const std::string& path) {
    gamePath_ = path;
}

void UE4SSInstaller::setProgressCallback(std::function<void(const std::string&)> callback) {
    progressCallback_ = callback;
}

void UE4SSInstaller::log(const std::string& message) {
    std::cout << "[UE4SS] " << message << std::endl;
    if (progressCallback_) {
        progressCallback_(message);
    }
}

// ============================================================================
// Game Path Detection
// ============================================================================

bool UE4SSInstaller::detectGamePath() {
    if (!gamePath_.empty() && validateGamePath(gamePath_)) {
        return true;
    }

    log("Searching for Hogwarts Legacy installation...");

    // Try Steam first
    std::string steamPath = findSteamGamePath();
    if (!steamPath.empty()) {
        gamePath_ = steamPath;
        log("Found Steam installation: " + gamePath_);
        return true;
    }

    // Try Epic Games
    std::string epicPath = findEpicGamePath();
    if (!epicPath.empty()) {
        gamePath_ = epicPath;
        log("Found Epic Games installation: " + gamePath_);
        return true;
    }

    // Search common paths
    std::string commonPath = searchCommonPaths();
    if (!commonPath.empty()) {
        gamePath_ = commonPath;
        log("Found installation in common path: " + gamePath_);
        return true;
    }

    return false;
}

std::string UE4SSInstaller::findSteamGamePath() {
    // Try registry first
    std::string steamPath = getRegistryString(
        HKEY_LOCAL_MACHINE,
        "SOFTWARE\\WOW6432Node\\Valve\\Steam",
        "InstallPath"
    );

    if (!steamPath.empty()) {
        std::string gamePath = steamPath + "/steamapps/common/Hogwarts Legacy/Phoenix/Binaries/Win64";
        if (validateGamePath(gamePath)) {
            return gamePath;
        }
    }

    // Try common Steam library paths
    for (const auto& basePath : STEAM_LIBRARY_PATHS) {
        std::string gamePath = basePath + "/Phoenix/Binaries/Win64";
        if (validateGamePath(gamePath)) {
            return gamePath;
        }
    }

    // Try to read Steam's libraryfolders.vdf for additional library paths
    std::string libraryFolders = steamPath.empty() ?
        "C:/Program Files (x86)/Steam/steamapps/libraryfolders.vdf" :
        steamPath + "/steamapps/libraryfolders.vdf";

    std::ifstream vdf(libraryFolders);
    if (vdf.is_open()) {
        std::string line;
        while (std::getline(vdf, line)) {
            // Look for "path" entries
            size_t pathPos = line.find("\"path\"");
            if (pathPos != std::string::npos) {
                size_t firstQuote = line.find('\"', pathPos + 6);
                size_t lastQuote = line.rfind('\"');
                if (firstQuote != std::string::npos && lastQuote > firstQuote) {
                    std::string libPath = line.substr(firstQuote + 1, lastQuote - firstQuote - 1);
                    // Replace double backslashes with forward slashes
                    std::replace(libPath.begin(), libPath.end(), '\\', '/');
                    std::string gamePath = libPath + "/steamapps/common/Hogwarts Legacy/Phoenix/Binaries/Win64";
                    if (validateGamePath(gamePath)) {
                        return gamePath;
                    }
                }
            }
        }
    }

    return "";
}

std::string UE4SSInstaller::findEpicGamePath() {
    // Try common Epic Games paths
    for (const auto& basePath : EPIC_LIBRARY_PATHS) {
        std::string gamePath = basePath + "/Phoenix/Binaries/Win64";
        if (validateGamePath(gamePath)) {
            return gamePath;
        }
    }

    // Try to find Epic Games launcher install location from registry
    std::string epicPath = getRegistryString(
        HKEY_LOCAL_MACHINE,
        "SOFTWARE\\WOW6432Node\\Epic Games\\EpicGamesLauncher",
        "AppDataPath"
    );

    if (!epicPath.empty()) {
        // Epic stores game manifests in a different location
        // Check the manifest files for Hogwarts Legacy
        std::string manifestDir = epicPath + "/Manifests";
        if (fs::exists(manifestDir)) {
            for (const auto& entry : fs::directory_iterator(manifestDir)) {
                if (entry.path().extension() == ".item") {
                    std::ifstream manifest(entry.path());
                    std::string content((std::istreambuf_iterator<char>(manifest)),
                                        std::istreambuf_iterator<char>());
                    if (content.find("Hogwarts") != std::string::npos) {
                        // Parse InstallLocation from manifest
                        size_t pos = content.find("\"InstallLocation\"");
                        if (pos != std::string::npos) {
                            size_t start = content.find(':', pos);
                            size_t firstQuote = content.find('\"', start);
                            size_t lastQuote = content.find('\"', firstQuote + 1);
                            if (firstQuote != std::string::npos && lastQuote > firstQuote) {
                                std::string installPath = content.substr(firstQuote + 1, lastQuote - firstQuote - 1);
                                std::replace(installPath.begin(), installPath.end(), '\\', '/');
                                std::string gamePath = installPath + "/Phoenix/Binaries/Win64";
                                if (validateGamePath(gamePath)) {
                                    return gamePath;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return "";
}

std::string UE4SSInstaller::searchCommonPaths() {
    // Additional common installation paths
    std::vector<std::string> additionalPaths = {
        "C:/Games/Hogwarts Legacy/Phoenix/Binaries/Win64",
        "D:/Games/Hogwarts Legacy/Phoenix/Binaries/Win64",
        "E:/Games/Hogwarts Legacy/Phoenix/Binaries/Win64"
    };

    for (const auto& path : additionalPaths) {
        if (validateGamePath(path)) {
            return path;
        }
    }

    return "";
}

bool UE4SSInstaller::validateGamePath(const std::string& path) {
    if (path.empty()) return false;

    std::string exePath = path + "/" + GAME_EXE_NAME;
    return fs::exists(exePath);
}

std::string UE4SSInstaller::getRegistryString(HKEY hKey, const std::string& subKey, const std::string& valueName) {
    HKEY key;
    if (RegOpenKeyExA(hKey, subKey.c_str(), 0, KEY_READ, &key) != ERROR_SUCCESS) {
        return "";
    }

    char buffer[MAX_PATH];
    DWORD bufferSize = sizeof(buffer);
    DWORD type;

    std::string result;
    if (RegQueryValueExA(key, valueName.c_str(), nullptr, &type, (LPBYTE)buffer, &bufferSize) == ERROR_SUCCESS) {
        if (type == REG_SZ || type == REG_EXPAND_SZ) {
            result = buffer;
        }
    }

    RegCloseKey(key);
    return result;
}

// ============================================================================
// Game Running Check
// ============================================================================

bool UE4SSInstaller::isGameRunning() const {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return false;
    }

    PROCESSENTRY32W pe32;
    pe32.dwSize = sizeof(pe32);

    bool found = false;
    if (Process32FirstW(snapshot, &pe32)) {
        do {
            // Convert wide string to narrow string for comparison
            std::wstring wProcessName = pe32.szExeFile;
            std::string processName(wProcessName.begin(), wProcessName.end());
            // Convert to lowercase for comparison
            std::transform(processName.begin(), processName.end(), processName.begin(), ::tolower);
            if (processName == "hogwartslegacy.exe") {
                found = true;
                break;
            }
        } while (Process32NextW(snapshot, &pe32));
    }

    CloseHandle(snapshot);
    return found;
}

// ============================================================================
// Version/Hash Comparison
// ============================================================================

size_t UE4SSInstaller::computeFileHash(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return 0;
    }

    // Simple hash based on file size and content samples
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read first 4KB and last 4KB for quick hash
    const size_t sampleSize = 4096;
    std::vector<char> buffer(sampleSize);

    size_t hash = fileSize;

    // Hash beginning
    file.read(buffer.data(), (std::min)(sampleSize, fileSize));
    size_t bytesRead = file.gcount();
    for (size_t i = 0; i < bytesRead; i++) {
        hash ^= (static_cast<size_t>(buffer[i]) << ((i % 8) * 8));
        hash = (hash << 5) | (hash >> (sizeof(size_t) * 8 - 5));
    }

    // Hash end if file is large enough
    if (fileSize > sampleSize * 2) {
        file.seekg(-static_cast<std::streamoff>(sampleSize), std::ios::end);
        file.read(buffer.data(), sampleSize);
        bytesRead = file.gcount();
        for (size_t i = 0; i < bytesRead; i++) {
            hash ^= (static_cast<size_t>(buffer[i]) << ((i % 8) * 8));
            hash = (hash << 5) | (hash >> (sizeof(size_t) * 8 - 5));
        }
    }

    return hash;
}

bool UE4SSInstaller::fileNeedsUpdate(const std::string& source, const std::string& target) {
    if (!fs::exists(target)) {
        return true;  // Target doesn't exist, needs install
    }

    if (!fs::exists(source)) {
        return false;  // Source doesn't exist, can't update
    }

    // Compare file sizes first (quick check)
    auto sourceSize = fs::file_size(source);
    auto targetSize = fs::file_size(target);
    if (sourceSize != targetSize) {
        return true;
    }

    // Compare hashes
    size_t sourceHash = computeFileHash(source);
    size_t targetHash = computeFileHash(target);
    return sourceHash != targetHash;
}

// ============================================================================
// Installation Check
// ============================================================================

InstallCheckResult UE4SSInstaller::checkInstallation() {
    InstallCheckResult result;
    result.ue4ssInstalled = false;
    result.spellCasterInstalled = false;
    result.ue4ssNeedsUpdate = false;
    result.spellCasterNeedsUpdate = false;

    // Check source files exist
    if (!fs::exists(sourcePath_)) {
        result.status = InstallStatus::ERROR_OCCURRED;
        result.message = "UE4SS Plugin source folder not found: " + sourcePath_;
        return result;
    }

    // Detect game path
    if (!detectGamePath()) {
        result.status = InstallStatus::GAME_NOT_FOUND;
        result.message = "Hogwarts Legacy installation not found. Please specify game_path in config.ini";
        return result;
    }
    result.gamePath = gamePath_;

    // Check if game is running
    if (isGameRunning()) {
        result.status = InstallStatus::GAME_RUNNING;
        result.message = "Hogwarts Legacy is currently running. Please close the game before installing.";
        return result;
    }

    // Check UE4SS installation
    std::string ue4ssPath = gamePath_ + "/" + UE4SS_DLL;
    std::string dwmapiPath = gamePath_ + "/" + DWMAPI_DLL;
    result.ue4ssInstalled = fs::exists(ue4ssPath) && fs::exists(dwmapiPath);

    // Check SpellCaster mod installation
    std::string spellCasterPath = gamePath_ + "/" + SPELLCASTER_DLL;
    result.spellCasterInstalled = fs::exists(spellCasterPath);

    // Check if updates are needed
    if (result.ue4ssInstalled) {
        std::string sourceUE4SS = sourcePath_ + "/" + UE4SS_DLL;
        std::string sourceDwmapi = sourcePath_ + "/" + DWMAPI_DLL;
        result.ue4ssNeedsUpdate = fileNeedsUpdate(sourceUE4SS, ue4ssPath) ||
                                  fileNeedsUpdate(sourceDwmapi, dwmapiPath);
    }

    if (result.spellCasterInstalled) {
        std::string sourceSpellCaster = sourcePath_ + "/" + SPELLCASTER_DLL;
        result.spellCasterNeedsUpdate = fileNeedsUpdate(sourceSpellCaster, spellCasterPath);
    }

    // Determine status
    if (!result.ue4ssInstalled) {
        result.status = InstallStatus::NOT_INSTALLED;
        result.message = "UE4SS not installed";
    } else if (!result.spellCasterInstalled) {
        result.status = InstallStatus::UE4SS_ONLY;
        result.message = "UE4SS installed but SpellCaster mod not found";
    } else if (result.ue4ssNeedsUpdate || result.spellCasterNeedsUpdate) {
        result.status = InstallStatus::OUTDATED;
        std::string updates;
        if (result.ue4ssNeedsUpdate) updates += "UE4SS";
        if (result.spellCasterNeedsUpdate) {
            if (!updates.empty()) updates += " and ";
            updates += "SpellCaster mod";
        }
        result.message = updates + " need(s) to be updated";
    } else {
        result.status = InstallStatus::UP_TO_DATE;
        result.message = "UE4SS and SpellCaster mod are up to date";
    }

    return result;
}

// ============================================================================
// Installation Operations
// ============================================================================

bool UE4SSInstaller::copyFile(const std::string& source, const std::string& target) {
    try {
        // Create parent directories if needed
        fs::path targetPath(target);
        if (targetPath.has_parent_path()) {
            fs::create_directories(targetPath.parent_path());
        }

        // Copy with overwrite
        fs::copy_file(source, target, fs::copy_options::overwrite_existing);
        return true;
    } catch (const std::exception& e) {
        log("Error copying " + source + " to " + target + ": " + e.what());
        return false;
    }
}

bool UE4SSInstaller::copyDirectory(const std::string& source, const std::string& target) {
    try {
        fs::create_directories(target);
        fs::copy(source, target, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        return true;
    } catch (const std::exception& e) {
        log("Error copying directory " + source + " to " + target + ": " + e.what());
        return false;
    }
}

bool UE4SSInstaller::backupFile(const std::string& filepath) {
    if (!fs::exists(filepath)) {
        return true;  // Nothing to backup
    }

    std::string backupPath = filepath + ".backup";
    try {
        fs::copy_file(filepath, backupPath, fs::copy_options::overwrite_existing);
        return true;
    } catch (const std::exception& e) {
        log("Warning: Could not backup " + filepath + ": " + e.what());
        return false;
    }
}

bool UE4SSInstaller::ensureModsEntry(const std::string& modsPath) {
    std::string modsTxtPath = modsPath + "/mods.txt";

    // Read existing mods.txt
    std::vector<std::string> lines;
    std::ifstream inFile(modsTxtPath);
    if (inFile.is_open()) {
        std::string line;
        while (std::getline(inFile, line)) {
            lines.push_back(line);
        }
        inFile.close();
    }

    // Check if SpellCaster is already enabled
    bool hasSpellCaster = false;
    for (const auto& line : lines) {
        // Skip comments and empty lines
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));
        if (trimmed.empty() || trimmed[0] == ';') continue;

        // Check for SpellCaster entry
        if (trimmed.find("SpellCaster") != std::string::npos) {
            // Check if it's enabled (starts with name, not with 0)
            size_t colonPos = trimmed.find(':');
            if (colonPos != std::string::npos) {
                hasSpellCaster = true;
                break;
            }
        }
    }

    // Add SpellCaster entry if not present
    if (!hasSpellCaster) {
        log("Adding SpellCaster to mods.txt...");
        std::ofstream outFile(modsTxtPath, std::ios::app);
        if (outFile.is_open()) {
            outFile << "\n; SpellCaster voice recognition mod\n";
            outFile << "SpellCaster : 1\n";
            outFile.close();
            return true;
        }
        return false;
    }

    return true;
}

InstallResult UE4SSInstaller::install(bool forceReinstall) {
    InstallResult result;
    result.success = false;

    // Check current status
    InstallCheckResult check = checkInstallation();

    if (check.status == InstallStatus::GAME_NOT_FOUND) {
        result.message = check.message;
        result.errors.push_back(check.message);
        return result;
    }

    if (check.status == InstallStatus::GAME_RUNNING) {
        result.message = check.message;
        result.errors.push_back(check.message);
        return result;
    }

    if (check.status == InstallStatus::ERROR_OCCURRED) {
        result.message = check.message;
        result.errors.push_back(check.message);
        return result;
    }

    if (check.status == InstallStatus::UP_TO_DATE && !forceReinstall) {
        result.success = true;
        result.message = "UE4SS and SpellCaster mod are already up to date";
        return result;
    }

    log("Installing UE4SS and SpellCaster mod to: " + gamePath_);

    bool needsUE4SS = !check.ue4ssInstalled || check.ue4ssNeedsUpdate || forceReinstall;
    bool needsSpellCaster = !check.spellCasterInstalled || check.spellCasterNeedsUpdate || forceReinstall;

    // Install/Update UE4SS core files
    if (needsUE4SS) {
        log("Installing UE4SS core files...");

        // UE4SS.dll
        std::string sourceUE4SS = sourcePath_ + "/" + UE4SS_DLL;
        std::string targetUE4SS = gamePath_ + "/" + UE4SS_DLL;
        if (fs::exists(sourceUE4SS)) {
            backupFile(targetUE4SS);
            if (copyFile(sourceUE4SS, targetUE4SS)) {
                if (check.ue4ssInstalled) {
                    result.filesUpdated.push_back(UE4SS_DLL);
                } else {
                    result.filesInstalled.push_back(UE4SS_DLL);
                }
            } else {
                result.errors.push_back("Failed to copy " + std::string(UE4SS_DLL));
            }
        }

        // dwmapi.dll
        std::string sourceDwmapi = sourcePath_ + "/" + DWMAPI_DLL;
        std::string targetDwmapi = gamePath_ + "/" + DWMAPI_DLL;
        if (fs::exists(sourceDwmapi)) {
            backupFile(targetDwmapi);
            if (copyFile(sourceDwmapi, targetDwmapi)) {
                if (check.ue4ssInstalled) {
                    result.filesUpdated.push_back(DWMAPI_DLL);
                } else {
                    result.filesInstalled.push_back(DWMAPI_DLL);
                }
            } else {
                result.errors.push_back("Failed to copy " + std::string(DWMAPI_DLL));
            }
        }

        // UE4SS-settings.ini (only if not exists - preserve user settings)
        std::string sourceSettings = sourcePath_ + "/" + UE4SS_SETTINGS;
        std::string targetSettings = gamePath_ + "/" + UE4SS_SETTINGS;
        if (fs::exists(sourceSettings) && !fs::exists(targetSettings)) {
            if (copyFile(sourceSettings, targetSettings)) {
                result.filesInstalled.push_back(UE4SS_SETTINGS);
            }
        }
    }

    // Create Mods directory if needed
    std::string modsDir = gamePath_ + "/Mods";
    if (!fs::exists(modsDir)) {
        fs::create_directories(modsDir);
    }

    // Install Keybinds mod (required by some UE4SS functionality)
    std::string sourceKeybinds = sourcePath_ + "/Mods/Keybinds";
    std::string targetKeybinds = modsDir + "/Keybinds";
    if (fs::exists(sourceKeybinds) && !fs::exists(targetKeybinds)) {
        log("Installing Keybinds mod...");
        if (copyDirectory(sourceKeybinds, targetKeybinds)) {
            result.filesInstalled.push_back("Mods/Keybinds/");
        }
    }

    // Install/Update SpellCaster mod
    if (needsSpellCaster) {
        log("Installing SpellCaster mod...");

        std::string sourceSpellCaster = sourcePath_ + "/Mods/SpellCaster";
        std::string targetSpellCaster = modsDir + "/SpellCaster";

        // Create SpellCaster directory structure
        fs::create_directories(targetSpellCaster + "/dlls");

        // Copy main.dll
        std::string sourceDll = sourceSpellCaster + "/dlls/main.dll";
        std::string targetDll = targetSpellCaster + "/dlls/main.dll";
        if (fs::exists(sourceDll)) {
            backupFile(targetDll);
            if (copyFile(sourceDll, targetDll)) {
                if (check.spellCasterInstalled) {
                    result.filesUpdated.push_back(SPELLCASTER_DLL);
                } else {
                    result.filesInstalled.push_back(SPELLCASTER_DLL);
                }
            } else {
                result.errors.push_back("Failed to copy SpellCaster main.dll");
            }
        }

        // Create enabled.txt if not exists
        std::string enabledPath = targetSpellCaster + "/enabled.txt";
        if (!fs::exists(enabledPath)) {
            std::ofstream enabledFile(enabledPath);
            enabledFile.close();
            result.filesInstalled.push_back("Mods/SpellCaster/enabled.txt");
        }
    }

    // Copy/update mods.txt and ensure SpellCaster entry
    std::string sourceModsTxt = sourcePath_ + "/Mods/mods.txt";
    std::string targetModsTxt = modsDir + "/mods.txt";
    if (!fs::exists(targetModsTxt) && fs::exists(sourceModsTxt)) {
        copyFile(sourceModsTxt, targetModsTxt);
        result.filesInstalled.push_back(MODS_TXT);
    }
    ensureModsEntry(modsDir);

    // Summary
    result.success = result.errors.empty();
    if (result.success) {
        std::stringstream ss;
        ss << "Installation complete! ";
        if (!result.filesInstalled.empty()) {
            ss << "Installed " << result.filesInstalled.size() << " file(s). ";
        }
        if (!result.filesUpdated.empty()) {
            ss << "Updated " << result.filesUpdated.size() << " file(s).";
        }
        result.message = ss.str();
    } else {
        result.message = "Installation completed with " + std::to_string(result.errors.size()) + " error(s)";
    }

    return result;
}
