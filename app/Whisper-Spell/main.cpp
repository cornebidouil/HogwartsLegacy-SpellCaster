#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "hid.lib")
#pragma comment(lib, "Xinput.lib")

#include <iostream>
#include <locale>
#include <vector>
#include <portaudio.h>
#include <algorithm>
#include <regex>
#include <numeric>
#include <cmath>
#include <limits>
#include <mutex>
#include <thread>
#include <chrono>
#include <signal.h>
#include <windows.h>
#include <tchar.h>
#include <soxr.h>
#include <tlhelp32.h>

#include "Audio.h"
#include "ITranscriber.h"
#include "MoonshineTranscriber.h"
#include "WhisperTranscriber.h"
#include "SpellTransmitter.h"
#include "UE4SSInstaller.h"
#include "CrowdsourcingManager.h"
#include "FirstRunDialog.h"
#include "Tools.h"
#include "ThreadPool.h"
#include "Config.h"
#include "cuda_tools.h"
#include "InteractiveMenu.h"
#include "PreferencePrompt.h"


// Global variables for audio processing
std::vector<float> resampledAudio; // Buffer for resampled audio
std::unique_ptr<ITranscriber> transcriber;  // Changed to use abstract interface
std::unique_ptr<SpellTransmitter> spellTransmitter;
std::unique_ptr<CrowdsourcingManager> crowdsourcingMgr;
std::unique_ptr<AudioResampler> audioResampler;
std::mutex bufferMutex;
ThreadPool pool(std::thread::hardware_concurrency());
Config cfg;


volatile bool stop = false;

// Factory function to create transcriber based on config
std::unique_ptr<ITranscriber> createTranscriber(const Config& cfg) {
    switch (cfg.models.engine) {
        case Config::ModelConfig::TranscriptionEngine::WHISPER:
            try {
                return std::make_unique<WhisperTranscriber>(
                    cfg.models.whisperModel,
                    cfg.transcription.moonshineVadThreshold,  // Use same VAD threshold as Moonshine
                    16000,  // Sample rate
                    cfg.transcription.whisperConfidenceThreshold,
                    true    // Use GPU if available
                );
            } catch (const std::exception& e) {
                std::cerr << "Failed to initialize Whisper: " << e.what() << std::endl;
                return nullptr;
            }

        case Config::ModelConfig::TranscriptionEngine::MOONSHINE:
            try {
                return std::make_unique<moonshine::MoonshineTranscriber>(
                    cfg.models.moonshineModel,
                    cfg.transcription.moonshineVadThreshold
                );
            } catch (const std::exception& e) {
                std::cerr << "Failed to initialize Moonshine: " << e.what() << std::endl;
                return nullptr;
            }

        default:
            std::cerr << "Unknown transcription engine" << std::endl;
            return nullptr;
    }
}

void transmitter_cleanup() {
    if (spellTransmitter) {
        spellTransmitter->shutdown();
        spellTransmitter.reset();
    }
    std::cout << "SpellTransmitter cleaned up before exit.\n";
}

LONG WINAPI exceptionHandler(EXCEPTION_POINTERS* ExceptionInfo) {
    std::cout << "Unhandled exception! Cleaning up SpellTransmitter...\n";
    transmitter_cleanup();
    return EXCEPTION_EXECUTE_HANDLER;
}

BOOL WINAPI CtrlHandler(DWORD fdwCtrlType)
{
    switch (fdwCtrlType)
    {
    case CTRL_C_EVENT:
        std::cout << std::endl << std::endl << "You've pressed CTRL+C the program is going to shutdown." << std::endl;
        stop = true;
        return TRUE;
    case CTRL_CLOSE_EVENT:
        std::cout << "CMD is closing. Cleaning up SpellTransmitter...\n";
        transmitter_cleanup();
        stop = true;
        return TRUE;
    default:
        return FALSE;
    }
}

bool resizeCMDSize(const int& nb_lines = 30) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO csbi;

    // Get current console info
    if (!GetConsoleScreenBufferInfo(hConsole, &csbi)) {
        std::cout << "Failed to get console info" << std::endl;
        return false;
    }

    // Get maximum possible window size
    COORD maxSize = GetLargestConsoleWindowSize(hConsole);

    // Keep the same number of columns, but set window height
    int defaultCols = csbi.dwSize.X;
    int newWindowHeight = nb_lines;

    // Clamp to maximum available height if necessary
    if (newWindowHeight > maxSize.Y) {
        newWindowHeight = maxSize.Y;
    }

    // Also clamp columns to maximum if necessary
    if (defaultCols > maxSize.X) {
        defaultCols = maxSize.X;
    }

    // Make sure buffer can accommodate the new window size
    if (csbi.dwSize.Y < newWindowHeight || csbi.dwSize.X < defaultCols) {
        COORD newBufferSize = {
            (std::max)(csbi.dwSize.X, (SHORT)defaultCols),
            (std::max)(csbi.dwSize.Y, (SHORT)newWindowHeight)
        };
        if (!SetConsoleScreenBufferSize(hConsole, newBufferSize)) {
            std::cout << "Failed to set buffer size" << std::endl;
            return false;
        }
    }

    // Set new window size
    SMALL_RECT newWindow = { 0, 0, defaultCols - 1, newWindowHeight - 1 };
    if (!SetConsoleWindowInfo(hConsole, TRUE, &newWindow)) {
        std::cout << "Failed to set window size" << std::endl;
        return false;
    }

    return true;
}


// Callback function for PortAudio
static int paCallback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) {

    if (!inputBuffer || !transcriber) return paContinue;

    const float* in = static_cast<const float*>(inputBuffer);

    // Resample to 16kHz mono
    size_t outputFrames = audioResampler->processAudio(in, framesPerBuffer, cfg.audio.inputChannels, resampledAudio);

    if (outputFrames == 0) return paContinue;

    // Feed audio incrementally to TEN VAD
    try {
        transcriber->processAudio(resampledAudio.data(), outputFrames);
    } catch (const std::exception& e) {
        std::cerr << "VAD error: " << e.what() << std::endl;
    }

    // Get completed transcriptions (non-blocking)
    auto completedTranscriptions = transcriber->getCompletedTranscriptions();

    // Queue spell transmission on ThreadPool with HIGH priority
    for (auto& result : completedTranscriptions) {
        pool.enqueue([result = std::move(result), trans = transcriber.get(), transmitter = spellTransmitter.get(),
                      crowdMgr = crowdsourcingMgr.get()]() mutable {

            if (trans->shouldAcceptTranscription(result)) {
                std::cout << "Spell: " << result.text
                         << " (conf: " << result.confidence << ", "
                         << result.duration << "s, "
                         << result.latency.count() << "ms)" << std::endl;

                transmitter->sendSpell(result.text);

                // Note: Crowdsourcing with Whisper would need audio data to be stored differently
                // For now, we'll skip crowdsourcing for Whisper transcriptions
                // TODO: Enhance to store raw audio in TranscriptionResult if needed
            }
        }, ThreadPool::Priority::HIGH);  // Spell transcription gets highest priority
    }

    return paContinue;
}

// =========================================================================
// Game Launch Helper Functions
// =========================================================================

bool isGameRunning(const std::string& processName) {
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE) {
        return false;
    }

    PROCESSENTRY32 pe32;
    pe32.dwSize = sizeof(PROCESSENTRY32);

    if (!Process32First(hSnapshot, &pe32)) {
        CloseHandle(hSnapshot);
        return false;
    }

    bool found = false;
    do {
        // Convert wide string to narrow string
        char narrowExe[MAX_PATH];
        WideCharToMultiByte(CP_UTF8, 0, pe32.szExeFile, -1, narrowExe, MAX_PATH, nullptr, nullptr);
        std::string currentProcess = narrowExe;
        std::transform(currentProcess.begin(), currentProcess.end(), currentProcess.begin(), ::tolower);

        std::string targetProcess = processName;
        std::transform(targetProcess.begin(), targetProcess.end(), targetProcess.begin(), ::tolower);

        if (currentProcess == targetProcess) {
            found = true;
            break;
        }
    } while (Process32Next(hSnapshot, &pe32));

    CloseHandle(hSnapshot);
    return found;
}

bool launchGame(const std::string& gamePath) {
    if (gamePath.empty()) {
        std::cerr << "Game path is empty, cannot launch game" << std::endl;
        return false;
    }

    // Extract game executable path
    std::string exePath = gamePath;
    if (exePath.back() == '\\' || exePath.back() == '/') {
        exePath += "HogwartsLegacy.exe";
    } else {
        exePath += "\\HogwartsLegacy.exe";
    }

    std::cout << "Launching game: " << exePath << std::endl;

    STARTUPINFOA si = {};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi = {};

    if (!CreateProcessA(
        exePath.c_str(),    // Application name
        nullptr,            // Command line
        nullptr,            // Process attributes
        nullptr,            // Thread attributes
        FALSE,              // Inherit handles
        0,                  // Creation flags
        nullptr,            // Environment
        gamePath.c_str(),   // Current directory (game folder)
        &si,                // Startup info
        &pi                 // Process information
    )) {
        std::cerr << "Failed to launch game. Error code: " << GetLastError() << std::endl;
        return false;
    }

    // Close process and thread handles (we don't need to wait)
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    std::cout << "Game launched successfully!" << std::endl;
    return true;
}

int main(int argc, char*argv[]) {
    std::setlocale(LC_ALL, ".UTF-8");

    std::atexit(transmitter_cleanup);
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);

    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
    
    SetUnhandledExceptionFilter(exceptionHandler);
#endif

    resizeCMDSize(75);

    cfg.load();

    // =========================================================================
    // Crowdsourcing First-Run Check (BEFORE Banner & Audio)
    // =========================================================================
    // Ask for consent at the very beginning to avoid overlapping with spell recognition
    if (!cfg.crowdsourcing.firstRunComplete) {
        std::cout << std::endl;
        std::cout << "\033[1;36m" << "=== Crowdsourcing First Run ===" << "\033[0m" << std::endl;
        std::cout << std::endl;

        // Create temporary manager just to show the dialog
        // (device name will be set properly later after audio device selection)
        CrowdsourcingManager tempMgr(cfg, "");
        if (tempMgr.checkFirstRun()) {
            std::cout << "\033[1;32m[Crowdsourcing] Thank you! Recordings will be saved and uploaded.\033[0m" << std::endl;
        } else {
            std::cout << "\033[1;33m[Crowdsourcing] Feature disabled. You can enable it later in config.ini\033[0m" << std::endl;
        }
        cfg.save();  // Save consent decision immediately

        std::cout << "\033[1;36m" << "===============================" << "\033[0m" << std::endl;
        std::cout << std::endl;
    }

    // An account password that could not be read back (config.ini copied from
    // another machine or Windows account, or the line removed by the user) is
    // asked for again here, before the audio setup takes over the console.
    if (cfg.crowdsourcing.enabled && cfg.crowdsourcing.consentGiven
        && cfg.crowdsourcing.authType == "account" && cfg.crowdsourcing.password.empty()) {
        cfg.crowdsourcing.password = FirstRunDialog::promptPassword(cfg.crowdsourcing.username);
        if (cfg.crowdsourcing.password.empty()) {
            std::cout << "\033[1;33m[Crowdsourcing] No password given: recordings stay on this machine and uploads are skipped until one is provided.\033[0m" << std::endl;
        } else {
            cfg.save();
        }
        std::cout << std::endl;
    }

    // =========================================================================
    // UE4SS + SpellCaster Mod Installation Check
    // =========================================================================
    if (cfg.game.autoInstallUE4SS) {
        std::cout << std::endl;
        std::cout << "\033[1;36m" << "=== UE4SS Mod Installation Check ===" << "\033[0m" << std::endl;

        UE4SSInstaller installer;

        // Use configured game path if set
        if (!cfg.game.gamePath.empty()) {
            installer.setGamePath(cfg.game.gamePath);
        }

        InstallCheckResult checkResult = installer.checkInstallation();

        switch (checkResult.status) {
            case InstallStatus::UP_TO_DATE:
                std::cout << "\033[1;32m" << "[OK] " << "\033[0m"
                          << "UE4SS and SpellCaster mod are up to date" << std::endl;
                std::cout << "     Game path: " << checkResult.gamePath << std::endl;
                break;

            case InstallStatus::NOT_INSTALLED:
            case InstallStatus::UE4SS_ONLY:
            case InstallStatus::OUTDATED: {
                std::cout << "\033[1;33m" << "[!] " << "\033[0m"
                          << checkResult.message << std::endl;

                if (!checkResult.gamePath.empty()) {
                    std::cout << "     Game path: " << checkResult.gamePath << std::endl;
                    std::cout << "     Installing/updating..." << std::endl;

                    InstallResult installResult = installer.install();

                    if (installResult.success) {
                        std::cout << "\033[1;32m" << "[OK] " << "\033[0m"
                                  << installResult.message << std::endl;

                        // List installed files
                        for (const auto& file : installResult.filesInstalled) {
                            std::cout << "     + " << file << std::endl;
                        }
                        for (const auto& file : installResult.filesUpdated) {
                            std::cout << "     ~ " << file << " (updated)" << std::endl;
                        }

                        // Save detected game path to config for future runs
                        if (cfg.game.gamePath.empty()) {
                            cfg.game.gamePath = installer.getGamePath();
                            cfg.save();
                            std::cout << "     Game path saved to config.ini" << std::endl;
                        }
                    } else {
                        std::cout << "\033[1;31m" << "[ERROR] " << "\033[0m"
                                  << installResult.message << std::endl;
                        for (const auto& error : installResult.errors) {
                            std::cout << "     - " << error << std::endl;
                        }
                    }
                }
                break;
            }

            case InstallStatus::GAME_NOT_FOUND:
                std::cout << "\033[1;33m" << "[!] " << "\033[0m"
                          << "Hogwarts Legacy installation not found" << std::endl;
                std::cout << "     Set game_path in config.ini to your Win64 folder" << std::endl;
                std::cout << "     Example: game_path=C:/Steam/.../Hogwarts Legacy/Phoenix/Binaries/Win64" << std::endl;
                break;

            case InstallStatus::GAME_RUNNING:
                std::cout << "\033[1;33m" << "[!] " << "\033[0m"
                          << "Game is running - skipping mod installation check" << std::endl;
                std::cout << "     Close the game and restart SpellCaster to check for updates" << std::endl;
                break;

            case InstallStatus::ERROR_OCCURRED:
                std::cout << "\033[1;31m" << "[ERROR] " << "\033[0m"
                          << checkResult.message << std::endl;
                break;
        }

        std::cout << "\033[1;36m" << "=====================================" << "\033[0m" << std::endl;
        std::cout << std::endl;
    }


    if (!SetConsoleCtrlHandler(CtrlHandler, TRUE)) {
        std::cerr << "ERROR: Could not set control handler" << std::endl;
#ifdef _WIN32
        system("PAUSE");
#endif
        return 1;
    }


    std::cout << std::endl;
    std::cout << "\033[1;37m" << "    +=======================================================================================+" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "                                                                                       " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;32m" << "    " << "\033[1;33m" << "*" << "\033[0m" << "    " << "\033[1;31m" << "+" << "\033[0m" << "         " << "\033[1;35m" << "*" << "\033[0m" << "   " << "\033[1;33m" << "+" << "\033[0m" << "            " << "\033[1;32m" << "Hogwarts Legacy             " << "\033[1;31m" << "*" << "\033[0m" << "    " << "\033[1;35m" << "+" << "\033[0m" << "         " << "\033[1;33m" << "*" << "\033[0m" << "       " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;36m" << "            " << "\033[1;33m" << "*" << "\033[0m" << "   " << "\033[1;31m" << "+" << "\033[0m" << "           " << "\033[1;36m" << "***  S P E L L C A S T E R ***        " << "\033[1;35m" << "*" << "\033[0m" << "    " << "\033[1;33m" << "+" << "\033[0m" << "     " << "\033[1;31m" << "*" << "\033[0m" << "     " << "\033[1;35m" << "+" << "\033[0m" << "   " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "      " << "\033[1;31m" << "+" << "\033[0m" << "                 " << "\033[1;35m" << "*" << "\033[0m" << "                                   " << "\033[1;33m" << "+" << "\033[0m" << "             " << "\033[1;31m" << "*" << "\033[0m" << "      " << "\033[1;35m" << "+" << "\033[0m" << "     " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;33m" << "  " << "\033[1;35m" << "*" << "\033[0m" << "        " << "\033[1;33m" << "+" << "\033[0m" << "        " << "\033[1;31m" << "*" << "\033[0m" << "            " << "\033[1;35m" << "+" << "\033[0m" << "     " << "\033[1;33m" << "/\\      " << "\033[1;33m" << "*" << "\033[0m" << "                  " << "\033[1;31m" << "+" << "\033[0m" << "          " << "\033[1;35m" << "*" << "\033[0m" << "         " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "       " << "\033[1;33m" << "+" << "\033[0m" << "      " << "\033[1;35m" << "+" << "\033[0m" << "       " << "\033[1;31m" << "*" << "\033[0m" << "     " << "\033[1;30m" << "^^    " << "\033[1;33m" << "    /  \\  " << "\033[1;31m" << "+" << "\033[0m" << "    " << "\033[1;30m" << "  ^^   " << "\033[1;33m" << " /\\   " << "\033[1;33m" << "*" << "\033[0m" << "      " << "\033[1;33m" << "+" << "\033[0m" << "             " << "\033[1;31m" << "*" << "\033[0m" << "   " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "   " << "\033[1;31m" << "*" << "\033[0m" << "             " << "\033[1;31m" << "+" << "\033[0m" << "         " << "\033[1;30m" << "/  \\   " << "\033[1;33m" << "+" << "\033[0m" << "" << "\033[1;33m" << "  /    \\     " << "\033[1;35m" << "*" << "\033[0m" << "" << "\033[1;30m" << " /  \\  " << "\033[1;33m" << "/  \\                  " << "\033[1;35m" << "+" << "\033[0m" << "        " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "          " << "\033[1;33m" << "*" << "\033[0m" << "            " << "\033[1;33m" << "+" << "\033[0m" << "   " << "\033[1;30m" << "|" << "\033[1;33m" << "[]" << "\033[1;30m" << "|   " << "\033[1;33m" << "   |" << "\033[1;33m" << "[][]" << "\033[1;33m" << "|     " << "\033[1;30m" << "  |" << "\033[1;33m" << "[]" << "\033[1;30m" << "| " << "\033[1;33m" << " |" << "\033[1;33m" << "[]" << "\033[1;33m" << "|    " << "\033[1;35m" << "*" << "\033[0m" << "       " << "\033[1;31m" << "+" << "\033[0m" << "              " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "     " << "\033[1;35m" << "+" << "\033[0m" << "            " << "\033[1;35m" << "*" << "\033[0m" << "        " << "\033[1;30m" << "|" << "\033[1;33m" << "  " << "\033[1;30m" << "|   " << "\033[1;33m" << "/\\ |" << "\033[1;33m" << "    " << "\033[1;33m" << "|/\\  " << "\033[1;33m" << "+" << "\033[0m" << "" << "\033[1;30m" << "  |" << "\033[1;33m" << "  " << "\033[1;30m" << "| " << "\033[1;33m" << " |" << "\033[1;33m" << "  " << "\033[1;33m" << "|       " << "\033[1;33m" << "*" << "\033[0m" << "               " << "\033[1;33m" << "+" << "\033[0m" << "   " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "             " << "\033[1;31m" << "*" << "\033[0m" << "            " << "\033[1;30m" << "/|" << "\033[1;33m" << "[]" << "\033[1;30m" << "|__" << "\033[1;33m" << "/  \\|" << "\033[1;33m" << "[][]" << "\033[1;33m" << "|  \\ " << "\033[1;30m" << "  /|" << "\033[1;33m" << "[]" << "\033[1;30m" << "|_" << "\033[1;33m" << "/|" << "\033[1;33m" << "[]" << "\033[1;33m" << "|                " << "\033[1;31m" << "*" << "\033[0m" << "          " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "    " << "\033[1;33m" << "+" << "\033[0m" << "                    " << "\033[1;30m" << "/ |" << "\033[1;33m" << "  " << "\033[1;30m" << "   " << "\033[1;33m" << "|" << "\033[1;33m" << "[]" << "\033[1;33m" << "|" << "\033[1;33m" << "     " << "\033[1;33m" << "|" << "\033[1;33m" << "[]" << "\033[1;33m" << " \\ " << "\033[1;30m" << "/ |" << "\033[1;33m" << "  " << "\033[1;30m" << "   " << "\033[1;33m" << "|" << "\033[1;33m" << "  " << "\033[1;33m" << "|      " << "\033[1;35m" << "+" << "\033[0m" << "                    " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "                   " << "\033[1;33m" << "*" << "\033[0m" << "    " << "\033[1;30m" << "/  |" << "\033[1;33m" << "[]" << "\033[1;30m" << "   " << "\033[1;33m" << "|" << "\033[1;33m" << "  " << "\033[1;33m" << "|_____" << "\033[1;33m" << "|" << "\033[1;33m" << "  " << "\033[1;33m" << "| " << "\033[1;30m" << "/  |" << "\033[1;33m" << "[]" << "\033[1;30m" << "   " << "\033[1;33m" << "|" << "\033[1;33m" << "[]" << "\033[1;33m" << "|                     " << "\033[1;35m" << "*" << "\033[0m" << "     " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "          " << "\033[1;31m" << "+" << "\033[0m" << "            " << "\033[1;30m" << "/   |" << "\033[1;33m" << "  " << "\033[1;30m" << "___" << "\033[1;33m" << "|" << "\033[1;33m" << "[]" << "\033[1;33m" << "|     |" << "\033[1;33m" << "[]" << "\033[1;33m" << "|" << "\033[1;30m" << "/   |" << "\033[1;33m" << "  " << "\033[1;30m" << "___" << "\033[1;33m" << "|" << "\033[1;33m" << "  " << "\033[1;33m" << "|              " << "\033[1;33m" << "+" << "\033[0m" << "            " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "                      /    |" << "\033[1;33m" << "[]" << "\033[1;30m" << "   " << "\033[1;33m" << "|" << "\033[1;33m" << "  " << "\033[1;33m" << "|     | " << "\033[1;30m" << " /    |" << "\033[1;33m" << "[]" << "\033[1;30m" << "   " << "\033[1;33m" << "|" << "\033[1;33m" << "[]" << "\033[1;33m" << "|                           " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "                     /     |" << "\033[1;33m" << "  " << "\033[1;30m" << "___" << "\033[1;33m" << "|" << "\033[1;33m" << "[]" << "\033[1;33m" << "|_____| " << "\033[1;30m" << "/     |" << "\033[1;33m" << "  " << "\033[1;30m" << "__" << "\033[1;33m" << " | " << "\033[1;33m" << " " << "\033[1;33m" << "|                           " << "\033[1;37m" << "| " << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "                    /      |" << "\033[1;33m" << "[]" << "\033[1;30m" << "   " << "\033[1;33m" << "|" << "\033[1;33m" << "  " << "\033[1;33m" << "|     |" << "\033[1;30m" << "/      |" << "\033[1;33m" << "[]" << "\033[1;30m" << "   " << "\033[1;33m" << "|" << "\033[1;33m" << "[]" << "\033[1;33m" << "|                           " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "                           |" << "\033[1;33m" << "  " << "\033[1;30m" << "___" << "\033[1;33m" << "|" << "\033[1;33m" << "[]" << "\033[1;33m" << "|_____| " << "\033[1;30m" << "      |" << "\033[1;33m" << "  " << "\033[1;30m" << "___" << "\033[1;33m" << "|" << "\033[1;33m" << "  " << "\033[1;33m" << "|                           " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "                           \\" << "\033[1;33m" << "[]" << "\033[1;30m" << "   " << "\033[1;33m" << "|" << "\033[1;33m" << "  " << "\033[1;33m" << "|     |  " << "\033[1;30m" << "     \\" << "\033[1;33m" << "[]" << "\033[1;30m" << "   " << "\033[1;33m" << "|" << "\033[1;33m" << "[]" << "\033[1;33m" << "/                           " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "                          ^^^^" << "\033[1;33m" << "^^^" << "\033[1;30m" << "^^" << "\033[1;33m" << "^^" << "\033[1;30m" << "^^^^^^" << "\033[1;33m" << "^^^^^^" << "\033[1;30m" << "^^^^^^^" << "\033[1;33m" << "^^^" << "\033[1;30m" << "^^                          " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "                         ^^^" << "\033[1;33m" << "^^^^" << "\033[1;30m" << "^^^" << "\033[1;33m" << "^" << "\033[1;30m" << "^^^^^^^" << "\033[1;33m" << "^^^^^^" << "\033[1;30m" << "^^^^^^" << "\033[1;33m" << "^^^" << "\033[1;30m" << "^^^^                         " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "                        ^^" << "\033[1;33m" << "^^^^" << "\033[1;30m" << "^^^^" << "\033[1;33m" << "^^" << "\033[1;30m" << "^^^^^^" << "\033[1;33m" << "^^^^^^^" << "\033[1;30m" << "^^^^^" << "\033[1;33m" << "^^^^" << "\033[1;30m" << "^^^^^                        " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "                                                                                       " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;35m" << "                                Created by Cornebidouil                                " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;35m" << "                        GitHub: " << "\033[0m\033[30;42m" << "https://github.com/pierre-cheneau" << "\033[0m\033[1;35m" << "                      " << "\033[0m" << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;35m" << "                 ETH Wallet: " << "\033[0m\033[30;42m" << "0x1F61fa7923d5E914A5Fdf36B584a1336fde20721" << "\033[0m\033[1;35m" << "                " << "\033[0m" << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    |" << "\033[1;30m" << "                                                                                       " << "\033[1;37m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;37m" << "    +=======================================================================================+" << "\033[0m" << std::endl << std::endl;


    // Enhanced Credits and Footer Section
    std::cout << "\033[1;33m" << "    +=======================================================================================+" << "\033[0m" << std::endl;
    std::cout << "\033[1;33m" << "    |" << "\033[1;36m" << "                                 *** HALL OF FAME ***                                  " << "\033[1;33m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;33m" << "    |" << "\033[1;32m" << "                              Beta Testers & Contributors                              " << "\033[1;33m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;33m" << "    |" << "\033[1;37m" << "                                                                                       " << "\033[1;33m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;33m" << "    |" << "\033[1;37m" << "                " << "\033[1;35m" << "*" << "\033[1;37m" << " Amecareth        " << "\033[1;35m" << "*" << "\033[1;37m" << " Darkenciels      " << "\033[1;35m" << "*" << "\033[1;37m" << " Thaxano                        " << "\033[1;33m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;33m" << "    |" << "\033[1;37m" << "                " << "\033[1;35m" << "*" << "\033[1;37m" << " Koba             " << "\033[1;35m" << "*" << "\033[1;37m" << " Meroshiro        " << "\033[1;35m" << "*" << "\033[1;37m" << " Vu Tran                        " << "\033[1;33m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;33m" << "    |" << "\033[1;37m" << "                " << "\033[1;35m" << "*" << "\033[1;37m" << " Avis320neo       " << "\033[1;35m" << "*" << "\033[1;37m" << " Paradoxius88     " << "\033[1;35m" << "*" << "\033[1;37m" << " jarommadsenand                 " << "\033[1;33m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;33m" << "    |" << "\033[1;37m" << "                " << "\033[1;35m" << "*" << "\033[1;37m" << " Miiko64340       " << "\033[1;35m" << "*" << "\033[1;37m" << " le_renard_rolist " << "\033[1;35m" << "*" << "\033[1;37m" << " Geko                           " << "\033[1;33m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;33m" << "    |" << "\033[1;37m" << "                " << "\033[1;35m" << "*" << "\033[1;37m" << " N7 Cmdr Sheppard " << "\033[1;35m" << "*" << "\033[1;37m" << " DragoWing        " << "\033[1;35m" << "*" << "\033[1;37m" << " MTGAh                          " << "\033[1;33m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;33m" << "    |" << "\033[1;37m" << "                                                                                       " << "\033[1;33m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;33m" << "    |" << "\033[1;32m" << "                 Thank you magical testers for your time & dedication!                 " << "\033[1;33m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;33m" << "    +=======================================================================================+" << "\033[0m" << std::endl;
    std::cout << std::endl;

    std::cout << "\033[1;36m" << "            =======================================================================" << "\033[0m" << std::endl;
    std::cout << "\033[1;36m" << "           ||" << "\033[1;33m" << "                        *** MAGICAL LINKS ***                        " << "\033[1;36m" << "||" << "\033[0m" << std::endl;
    std::cout << "\033[1;36m" << "           ||" << "\033[1;37m" << "                                                                     " << "\033[1;36m" << "||" << "\033[0m" << std::endl;
    std::cout << "\033[1;36m" << "           ||" << "\033[1;35m" << "    " << "\033[1;31m" << "*" << "\033[1;35m" << "  Training Portal: " << "\033[0m\033[30;42m" << "http://hogwartslegacyspellcaster.xyz" << "\033[0m" << "\033[1;35m" << "  " << "\033[1;31m" << "*" << "\033[1;35m" << "      " << "\033[1;36m" << "||" << "\033[0m" << std::endl;
    std::cout << "\033[1;36m" << "           ||" << "\033[1;37m" << "                                                                     " << "\033[1;36m" << "||" << "\033[0m" << std::endl;
    std::cout << "\033[1;36m" << "           ||" << "\033[1;35m" << "             Want to participate in recognition training?            " << "\033[1;36m" << "||" << "\033[0m" << std::endl;
    std::cout << "\033[1;36m" << "           ||" << "\033[1;35m" << "             Help us improve spell recognition accuracy!             " << "\033[1;36m" << "||" << "\033[0m" << std::endl;
    std::cout << "\033[1;36m" << "            =======================================================================" << "\033[0m" << std::endl;
    std::cout << std::endl;

    std::cout << "\033[1;35m" << "           *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << "\033[0m" << std::endl;
    std::cout << "\033[1;35m" << "           |" << "\033[1;33m" << "                          *** SUPPORT ZONE ***                         " << "\033[1;35m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;35m" << "           |" << "\033[1;37m" << "                                                                       " << "\033[1;35m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;35m" << "           |" << "\033[1;36m" << "   Need help? Having trouble? We've got you covered!                   " << "\033[1;35m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;35m" << "           |" << "\033[1;32m" << "   Join our Discord community for magical support & solutions:         " << "\033[1;35m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;35m" << "           |" << "\033[1;37m" << "                                                                       " << "\033[1;35m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;35m" << "           |" << "\033[1;31m" << "          " << "\033[1;33m" << "*" << "\033[1;31m" << "  Discord Server: " << "\033[0m\033[30;42m" << "https://discord.gg/zE4NRsTGdw" << "\033[0m" << "\033[1;31m" << "  " << "\033[1;33m" << "*" << "\033[1;31m" << "          " << "\033[1;35m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;35m" << "           |" << "\033[1;37m" << "                                                                       " << "\033[1;35m" << "|" << "\033[0m" << std::endl;
    std::cout << "\033[1;35m" << "           *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << "\033[0m" << std::endl;
    std::cout << std::endl << std::endl;


    std::cout << " * Launching the program before launching the game is highly recommended." << std::endl;

    if (isWindowOpen("Hogwarts Legacy")) {
        std::cout << std::endl;
        std::cout << ORANGE << BOLD << "WARNING: " << RESET
            << ORANGE << "Launching the program before launching the game is highly recommended."
            << std::endl;
#ifdef _WIN32
        system("PAUSE");
#elif defined(__APPLE__) || defined(__MACH__)
        system("read -p 'Press Enter to continue...'");
#elif defined(__linux__)
        system("read -p 'Press Enter to continue...'");
#else
        std::cout << "Press Enter to continue...";
        std::cin.get();
#endif
        std::cout << RESET;
    }

    std::cout << std::endl << std::endl;

    // Initialize SpellTransmitter for direct spell transmission to UE4SS mod
    spellTransmitter = std::make_unique<SpellTransmitter>();
    spellTransmitter->setDebugLogging(cfg.spellTransmitter.debugLogging);
    if (!spellTransmitter->initialize()) {
        std::cerr << "WARNING: Failed to initialize SpellTransmitter. Spells will be queued when UE4SS mod connects." << std::endl;
    }

    // --- Initialize PortAudio (EARLY - needed for preference check)
    PaError err;
    err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
#ifdef _WIN32
        system("PAUSE");
#endif // !_WIN32
        return 1;
    }

    // =========================================================================
    // Preference Check & Quick-Start Flow (AFTER PortAudio Init)
    // =========================================================================
    bool needsModelSelection = true;
    bool needsDeviceSelection = true;
    bool needsAutoLaunchSelection = true;

    if (cfg.preferences.firstRunComplete && !cfg.audio.selectedDeviceName.empty()) {
        // Check if device list changed
        int currentHash = computeDeviceListHash();
        bool deviceListChanged = (currentHash != cfg.audio.deviceListHash);

        if (!deviceListChanged) {
            // Device list unchanged - offer quick-start
            std::string modelName = (cfg.models.engine == Config::ModelConfig::TranscriptionEngine::WHISPER)
                                    ? "Whisper - High Accuracy"
                                    : "Moonshine - Fast & Lightweight";

            PreferencePrompt prompt(modelName, cfg.audio.selectedDeviceName, cfg.audio.selectedDeviceAPI, cfg.preferences.autoLaunchGame);

            if (prompt.show()) {
                // User accepted - skip all menus
                needsModelSelection = false;
                needsDeviceSelection = false;
                needsAutoLaunchSelection = false;
            }
            // else: User pressed ESC - allow reconfiguration of all preferences
        } else {
            // Device list changed - notify and force device re-selection
            std::cout << "\n\033[1;33m⚠️  Audio device list has changed!\033[0m" << std::endl;
            std::cout << "Your saved device may no longer be available." << std::endl;
            std::cout << "Please re-select your audio device.\n" << std::endl;

            needsModelSelection = false;  // Keep model choice
            needsDeviceSelection = true;   // Force device selection
            needsAutoLaunchSelection = false; // Keep auto-launch setting

            std::cout << "Press any key to continue...";
            _getch();
            std::cout << std::endl;
        }
    }

    // =========================================================================
    // Model Selection (conditional)
    // =========================================================================
    if (needsModelSelection) {
        std::cout << std::endl;

        std::vector<InteractiveMenu::MenuItem> modelItems = {
            InteractiveMenu::MenuItem(
                "Whisper - High Accuracy (Recommended)",
                "Best accuracy, ~100-200ms latency with GPU | For: Desktop PCs with dedicated GPU"
            ),
            InteractiveMenu::MenuItem(
                "Moonshine - Fast & Lightweight",
                "Optimized for speed, ~30-50ms latency on CPU | For: Laptops and systems without powerful GPU"
            )
        };

        InteractiveMenu modelMenu("Select Speech Recognition Model:", modelItems, 0);
        int modelChoice = modelMenu.show();

        // Update config based on selection
        if (modelChoice == 0) {
            cfg.models.engine = Config::ModelConfig::TranscriptionEngine::WHISPER;
        } else {
            cfg.models.engine = Config::ModelConfig::TranscriptionEngine::MOONSHINE;
        }
    }

    // =========================================================================
    // Transcriber Initialization
    // =========================================================================
    std::cout << std::endl << "  --- Transcriber Initialization ---" << std::endl << std::endl;

    transcriber = createTranscriber(cfg);
    if (!transcriber) {
        std::cerr << "Failed to initialize transcription engine" << std::endl;
#ifdef _WIN32
        system("PAUSE");
#endif
        return 1;
    }

    std::cout << "Using " << transcriber->getModelName() << " transcription engine" << std::endl;
    std::cout << "Initialization complete!" << std::endl << std::endl;

    // List all available input devices

    // Device selection with resampling support
    PaDeviceIndex selectedDevice;
    if (needsDeviceSelection) {
        selectedDevice = selectInputDevice(cfg.audio, true);  // Force selection
    } else {
        selectedDevice = selectInputDevice(cfg.audio, false); // Use saved
    }

    if (selectedDevice == paNoDevice) {
        std::cerr << "Failed to select input device" << std::endl;
        Pa_Terminate();
#ifdef _WIN32
        system("PAUSE");
#endif
        return 1;
    }

    // Initialize audio resampler
    audioResampler = std::make_unique<AudioResampler>(&cfg.audio);
    if (!audioResampler->initialize(cfg.audio.inputSampleRate, cfg.audio.inputChannels)) {
        std::cerr << "Failed to initialize audio resampler" << std::endl;
        Pa_Terminate();
#ifdef _WIN32
        system("PAUSE");
#endif
        return 1;
    }

    // === SAVE CONFIGURATION ===
    cfg.audio.deviceListHash = computeDeviceListHash();

    // Ask about auto-launch preference if needed
    if (needsAutoLaunchSelection) {
        std::cout << std::endl;
        std::vector<InteractiveMenu::MenuItem> autoLaunchItems = {
            InteractiveMenu::MenuItem(
                "Yes, launch game automatically",
                "Game will start before voice listening begins"
            ),
            InteractiveMenu::MenuItem(
                "No, I'll launch it manually",
                "You'll need to start the game yourself"
            )
        };

        InteractiveMenu autoLaunchMenu("Auto-launch Hogwarts Legacy before listening?", autoLaunchItems, 0);
        int autoLaunchChoice = autoLaunchMenu.show();

        cfg.preferences.autoLaunchGame = (autoLaunchChoice == 0);
    }

    if (!cfg.preferences.firstRunComplete) {
        cfg.preferences.firstRunComplete = true;
    }

    cfg.save();
    std::cout << "✅ Configuration saved\n" << std::endl;


    // Setup audio stream with selected device
    const PaDeviceInfo* selectedDeviceInfo = Pa_GetDeviceInfo(selectedDevice);

    PaStreamParameters inputParameters;
    inputParameters.device = selectedDevice;
    inputParameters.channelCount = cfg.audio.inputChannels;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = selectedDeviceInfo->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = NULL;


    // Open an audio I/O stream
    PaStream* stream;
    err = Pa_OpenStream(&stream, &inputParameters, NULL, cfg.audio.inputSampleRate,
                        cfg.audio.framesPerBuffer, paClipOff, paCallback, nullptr);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        std::cerr << "Failed to open stream for device: " << selectedDeviceInfo->name << std::endl;
        Pa_Terminate();
#ifdef _WIN32
        system("PAUSE");
#endif // !_WIN32
        return -1;
    }

    // =========================================================================
    // Auto-Launch Game (if configured)
    // =========================================================================
    if (cfg.preferences.autoLaunchGame && !cfg.game.gamePath.empty()) {
        std::cout << "\n=== Checking Game Status ===" << std::endl;

        if (isGameRunning("HogwartsLegacy.exe")) {
            std::cout << "✅ Hogwarts Legacy is already running" << std::endl;
        } else {
            std::cout << "Game not running. Launching..." << std::endl;

            if (launchGame(cfg.game.gamePath)) {
                std::cout << "⏳ Waiting 5 seconds for game to initialize..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(5));
            } else {
                std::cout << "⚠️  Failed to launch game. Please start it manually." << std::endl;
                std::cout << "Press any key to continue anyway...";
                _getch();
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // Start the stream
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
#ifdef _WIN32
        system("PAUSE");
#endif // !_WIN32
        return 1;
    }

    const PaStreamInfo* streamInfo = Pa_GetStreamInfo(stream);
    std::cout << std::endl << "Input device: " << selectedDeviceInfo->name << std::endl;
    std::cout << "Input format: " << cfg.audio.inputSampleRate << "Hz, " << cfg.audio.inputChannels << " channels" << std::endl;
    std::cout << "Output format: " << cfg.audio.sampleRate << "Hz, 1 channel" << std::endl;
    std::cout << "Input latency: " << streamInfo->inputLatency << " seconds" << std::endl;
    std::cout << "Sample rate: " << streamInfo->sampleRate << " Hz" << std::endl;

    if (cfg.audio.inputSampleRate != cfg.audio.sampleRate) {
        std::cout << "Resampling: " << cfg.audio.inputSampleRate << "Hz -> " << cfg.audio.sampleRate << "Hz" << std::endl;
    }
    else {
        std::cout << "Direct processing: No resampling needed" << std::endl;
    }
    std::cout << std::endl;

    // =========================================================================
    // Crowdsourcing Manager Initialization (After Audio Device Selection)
    // =========================================================================
    // Initialize manager with actual device name if user consented
    if (cfg.crowdsourcing.enabled && cfg.crowdsourcing.consentGiven) {
        if (cfg.crowdsourcing.debugLogging) {
            std::cout << "\033[1;36m" << "=== Crowdsourcing Active ===" << "\033[0m" << std::endl;
        }

        crowdsourcingMgr = std::make_unique<CrowdsourcingManager>(cfg, selectedDeviceInfo->name);
        if (!crowdsourcingMgr->initialize()) {
            if (cfg.crowdsourcing.debugLogging) {
                std::cerr << "\033[1;31m[Crowdsourcing] Failed to initialize\033[0m" << std::endl;
            }
            crowdsourcingMgr.reset();
        } else {
            if (cfg.crowdsourcing.debugLogging) {
                std::cout << "\033[1;32m[Crowdsourcing] Recordings will be saved and uploaded\033[0m" << std::endl;
            }

            // Sync in background if auto-sync enabled
            if (cfg.crowdsourcing.autoSync) {
                if (cfg.crowdsourcing.debugLogging) {
                    std::cout << "\033[1;36m[Crowdsourcing] Starting background sync...\033[0m" << std::endl;
                }
                crowdsourcingMgr->syncAsync();
            }
        }

        if (cfg.crowdsourcing.debugLogging) {
            std::cout << "\033[1;36m" << "============================" << "\033[0m" << std::endl;
            std::cout << std::endl;
        }
    }

    // --- Find Whisper-Spell.exe path
    //
    TCHAR buffer[MAX_PATH];
    GetCurrentDirectory(MAX_PATH, buffer);
    #ifdef UNICODE
    std::wstring wPath(buffer);
    std::string exePath(wPath.begin(), wPath.end());
    #else
    std::string exePath(buffer);
    #endif

    // Note: SpellCasterGUI is no longer needed in direct spell transmission mode

    // --- Direct Spell Transmission Mode ---
    std::cout << std::endl << "  --- Direct Spell Transmission Mode ---" << std::endl << std::endl;
    std::cout << "\tSpells are sent directly to the UE4SS mod via shared memory." << std::endl;
    std::cout << "\tNo keybinding configuration needed - the mod handles spell execution." << std::endl;
    std::cout << std::endl;
    std::cout << "\tSupported spells include: Lumos, Incendio, Accio, Levioso, Depulso," << std::endl;
    std::cout << "\tDescendo, Flipendo, Glacius, Confringo, Diffindo, Stupefy," << std::endl;
    std::cout << "\tExpelliarmus, Protego, Revelio, Reparo, and more..." << std::endl;
    std::cout << std::endl;

    // Check connection status
    if (spellTransmitter->isConnected()) {
        std::cout << "\t*** UE4SS Mod: CONNECTED ***" << std::endl;
    } else {
        std::cout << "\t*** UE4SS Mod: Not yet connected (will auto-connect when game loads) ***" << std::endl;
    }

    std::cout << std::endl << std::endl;
    std::cout << "Listening... Press Ctrl+C to stop." << std::endl << std::endl;

    // Keep the stream active
    while (!stop) {
        Pa_Sleep(500);

        // Update SpellTransmitter heartbeat and connection state
        if (spellTransmitter) {
            spellTransmitter->update();
        }
    }

    std::cout << "Shutting down..." << std::endl;

    // Shutdown SpellTransmitter
    if (spellTransmitter) {
        auto stats = spellTransmitter->getStatistics();
        std::cout << "SpellTransmitter stats: sent=" << stats.spellsSent
                  << ", dropped=" << stats.spellsDropped << std::endl;
        spellTransmitter->shutdown();
    }

    // Shutdown CrowdsourcingManager
    if (crowdsourcingMgr) {
        auto crowdStats = crowdsourcingMgr->getStats();
        if (crowdStats.pendingCount > 0 || crowdStats.syncedCount > 0) {
            std::cout << "\033[1;36m[Crowdsourcing] Stats: "
                      << crowdStats.pendingCount << " pending, "
                      << crowdStats.syncedCount << " synced\033[0m" << std::endl;
        }
        crowdsourcingMgr->shutdown();
    }

    // Stop the stream
    err = Pa_StopStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
#ifdef _WIN32
        system("PAUSE");
#endif // !_WIN32
        return 1;
    }

    // Close the stream
    err = Pa_CloseStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
#ifdef _WIN32
        system("PAUSE");
#endif // !_WIN32
        return 1;
    }

    // Terminate PortAudio
    Pa_Terminate();

    // Print performance statistics before cleanup
    if (transcriber) {
        transcriber->printPerformanceStats();
    }

    // Cleanup completed by unique_ptr destructors

    std::cout << "\nShutdown complete." << std::endl;

#ifdef _WIN32
    system("PAUSE");
#endif // !_WIN32


    return 0;
}
