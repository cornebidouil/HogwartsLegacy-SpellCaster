#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <Windows.h>

#define RED "\033[1;31m"
#define GREEN "\033[30;42m"
#define YELLOW "\033[1;33m"
#define ORANGE "\033[38;5;208m"
#define RESET "\033[0m"
#define BOLD "\033[1m"

#define IoControlDeviceType 32769

#define IOCTL_GET_WHITELIST CTL_CODE(IoControlDeviceType, 2048, METHOD_BUFFERED, FILE_READ_DATA)
#define IOCTL_SET_WHITELIST CTL_CODE(IoControlDeviceType, 2049, METHOD_BUFFERED, FILE_READ_DATA)
#define IOCTL_GET_BLACKLIST CTL_CODE(IoControlDeviceType, 2050, METHOD_BUFFERED, FILE_READ_DATA)
#define IOCTL_SET_BLACKLIST CTL_CODE(IoControlDeviceType, 2051, METHOD_BUFFERED, FILE_READ_DATA)
#define IOCTL_GET_ACTIVE    CTL_CODE(IoControlDeviceType, 2052, METHOD_BUFFERED, FILE_READ_DATA)
#define IOCTL_SET_ACTIVE    CTL_CODE(IoControlDeviceType, 2053, METHOD_BUFFERED, FILE_READ_DATA)

class HidHideControl {
private:
    HANDLE filterHandle;
    std::vector<std::wstring> sessionBlacklistDevices, sessionWhitelistApplications;

public:
    HidHideControl() {
        filterHandle = CreateFile(
            L"\\\\.\\HidHide",
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL
        );

        if (filterHandle == INVALID_HANDLE_VALUE) {
            std::cerr << RED << "Failed to open HidHide filter." << RESET << std::endl << std::endl;
            std::cerr << "Please install HidHide program thanks to :" << std::endl << "\t- " << GREEN <<"'HidHide_1.5.230_x64.exe'" << RESET << " in the mod's folder" << std::endl << "\t- or from " << GREEN << "https://github.com/ViGEm/HidHide/releases" << RESET << std::endl << std::endl;
#ifdef _WIN32
            system("PAUSE");
#endif
            exit(1);
        }

        setStatus(false);
    }

    ~HidHideControl() {
        resetWhitelist();
        resetBlacklist();
        setStatus(false);
        if (filterHandle != INVALID_HANDLE_VALUE) {
            CloseHandle(filterHandle);
        }
    }

    bool setStatus(bool status) {
        DWORD bytesReturned;
        BOOLEAN state = status ? TRUE : FALSE;
        return DeviceIoControl(
            filterHandle,
            IOCTL_SET_ACTIVE,
            &state,
            sizeof(BOOLEAN),
            NULL,
            0,
            &bytesReturned,
            NULL
        );
    }

    bool getStatus() {
        DWORD bytesReturned;
        BOOLEAN state;
        DeviceIoControl(
            filterHandle,
            IOCTL_GET_ACTIVE,
            NULL,
            0,
            &state,
            sizeof(BOOLEAN),
            &bytesReturned,
            NULL
        );
        return state == TRUE;
    }

    std::vector<std::wstring> getBlacklist() {
        std::vector<std::wstring> blacklist;
        DWORD bytesNeeded = 0;

        DeviceIoControl(
            filterHandle,
            IOCTL_GET_BLACKLIST,
            NULL,
            0,
            NULL,
            0,
            &bytesNeeded,
            NULL
        );

        if (bytesNeeded > 0) {
            std::vector<wchar_t> buffer(bytesNeeded / sizeof(wchar_t));
            DeviceIoControl(
                filterHandle,
                IOCTL_GET_BLACKLIST,
                NULL,
                0,
                buffer.data(),
                bytesNeeded,
                &bytesNeeded,
                NULL
            );

            const wchar_t* current = buffer.data();
            while (*current) {
                blacklist.push_back(current);
                current += wcslen(current) + 1;
            }
        }

        return blacklist;
    }

    bool addToBlacklist(const std::wstring& deviceInstanceId) {
        sessionBlacklistDevices.push_back(deviceInstanceId);

        // Get current blacklist
        std::vector<std::wstring> current = getBlacklist();

        // Add new device if not already present
        if (std::find(current.begin(), current.end(), deviceInstanceId) == current.end()) {
            current.push_back(deviceInstanceId);
        }

        // Create double-null-terminated string
        std::wstring buffer;
        for (const auto& id : current) {
            buffer += id;
            buffer += L'\0';
        }
        buffer += L'\0';  // Add final null terminator

        DWORD bytesReturned;
        return DeviceIoControl(
            filterHandle,
            IOCTL_SET_BLACKLIST,
            (LPVOID)buffer.c_str(),
            (buffer.length() * sizeof(wchar_t)),
            NULL,
            0,
            &bytesReturned,
            NULL
        );
    }

    bool removeFromBlacklist(const std::wstring& deviceInstanceId) {
        // Get current blacklist
        std::vector<std::wstring> current = getBlacklist();

        // Remove the specified device and create new list
        current.erase(std::remove(current.begin(), current.end(), deviceInstanceId), current.end());

        // Create double-null-terminated string
        std::wstring buffer;
        for (const auto& id : current) {
            buffer += id;
            buffer += L'\0';
        }
        buffer += L'\0';  // Add final null terminator

        DWORD bytesReturned;
        return DeviceIoControl(
            filterHandle,
            IOCTL_SET_BLACKLIST,
            (LPVOID)buffer.c_str(),
            (buffer.length() * sizeof(wchar_t)),
            NULL,
            0,
            &bytesReturned,
            NULL
        );
    }

    void resetBlacklist() {
        std::vector<std::wstring> current = getBlacklist();

        for (const auto& device : sessionBlacklistDevices) {
            current.erase(std::remove(current.begin(), current.end(), device), current.end());
        }

        // Create double-null-terminated string
        std::wstring buffer;
        for (const auto& id : current) {
            buffer += id;
            buffer += L'\0';
        }
        buffer += L'\0';

        DWORD bytesReturned;
        DeviceIoControl(
            filterHandle,
            IOCTL_SET_BLACKLIST,
            (LPVOID)buffer.c_str(),
            (buffer.length() * sizeof(wchar_t)),
            NULL,
            0,
            &bytesReturned,
            NULL
        );

        sessionBlacklistDevices.clear();
    }

    std::vector<std::wstring> getWhitelist() {
        std::vector<std::wstring> whitelist;
        DWORD bytesNeeded = 0;

        DeviceIoControl(
            filterHandle,
            IOCTL_GET_WHITELIST,
            NULL,
            0,
            NULL,
            0,
            &bytesNeeded,
            NULL
        );

        if (bytesNeeded > 0) {
            std::vector<wchar_t> buffer(bytesNeeded / sizeof(wchar_t));
            DeviceIoControl(
                filterHandle,
                IOCTL_GET_WHITELIST,
                NULL,
                0,
                buffer.data(),
                bytesNeeded,
                &bytesNeeded,
                NULL
            );

            const wchar_t* current = buffer.data();
            while (*current) {
                whitelist.push_back(current);
                current += wcslen(current) + 1;
            }
        }

        return whitelist;
    }

    bool addToWhitelist(const std::wstring& applicationPath) {
        sessionWhitelistApplications.push_back(applicationPath);

        std::vector<std::wstring> current = getWhitelist();

        if (std::find(current.begin(), current.end(), applicationPath) == current.end()) {
            current.push_back(applicationPath);
        }

        std::wstring buffer;
        for (const auto& path : current) {
            buffer += path;
            buffer += L'\0';
        }
        buffer += L'\0';

        DWORD bytesReturned;
        return DeviceIoControl(
            filterHandle,
            IOCTL_SET_WHITELIST,
            (LPVOID)buffer.c_str(),
            (buffer.length() * sizeof(wchar_t)),
            NULL,
            0,
            &bytesReturned,
            NULL
        );
    }


    bool removeFromWhitelist(const std::wstring& applicationPath) {
        std::vector<std::wstring> current = getWhitelist();
        current.erase(std::remove(current.begin(), current.end(), applicationPath), current.end());

        std::wstring buffer;
        for (const auto& path : current) {
            buffer += path;
            buffer += L'\0';
        }
        buffer += L'\0';

        DWORD bytesReturned;
        return DeviceIoControl(
            filterHandle,
            IOCTL_SET_WHITELIST,
            (LPVOID)buffer.c_str(),
            (buffer.length() * sizeof(wchar_t)),
            NULL,
            0,
            &bytesReturned,
            NULL
        );
    }

    void resetWhitelist() {
        std::vector<std::wstring> current = getWhitelist();

        for (const auto& app : sessionWhitelistApplications) {
            current.erase(std::remove(current.begin(), current.end(), app), current.end());
        }

        std::wstring buffer;
        for (const auto& path : current) {
            buffer += path;
            buffer += L'\0';
        }
        buffer += L'\0';

        DWORD bytesReturned;
        DeviceIoControl(
            filterHandle,
            IOCTL_SET_WHITELIST,
            (LPVOID)buffer.c_str(),
            (buffer.length() * sizeof(wchar_t)),
            NULL,
            0,
            &bytesReturned,
            NULL
        );

        sessionWhitelistApplications.clear();
    }
};
