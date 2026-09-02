#pragma once

#include <string>
#include <vector>

class FirstRunDialog {
public:
    struct UserConsent {
        bool consentGiven = false;
        bool hasAccount = false;
        std::string username;
        std::string password;
        std::string nationality;
        std::string gender; // "male", "female", "other", "prefer_not_to_say"
    };

    /**
     * Show console-based first-run consent dialog
     * Asks user about crowdsourcing participation, account status, and metadata
     * @param outConsent Output structure to receive user's choices
     * @return true if user provided consent, false if declined or error
     */
    static bool show(UserConsent& outConsent);

    /**
     * Ask for the password of an already configured account, for example when
     * the protected value in config.ini cannot be decrypted on this machine.
     * @param username Account the password belongs to (shown to the user)
     * @return The password typed, or an empty string if the user skipped
     */
    static std::string promptPassword(const std::string& username);

private:
    static std::string getInput(const std::string& prompt, const std::string& defaultValue = "");
    static std::string getSecureInput(const std::string& prompt);
};
