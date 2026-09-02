#include "FirstRunDialog.h"
#include "CrowdsourcingUtils.h"
#include "InteractiveMenu.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <windows.h>
#include <conio.h>

bool FirstRunDialog::show(UserConsent& outConsent) {
    // Clear screen and show welcome message
    system("cls");

    std::cout << "\n";
    std::cout << "\033[1;36m" << "╔════════════════════════════════════════════════════════════════════════╗" << "\033[0m" << std::endl;
    std::cout << "\033[1;36m" << "║           Welcome to SpellCaster Voice Crowdsourcing!                  ║" << "\033[0m" << std::endl;
    std::cout << "\033[1;36m" << "╚════════════════════════════════════════════════════════════════════════╝" << "\033[0m" << std::endl;
    std::cout << "\n";
    std::cout << "Help us improve spell recognition by contributing your voice recordings!\n\n";
    std::cout << "Your contribution will:\n";
    std::cout << "  • Help train better speech recognition models\n";
    std::cout << "  • Improve accuracy for all players\n";
    std::cout << "  • Support diverse accents and speaking styles\n\n";
    std::cout << "Privacy & Data:\n";
    std::cout << "  • Only successfully recognized spell commands are saved\n";
    std::cout << "  • Audio recordings are 1-3 seconds long\n";
    std::cout << "  • You can participate anonymously or link your account\n";
    std::cout << "  • You can disable this feature at any time in config.ini\n\n";

    // Ask for consent
    std::vector<InteractiveMenu::MenuItem> consentItems = {
        InteractiveMenu::MenuItem(
            "Yes, I agree to contribute",
            "Help improve spell recognition for everyone"
        ),
        InteractiveMenu::MenuItem(
            "No, I decline",
            "You can enable this later in config.ini"
        )
    };

    InteractiveMenu consentMenu("Do you agree to contribute your voice recordings?", consentItems, 0);
    int consentChoice = consentMenu.show();

    if (consentChoice != 0) {
        std::cout << "\nCrowdsourcing disabled. You can enable it later in config.ini\n";
        outConsent.consentGiven = false;
        std::cout << "\nPress any key to continue...";
        _getch();
        return false;
    }

    outConsent.consentGiven = true;
    std::cout << "\n\033[1;32m" << "Thank you for contributing!" << "\033[0m" << "\n\n";

    // Ask about account
    std::vector<InteractiveMenu::MenuItem> accountItems = {
        InteractiveMenu::MenuItem(
            "I have an account",
            "Link recordings to your hogwartslegacyspellcaster.xyz profile"
        ),
        InteractiveMenu::MenuItem(
            "Use anonymously",
            "Contribute without account authentication"
        )
    };

    InteractiveMenu accountMenu("Account Selection:", accountItems, 1);
    int accountChoice = accountMenu.show();

    outConsent.hasAccount = (accountChoice == 0);

    if (outConsent.hasAccount) {
        std::cout << "\n\033[1;33m" << "Account Authentication" << "\033[0m" << "\n";
        std::cout << "Recordings will count toward your website profile limits.\n\n";

        outConsent.username = getInput("Username: ");
        if (outConsent.username.empty()) {
            std::cout << "\033[1;31m" << "Invalid username. Switching to anonymous mode." << "\033[0m" << "\n";
            outConsent.hasAccount = false;
        } else {
            outConsent.password = getSecureInput("Password: ");
            if (outConsent.password.empty()) {
                std::cout << "\033[1;31m" << "Invalid password. Switching to anonymous mode." << "\033[0m" << "\n";
                outConsent.hasAccount = false;
                outConsent.username.clear();
            }
        }
    }

    // Optional metadata (for both account and anonymous users)
    std::cout << "\n\033[1;33m" << "Optional Information" << "\033[0m" << "\n";
    std::cout << "This helps us understand our diverse community (optional, press Enter to skip).\n\n";

    outConsent.nationality = getInput("Country/Nationality: ");

    // Gender selection
    std::vector<InteractiveMenu::MenuItem> genderItems = {
        InteractiveMenu::MenuItem("Male", ""),
        InteractiveMenu::MenuItem("Female", ""),
        InteractiveMenu::MenuItem("Other", ""),
        InteractiveMenu::MenuItem("Prefer not to say", ""),
        InteractiveMenu::MenuItem("Skip", "")
    };

    InteractiveMenu genderMenu("Gender (Optional):", genderItems, 4);
    int genderChoice = genderMenu.show();

    switch (genderChoice) {
        case 0: outConsent.gender = "male"; break;
        case 1: outConsent.gender = "female"; break;
        case 2: outConsent.gender = "other"; break;
        case 3: outConsent.gender = "prefer_not_to_say"; break;
        case 4: outConsent.gender = ""; break;
        default: outConsent.gender = ""; break;
    }

    // Summary
    std::cout << "\n\n\033[1;32m" << "Setup Complete!" << "\033[0m" << "\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Mode: " << (outConsent.hasAccount ? "Account (" + outConsent.username + ")" : "Anonymous") << "\n";
    if (!outConsent.nationality.empty()) {
        std::cout << "Country: " << outConsent.nationality << "\n";
    }
    if (!outConsent.gender.empty() && outConsent.gender != "prefer_not_to_say") {
        std::cout << "Gender: " << outConsent.gender << "\n";
    }
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    std::cout << "Settings saved to config.ini\n";
    std::cout << "Audio recordings will be uploaded automatically on startup.\n\n";
    std::cout << "Press any key to continue...";
    _getch();

    return true;
}

std::string FirstRunDialog::getInput(const std::string& prompt, const std::string& defaultValue) {
    std::cout << prompt;
    if (!defaultValue.empty()) {
        std::cout << " [" << defaultValue << "]";
    }
    std::cout << " ";

    std::string input;
    std::getline(std::cin, input);

    // Trim whitespace
    input.erase(0, input.find_first_not_of(" \t\r\n"));
    input.erase(input.find_last_not_of(" \t\r\n") + 1);

    if (input.empty() && !defaultValue.empty()) {
        return defaultValue;
    }

    return input;
}

std::string FirstRunDialog::getSecureInput(const std::string& prompt) {
    std::cout << prompt;

    std::string input;
    char ch;

    while ((ch = _getch()) != '\r') { // Enter key
        if (ch == '\b') { // Backspace
            if (!input.empty()) {
                input.pop_back();
                std::cout << "\b \b"; // Erase character from console
            }
        } else if (ch >= 32 && ch <= 126) { // Printable characters
            input += ch;
            std::cout << '*'; // Show asterisk instead of character
        }
    }

    std::cout << std::endl;
    return input;
}

std::string FirstRunDialog::promptPassword(const std::string& username) {
    std::cout << "\n\033[1;33m" << "Crowdsourcing account" << "\033[0m" << "\n";
    std::cout << "The password for '" << username << "' is not available on this machine.\n";
    std::cout << "Enter it to keep uploading recordings to your profile, or press Enter to skip.\n\n";
    return getSecureInput("Password: ");
}

