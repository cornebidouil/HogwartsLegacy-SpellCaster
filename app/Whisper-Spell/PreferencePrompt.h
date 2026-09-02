#ifndef PREFERENCE_PROMPT_H
#define PREFERENCE_PROMPT_H

#include <iostream>
#include <string>
#include <windows.h>
#include <conio.h>

/**
 * PreferencePrompt - Show saved preferences and let user continue or reconfigure
 *
 * Displays saved model and audio device choices, prompting user to:
 * - Press any key (except ESC) to continue with saved preferences
 * - Press ESC to reconfigure preferences
 */
class PreferencePrompt {
private:
    std::string modelName_;
    std::string deviceName_;
    std::string deviceAPI_;
    bool autoLaunchGame_;

    void hideCursor() {
        std::cout << "\033[?25l";
        std::cout.flush();
    }

    void showCursor() {
        std::cout << "\033[?25h";
        std::cout.flush();
    }

public:
    PreferencePrompt(const std::string& modelName,
                    const std::string& deviceName,
                    const std::string& deviceAPI,
                    bool autoLaunchGame)
        : modelName_(modelName), deviceName_(deviceName), deviceAPI_(deviceAPI), autoLaunchGame_(autoLaunchGame) {}

    /**
     * Show preference confirmation prompt
     * @return true to use saved preferences, false to reconfigure
     */
    bool show() {
        hideCursor();

        std::cout << "\n\033[1;36m=== Saved Preferences ===\033[0m" << std::endl;
        std::cout << "\033[1;32m✓\033[0m Model: \033[1m" << modelName_ << "\033[0m" << std::endl;
        std::cout << "\033[1;32m✓\033[0m Audio: \033[1m" << deviceName_ << "\033[0m" << std::endl;
        std::cout << "  API: " << deviceAPI_ << std::endl;
        std::cout << "\033[1;32m✓\033[0m Auto-launch: " << (autoLaunchGame_ ? "Enabled" : "Disabled") << std::endl;
        std::cout << std::endl;
        std::cout << "\033[1;37mPress any key to continue, or ESC to reconfigure\033[0m" << std::endl;

        int key = _getch();
        showCursor();

        bool useSaved = (key != 27); // ESC = 27

        if (useSaved) {
            std::cout << "\033[1;32m\n✓ Using saved preferences\033[0m\n" << std::endl;
        } else {
            std::cout << "\033[1;33m\n⚙️  Reconfiguring...\033[0m\n" << std::endl;
        }

        return useSaved;
    }
};

#endif // PREFERENCE_PROMPT_H
