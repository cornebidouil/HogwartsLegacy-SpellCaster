#ifndef INTERACTIVE_MENU_H
#define INTERACTIVE_MENU_H

#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <conio.h>

// ANSI Color codes
#define MENU_RESET "\033[0m"
#define MENU_BOLD "\033[1m"
#define MENU_LIGHT_BLUE "\033[1;36m"
#define MENU_WHITE "\033[1;37m"
#define MENU_DIM "\033[2m"
#define MENU_GREEN "\033[1;32m"

// Arrow symbols
#define MENU_ARROW " > "
#define MENU_NO_ARROW "   "

/**
 * InteractiveMenu - Arrow key navigation menu system
 *
 * Features:
 * - Arrow up/down navigation
 * - Visual highlighting of selected option
 * - Enter to confirm selection
 * - Automatic cursor management
 */
class InteractiveMenu {
public:
    struct MenuItem {
        std::string title;
        std::string description;

        MenuItem(const std::string& t, const std::string& d = "")
            : title(t), description(d) {}
    };

private:
    std::vector<MenuItem> items_;
    std::string title_;
    int selectedIndex_;
    int defaultIndex_;

    void clearAndRepositionCursor(int totalLines) {
        // Move cursor up to the beginning of the menu
        for (int i = 0; i < totalLines; ++i) {
            std::cout << "\033[1A"; // Move up one line
        }
        // Clear from cursor to end of screen
        std::cout << "\033[0J";
        std::cout.flush();
    }

    void hideCursor() {
        std::cout << "\033[?25l";
        std::cout.flush();
    }

    void showCursor() {
        std::cout << "\033[?25h";
        std::cout.flush();
    }

    void renderMenu() {
        // Render title
        if (!title_.empty()) {
            std::cout << MENU_BOLD << title_ << MENU_RESET << std::endl << std::endl;
        }

        // Render each menu item
        for (size_t i = 0; i < items_.size(); ++i) {
            bool isSelected = (i == selectedIndex_);

            // Arrow indicator
            if (isSelected) {
                std::cout << MENU_LIGHT_BLUE << MENU_ARROW;
            } else {
                std::cout << MENU_NO_ARROW;
            }

            // Item number and title
            if (isSelected) {
                std::cout << MENU_BOLD << MENU_LIGHT_BLUE << (i + 1) << ". " << items_[i].title << MENU_RESET << std::endl;
            } else {
                std::cout << MENU_WHITE << (i + 1) << ". " << items_[i].title << MENU_RESET << std::endl;
            }

            // Description (indented)
            if (!items_[i].description.empty()) {
                if (isSelected) {
                    std::cout << MENU_LIGHT_BLUE << "     " << items_[i].description << MENU_RESET << std::endl;
                } else {
                    std::cout << MENU_DIM << "     " << items_[i].description << MENU_RESET << std::endl;
                }
            }
        }

        std::cout << std::endl;
        std::cout << MENU_DIM << "Use UP/DOWN arrow keys to navigate, Enter to select" << MENU_RESET << std::endl;
    }

    int countMenuLines() const {
        size_t lines = 0;
        if (!title_.empty()) {
            lines += 2; // Title + blank line
        }
        for (const auto& item : items_) {
            lines++; // Title line
            if (!item.description.empty()) {
                lines++; // Description line
            }
        }
        lines += 2; // Blank line + instruction line
        return static_cast<int>(lines);
    }

public:
    InteractiveMenu(const std::string& title, const std::vector<MenuItem>& items, int defaultIndex = 0)
        : title_(title), items_(items), selectedIndex_(defaultIndex), defaultIndex_(defaultIndex) {
        if (selectedIndex_ < 0 || selectedIndex_ >= static_cast<int>(items_.size())) {
            selectedIndex_ = 0;
        }
    }

    /**
     * Display menu and wait for user selection
     * @return Index of selected item (0-based)
     */
    int show() {
        hideCursor();

        // Initial render
        renderMenu();

        bool done = false;
        while (!done) {
            // Wait for key press
            int key = _getch();

            // Handle arrow keys (two-byte sequence on Windows)
            if (key == 0 || key == 0xE0) {
                key = _getch(); // Get the actual arrow key code

                int oldIndex = selectedIndex_;

                switch (key) {
                    case 72: // Up arrow
                        selectedIndex_--;
                        if (selectedIndex_ < 0) {
                            selectedIndex_ = items_.size() - 1; // Wrap to bottom
                        }
                        break;

                    case 80: // Down arrow
                        selectedIndex_++;
                        if (selectedIndex_ >= static_cast<int>(items_.size())) {
                            selectedIndex_ = 0; // Wrap to top
                        }
                        break;
                }

                // Redraw if selection changed
                if (oldIndex != selectedIndex_) {
                    int totalLines = countMenuLines();
                    clearAndRepositionCursor(totalLines);
                    renderMenu();
                }
            }
            // Handle Enter key
            else if (key == 13) {
                done = true;
            }
            // Handle number keys (1-9)
            else if (key >= '1' && key <= '9') {
                int numericChoice = key - '1'; // Convert to 0-based index
                if (numericChoice < static_cast<int>(items_.size())) {
                    selectedIndex_ = numericChoice;
                    done = true;
                }
            }
            // Handle Escape key (use default)
            else if (key == 27) {
                selectedIndex_ = defaultIndex_;
                done = true;
            }
        }

        showCursor();

        // Show final selection
        std::cout << MENU_GREEN << "\nSelected: " << items_[selectedIndex_].title << MENU_RESET << std::endl;

        return selectedIndex_;
    }

    /**
     * Get the selected item
     */
    const MenuItem& getSelectedItem() const {
        return items_[selectedIndex_];
    }
};

#endif // INTERACTIVE_MENU_H
