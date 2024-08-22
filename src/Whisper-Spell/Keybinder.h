#ifndef DEF_KEYBINDER
#define DEF_KEYBINDER

#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <future>
#include <string>
#include <thread>
#include <fstream>
#include <sstream>
#include <Windows.h>
#include <winuser.h>

#include "Tools.h"

//https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes

const std::vector<std::string> NEED_TO_WAIT{"accio"};

const std::map<std::string, WORD> unrealKeyMap = {
    {"None", 0},
    {"LeftMouseButton", VK_LBUTTON},
    {"RightMouseButton", VK_RBUTTON},
    {"Cancel", VK_CANCEL},
    {"MiddleMouseButton", VK_MBUTTON},
    {"ThumbMouseButton", VK_XBUTTON1},
    {"ThumbMouseButton2", VK_XBUTTON2},
    {"BackSpace", VK_BACK},
    {"Tab", VK_TAB},
    {"Enter", VK_RETURN},
    {"Pause", VK_PAUSE},
    {"CapsLock", VK_CAPITAL},
    {"Escape", VK_ESCAPE},
    {"SpaceBar", VK_SPACE},
    {"PageUp", VK_PRIOR},
    {"PageDown", VK_NEXT},
    {"End", VK_END},
    {"Home", VK_HOME},
    {"Left", VK_LEFT},
    {"Up", VK_UP},
    {"Right", VK_RIGHT},
    {"Down", VK_DOWN},
    {"Insert", VK_INSERT},
    {"Delete", VK_DELETE},
    {"Zero", '0'},
    {"One", '1'},
    {"Two", '2'},
    {"Three", '3'},
    {"Four", '4'},
    {"Five", '5'},
    {"Six", '6'},
    {"Seven", '7'},
    {"Eight", '8'},
    {"Nine", '9'},
    {"A", 'A'},
    {"B", 'B'},
    {"C", 'C'},
    {"D", 'D'},
    {"E", 'E'},
    {"F", 'F'},
    {"G", 'G'},
    {"H", 'H'},
    {"I", 'I'},
    {"J", 'J'},
    {"K", 'K'},
    {"L", 'L'},
    {"M", 'M'},
    {"N", 'N'},
    {"O", 'O'},
    {"P", 'P'},
    {"Q", 'Q'},
    {"R", 'R'},
    {"S", 'S'},
    {"T", 'T'},
    {"U", 'U'},
    {"V", 'V'},
    {"W", 'W'},
    {"X", 'X'},
    {"Y", 'Y'},
    {"Z", 'Z'},
    {"LeftWindows", VK_LWIN},
    {"RightWindows", VK_RWIN},
    {"Applications", VK_APPS},
    {"Sleep", VK_SLEEP},
    {"NumPadZero", VK_NUMPAD0},
    {"NumPadOne", VK_NUMPAD1},
    {"NumPadTwo", VK_NUMPAD2},
    {"NumPadThree", VK_NUMPAD3},
    {"NumPadFour", VK_NUMPAD4},
    {"NumPadFive", VK_NUMPAD5},
    {"NumPadSix", VK_NUMPAD6},
    {"NumPadSeven", VK_NUMPAD7},
    {"NumPadEight", VK_NUMPAD8},
    {"NumPadNine", VK_NUMPAD9},
    {"Multiply", VK_MULTIPLY},
    {"Add", VK_ADD},
    {"Subtract", VK_SUBTRACT},
    {"Decimal", VK_DECIMAL},
    {"Divide", VK_DIVIDE},
    {"F1", VK_F1},
    {"F2", VK_F2},
    {"F3", VK_F3},
    {"F4", VK_F4},
    {"F5", VK_F5},
    {"F6", VK_F6},
    {"F7", VK_F7},
    {"F8", VK_F8},
    {"F9", VK_F9},
    {"F10", VK_F10},
    {"F11", VK_F11},
    {"F12", VK_F12},
    {"F13", VK_F13},
    {"F14", VK_F14},
    {"F15", VK_F15},
    {"F16", VK_F16},
    {"F17", VK_F17},
    {"F18", VK_F18},
    {"F19", VK_F19},
    {"F20", VK_F20},
    {"F21", VK_F21},
    {"F22", VK_F22},
    {"F23", VK_F23},
    {"F24", VK_F24},
    {"NumLock", VK_NUMLOCK},
    {"ScrollLock", VK_SCROLL},
    {"LeftShift", VK_LSHIFT},
    {"RightShift", VK_RSHIFT},
    {"LeftControl", VK_LCONTROL},
    {"RightControl", VK_RCONTROL},
    {"LeftAlt", VK_LMENU},
    {"RightAlt", VK_RMENU},
    {"Semicolon", VK_OEM_1},
    {"Equals", VK_OEM_PLUS},
    {"Comma", VK_OEM_COMMA},
    {"Hyphen", VK_OEM_MINUS},
    {"Period", VK_OEM_PERIOD},
    {"Slash", VK_OEM_2},
    {"Tilde", VK_OEM_3},
    {"LeftBracket", VK_OEM_4},
    {"Backslash", VK_OEM_5},
    {"RightBracket", VK_OEM_6},
    {"Quote", VK_OEM_7},
    {"Unknown", VK_OEM_8},
};


const std::map<std::string, std::vector<WORD>> defaultBinding = {
    {"columns", { 0x31, 0x32, 0x33, 0x34 }},
    {"lines", { VK_F1, VK_F2, VK_F3, VK_F4 }},
    {"accio broomstick", { VK_TAB , 0x33 }},
    {"smash", {'X'}},
    {"revelio", {'R'}},
    {"protego", {'A'}},
    {"appare vestigium", {'V'}},
    {"petrificus totalus", {'F'}},
    {"oppugno", {'W'}}
};



class Keybinder
{
public:
	Keybinder(std::string conf_path);
	~Keybinder();

    std::map<std::string, std::vector<WORD>> loadGameBindings();
    void loadConfBindings(const std::string& conf_path, const std::map<std::string, std::vector<WORD>>& game_bindings);

	bool decode(const std::string& word, const bool& final_record = false);
	void start();
	void stop();

	void pressKey(WORD key_code, int duration_ms = 0);
	void combinationKey(std::vector<WORD> key_codes, int duration_ms = 0);
	void holdRightClick(int duration_ms = 0);

	void checkHold();
private:
	bool _is_working, _lumos_status;
	std::vector<std::future<void>> _hold_thread;
    std::map<std::string, std::vector<WORD>> _game_bindings;
	std::map<std::string, std::vector<WORD>> _principal_bindings, _secondary_bindings;
    std::string _conf_path;
    std::size_t _conf_hash;
};

#endif // !DEF_KEYBINDER
