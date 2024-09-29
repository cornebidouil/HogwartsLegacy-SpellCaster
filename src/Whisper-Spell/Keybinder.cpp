#include "Keybinder.h"

std::string getAzertyEquivalent(const std::string& key)
{
	if (key == "Q")
		return "A";
	else if (key == "A")
		return "Q";
	else if (key == "Z")
		return "W";
	else if (key == "W")
		return "Z";
	else if (key == "M")
		return ";";
	else if (key == ";")
		return "M";
	// add other keys as needed...

	// If no AZERTY equivalent is found, return the original key
	return key;
}

/*
VK_LBUTTON     0x01    // Left mouse button
VK_RBUTTON     0x02    // Right mouse button
VK_CANCEL      0x03    // Control-break processing
VK_MBUTTON     0x04    // Middle mouse button (three-button mouse)
VK_XBUTTON1    0x05    // X1 mouse button
VK_XBUTTON2    0x06    // X2 mouse button
VK_BACK        0x08    // BACKSPACE key
VK_TAB         0x09    // TAB key
VK_CLEAR       0x0C    // CLEAR key
VK_RETURN      0x0D    // ENTER key
VK_SHIFT       0x10    // SHIFT key
VK_CONTROL     0x11    // CTRL key
VK_MENU        0x12    // ALT key
VK_PAUSE       0x13    // PAUSE key
VK_CAPITAL     0x14    // CAPS LOCK key
VK_ESCAPE      0x1B    // ESC key
VK_SPACE       0x20    // SPACEBAR
VK_PRIOR       0x21    // PAGE UP key
VK_NEXT        0x22    // PAGE DOWN key
VK_END         0x23    // END key
VK_HOME        0x24    // HOME key
VK_LEFT        0x25    // LEFT ARROW key
VK_UP          0x26    // UP ARROW key
VK_RIGHT       0x27    // RIGHT ARROW key
VK_DOWN        0x28    // DOWN ARROW key
VK_SELECT      0x29    // SELECT key
VK_PRINT       0x2A    // PRINT key
VK_EXECUTE     0x2B    // EXECUTE key
VK_SNAPSHOT    0x2C    // PRINT SCREEN key
VK_INSERT      0x2D    // INS key
VK_DELETE      0x2E    // DEL key
VK_HELP        0x2F    // HELP key
VK_0           0x30    // 0 key
VK_1           0x31    // 1 key
VK_2           0x32    // 2 key
VK_3           0x33    // 3 key
VK_4           0x34    // 4 key
VK_5           0x35    // 5 key
VK_6           0x36    // 6 key
VK_7           0x37    // 7 key
VK_8           0x38    // 8 key
VK_9           0x39    // 9 key
VK_A           0x41    // A key
VK_B           0x42    // B key
VK_C           0x43    // C key
VK_D           0x44    // D key
VK_E           0x45    // E key
VK_F           0x46    // F key
VK_G           0x47    // G key
VK_H           0x48    // H key
VK_I           0x49    // I key
VK_J           0x4A    // J key
VK_K           0x4B    // K key
VK_L           0x4C    // L key
VK_M           0x4D    // M key
VK_N           0x4E    // N key
VK_O           0x4F    // O key
VK_P           0x50    // P key
VK_Q           0x51    // Q key
VK_R           0x52    // R key
VK_S           0x53    // S key
VK_T           0x54    // T key
VK_U           0x55    // U key
VK_V           0x56    // V key
VK_W           0x57    // W key
VK_X           0x58    // X key
VK_Y           0x59    // Y key
VK_Z           0x5A    // Z key
VK_LWIN        0x5B    // Left Windows key
VK_RWIN        0x5C    // Right Windows key
VK_APPS        0x5D    // Applications key (Windows context menu key)
VK_SLEEP       0x5F    // Computer Sleep key
VK_NUMPAD0     0x60    // Numeric keypad 0 key
VK_NUMPAD1     0x61    // Numeric keypad 1 key
VK_NUMPAD2     0x62    // Numeric keypad 2 key
VK_NUMPAD3     0x63    // Numeric keypad 3 key
VK_NUMPAD4     0x64    // Numeric keypad 4 key
VK_NUMPAD5     0x65    // Numeric keypad 5 key
VK_NUMPAD6     0x66    // Numeric keypad 6 key
VK_NUMPAD7     0x67    // Numeric keypad 7 key
VK_NUMPAD8     0x68    // Numeric keypad 8 key
VK_NUMPAD9     0x69    // Numeric keypad 9 key
VK_MULTIPLY    0x6A    // Multiply key
VK_ADD         0x6B    // Add key
VK_SEPARATOR   0x6C    // Separator key
VK_SUBTRACT    0x6D    // Subtract key
VK_DECIMAL     0x6E    // Decimal key
VK_DIVIDE      0x6F    // Divide key
VK_F1          0x70    // F1 key
VK_F2          0x71    // F2 key
VK_F3          0x72    // F3 key
VK_F4          0x73    // F4 key
VK_F5          0x74    // F5 key
VK_F6          0x75    // F6 key
VK_F7          0x76    // F7 key
VK_F8          0x77    // F8 key
VK_F9          0x78    // F9 key
VK_F10         0x79    // F10 key
VK_F11         0x7A    // F11 key
VK_F12         0x7B    // F12 key
VK_NUMLOCK     0x90    // NUM LOCK key
VK_SCROLL      0x91    // SCROLL LOCK key
*/

Keybinder::Keybinder(std::string conf_path) :
_is_working(false),
_lumos_status(false),
_conf_path(conf_path),
_device_controller(-1)
{
	_principal_bindings.clear();
	_secondary_bindings.clear();
	_game_bindings.clear();

	selectingGamepad();

	_game_bindings = loadGameBindings();
	loadConfBindings(conf_path, _game_bindings);
}

Keybinder::~Keybinder() {
	if (_device_controller > DEVICE_GAMEPAD_UNDEFINED) {
		vigem_target_remove(_client, _pad);
		vigem_target_free(_pad);
		vigem_free(_client);
	}
}


void Keybinder::selectingGamepad() {
	std::cout << std::endl << "Do you play with a :" << std::endl << "\t" << DEVICE_KEYBOARD << " -> Keyboard" << std::endl << "\t" << DEVICE_GAMEPAD_UNDEFINED << " -> Gamepad" << std::endl;
	std::cout << std::endl << "Enter the playing device : ";
	std::cin  >> _device_controller;

	if (_device_controller != DEVICE_KEYBOARD and _device_controller != DEVICE_GAMEPAD_UNDEFINED) {
		std::cerr << "Invalid device controller." << std::endl;
#ifdef _WIN32
		system("PAUSE");
#endif // _WIN32
		exit(-1);
	}

	if (_device_controller == DEVICE_GAMEPAD_UNDEFINED) {
		// List all available gamepads
		std::cout << std::endl << "Available gamepads:" << std::endl;
		for (int i = 0; i < SDL_NumJoysticks(); ++i) {
			if (SDL_IsGameController(i)) {
				SDL_GameController* controller = SDL_GameControllerOpen(i);
				if (controller) {
					const char* name = SDL_GameControllerName(controller);
					std::cout << "\t" << i << ": " << name << std::endl;

					// Determine if it's a DualShock or Xbox controller
					if (strstr(name, "DualShock") != nullptr || 
						strstr(name, "Sony") != nullptr) {

						std::cout << "   Type: DualShock" << std::endl;
						_device_controller = DEVICE_GAMEPAD_DUALSHOCK;
					}
					else if (strstr(name, "Xbox") != nullptr || 
							 strstr(name, "Microsoft") != nullptr) {

						std::cout << "   Type: Xbox" << std::endl;
						_device_controller = DEVICE_GAMEPAD_XBOX;
					}
					else {
						std::cout << "   Type: Unknown" << std::endl;
#ifdef _WIN32
						system("PAUSE");
#endif // _WIN32
						_device_controller = DEVICE_GAMEPAD_UNKNOWN;
					}

					SDL_GameControllerClose(controller);
				}
			}
		}



		if (_device_controller == DEVICE_GAMEPAD_UNDEFINED) { 
			std::cout << "\tNo gamepad found." << std::endl;

			_device_controller = DEVICE_GAMEPAD_XBOX;
		}
		//else {

		int manual_selection = -1;
		std::cout << std::endl << "Do you want to use the :" << std::endl;
		std::cout << "\t0 -> Automatically selected controller: " << (_device_controller == DEVICE_GAMEPAD_DUALSHOCK ? "DualShock" : "Xbox") << " controller" << std::endl;
		std::cout << "\t1 -> " << (_device_controller == DEVICE_GAMEPAD_DUALSHOCK ? "Xbox" : "DualShock") << " controller" << std::endl;
		std::cout << std::endl << "Enter the controller's type : ";
		std::cin >> manual_selection;

		if (manual_selection == 1) {
			if (_device_controller == DEVICE_GAMEPAD_DUALSHOCK) {
				_device_controller = DEVICE_GAMEPAD_XBOX;
			}
			else if (_device_controller == DEVICE_GAMEPAD_XBOX) {
				_device_controller = DEVICE_GAMEPAD_DUALSHOCK;
			}
			std::cout << "Controller type switched to: " << (_device_controller == DEVICE_GAMEPAD_DUALSHOCK ? "DualShock" : "Xbox") << std::endl;
		}


		_client = vigem_alloc();
		if (vigem_connect(_client) != VIGEM_ERROR_NONE) {
			std::cerr << "Failed to connect to ViGEm Bus" << std::endl;
#ifdef _WIN32
			system("PAUSE");
#endif
			exit(-1);
		}

		if (_device_controller == DEVICE_GAMEPAD_DUALSHOCK) {
			_pad = vigem_target_ds4_alloc();
			if (vigem_target_add(_client, _pad) != VIGEM_ERROR_NONE) {
				std::cerr << "Failed to add virtual DualShock 4 controller" << std::endl;
				vigem_free(_client);
#ifdef _WIN32
				system("PAUSE");
#endif
				exit(-1);
			}
		}
		else { // DEVICE_GAMEPAD_XBOX || DEVICE_GAMEPAD_UNKNOWN
			_pad = vigem_target_x360_alloc();
			if (vigem_target_add(_client, _pad) != VIGEM_ERROR_NONE) {
				std::cerr << "Failed to add virtual Xbox 360 controller" << std::endl;
				vigem_free(_client);
#ifdef _WIN32
				system("PAUSE");
#endif
				exit(-1);
			}
		}
		//}
	}
	else {
		std::cout << std::endl;
	}
}

std::unordered_map<std::string, std::vector<WORD>> Keybinder::loadGameBindings() {
	
	std::unordered_map<std::string, std::vector<WORD>> binding = {
		{"columns", { UNDEFINED_BINDING, UNDEFINED_BINDING, UNDEFINED_BINDING, UNDEFINED_BINDING }},
		{"lines", { UNDEFINED_BINDING, UNDEFINED_BINDING, UNDEFINED_BINDING, UNDEFINED_BINDING }},
		{"accio broomstick", { UNDEFINED_BINDING , UNDEFINED_BINDING }},
		{"smash", {UNDEFINED_BINDING}},
		{"revelio", {UNDEFINED_BINDING}},
		{"protego", {UNDEFINED_BINDING}},
		{"appare vestigium", {UNDEFINED_BINDING}},
		{"petrificus totalus", {UNDEFINED_BINDING}},
		{"oppugno", {UNDEFINED_BINDING}},
		{"alohomora", {UNDEFINED_BINDING}}
	};

	char* userProfile;
	size_t profile_size;

	// Get the value of the "USERPROFILE" environment variable
	if (_dupenv_s(&userProfile, &profile_size, "USERPROFILE") != 0 || userProfile == nullptr) {
		std::cerr << "Failed to get USERPROFILE environment variable" << std::endl;
		exit(-1);
	}

	std::string game_binding_path = std::string(userProfile) + "\\AppData\\Local\\Hogwarts Legacy\\Saved\\Config\\WindowsNoEditor\\Input.ini";
	
	if (userProfile != nullptr) {
		free(userProfile);
	}

	std::cout << "Game bindings path : " << game_binding_path << std::endl;

	std::ifstream inputFile(game_binding_path);

	if (!inputFile) {
		std::cout << "*** Game config not found : default key binding used ***" << std::endl;
		if (_device_controller == DEVICE_GAMEPAD_DUALSHOCK)
			return defaultDS4Binding;
		else if (_device_controller == DEVICE_GAMEPAD_XBOX || _device_controller == DEVICE_GAMEPAD_UNKNOWN)
			return defaultXUSBBinding;
		else 
			return defaultBinding;
	}

	WORD currentLayout = PRIMARYLANGID(HIWORD(GetKeyboardLayout(0))); //GetKeyboardLayout(0);
	if (_device_controller == DEVICE_KEYBOARD && currentLayout == LANG_FRENCH)
		std::cout << "*** AZERTY keyboard detected ***" << std::endl;

	std::vector<std::string> actionChecked;
	std::string line;
	while (std::getline(inputFile, line)) {
		std::string actionName;
		std::string key;

		if (line.find("ActionName=\"") == std::string::npos || line.find("Key=") == std::string::npos)
			continue;

		// Find the start and end position of the ActionName and Key strings
		size_t actionNameStart = line.find("ActionName=\"") + std::strlen("ActionName=\"");
		size_t actionNameEnd = line.find("\"", actionNameStart);
		size_t keyStart = line.find("Key=") + std::strlen("Key=");
		size_t keyEnd = line.find(",", keyStart);


		// Extract the ActionName and Key strings
		actionName = line.substr(actionNameStart, actionNameEnd - actionNameStart);
		key = line.substr(keyStart, keyEnd - keyStart);

		auto it = std::find(actionChecked.begin(), actionChecked.end(), actionName);

		if ((_device_controller == DEVICE_KEYBOARD && key.find("Gamepad") != std::string::npos) ||
			(_device_controller != DEVICE_KEYBOARD && key.find("Gamepad") == std::string::npos) ||
			it != actionChecked.end())
			continue;

		// Convert the key to the AZERTY equivalent if necessary
		if (_device_controller == DEVICE_KEYBOARD && currentLayout == LANG_FRENCH) { // 0x40c is the identifier for the French (France) AZERTY layout
			key = getAzertyEquivalent(key);
		}

		if (updateBinding(binding, actionName, key))
			actionChecked.push_back(actionName);
		else 
			continue;

	}

	// Check if all the bindings were found
	bool error_found = false;
	for (auto it = binding.begin(); it != binding.end(); ++it) {
		for (auto it_vec = it->second.begin(); it_vec != it->second.end(); ++it_vec) {
			if (*it_vec == UNDEFINED_BINDING)
				error_found = true; break; break;
		}
	}
	if (error_found) {
		std::cout << "*** Binding not assignated in game ***" << std::endl;
		for (auto it = binding.begin(); it != binding.end(); ++it) {
			std::cout << "  * " << it->first << " : ";
			for (auto it_vec = it->second.begin(); it_vec != it->second.end(); ++it_vec) {
				std::cout << *it_vec << " | ";
			}
			std::cout << std::endl;
		}
#ifdef _WIN32
		system("PAUSE");
#endif
		std::cout << std::endl;
	}

	inputFile.close();

	return binding;
}


WORD Keybinder::keyToBind(const std::string& key) {
	if (_device_controller == DEVICE_KEYBOARD) {
		return unrealKeyMap.at(key);
	}
	else if (_device_controller == DEVICE_GAMEPAD_DUALSHOCK) {
		return ds4_gamepadButtonMap.at(key);
	}
	else {
		return xusb_gamepadButtonMap.at(key);
	}
}

bool Keybinder::updateBinding(std::unordered_map<std::string, std::vector<WORD>>& binding, const std::string& actionName, const std::string& key) {
	static const std::unordered_map<std::string, std::pair<std::string, int>> actionMap = {
        {"AM_Loadout1", {"lines", 0}},
        {"AM_Loadout2", {"lines", 1}},
        {"AM_Loadout3", {"lines", 2}},
        {"AM_Loadout4", {"lines", 3}},
        {"AM_SpellButton1", {"columns", 0}},
        {"AM_SpellButton2", {"columns", 1}},
        {"AM_SpellButton3", {"columns", 2}},
        {"AM_SpellButton4", {"columns", 3}},
        {"AM_ItemMenu", {"accio broomstick", 0}},
        {"UMGGadgetWheelMountSlot3", {"accio broomstick", 1}},
        {"AM_Navigation", {"appare vestigium", 0}},
        {"AM_Interact", {"alohomora", 0}},
        {"AM_Oppugno", {"oppugno", 0}},
        {"AM_Protego", {"protego", 0}},
        {"AM_Revelio", {"revelio", 0}},
        {"AM_CriticalFinisher", {"smash", 0}}
    };

    auto it = actionMap.find(actionName);
    if (it != actionMap.end()) {
		//std::cout << "action: " << actionName << " " << key << " " << keyToBind(key) << std::endl;
        binding[it->second.first][it->second.second] = keyToBind(key);
        if (actionName == "AM_Interact") {
            binding["petrificus totalus"][0] = keyToBind(key);
        }
        return true;
    }
    return false;
}

void Keybinder::loadConfBindings(const std::string& conf_path, const std::unordered_map<std::string, std::vector<WORD>>& game_bindings) {
	std::vector<WORD> key_columns = game_bindings.at("columns"); // { 0x31, 0x32, 0x33, 0x34 };
	std::vector<WORD> key_lines = game_bindings.at("lines");     // { VK_F1, VK_F2, VK_F3, VK_F4 };

	std::ifstream file(conf_path);

	if (file.is_open()) {
		std::string line;
		//std::getline(file, line);

		int line_count = 0;
		while (std::getline(file, line)) {
			if (line_count > 3) {
				std::cerr << std::endl << "Too many lines in the keybinding.txt, should only be 4 for the 4 spell loadout." << std::endl << std::endl;
				break;
			}

			std::stringstream ss(line);
			std::string formula_name;

			int column_count = 0;
			while (std::getline(ss, formula_name, ';')) {
				if (column_count > 3) {
					std::cout << std::endl << "Too many columns in the keybinding.txt, should only be 4 spell by loadout." << std::endl << std::endl;
					break;
				}

				std::transform(formula_name.begin(), formula_name.end(), formula_name.begin(),
				[](unsigned char c) { return std::tolower(c); });

				_principal_bindings[formula_name] = std::vector<WORD>({ key_lines[line_count], key_columns[column_count] });
				//std::cout << formula_name << " : " << key_lines[line_count] << " | " << key_columns[column_count] << std::endl;
				column_count++;
			}
			line_count++;
		}
		file.close();
	}
	else {
		std::cerr << "Failed to open the conf file : " << conf_path << std::endl;
		exit(-1);
	}

	_conf_hash = hashFile(conf_path);

	for (auto it = game_bindings.begin(); it != game_bindings.end(); ++it) {
		if (it->first == "columns" || it->first == "lines")
			continue;

		_secondary_bindings[it->first] = it->second;
	}
}




void Keybinder::start() {
	_is_working = true;

	if (hashFile(_conf_path) != _conf_hash) {
		std::cout << "Spell Bindings modified" << std::endl;

		_principal_bindings.clear();
		_secondary_bindings.clear();

		loadConfBindings(_conf_path, _game_bindings);
	}
}

void Keybinder::stop() {
	_is_working = false;
}




bool Keybinder::decode(const std::string& word, const bool& final_record) {
	bool lumos_cast = false;

	//std::cout << "'" << trimTrailingSpaces( GetActiveWindowTitle() ) << "'" << std::endl;
	if (trimTrailingSpaces(GetActiveWindowTitle()) != "Hogwarts Legacy") {
		std::cout << "Not ingame => input skipped" << std::endl;
		return false;
	}

	if (_is_working) {
		_is_working = false;

		// --- Word which need wait
		if (std::find(NEED_TO_WAIT.begin(), NEED_TO_WAIT.end(), word) != NEED_TO_WAIT.end() && !final_record) {
			_is_working = true;
			return _is_working;
		}

		// --- Special case of lumos
		if (word == "lumos") {
			lumos_cast = true;
			_lumos_status = true;
		}
		
		auto it = _principal_bindings.find(word);
		if (it != _principal_bindings.end()) {
			if (_device_controller == DEVICE_KEYBOARD) {
				for (const WORD& key : it->second)
					pressKey(key, 0);
			} else if (_device_controller == DEVICE_GAMEPAD_DUALSHOCK) {
				sendDS4PrincipalSpells(it->second);
			} else {
				sendXUSBPrincipalSpells(it->second);
			}
			
		}

		else if (word == "accio broomstick" || word == "accio balais") {
			if (_device_controller == DEVICE_KEYBOARD)
				combinationKey(_secondary_bindings["accio broomstick"], 500);
			else if (_device_controller == DEVICE_GAMEPAD_DUALSHOCK)
				combinationDS4Button(_secondary_bindings["accio broomstick"], 500);
			else
				combinationXUSBButton(_secondary_bindings["accio broomstick"], 500);
		}

		else if (word == "nox") {
			auto it = _principal_bindings.find("lumos");
			if (it != _principal_bindings.end()) {
				if (_device_controller == DEVICE_KEYBOARD) {
					for (WORD key : it->second)
						pressKey(key);
				}
				else if (_device_controller == DEVICE_GAMEPAD_DUALSHOCK) {
					sendDS4PrincipalSpells(it->second);
				}
				else {
					sendXUSBPrincipalSpells(it->second);
				}
			}

			_lumos_status = false;
		}
		else if (word == "appare vestigium") {
			if (_device_controller == DEVICE_KEYBOARD)
				pressKey(_secondary_bindings["appare vestigium"][0], 50);
			else if (_device_controller == DEVICE_GAMEPAD_DUALSHOCK)
				pressDS4Button(_secondary_bindings["appare vestigium"][0], 50);
			else
				pressXUSBButton(_secondary_bindings["appare vestigium"][0], 50);
		}
		else if (word == "smash") {
			if (_device_controller == DEVICE_KEYBOARD)
				pressKey(_secondary_bindings[word][0]);
			else if (_device_controller == DEVICE_GAMEPAD_DUALSHOCK) {
				simultaneousPressDS4Button({ DS4_BUTTON_SHOULDER_LEFT, DS4_BUTTON_SHOULDER_RIGHT });
			}
			else {
				simultaneousPressXUSBButton({ XUSB_GAMEPAD_LEFT_SHOULDER, XUSB_GAMEPAD_RIGHT_SHOULDER });
			}
		}
		else if (word == "petrificus totalus" || 
				 word == "revelio" ||
			     word == "oppugno" || 
				 word == "alohomora" ||
				 word == "protego") {
			if (_device_controller == DEVICE_KEYBOARD)
				pressKey(_secondary_bindings[word][0]);
			else if (_device_controller == DEVICE_GAMEPAD_DUALSHOCK)
				pressDS4Button(_secondary_bindings[word][0]);
			else
				pressXUSBButton(_secondary_bindings[word][0]);

		}
		else 
			std::cout << "*** Binding not found ... ***" << std::endl;
		


		// --- Reset of lumos status if another spell is cast
		if (_lumos_status && !_is_working && !lumos_cast)
			_lumos_status = false;

	}

	return _is_working;
}


// --- Send Keyboard Inputs --- //
//

void Keybinder::pressKey(WORD key_code, int duration_ms) {
	INPUT input;
	ZeroMemory(&input, sizeof(INPUT));
	input.type = INPUT_KEYBOARD;
	input.ki.wVk = key_code;
	input.ki.dwFlags = 0;

	// Convert key code to character
	char keyName[32];
	if (GetKeyNameTextA(MapVirtualKey(key_code, MAPVK_VK_TO_VSC) << 16, keyName, sizeof(keyName)) == 0 ) {
		std::cerr << "Failed to get key name for key code: " << key_code << std::endl;
        strcpy_s(keyName, "Unknown Key");
	}

	if (SendInput(1, &input, sizeof(INPUT)) != 1) {
		std::cerr << "Failed to send keyboard input : " << keyName << "(code: " << key_code << ")"  << std::endl;
		return;
	}
	
	
	SHORT keyState = GetAsyncKeyState(key_code);
	bool isPressed = (keyState & 0x8000) != 0;
	
	std::cout << "Input : " << keyName << " (code: " << key_code << ") -> " << isPressed << std::endl;
	
	//input.ki.time += duration_ms; // Set the time stamp for the key release
	if (duration_ms != 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
	
	input.ki.dwFlags = KEYEVENTF_KEYUP;
	if (SendInput(1, &input, sizeof(INPUT)) != 1) {
		std::cerr << "Failed to send key release event for input: " << keyName << "(code: " << key_code << ")" << std::endl;
	}

}

void Keybinder::holdRightClick(int duration_ms) {
	INPUT input;
	ZeroMemory(&input, sizeof(INPUT));
	input.type = INPUT_MOUSE;
	input.ki.dwFlags = MOUSEEVENTF_RIGHTDOWN;
	SendInput(1, &input, sizeof(INPUT));
	//input.ki.time += duration_ms; // Set the time stamp for the key release
	std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
	input.ki.dwFlags = MOUSEEVENTF_RIGHTUP;
	SendInput(1, &input, sizeof(INPUT));
}

void Keybinder::combinationKey(std::vector<WORD> key_codes, int duration_ms) {
	INPUT input;
	ZeroMemory(&input, sizeof(INPUT));
	input.type = INPUT_KEYBOARD;

	// --- PRESS
	int press_count = 0;
	for (WORD key : key_codes) {
		input.ki.wVk = key;
		input.ki.dwFlags = 0;
		SendInput(1, &input, sizeof(INPUT));

		if (press_count != key_codes.size()-1)
			std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

		press_count++;
	}

	// --- RELEASE
	for (auto it = key_codes.rbegin(); it != key_codes.rend(); ++it) {
		input.ki.wVk = *it;
		input.ki.dwFlags = KEYEVENTF_KEYUP;
		SendInput(1, &input, sizeof(INPUT));
	}
}

void Keybinder::checkHold() {
	auto iter = _hold_thread.begin();
	while (iter != _hold_thread.end()) {
		// Check whether the future has finished executing
		if (iter->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
			// Remove the future from the vector
			iter = _hold_thread.erase(iter);
		}
		else {
			++iter;
		}
	}
}


// --- Send XBOX Inputs --- //
//

void Keybinder::pressXUSBButton(WORD button, int duration_ms, bool combined) {
	if (!combined) {
		XUSB_REPORT_INIT(&_x360_report);
	}

	_x360_report.wButtons |= button;
	vigem_target_x360_update(_client, _pad, _x360_report);
	
	if (duration_ms == 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(25)); //15
	else
		std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

	if(!combined) {
		XUSB_REPORT_INIT(&_x360_report);
		vigem_target_x360_update(_client, _pad, _x360_report);
	} else {
		_x360_report.wButtons ^= button;
	}
}

void Keybinder::combinationXUSBButton(std::vector<WORD> buttons, int duration_ms) {
	XUSB_REPORT_INIT(&_x360_report);

	int press_count = 0;
	for (WORD button : buttons) {
		_x360_report.wButtons |= button;
		vigem_target_x360_update(_client, _pad, _x360_report);

		if (press_count != buttons.size()-1)
			std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

		press_count++;
	}

	std::this_thread::sleep_for(std::chrono::milliseconds(25));

	XUSB_REPORT_INIT(&_x360_report);
	vigem_target_x360_update(_client, _pad, _x360_report);
}

void Keybinder::sendXUSBPrincipalSpells(std::vector<WORD> buttons, int duration_ms) {
	XUSB_REPORT_INIT(&_x360_report);

	_x360_report.bRightTrigger = 255;
	vigem_target_x360_update(_client, _pad, _x360_report);
	std::this_thread::sleep_for(std::chrono::milliseconds(15));

	for (const WORD& button : buttons) {
		pressXUSBButton(button, 0, true);
	}

	XUSB_REPORT_INIT(&_x360_report);
	vigem_target_x360_update(_client, _pad, _x360_report);
}

void Keybinder::simultaneousPressXUSBButton(std::vector<WORD> buttons, int duration_ms) {
	XUSB_REPORT_INIT(&_x360_report);
	for (const WORD& button : buttons) {
		_x360_report.wButtons |= button;
	}
	vigem_target_x360_update(_client, _pad, _x360_report);

	if (duration_ms == 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(25)); //15
	else
		std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

	XUSB_REPORT_INIT(&_x360_report);
	vigem_target_x360_update(_client, _pad, _x360_report);
}



// --- Send Dualshocks Inputs --- //
//

void Keybinder::pressDS4Button(WORD button, int duration_ms, bool combined){
	if (!combined) {
		DS4_REPORT_INIT(&_ds4_report);
	}

	bool is_dpad = false; 
	if (button & (
		DS4_BUTTON_DPAD_NONE |
		DS4_BUTTON_DPAD_NORTHWEST |
		DS4_BUTTON_DPAD_WEST |
		DS4_BUTTON_DPAD_SOUTHWEST |
		DS4_BUTTON_DPAD_SOUTH |
		DS4_BUTTON_DPAD_SOUTHEAST |
		DS4_BUTTON_DPAD_EAST |
		DS4_BUTTON_DPAD_NORTHEAST |
		DS4_BUTTON_DPAD_NORTH))
		is_dpad = true;

	if (is_dpad)
		DS4_SET_DPAD(&_ds4_report, static_cast<DS4_DPAD_DIRECTIONS>(button));
	else
		_ds4_report.wButtons |= button;

	vigem_target_ds4_update(_client, _pad, _ds4_report);
	
	if (duration_ms == 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(25)); //15
	else
		std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

	if(!combined) {
		DS4_REPORT_INIT(&_ds4_report);
		vigem_target_ds4_update(_client, _pad, _ds4_report);
	} else {
		if (is_dpad)
			DS4_SET_DPAD(&_ds4_report, DS4_BUTTON_DPAD_NONE);
		else 
			_ds4_report.wButtons ^= button;
	}
}

void Keybinder::combinationDS4Button(std::vector<WORD> buttons, int duration_ms) {
	DS4_REPORT_INIT(&_ds4_report);

	int press_count = 0;
	for (WORD button : buttons) {
		if (button & (
			DS4_BUTTON_DPAD_NONE |
			DS4_BUTTON_DPAD_NORTHWEST |
			DS4_BUTTON_DPAD_WEST |
			DS4_BUTTON_DPAD_SOUTHWEST |
			DS4_BUTTON_DPAD_SOUTH |
			DS4_BUTTON_DPAD_SOUTHEAST |
			DS4_BUTTON_DPAD_EAST |
			DS4_BUTTON_DPAD_NORTHEAST |
			DS4_BUTTON_DPAD_NORTH))
			DS4_SET_DPAD(&_ds4_report, static_cast<DS4_DPAD_DIRECTIONS>(button));
		else 
			_ds4_report.wButtons |= button;
		vigem_target_ds4_update(_client, _pad, _ds4_report);

		if (press_count != buttons.size()-1)
			std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

		press_count++;
	}

	std::this_thread::sleep_for(std::chrono::milliseconds(25));

	DS4_REPORT_INIT(&_ds4_report);
	vigem_target_ds4_update(_client, _pad, _ds4_report);
}

void Keybinder::sendDS4PrincipalSpells(std::vector<WORD> buttons, int duration_ms) {
	DS4_REPORT_INIT(&_ds4_report);

	_ds4_report.bTriggerR = 255;
	vigem_target_ds4_update(_client, _pad, _ds4_report);
	std::this_thread::sleep_for(std::chrono::milliseconds(15));
	//std::this_thread::sleep_for(std::chrono::seconds(5));

	for (const WORD& button : buttons) {
		pressDS4Button(button, 0, true);
	}

	DS4_REPORT_INIT(&_ds4_report);
	vigem_target_ds4_update(_client, _pad, _ds4_report);
}


void Keybinder::simultaneousPressDS4Button(std::vector<WORD> buttons, int duration_ms) {
	DS4_REPORT_INIT(&_ds4_report);
	for (const WORD& button : buttons) {
		if (button & (
			DS4_BUTTON_DPAD_NONE |
			DS4_BUTTON_DPAD_NORTHWEST |
			DS4_BUTTON_DPAD_WEST |
			DS4_BUTTON_DPAD_SOUTHWEST |
			DS4_BUTTON_DPAD_SOUTH |
			DS4_BUTTON_DPAD_SOUTHEAST |
			DS4_BUTTON_DPAD_EAST |
			DS4_BUTTON_DPAD_NORTHEAST |
			DS4_BUTTON_DPAD_NORTH))
			DS4_SET_DPAD(&_ds4_report, static_cast<DS4_DPAD_DIRECTIONS>(button));
		else 
			_ds4_report.wButtons |= button;
	}
	vigem_target_ds4_update(_client, _pad, _ds4_report);

	if (duration_ms == 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(25)); //15
	else
		std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

	DS4_REPORT_INIT(&_ds4_report);
	vigem_target_ds4_update(_client, _pad, _ds4_report);
}



// --- Auxiliary Functions --- //
//

void Keybinder::checkAllBindings() {
	if (_device_controller == 0) {
		std::cout << "Checking all bindings:" << std::endl;

		// Check principal bindings
		std::cout << std::endl << "\tPrincipal bindings:" << std::endl;
		for (const auto& binding : _principal_bindings) {
			std::cout << "\t   * " << binding.first << ": ";
			for (const auto& key : binding.second) {
				std::cout << getKeyName(key) << " ";
			}
			std::cout << std::endl;
		}

		// Check secondary bindings
		std::cout << std::endl << "\tSecondary bindings:" << std::endl;
		for (const auto& binding : _secondary_bindings) {
			std::cout << "\t   * " << binding.first << ": ";
			for (const auto& key : binding.second) {
				std::cout << getKeyName(key) << " ";
			}
			std::cout << std::endl;
		}

		// Check game bindings
		std::cout << std::endl << "\tGame bindings:" << std::endl;
		for (const auto& binding : _game_bindings) {
			std::cout << "\t   * " << binding.first << ": ";
			for (const auto& key : binding.second) {
				std::cout << getKeyName(key) << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl << std::endl << std::endl;
	}
}


std::string Keybinder::getKeyName(WORD key_code) {
	char keyName[32];
	UINT scanCode = MapVirtualKey(key_code, MAPVK_VK_TO_VSC);
	LONG lParam = (scanCode << 16);

	if (GetKeyNameTextA(lParam, keyName, sizeof(keyName)) == 0) {
		return "Unknown Key";
	}

	return std::string(keyName);
}