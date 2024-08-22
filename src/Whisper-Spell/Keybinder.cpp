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
_conf_path(conf_path)
{
	_principal_bindings.clear();
	_secondary_bindings.clear();
	_game_bindings.clear();

	_game_bindings = loadGameBindings();
	loadConfBindings(conf_path, _game_bindings);
}

Keybinder::~Keybinder() {

}

std::map<std::string, std::vector<WORD>> Keybinder::loadGameBindings() {
	
	std::map<std::string, std::vector<WORD>> binding = {
		{"columns", { 0, 0, 0, 0 }},
		{"lines", { 0, 0, 0, 0 }},
		{"accio broomstick", { 0 , 0 }},
		{"smash", {0}},
		{"revelio", {0}},
		{"protego", {0}},
		{"appare vestigium", {0}},
		{"petrificus totalus", {0}},
		{"oppugno", {0}},
		{"alohomora", {0}}
	};
	//{"stupefy", {0}}

	/*const char* userProfile = std::getenv("USERPROFILE");
	if (!userProfile) {
		std::cerr << "Failed to get USERPROFILE environment variable" << std::endl;
		exit(-1);
	}*/
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
		return defaultBinding;
	}

	WORD currentLayout = PRIMARYLANGID(HIWORD(GetKeyboardLayout(0))); //GetKeyboardLayout(0);
	if (currentLayout == LANG_FRENCH)
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
		if (key.find("Gamepad") != std::string::npos || it != actionChecked.end())
			continue;

		// Convert the key to the AZERTY equivalent if necessary
		if (currentLayout == LANG_FRENCH) { // 0x40c is the identifier for the French (France) AZERTY layout
			key = getAzertyEquivalent(key);
		}

		// Identify the action
		if (actionName == "AM_Loadout1") {
			binding["lines"][0] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "AM_Loadout2") {
			binding["lines"][1] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "AM_Loadout3") {
			binding["lines"][2] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "AM_Loadout4") {
			binding["lines"][3] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "AM_SpellButton1") {
			binding["columns"][0] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "AM_SpellButton2") {
			binding["columns"][1] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "AM_SpellButton3") {
			binding["columns"][2] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "AM_SpellButton4") {
			binding["columns"][3] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "AM_ItemMenu") {
			binding["accio broomstick"][0] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "UMGGadgetWheelMountSlot3") {
			binding["accio broomstick"][1] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "AM_Navigation") {
			binding["appare vestigium"][0] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "AM_Interact") {
			binding["alohomora"][0] = unrealKeyMap.at(key);
			binding["petrificus totalus"][0] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "AM_Oppugno") {
			binding["oppugno"][0] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "AM_Protego") {
			binding["protego"][0] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "AM_Revelio") {
			binding["revelio"][0] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else if (actionName == "AM_CriticalFinisher") {
			binding["smash"][0] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		else {
			continue;
		}

		/*
		else if (actionName == "AM_Stupefy") {
			binding["stupefy"][0] = unrealKeyMap.at(key);
			actionChecked.push_back(actionName);
		}
		*/
	}

	// Check if all the bindings were found
	bool error_found = false;
	for (auto it = binding.begin(); it != binding.end(); ++it) {
		for (auto it_vec = it->second.begin(); it_vec != it->second.end(); ++it_vec) {
			if (*it_vec == 0)
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
	}

	inputFile.close();

	return binding;
}

void Keybinder::loadConfBindings(const std::string& conf_path, const std::map<std::string, std::vector<WORD>>& game_bindings) {
	std::vector<WORD> key_columns = game_bindings.at("columns"); // { 0x31, 0x32, 0x33, 0x34 };
	std::vector<WORD> key_lines = game_bindings.at("lines");     //{ VK_F1, VK_F2, VK_F3, VK_F4 };

	std::ifstream file(conf_path);

	if (file.is_open()) {
		std::string line;
		std::getline(file, line);

		int line_count = 0;
		while (std::getline(file, line)) {
			if (line_count > 3)
				break;

			std::stringstream ss(line);
			std::string formula_name;

			int column_count = 0;
			while (std::getline(ss, formula_name, ';')) {
				if (column_count > 3)
					break;

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
			/*if (_lumos_status) { // If lumos cast over lumos => cancel
				_is_working = true;
				return _is_working;
			}*/
			_lumos_status = true;
		}
		
		auto it = _principal_bindings.find(word);
		if (it != _principal_bindings.end()) {
			for (WORD key : it->second)
				pressKey(key, 0);
		}

		else if (word == "accio broomstick" || word == "accio balais") {
			_hold_thread.emplace_back(std::async(std::launch::async, &Keybinder::combinationKey, this, _secondary_bindings["accio broomstick"], 500));
			//_hold_thread.emplace_back(std::async(std::launch::async, &Keybinder::combinationKey, this, std::vector<WORD>({ VK_TAB , 0x33 }), 500));
		}
		else if (word == "smash") {
			pressKey(_secondary_bindings["smash"][0], 0);
			//pressKey('X', 0);
		}
		else if (word == "revelio") {
			pressKey(_secondary_bindings["revelio"][0], 0);
			//pressKey('R', 0);
		}
		else if (word == "protego") {
			//pressKey('A', 0);
			_hold_thread.emplace_back(std::async(std::launch::async, &Keybinder::pressKey, this, _secondary_bindings["protego"][0], 1000));
			//_hold_thread.emplace_back(std::async(std::launch::async, &Keybinder::pressKey, this, 'A', 1000));
			
			//_hold_thread.emplace_back(std::async(std::launch::async, &Keybinder::holdRightClick, this, 3000));
		}
		else if (word == "nox") {
			auto it = _principal_bindings.find("lumos");
			if (it != _principal_bindings.end()) {
				for (WORD key : it->second)
					pressKey(key, 0);
			}

			_lumos_status = false;
		}
		else if (word == "appare vestigium") {
			_hold_thread.emplace_back(std::async(std::launch::async, &Keybinder::pressKey, this, _secondary_bindings["appare vestigium"][0], 50));
			//_hold_thread.emplace_back(std::async(std::launch::async, &Keybinder::pressKey, this, 'V', 50));
			//apressKey('V', 0);
		}
		else if (word == "petrificus totalus") {
			pressKey(_secondary_bindings["petrificus totalus"][0], 0);
			//pressKey('F', 0);
		}
		else if (word == "oppugno") {
			pressKey(_secondary_bindings["oppugno"][0], 0);
			//pressKey('W', 0);
		}
		else if (word == "alohomora") {
			pressKey(_secondary_bindings["alohomora"][0], 0);
		}
		else
			_is_working = true;

		// --- Reset of lumos status if another spell is cast
		//std::cout << _lumos_status << " " << _is_working << 
		if (_lumos_status && !_is_working && !lumos_cast)
			_lumos_status = false;

		//std::cout << "Lumos status : " << _lumos_status << std::endl;
	}

	return _is_working;
}




void Keybinder::pressKey(WORD key_code, int duration_ms) {
	INPUT input = { 0 };
	input.type = INPUT_KEYBOARD;
	input.ki.wVk = key_code;
	input.ki.dwFlags = 0;
	SendInput(1, &input, sizeof(INPUT));
	//std::cout << "Input 1 : " << key_code << std::endl;
	//input.ki.time += duration_ms; // Set the time stamp for the key release
	if (duration_ms != 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
	input.ki.dwFlags = KEYEVENTF_KEYUP;
	SendInput(1, &input, sizeof(INPUT));
	//std::cout << "Input 2 : " << key_code << std::endl;
}

void Keybinder::holdRightClick(int duration_ms) {
	INPUT input = { 0 };
	input.type = INPUT_MOUSE;
	input.ki.dwFlags = MOUSEEVENTF_RIGHTDOWN;
	SendInput(1, &input, sizeof(INPUT));
	//input.ki.time += duration_ms; // Set the time stamp for the key release
	std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
	input.ki.dwFlags = MOUSEEVENTF_RIGHTUP;
	SendInput(1, &input, sizeof(INPUT));
}

void Keybinder::combinationKey(std::vector<WORD> key_codes, int duration_ms) {
	INPUT input = { 0 };
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