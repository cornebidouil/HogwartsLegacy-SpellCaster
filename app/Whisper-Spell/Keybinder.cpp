#include "Keybinder.h"
#include <iostream>

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
		return "Comma";
	else if (key == ",")
		return "M";
	// add other keys as needed...

	// If no AZERTY equivalent is found, return the original key
	return key;
}

Keybinder::Keybinder(std::string conf_path, Config& cfg) :
_is_working(false),
_lumos_status(false),
_conf_path(conf_path),
_device_controller(-1),
_conf_hash(0),
_game_binding_hash(0),
_game_binding_path(cfg.models.gameBindingPath)
{
	_hidHide = std::make_unique<HidHideControl>();
    
    // Setup HidHide protection
    wchar_t exePath[MAX_PATH];
    GetModuleFileName(NULL, exePath, MAX_PATH);
    _hidHide->addToWhitelist(exePath);
    
    // Initialize DirectInput
    DirectInput8Create(GetModuleHandle(nullptr), DIRECTINPUT_VERSION, 
                       IID_IDirectInput8, (VOID**)&_directInput, nullptr);

	_principal_bindings.clear();
	_secondary_bindings.clear();
	_game_bindings.clear();

	_recurrent_spells_mapping = buildReccurentSpellMap();

	selectingGamepad(cfg);

	_game_bindings = loadGameBindings();
	loadConfBindings(conf_path, _game_bindings);

}

Keybinder::~Keybinder() {
	stopRedirectingController();
    _inputRedirector.reset();
    _vrInputHandler.reset();
    _hidHide.reset();
    if (_directInput) {
        _directInput->Release();
    }
}


void Keybinder::selectingGamepad(Config& cfg) {

	std::cout << "Do you play with a :" << std::endl << "\t" << DEVICE_KEYBOARD << " -> Keyboard" << std::endl << "\t" << DEVICE_GAMEPAD << " -> Gamepad" << std::endl << "\t" << DEVICE_VR << " -> VR" << std::endl;
	std::cout << std::endl << "Enter the playing device : ";
	std::cin  >> _device_controller;

	if (_device_controller != DEVICE_KEYBOARD and _device_controller != DEVICE_GAMEPAD and _device_controller != DEVICE_VR) {
		std::cerr << "Invalid device controller." << std::endl;
#ifdef _WIN32
		system("PAUSE");
#endif // _WIN32
		exit(-1);
	}

	std::cout << std::endl;

	if (_device_controller == DEVICE_VR) {

		std::string UEVR_path = DllUtils::getRoamingAppDataPath() + "/UnrealVRMod/HogwartsLegacy/plugins/";
		if (!DllUtils::folderExists(UEVR_path)) {
			std::cout << "\033[1;33m" << "Hogwarts Legacy's UEVR folder not found : " << UEVR_path << std::endl << "Please manually copy UEVRSpellCasterPlugin.dll (from UEVR Plugin/ subfolder of Spellcaster folder) the inside the .../UnrealVRMod/HogwartsLegacy/plugins/ folder before launching the game." << "\033[0m" << std::endl << std::endl;
			std::cout << "Press Enter to continue...";
			std::cin.ignore((std::numeric_limits<std::streamsize>::max)(), '\n');
			std::cin.get();
		}
		else {
			std::filesystem::path executablePath = std::filesystem::current_path();
			std::filesystem::path pluginPath = executablePath / "UEVR Plugin" / "UEVRSpellCasterPlugin.dll";
			std::string original_plugin_path = pluginPath.string();

			std::string UEVR_plugin_path = UEVR_path + "UEVRSpellCasterPlugin.dll";
			
			std::string current_plugin_version = DllUtils::getFileVersion(original_plugin_path);
			if (!DllUtils::fileExistsInDirectory(UEVR_path, "UEVRSpellCasterPlugin.dll") || current_plugin_version > DllUtils::getFileVersion(UEVR_plugin_path)) {
				if (!DllUtils::copyFile(original_plugin_path, UEVR_plugin_path)) {
					std::cout << "\033[1;33m" << "Unable to copy the UEVRSpellCasterPlugin.dll to the UEVR folder." << std::endl << "Please manually copy UEVRSpellCasterPlugin.dll (from UEVR Plugin/ subfolder of Spellcaster folder) the inside the .../UnrealVRMod/HogwartsLegacy/plugins/ folder before launching the game." << "\033[0m" << std::endl << std::endl;
					std::cout << "Press Enter to continue...";
					std::cin.ignore((std::numeric_limits<std::streamsize>::max)(), '\n');
					std::cin.get();
				}
			}
		}


		// Create and initialize VR input handler
		_vrInputHandler = std::make_unique<VRInputHandler>(cfg);
		if (!_vrInputHandler->initialize()) {
			std::cerr << "Failed to initialize VR input handler. Check VR configuration." << std::endl;
#ifdef _WIN32
			system("PAUSE");
#endif
			exit(-1);
		}
		
		std::cout << "VR Input Handler initialized successfully!" << std::endl;
	}
	else if (_device_controller == DEVICE_GAMEPAD) {

		auto controllers = enumerateControllers(_directInput);
		if (controllers.empty()) {
			std::cout << "\nNo standard controllers detected." << std::endl;

			std::cerr << "No gamepads found." << std::endl;
#ifdef _WIN32
			system("PAUSE");
#endif
			exit(-1);
		}

		int selection;
		std::cout << "Select controller (0-" << controllers.size() - 1 << "): ";
		std::cin >> selection;

		if (selection >= 0 && selection < controllers.size()) {
			_selectedController = controllers[selection]; // Or implement selection logic

			if (InputRedirector::detectInputType(_selectedController) == InputType::DirectInput) {
				ShowInputChecker(_directInput, _selectedController, cfg);
				cfg.load();
			}

			bool isAdmin = checkAdministratorPrivileges();
			std::cout << "Administrator privileges: " << (isAdmin ? "AVAILABLE" : "MISSING") << std::endl;

			std::cout << "\nStep 1: Working device path enumeration..." << std::endl;
			auto allPaths = getAllDeviceInstancePaths(_selectedController);

			std::cout << "Found " << allPaths.size() << " device access paths:" << std::endl;
			for (const auto& path : allPaths) {
				std::wcout << L"  Path: " << path << std::endl;
			}

			// Step 2: HidHide blacklisting
			std::cout << "\nStep 2: Adding paths to HidHide blacklist..." << std::endl;
			restoreController();
			bool hidHideSuccess = true;
			for (const auto& path : allPaths) {
				if (_hidHide->addToBlacklist(path)) {
					std::wcout << L"  Blacklisted: " << path << std::endl;
					_hidHide_blocked_paths.push_back(path);
				}
				else {
					std::wcout << L"  Failed: " << path << std::endl;
					hidHideSuccess = false;
				}
			}
			
			_inputRedirector = std::make_unique<InputRedirector>(_directInput, _selectedController, cfg);
			

		} else {
			std::cerr << "Invalid controller selection." << std::endl;
	#ifdef _WIN32
			system("PAUSE");
	#endif
			exit(-1);
		}
	}
}

void Keybinder::startRedirectingController() {
    if (_inputRedirector) {
        _inputRedirector->start();
		Sleep(1000);
        _hidHide->setStatus(true);
    }
}

void Keybinder::stopRedirectingController() {
    if (_inputRedirector) {
        _hidHide->setStatus(false);
        _inputRedirector->stop();
    }
    
    // Ensure VR input handler is properly shut down to prevent thread hang
    if (_vrInputHandler) {
        _vrInputHandler->shutdown();
    }
}

void Keybinder::restoreController() {
	for (const auto& path : _hidHide_blocked_paths) {
		if (_hidHide->removeFromBlacklist(path)) {
			std::wcout << L"  Removed: " << path << std::endl;
		}
	}

	_hidHide_blocked_paths.clear();
}


void Keybinder::localizeGameBindings() {
	char* userProfile;
	size_t profile_size;

	// Get the value of the "USERPROFILE" environment variable
	if (_dupenv_s(&userProfile, &profile_size, "USERPROFILE") != 0 || userProfile == nullptr) {
		std::cerr << "Failed to get USERPROFILE environment variable" << std::endl;
#ifdef _WIN32
		system("PAUSE");
#endif
		exit(-1);
	}

	_game_binding_path = std::string(userProfile) + "\\AppData\\Local\\Hogwarts Legacy\\Saved\\Config\\WindowsNoEditor\\Input.ini";
	
	if (userProfile != nullptr) {
		free(userProfile);
	}

	std::cout << "Game bindings path : " << _game_binding_path << std::endl << std::endl;
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
		{"alohomora", {UNDEFINED_BINDING}},
		{"protego contra", {UNDEFINED_BINDING, UNDEFINED_BINDING}},
		{"episkey", {UNDEFINED_BINDING}},
		{"apperta liber", {UNDEFINED_BINDING}},
		{"finite liber", {UNDEFINED_BINDING}},
		{"apperta mappa", {UNDEFINED_BINDING}},
		{"apperta sacculus", {UNDEFINED_BINDING}},
		{"apperta vestarium", {UNDEFINED_BINDING}},
		{"apperta meritas", {UNDEFINED_BINDING}},
		{"apperta codex", {UNDEFINED_BINDING}},
		{"apperta compendium", {UNDEFINED_BINDING}},
		{"apperta literae", {UNDEFINED_BINDING}},
		{"apperta facultates", {UNDEFINED_BINDING}},
		{"apperta configuratio", {UNDEFINED_BINDING}},
		{"apperta incantatem", {UNDEFINED_BINDING}},
	};

	std::cout << "Game binding path : " << _game_binding_path << std::endl << std::endl;
	
	std::ifstream inputFile(_game_binding_path);

	if (!inputFile) {
		std::cout << "\033[1;41m" << "*** Game config not found (Input.ini) : default key binding used ***" << "\033[0m" << std::endl << std::endl;
#ifdef _WIN32
		system("PAUSE");
#elif defined(__linux__)
		std::cout << "Press ENTER to continue ...";
		getchar();
#endif // _WIN32
		if (_device_controller == DEVICE_KEYBOARD)
			return defaultBinding;
		else 
			return defaultXUSBBinding;

	}

	WORD currentLayout = PRIMARYLANGID(HIWORD(GetKeyboardLayout(0)));
	if (_device_controller == DEVICE_KEYBOARD && currentLayout == LANG_FRENCH)
		std::cout << "*** AZERTY keyboard detected ***" << std::endl << std::endl;


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

		
		if (key != "None" && key.find("Gamepad") == std::string::npos && std::find(special_menu_list.begin(), special_menu_list.end(), actionName) != special_menu_list.end() ) {
			if (currentLayout == LANG_FRENCH)
				key = getAzertyEquivalent(key);
			if (updateBinding(binding, actionName, key))
				actionChecked.push_back(actionName);
			continue;
		}

		if (key == "None" || 
			(_device_controller == DEVICE_KEYBOARD && key.find("Gamepad") != std::string::npos) ||
			(_device_controller != DEVICE_KEYBOARD && key.find("Gamepad") == std::string::npos) ||
			std::find(special_menu_list.begin(), special_menu_list.end(), actionName) != special_menu_list.end() ||
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
		std::cout << "\033[1;41m" << "--- Please rebind one key ingame to re-save the game configuration, it will automatically be updated (it's a bug from the game). ---" << "\033[0m" << std::endl;
#ifdef _WIN32
		system("PAUSE");
#endif
		std::cout << std::endl;
	}

	inputFile.close();

	_game_binding_hash = hashFile(_game_binding_path);

	std::cout << "Game bindings loaded." << std::endl;

	return binding;
}


std::unordered_map<std::string, std::vector<std::string>> Keybinder::buildReccurentSpellMap() {
    std::unordered_map<std::string, std::vector<std::string>> bindingMap;
    for (const auto& group : spell_equivalences) {
        for (const auto& spell : group) {
            // Create a new vector excluding the current spell
            std::vector<std::string> otherSpells;
            std::copy_if(group.begin(), group.end(), 
                        std::back_inserter(otherSpells),
                        [&spell](const std::string& s) { return s != spell; });
            bindingMap[spell] = otherSpells;
        }
    }
    return bindingMap;
}

WORD Keybinder::keyToBind(const std::string& key, const std::string& actionName) {

	try {
        if (_device_controller == DEVICE_KEYBOARD || std::find(special_menu_list.begin(), special_menu_list.end(), actionName) != special_menu_list.end()) {
            if (unrealKeyMap.find(key) != unrealKeyMap.end()) {
                return unrealKeyMap.at(key);
            } else {
                std::cerr << "\033[1;31mERROR: Unhandled keyboard key '" << key << "' in keyToBind function\033[0m" << std::endl;
#ifdef _WIN32
                system("PAUSE");
#endif
                exit(-1);
            }
        } else { // _device_controller == DEVICE_GAMEPAD || _device_controller == DEVICE_VR
            if (xusb_gamepadButtonMap.find(key) != xusb_gamepadButtonMap.end()) {
                return xusb_gamepadButtonMap.at(key);
            } else {
                std::cerr << "\033[1;31mERROR: Unhandled gamepad button '" << key << "' in keyToBind function\033[0m" << std::endl;
#ifdef _WIN32
                system("PAUSE");
#endif
                exit(-1);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "\033[1;31mEXCEPTION in keyToBind function: " << e.what() << "\033[0m" << std::endl;
        std::cerr << "Key that caused the exception: '" << key << "'" << std::endl;
#ifdef _WIN32
        system("PAUSE");
#endif
        exit(-1);
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
        {"AM_CriticalFinisher", {"smash", 0}},
		{"UMGMapScreenToggle", {"apperta mappa", 0}},
		{"UMGPauseMenu", {"apperta liber", 0}},
		{"AM_Stupefy", {"protego contra", 1}},
		{"AM_Health", {"episkey", 0}},
		{"UMGOwlMailScreenToggle", {"apperta literae", 0}},
		{"UMGInventoryScreenToggle", {"apperta sacculus", 0}},
		{"UMGCharacterScreenToggle", {"apperta vestarium", 0}},
		{"UMGChallengeScreenToggle", {"apperta meritas", 0}},
		{"UMGQuestScreenToggle", {"apperta codex", 0}},
		{"UMGCompendiumScreenToggle", {"apperta compendium", 0}},
		{"UMGTalentsScreenToggle", {"apperta facultates", 0}},
		{"UMGSettingsScreenToggle", {"apperta configuratio", 0}},
		{"UMGActionScreenToggle", {"apperta incantatem", 0}}
    };

    auto it = actionMap.find(actionName);
    if (it != actionMap.end()) {
        binding[it->second.first][it->second.second] = keyToBind(key, actionName);
        if (actionName == "AM_Interact") {
            binding["petrificus totalus"][0] = keyToBind(key, actionName);
        }
		if (actionName == "AM_Protego") {
			binding["protego contra"][0] = keyToBind(key, actionName);
		}
		if (actionName == "UMGPauseMenu") {
			binding["finite liber"][0] = keyToBind(key, actionName);
		}
        return true;
    }
    return false;
}

void Keybinder::loadConfBindings(const std::string& conf_path, const std::unordered_map<std::string, std::vector<WORD>>& game_bindings) {
	std::vector<WORD> key_columns = game_bindings.at("columns");
	std::vector<WORD> key_lines = game_bindings.at("lines");
	_user_binding_txt.clear();
	_principal_bindings.clear(); // Clear the vector

	std::ifstream file(conf_path);
	if (file.is_open()) {
		std::string line;
		int line_count = 0;
		while (std::getline(file, line)) {
			_user_binding_txt.push_back(line);
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

				size_t underscore_pos = formula_name.find('_');
				std::string readable_formula_name = (underscore_pos != std::string::npos)
					? formula_name.substr(0, underscore_pos)
					: formula_name;

				// Add to vector (preserves insertion order)
				_principal_bindings.emplace_back(readable_formula_name, std::vector<WORD>({ key_lines[line_count], key_columns[column_count] }));

				column_count++;
			}
			line_count++;
		}
		file.close();
	}
	else {
		std::cerr << "Failed to open the conf file : " << conf_path << std::endl;
#ifdef _WIN32
		system("PAUSE");
#endif
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

	if (!_is_working) {
		_is_working = true;

		if (hashFile(_game_binding_path) != _game_binding_hash) {
			std::cout << std::endl << "Game Bindings modified" << std::endl;

			_principal_bindings.clear();
			_secondary_bindings.clear();
			_game_bindings.clear();

			_game_bindings = loadGameBindings();
			loadConfBindings(_conf_path, _game_bindings);
			std::cout << std::endl;
		}
		if (hashFile(_conf_path) != _conf_hash) {
			std::cout << std::endl << "Spell Bindings modified" << std::endl;

			_principal_bindings.clear();
			_secondary_bindings.clear();

			loadConfBindings(_conf_path, _game_bindings);
			std::cout << std::endl;
		}
	}
}

void Keybinder::stop() {
	_is_working = false;
}


auto Keybinder::findInPrincipalBindings(const std::string& word) {
	for (auto it = _principal_bindings.begin(); it != _principal_bindings.end(); ++it) {
		if (it->first == word) {
			return it; // Returns the FIRST occurrence found
		}
	}
	return _principal_bindings.end();
}

bool Keybinder::decode(const std::string& word, const bool& final_record) {
	bool lumos_cast = false;

	if (trimTrailingSpaces(GetActiveWindowTitle()) != "Hogwarts Legacy") {
		std::cout << "Not ingame => input skipped (Go ingame to cast spells!)" << std::endl;
		return false;
	}


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
	
	auto it = findInPrincipalBindings(word);
	auto it_equivalence = _recurrent_spells_mapping.find(word);
	if (it != _principal_bindings.end()) {
		std::cout << word << " : " << it->second[0] << " | " << it->second[1] << std::endl;
		if (_device_controller == DEVICE_KEYBOARD) {
			for (const WORD& key : it->second)
				pressKey(key, 0);
		} else if (_device_controller == DEVICE_GAMEPAD) {
			if (_inputRedirector){
				_inputRedirector->queuePrincipalSpell(it->second);
			}
		} else { // _device_controller == DEVICE_VR
			if (_vrInputHandler && _vrInputHandler->isReady()) {
				_vrInputHandler->queuePrincipalSpell(it->second);
			}
		}
	}

	else if (it_equivalence != _recurrent_spells_mapping.end()) {
		for (const auto& equivalent : it_equivalence->second) {
			auto it_search = findInPrincipalBindings(equivalent);
			if (it_search != _principal_bindings.end()) {
				if (_device_controller == DEVICE_KEYBOARD) {
					for (const WORD& key : it_search->second)
						pressKey(key, 0);
				} else if (_device_controller == DEVICE_GAMEPAD) {
					if (_inputRedirector){
						_inputRedirector->queuePrincipalSpell(it_search->second);
					}
				} else { // _device_controller == DEVICE_VR
					if (_vrInputHandler && _vrInputHandler->isReady()) {
						_vrInputHandler->queuePrincipalSpell(it_search->second);
					}
				}
			}
		}
	}

	else if (word == "accio broomstick" || word == "accio balais" || word == "accio eclair de feu" || word == "accio firebolt") {
		if (_device_controller == DEVICE_KEYBOARD)
			combinationKey(_secondary_bindings["accio broomstick"], 500);
		else if (_device_controller == DEVICE_GAMEPAD) {
			if (_inputRedirector){
				_inputRedirector->queueButtonCombination(_secondary_bindings["accio broomstick"], 500);
			}
		} else { // _device_controller == DEVICE_VR
			if (_vrInputHandler && _vrInputHandler->isReady()) {
				_vrInputHandler->queueButtonCombination(_secondary_bindings["accio broomstick"], 500);
			}
		}
	}
	else if (word == "nox") {
		auto it = findInPrincipalBindings("lumos");
		if (it != _principal_bindings.end()) {
			if (_device_controller == DEVICE_KEYBOARD) {
				for (WORD key : it->second)
					pressKey(key);
			} else if (_device_controller == DEVICE_GAMEPAD) {
				if (_inputRedirector){
					_inputRedirector->queuePrincipalSpell(it->second);
				}
			} else { // _device_controller == DEVICE_VR
				if (_vrInputHandler && _vrInputHandler->isReady()) {
					_vrInputHandler->queuePrincipalSpell(it->second);
				}
			}
		}

		_lumos_status = false;
	}
	else if (word == "appare vestigium") {
		if (_device_controller == DEVICE_KEYBOARD)
			pressKey(_secondary_bindings["appare vestigium"][0], 50);
		else if (_device_controller == DEVICE_GAMEPAD) {
			if (_inputRedirector){
				_inputRedirector->queueSingleButton(_secondary_bindings["appare vestigium"][0], 50);
			}
		} else { // _device_controller == DEVICE_VR
			if (_vrInputHandler && _vrInputHandler->isReady()) {
				_vrInputHandler->queueSingleButton(_secondary_bindings["appare vestigium"][0], 50);
			}
		}
	}
	else if (word == "smash") {
		if (_device_controller == DEVICE_KEYBOARD)
			pressKey(_secondary_bindings[word][0]);
		else if (_device_controller == DEVICE_GAMEPAD) {
			if (_inputRedirector){
				_inputRedirector->queueSimultaneousButtons({ XUSB_GAMEPAD_LEFT_SHOULDER, XUSB_GAMEPAD_RIGHT_SHOULDER });
			}
		} else { // _device_controller == DEVICE_VR
			if (_vrInputHandler && _vrInputHandler->isReady()) {
				_vrInputHandler->queueSimultaneousButtons({ XUSB_GAMEPAD_LEFT_SHOULDER, XUSB_GAMEPAD_RIGHT_SHOULDER });
			}
		}
	}
	else if (word == "protego contra") {
		if (_device_controller == DEVICE_KEYBOARD)
			pressKey(_secondary_bindings["protego"][0], 3000);
		else if (_device_controller == DEVICE_GAMEPAD) {
			if (_inputRedirector) {
				_inputRedirector->queueSingleButton(_secondary_bindings["protego"][0], 3000);
			}
		}
		else { // _device_controller == DEVICE_VR
			if (_vrInputHandler && _vrInputHandler->isReady()) {
				_vrInputHandler->queueSingleButton(_secondary_bindings["protego"][0], 3000);
			}
		}
	}
	else if (word == "apperta mappa" ||
			 word == "je jure solennellement que mes intentions sont mauvaises" ||
			 word == "i solemnly swear that i am up to no good") {
		if (_device_controller == DEVICE_KEYBOARD)
			pressKey(_secondary_bindings["apperta mappa"][0]);
		else if (_device_controller == DEVICE_GAMEPAD) {
			if (_inputRedirector) {
				_inputRedirector->queueSingleButton(_secondary_bindings["apperta mappa"][0]);
			}
		}
		else { // _device_controller == DEVICE_VR
			if (_vrInputHandler && _vrInputHandler->isReady()) {
				_vrInputHandler->queueSingleButton(_secondary_bindings["apperta mappa"][0]);
			}
		}
	}
	else if (word == "mischief managed" ||
			 word == "mefait accompli") {
		if (_device_controller == DEVICE_KEYBOARD)
			pressKey(_secondary_bindings["finite liber"][0]);
		else if (_device_controller == DEVICE_GAMEPAD) {
			if (_inputRedirector) {
				_inputRedirector->queueSingleButton(_secondary_bindings["apperta liber"][0]);
			}
		}
		else { // _device_controller == DEVICE_VR
			if (_vrInputHandler && _vrInputHandler->isReady()) {
				_vrInputHandler->queueSingleButton(_secondary_bindings["apperta liber"][0]);
			}
		}
	}
	else if (std::find(special_spell_list.begin(), special_spell_list.end(), word) != special_spell_list.end()) {
		pressKey(_secondary_bindings[word][0], 50);
	}
	else if (word == "petrificus totalus" || 
             word == "revelio" ||
             word == "oppugno" || 
             word == "alohomora" ||
		     word == "protego" ||
			 word == "episkey" ||
			 word == "apperta liber" ||
			 word == "finite liber") {
		if (_device_controller == DEVICE_KEYBOARD)
			pressKey(_secondary_bindings[word][0]);
		else if (_device_controller == DEVICE_GAMEPAD) {
			if (_inputRedirector){
				_inputRedirector->queueSingleButton(_secondary_bindings[word][0]);
			}
		} else { // _device_controller == DEVICE_VR
			if (_vrInputHandler && _vrInputHandler->isReady()) {
				_vrInputHandler->queueSingleButton(_secondary_bindings[word][0]);
			}
		}

	}
	else 
		std::cout << "*** Binding not found ... ***" << std::endl;
	
	// --- Reset of lumos status if another spell is cast
	if (_lumos_status && !_is_working && !lumos_cast)
		_lumos_status = false;

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

void Keybinder::printUserBindings() {
	for (const auto& line: _user_binding_txt) {
		std::cout << "\t\t" << line << std::endl;
	}
}
