#include "VRInputHandler_SharedMemory.h"
#include "VRSharedMemoryServer.h"
#include "SharedMemoryStructure.h"
#include <ViGEm/Client.h>
#include <Xinput.h>

// Direct mapping from XUSB to XInput buttons (values are identical)
// Note: GUIDE button (XUSB_GAMEPAD_GUIDE) is not supported in standard XInput API
const std::unordered_map<WORD, WORD> VRInputHandler::XUSB_TO_XINPUT = {
    {XUSB_GAMEPAD_A, XINPUT_GAMEPAD_A},
    {XUSB_GAMEPAD_B, XINPUT_GAMEPAD_B},
    {XUSB_GAMEPAD_X, XINPUT_GAMEPAD_X},
    {XUSB_GAMEPAD_Y, XINPUT_GAMEPAD_Y},
    {XUSB_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_SHOULDER},
    {XUSB_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_RIGHT_SHOULDER},
    {XUSB_GAMEPAD_BACK, XINPUT_GAMEPAD_BACK},
    {XUSB_GAMEPAD_START, XINPUT_GAMEPAD_START},
    {XUSB_GAMEPAD_LEFT_THUMB, XINPUT_GAMEPAD_LEFT_THUMB},
    {XUSB_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_RIGHT_THUMB},
    {XUSB_GAMEPAD_DPAD_UP, XINPUT_GAMEPAD_DPAD_UP},
    {XUSB_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_DOWN},
    {XUSB_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_LEFT},
    {XUSB_GAMEPAD_DPAD_RIGHT, XINPUT_GAMEPAD_DPAD_RIGHT}
    // XUSB_GAMEPAD_GUIDE is not included - not supported in standard XInput API
};

VRInputHandler::VRInputHandler(const Config& cfg)
    : cfg_(cfg)
    , sharedMemoryServer_(nullptr) {
}

VRInputHandler::~VRInputHandler() {
    shutdown();
}

VRInputHandler::VRInputHandler(VRInputHandler&& other) noexcept
    : cfg_(other.cfg_)
    , isConnected_(other.isConnected_.load())
    , isInitialized_(other.isInitialized_.load())
    , sharedMemoryServer_(std::move(other.sharedMemoryServer_)) {
    
    other.isConnected_.store(false);
    other.isInitialized_.store(false);
}

VRInputHandler& VRInputHandler::operator=(VRInputHandler&& other) noexcept {
    if (this != &other) {
        shutdown();
        
        isConnected_.store(other.isConnected_.load());
        isInitialized_.store(other.isInitialized_.load());
        sharedMemoryServer_ = std::move(other.sharedMemoryServer_);
        
        other.isConnected_.store(false);
        other.isInitialized_.store(false);
    }
    return *this;
}

bool VRInputHandler::initialize() {
    if (isInitialized_.load()) {
        return true;
    }

    try {
        // Create shared memory server (no admin privileges required!)
        sharedMemoryServer_ = std::make_unique<whisper_spell::VRSharedMemoryServer>();
        
        // Mapping name from the VR pipe_name setting ("UEVRSpellCaster" by default); the UEVR plugin opens the same name
        std::string memoryName = cfg_.vr.pipeName.empty() ? "WhisperSpellVR" : cfg_.vr.pipeName;
        
        bool started = sharedMemoryServer_->start(
            memoryName,
            // Message handler - process incoming messages from UEVR plugin
            [this](const std::string& message) {
                handleIncomingMessage(message);
            },
            // Connection handler - track UEVR plugin connection state
            [this](bool connected) {
                handleConnectionChange(connected);
            }
        );

        if (!started) {
            std::cerr << "[VR-SM] Failed to start shared memory server with name: " << memoryName << std::endl;
            return false;
        }

        isInitialized_.store(true);
        std::cout << "[VR-SM] VR Input Handler initialized successfully (Shared Memory)" << std::endl;
        std::cout << "[VR-SM] Memory name: " << memoryName << std::endl;
        std::cout << "[VR-SM] Waiting for UEVR plugin connection..." << std::endl;
        std::cout << "[VR-SM] *** NO ADMIN PRIVILEGES REQUIRED ***" << std::endl;
        
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[VR-SM] Exception during initialization: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "[VR-SM] Unknown exception during initialization" << std::endl;
        return false;
    }
}

void VRInputHandler::shutdown() {
    if (!isInitialized_.load()) {
        return;
    }

    isConnected_.store(false);
    
    if (sharedMemoryServer_) {
        std::cout << "[VR-SM] Stopping VR input handler..." << std::endl;
        sharedMemoryServer_->stop();
        sharedMemoryServer_.reset();
    }
    
    isInitialized_.store(false);
}

uint16_t VRInputHandler::convertButton(WORD button) const {
    auto it = XUSB_TO_XINPUT.find(button);
    if (it != XUSB_TO_XINPUT.end()) {
        return it->second;  // Direct XInput button value
    } else {
        std::cerr << "[VR-SM] Warning: Unknown XUSB button code " << std::hex << button << std::endl;
        return 0;
    }
}

void VRInputHandler::fillButtonSequence(SharedVRCommand& cmd, const std::vector<WORD>& buttons) const {
    // Clear the sequence first
    for (int i = 0; i < 4; i++) {
        cmd.buttonSequence[i] = 0;
    }
    
    // Fill with buttons in order (max 4 buttons supported)
    size_t count = (std::min)(buttons.size(), size_t(4));
    for (size_t i = 0; i < count; i++) {
        cmd.buttonSequence[i] = convertButton(buttons[i]);
    }
    
    cmd.buttonCount = static_cast<uint8_t>(count);
}

bool VRInputHandler::sendCommand(const SharedVRCommand& cmd) {
    if (!isConnected_.load() || !sharedMemoryServer_) {
        // Fire-and-forget: silently drop when UEVR disconnected
        return false;
    }

    try {
        // Send binary command directly for maximum performance
        // Convert our VRCommand to SharedMemory VRCommand format
        whisper_spell::SharedVRMemory::VRCommand sharedCmd{};
        sharedCmd.type = cmd.type;
        sharedCmd.buttonCount = cmd.buttonCount;
        sharedCmd.duration = cmd.duration;
        for (int i = 0; i < 4; i++) {
            sharedCmd.buttonSequence[i] = cmd.buttonSequence[i];
        }
        sharedCmd.reserved = cmd.reserved;
        
        std::string message(reinterpret_cast<const char*>(&sharedCmd), sizeof(sharedCmd));
        return sharedMemoryServer_->sendMessage(message);
    } catch (const std::exception& e) {
        std::cerr << "[VR-SM] Exception sending command: " << e.what() << std::endl;
        return false;
    }
}

void VRInputHandler::queueSingleButton(WORD button, int duration_ms) {
    SharedVRCommand cmd{
        1, // type: single button
        0, // buttonCount (filled by fillButtonSequence)
        static_cast<uint16_t>(duration_ms),
        {0, 0, 0, 0}, // buttonSequence (filled by fillButtonSequence)
        0  // reserved
    };
    
    fillButtonSequence(cmd, {button});
    sendCommand(cmd);
}

void VRInputHandler::queueButtonCombination(const std::vector<WORD>& buttons, int duration_ms) {
    SharedVRCommand cmd{
        2, // type: button combination
        0, // buttonCount (filled by fillButtonSequence)
        static_cast<uint16_t>(duration_ms),
        {0, 0, 0, 0}, // buttonSequence (filled by fillButtonSequence)
        0  // reserved
    };
    
    fillButtonSequence(cmd, buttons);
    sendCommand(cmd);
}

void VRInputHandler::queuePrincipalSpell(const std::vector<WORD>& buttons, int duration_ms) {
    SharedVRCommand cmd{
        3, // type: principal spell (with right trigger)
        0, // buttonCount (filled by fillButtonSequence)
        static_cast<uint16_t>(duration_ms),
        {0, 0, 0, 0}, // buttonSequence (filled by fillButtonSequence)
        0  // reserved
    };
    
    fillButtonSequence(cmd, buttons);
    sendCommand(cmd);
}

void VRInputHandler::queueSimultaneousButtons(const std::vector<WORD>& buttons, int duration_ms) {
    SharedVRCommand cmd{
        4, // type: simultaneous buttons
        0, // buttonCount (filled by fillButtonSequence)
        static_cast<uint16_t>(duration_ms),
        {0, 0, 0, 0}, // buttonSequence (filled by fillButtonSequence)
        0  // reserved
    };
    
    fillButtonSequence(cmd, buttons);
    sendCommand(cmd);
}

void VRInputHandler::handleIncomingMessage(const std::string& message) {
    // Currently, we only send commands to UEVR plugin
    // Future: Could handle acknowledgments or status updates
    if (cfg_.vr.debugLogging) {
        std::cout << "[VR-SM] Received message from UEVR: " << message << std::endl;
    }
}

void VRInputHandler::handleConnectionChange(bool connected) {
    isConnected_.store(connected);
    
    if (connected) {
        std::cout << "[VR-SM] SUCCESS: UEVR Plugin connected - spells will be sent via VR controllers" << std::endl;
    } else {
        std::cout << "[VR-SM] WARNING: UEVR Plugin disconnected - spells will be dropped (fire-and-forget)" << std::endl;
    }
}

whisper_spell::PipeStatistics VRInputHandler::getStatistics() const {
    if (sharedMemoryServer_) {
        return sharedMemoryServer_->getStatistics();
    }
    return whisper_spell::PipeStatistics{};
}
