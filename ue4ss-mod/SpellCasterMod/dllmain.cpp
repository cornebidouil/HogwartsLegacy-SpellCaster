/**
 * @file    dllmain.cpp
 * @brief   SpellCasterMod - UE4SS C++ mod for Hogwarts Legacy
 * @author  Cornebidouil
 * @version 1.1
 *
 * Casts spells on behalf of the SpellCaster desktop application:
 *   - Receives spell names over shared memory (see docs/ipc-protocol.md)
 *   - Checks unlock status via SpellManagerBPInterface
 *   - Casts through the player's WandTool
 *   - Handles actions that are not spells: Lumos/Nox toggling, Finite,
 *     Field Guide pages (Apperta*), the objective path, broom and mount
 *
 * Installed as Mods/SpellCaster/dlls/main.dll; log lines are prefixed [SpellCaster].
 */

// ============================================================================
// [1] Platform Configuration
// ============================================================================

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

// ============================================================================
// [2] Includes
// ============================================================================

// Platform
#include <Windows.h>

// Standard Library
#include <atomic>
#include <queue>
#include <string>
#include <unordered_map>

// UE4SS Framework
#include <Mod/CppUserModBase.hpp>
#include <DynamicOutput/DynamicOutput.hpp>

// Unreal Engine Types
#include <Unreal/UObjectGlobals.hpp>
#include <Unreal/UObject.hpp>
#include <Unreal/CoreUObject/UObject/Class.hpp>

// ============================================================================
// [3] Using Declarations
// ============================================================================

using namespace RC;
using namespace RC::Unreal;

// ============================================================================
// [4] Constants & Configuration
// ============================================================================

namespace Config
{
    // Shared Memory
    constexpr const char* SHARED_MEMORY_NAME = "SpellCasterSharedMemory";
    constexpr uint32_t COMMAND_BUFFER_SIZE = 256;
    constexpr uint32_t COMMAND_BUFFER_MASK = COMMAND_BUFFER_SIZE - 1;

    // Timing
    constexpr int HEARTBEAT_UPDATE_INTERVAL = 5;  // Frames between heartbeat updates (~83ms at 60fps, ~166ms at 30fps)

    // Limits
    constexpr size_t SPELL_NAME_MAX_LENGTH = 27;
    constexpr size_t PARAM_BUFFER_ALIGNMENT = 8;

    // ProcessEvent parameter buffer sizes (determined by UE4 function signatures)
    namespace ParamSize
    {
        constexpr size_t IS_UNLOCKED = 16;           // FName(8) + bool(1) padded
        constexpr size_t ACTIVATE_SPELL_TOOL = 16;   // UObject*(8) + bool(1) padded
        constexpr size_t IS_SPELL_AVAILABLE = 16;    // UObject*(8) + bool(1) + bool(1) padded
        constexpr size_t CAST_ACTIVE_SPELL = 32;     // Safety buffer for unknown params
    }
}

// ============================================================================
// [5] Data Structures
// ============================================================================

/**
 * Spell database entry containing all information needed to check and cast a spell.
 */
struct SpellInfo
{
    const wchar_t* name;        ///< Display name (e.g., "Lumos")
    const wchar_t* recordPath;  ///< Full path to SpellToolRecord asset
    const wchar_t* lockName;    ///< Lock name for unlock check (e.g., "Spell_Lumos")
    bool requiresTarget;        ///< True if spell needs a valid target (Transformation, Conjuration, etc.)
};

/**
 * Result of attempting to cast a spell.
 */
enum class SpellResult
{
    Cast,           ///< Spell was cast successfully
    NotUnlocked,    ///< Spell exists but player hasn't learned it (discard)
    Unknown,        ///< Spell not in database (discard)
    Busy,           ///< Spell tool not available (should retry)
    Failed          ///< Error occurred during casting
};

/**
 * Pause menu pages for UIManager::ToggleMenuTab
 */
enum class EPauseMenuPage : uint8_t
{
    PAGE_INVENTORY = 0,
    PAGE_CHARACTER = 1,
    PAGE_TALENTS = 2,
    PAGE_QUESTS = 3,
    PAGE_MAP = 4,
    PAGE_MAIL = 5,
    PAGE_COLLECTIONS = 6,
    PAGE_CHALLENGES = 7,
    PAGE_SETTINGS = 8,
    PAGE_MAX = 9
};

/**
 * Season enum for FieldGuideMenuStart
 */
enum class ESeasonEnum : uint8_t
{
    SEASON_SPRING = 0,
    SEASON_SUMMER = 1,
    SEASON_FALL = 2,
    SEASON_WINTER = 3,
    SEASON_MAX = 4
};

/**
 * UMG Input Action enum for ChangeMenuPage navigation
 * Only includes menu-related values (subset of full enum)
 */
enum class EUMGInputAction : uint8_t
{
    UMGMapScreenToggle = 13,
    UMGInventoryScreenToggle = 14,
    UMGCharacterScreenToggle = 15,
    UMGChallengeScreenToggle = 16,
    UMGQuestScreenToggle = 17,
    UMGCompendiumScreenToggle = 18,
    UMGOwlMailScreenToggle = 19,
    UMGTalentsScreenToggle = 20,
    UMGSettingsScreenToggle = 21
};

// ============================================================================
// [6] Shared Memory Protocol
// ============================================================================

namespace SharedMemory
{
    /**
     * Single spell command in the circular buffer.
     * Size: 32 bytes (28 name + 4 flags)
     */
    struct SpellCommand
    {
        char spellName[28];     ///< Null-terminated spell name
        uint32_t flags;         ///< Reserved for future use (priority, etc.)
    };

    /**
     * Shared memory layout for IPC with external spell casting applications.
     *
     * Memory Layout:
     *   [0x0000 - 0x003F] Control structure (64 bytes)
     *   [0x0040 - 0x203F] Command buffer (256 * 32 = 8192 bytes)
     *   [0x2040 - 0x3040] Status area (4096 bytes)
     */
    struct Layout
    {
        // Control structure (64 bytes, cache-line aligned)
        struct
        {
            std::atomic<uint32_t> writeIndex{0};        ///< Producer write position
            std::atomic<uint32_t> readIndex{0};         ///< Consumer read position
            std::atomic<uint32_t> producerHeartbeat{0}; ///< Producer alive signal
            std::atomic<uint32_t> consumerHeartbeat{0}; ///< Consumer (mod) alive signal
            std::atomic<uint32_t> messagesDropped{0};   ///< Buffer overflow counter
            std::atomic<uint32_t> totalMessages{0};     ///< Total messages sent
            uint32_t padding[10];                       ///< Pad to 64 bytes
        } control;

        // Circular command buffer
        SpellCommand commandBuffer[Config::COMMAND_BUFFER_SIZE];

        // Status feedback area
        struct
        {
            std::atomic<uint32_t> statusWriteIndex{0};
            std::atomic<uint32_t> statusReadIndex{0};
            char statusBuffer[4096 - 8];
        } status;

        /// Check if there are pending commands to process
        bool HasPendingCommands() const
        {
            uint32_t write = control.writeIndex.load(std::memory_order_acquire);
            uint32_t read = control.readIndex.load(std::memory_order_acquire);
            return write != read;
        }
    };
}

// ============================================================================
// [7] Spell Database
// ============================================================================

namespace SpellDatabase
{
    static const SpellInfo SPELLS[] =
    {
        // ---- Control Spells ----
        { STR("Accio"),           STR("/Game/Gameplay/ToolSet/Spells/Accio/DA_AccioSpellRecord.DA_AccioSpellRecord"),                               STR("Spell_Accio"),           false },
        { STR("Levioso"),         STR("/Game/Gameplay/ToolSet/Spells/Levioso/DA_LeviosoSpellRecord.DA_LeviosoSpellRecord"),                         STR("Spell_Levioso"),         false },
        { STR("Depulso"),         STR("/Game/Gameplay/ToolSet/Spells/Depulso/DA_DepulsoSpellRecord.DA_DepulsoSpellRecord"),                         STR("Spell_Depulso"),         false },
        { STR("Descendo"),        STR("/Game/Gameplay/ToolSet/Spells/Descendo/DA_DescendoSpellRecord.DA_DescendoSpellRecord"),                      STR("Spell_Descendo"),        false },
        { STR("Flipendo"),        STR("/Game/Gameplay/ToolSet/Spells/Flipendo/DA_FlipendoSpellRecord.DA_FlipendoSpellRecord"),                      STR("Spell_Flipendo"),        false },
        { STR("Glacius"),         STR("/Game/Gameplay/ToolSet/Spells/Glacius/DA_GlaciusSpellRecord.DA_GlaciusSpellRecord"),                         STR("Spell_Glacius"),         false },
        { STR("ArrestoMomentum"), STR("/Game/Gameplay/ToolSet/Spells/ArrestoMomentum/DA_ArrestoMomentumSpellRecord.DA_ArrestoMomentumSpellRecord"), STR("Spell_ArrestoMomentum"), false },

        // ---- Damage Spells ----
        { STR("Incendio"),        STR("/Game/Gameplay/ToolSet/Spells/Incendio/DA_IncendioSpellRecord.DA_IncendioSpellRecord"),                      STR("Spell_Incendio"),  false },
        { STR("Confringo"),       STR("/Game/Gameplay/ToolSet/Spells/Confringo/DA_ConfringoSpellRecord.DA_ConfringoSpellRecord"),                   STR("Spell_Confringo"), false },
        { STR("Diffindo"),        STR("/Game/Gameplay/ToolSet/Spells/Diffindo/DA_DiffindoSpellRecord.DA_DiffindoSpellRecord"),                      STR("Spell_Diffindo"),  false },
        { STR("Bombarda"),         STR("/Game/Gameplay/ToolSet/Spells/Expulso/DA_ExpulsoSpellRecord.DA_ExpulsoSpellRecord"),                         STR("Spell_Expulso"),   false },

        // ---- Combat Spells ----
        { STR("Stupefy"),         STR("/Game/Gameplay/ToolSet/Spells/Stupefy/DA_StupefySpellRecord.DA_StupefySpellRecord"),                         STR("Spell_Stupefy"),      false },
        { STR("Expelliarmus"),    STR("/Game/Gameplay/ToolSet/Spells/Expelliarmus/DA_ExpelliarmusSpellRecord.DA_ExpelliarmusSpellRecord"),          STR("Spell_Expelliarmus"), false },
        { STR("Protego"),         STR("/Game/Gameplay/ToolSet/Spells/Protego/DA_ProtegoSpellRecord.DA_ProtegoSpellRecord"),                         STR("Spell_Protego"),      false },
        { STR("Oppugno"),         STR("/Game/Gameplay/ToolSet/Spells/Oppugno/DA_OppugnoSpellRecord.DA_OppugnoSpellRecord"),                         STR("Spell_Oppugno"),      false },

        // ---- Utility Spells ----
        { STR("Lumos"),           STR("/Game/Gameplay/ToolSet/Spells/Lumos/DA_LumosSpellRecord.DA_LumosSpellRecord"),                               STR("Spell_Lumos"),            false },
        { STR("Nox"),             STR("/Game/Gameplay/ToolSet/Spells/Lumos/DA_LumosSpellRecord.DA_LumosSpellRecord"),                               STR("Spell_Lumos"),            false },
        { STR("Reparo"),          STR("/Game/Gameplay/ToolSet/Spells/Reparo/DA_ReparoSpellRecord.DA_ReparoSpellRecord"),                            STR("Spell_Reparo"),           false },
        { STR("Revelio"),         STR("/Game/Gameplay/ToolSet/Spells/Revelio/DA_RevelioSpellRecord.DA_RevelioSpellRecord"),                         STR("Spell_Revelio"),          false },
        { STR("Invisica"),        STR("/Game/Gameplay/ToolSet/Spells/Disillusionment/DA_DisillusionmentSpellRecord.DA_DisillusionmentSpellRecord"), STR("Spell_Disillusionment"),  false },
        { STR("WingardiumLeviosa"),      STR("/Game/Gameplay/ToolSet/Spells/Wingardium/DA_WingardiumSpellRecord.DA_WingardiumSpellRecord"),                STR("Spell_Wingardium"), false },

        // ---- Transfiguration Spells ----
        { STR("TransfiguraVerto"),  STR("/Game/Gameplay/ToolSet/Spells/Transformation/DA_TransformationSpellRecord.DA_TransformationSpellRecord"),   STR("Spell_Transformation"), true  },  // Requires target
        { STR("Conjuration"),       STR("/Game/Gameplay/ToolSet/Spells/Conjuration/DA_ConjurationSpellRecord.DA_ConjurationSpellRecord"),             STR("Spell_Conjuration"),    false },
        { STR("Vanishment"),        STR("/Game/Gameplay/ToolSet/Spells/Vanishment/DA_VanishmentSpellRecord.DA_VanishmentSpellRecord"),               STR("Spell_Vanishment"),     false },

        // ---- Unforgivable Curses ----
        { STR("AvadaKedavra"),    STR("/Game/Gameplay/ToolSet/Spells/AvadaKedavra/DA_AvadaKedavraSpellRecord.DA_AvadaKedavraSpellRecord"),          STR("Spell_AvadaKedavra"), false },
        { STR("Crucio"),          STR("/Game/Gameplay/ToolSet/Spells/Crucio/DA_CrucioSpellRecord.DA_CrucioSpellRecord"),                            STR("Spell_Crucio"),       false },
        { STR("Imperio"),         STR("/Game/Gameplay/ToolSet/Spells/Imperious/DA_ImperiusSpellRecord.DA_ImperiusSpellRecord"),                     STR("Spell_Imperius"),     false },

        // ---- Special Spells ----
        { STR("Smash"),           STR("/Game/Gameplay/ToolSet/Spells/AMBossKiller/DA_AMBossKillerSpellRecord.DA_AMBossKillerSpellRecord"),          STR("FinisherAMBossKiller"),   false },
        { STR("StealthTakedown"), STR("/Game/Gameplay/ToolSet/Spells/StealthTakedown/DA_StealthTakedownSpellRecord.DA_StealthTakedownSpellRecord"), STR("Spell_StealthTakedown"),  false },
        { STR("Confundo"),        STR("/Game/Gameplay/ToolSet/Spells/Confundo/DA_ConfundoSpellRecord.DA_ConfundoSpellRecord"),                      STR("Spell_Confundo"),         false },
        { STR("Episkey"),         STR("/Game/Gameplay/ToolSet/Spells/Episkey/DA_EpiskeySpellRecord.DA_EpiskeySpellRecord"),                         STR("Spell_Episkey"),          false },

        // ---- SpellsEnhanced (Khione Mod) ----
        { STR("Inflatus"),            STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_InflatingCharm_Khione.DA_InflatingCharm_Khione"),                   STR("Spell_InflatingCharm"),    false },
        { STR("PetrificusTotalus"),   STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_PetrificusSpellRecord_Khione.DA_PetrificusSpellRecord_Khione"),    STR("Spell_Petrificus"),        false },
        { STR("ConfundoEnhanced"),    STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_ConfundoSpellRecord_Khione.DA_ConfundoSpellRecord_Khione"),        STR("Spell_Confundo"),          false },
        { STR("IncendioIncantatem"),  STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_IncendioIncantaten_Khione.DA_IncendioIncantaten_Khione"),          STR("Spell_IncendioIncantatem"), false },
        { STR("Baubillious"),         STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_Baubillious_Khione.DA_Baubillious_Khione"),                        STR("Spell_Baubillious"),       false },
        { STR("ExpulsoEnhanced"),     STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_Expulso_Khione.DA_Expulso_Khione"),                                STR("Spell_Expulso"),           false },
        { STR("Reducto"),             STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_Reducto_Khione.DA_Reducto_Khione"),                                STR("Spell_Reducto"),           false },
        { STR("Verdimillious"),       STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_Verdimillious_Khione.DA_Verdimillious_Khione"),                    STR("Spell_Verdimillious"),     false },
        { STR("Obliviate"),           STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_ObliviateSpellRecord_Khione.DA_ObliviateSpellRecord_Khione"),      STR("Spell_Obliviate"),         false },
        { STR("LevicorpusMaxima"),    STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_LevicorpusMaxima_Khione.DA_LevicorpusMaxima_Khione"),              STR("Spell_LevicorpusMaxima"),  false },
        { STR("Ventus"),              STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_Venuts_Khione.DA_Venuts_Khione"),                                  STR("Spell_Ventus"),            false },
        { STR("AquaEructo"),          STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_AquaEructo_Khione.DA_AquaEructo_Khione"),                          STR("Spell_AquaEructo"),        false },
        { STR("Apparate"),            STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_ApparitionSpellRecord_Khione.DA_ApparitionSpellRecord_Khione"),    STR("Spell_Apparition"),        false },
        { STR("ApparateMaxima"),      STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_ApparitionMaximaSpellRecord_Khione.DA_ApparitionMaximaSpellRecord_Khione"), STR("Spell_ApparitionMaxima"), false },
        { STR("EpiskeyEnhanced"),     STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_EpiskeySpellRecord_Khione.DA_EpiskeySpellRecord_Khione"),          STR("Spell_Episkey"),           false },
        { STR("TripJinx"),            STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_TripJinxSpellRecord_Khione.DA_TripJinxSpellRecord_Khione"),        STR("Spell_TripJinx"),          false },
        { STR("StupefyEnhanced"),     STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_Stupefy_Khione.DA_Stupefy_Khione"),                                STR("Spell_Stupefy"),           false },
        { STR("AOEIce"),              STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_AOEAttack_Ice_Khione.DA_AOEAttack_Ice_Khione"),                    STR("Spell_AOEIce"),            false },
        { STR("AOELightning"),        STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_AOEAttack_Lightning_Khione.DA_AOEAttack_Lightning_Khione"),        STR("Spell_AOELightning"),      false },
        { STR("AOEFlame"),            STR("/SpellsEnhanced/Spells/Spell_DataAssets/DA_AOEAttack_Flame_Khione.DA_AOEAttack_Flame_Khione"),                STR("Spell_AOEFlame"),          false },

        // ---- HRBSpellPack ----
        { STR("LumosMaxima"),         STR("/HRBSpellPack/DA_LumosMaximaSpellRecord.DA_LumosMaximaSpellRecord"),                                          STR("Spell_LumosMaxima"),    false },
        { STR("Ascendio"),            STR("/HRBSpellPack/DA_AscendioSpellRecord.DA_AscendioSpellRecord"),                                                STR("Spell_Ascendio"),       false },
        { STR("BombardaMaxima"),      STR("/HRBSpellPack/BP_BombardaMaximaSpellRecord.BP_BombardaMaximaSpellRecord"),                                    STR("Spell_BombardaMaxima"), false },

        // ---- HermitHollow ----
        { STR("TempusMutatio"),       STR("/HermitHollow/DA_TempusMutatioSpellRecord.DA_TempusMutatioSpellRecord"),                                      STR("Spell_TempusMutatio"), false },

        // ---- WFM_SpellMod_01 ----
        { STR("Fulminous"),           STR("/WFM_SpellMod_01/DA_Spell_FulminousSpellRecord.DA_Spell_FulminousSpellRecord"),                               STR("Spell_Fulminous"), false },

        // ---- SpM_Test-01 ----
        { STR("Sectumsempra"),        STR("/SpM_Test-01/Sectumsempra/DA_Spell_SectumsempraSpellRecord.DA_Spell_SectumsempraSpellRecord"),                STR("Spell_Sectumsempra"), false },
        { STR("Impedimenta"),         STR("/SpM_Test-01/Impedimenta/DA_Spell_ImpedimentaSpellRecord.DA_Spell_ImpedimentaSpellRecord"),                  STR("Spell_Impedimenta"),  false },
    };

    static constexpr size_t COUNT = sizeof(SPELLS) / sizeof(SPELLS[0]);

    /**
     * Find a spell by name (case-insensitive).
     * @param name Spell name to search for
     * @return Pointer to SpellInfo or nullptr if not found
     */
    const SpellInfo* FindByName(const wchar_t* name)
    {
        std::wstring searchName(name);
        for (auto& c : searchName) c = towlower(c);

        for (size_t i = 0; i < COUNT; ++i)
        {
            std::wstring dbName(SPELLS[i].name);
            for (auto& c : dbName) c = towlower(c);

            if (searchName == dbName)
                return &SPELLS[i];
        }
        return nullptr;
    }
}

// ============================================================================
// [8] SpellCasterMod Class
// ============================================================================

class SpellCasterMod : public CppUserModBase
{
    // ========================================================================
    // [8.1] Member Variables
    // ========================================================================

private:
    // ---- Spell Unlock System ----
    UFunction* m_funcIsUnlocked = nullptr;
    UObject* m_interfaceCDO = nullptr;
    bool m_unlockSystemReady = false;

    // ---- Spell Casting System ----
    UObject* m_wandTool = nullptr;
    UFunction* m_funcCancelCurrentSpell = nullptr;
    UFunction* m_funcActivateSpellTool = nullptr;
    UFunction* m_funcCastActiveSpell = nullptr;
    UFunction* m_funcIsSpellToolAvailable = nullptr;
    UFunction* m_funcGetSpellTool = nullptr;
    UFunction* m_funcIsLumosActive = nullptr;
    bool m_castingSystemReady = false;
    std::unordered_map<std::wstring, UObject*> m_spellRecordCache;

    // ---- Shared Memory IPC ----
    HANDLE m_hSharedMemory = INVALID_HANDLE_VALUE;
    SharedMemory::Layout* m_pSharedMemory = nullptr;
    bool m_sharedMemoryConnected = false;
    uint32_t m_heartbeatCounter = 0;
    std::queue<std::string> m_spellQueue;

    // ---- Special Spell State (extensibility point) ----
    // Future: Add toggle states, cooldowns, spell aliases here

    // ---- UI System ----
    UObject* m_uiManager = nullptr;
    UFunction* m_funcToggleMenuTab = nullptr;
    UFunction* m_funcFieldGuideMenuStart = nullptr;
    UFunction* m_funcExitFieldGuide = nullptr;
    UFunction* m_funcGetFieldGuideWidget = nullptr;
    UFunction* m_funcChangeMenuPage = nullptr;
    UFunction* m_funcSetActiveMenu = nullptr;
    UFunction* m_funcOnInputAction = nullptr;
    UFunction* m_funcLoadFieldGuideScreen = nullptr;
    UFunction* m_funcTogglePathActive = nullptr;  // Toggle objective path display
    UFunction* m_funcShowPathSelectionPressed = nullptr;  // Show path (V key pressed)
    UFunction* m_funcShowPathSelectionReleased = nullptr; // Show path (V key released)
    bool m_uiSystemReady = false;

    // ---- Broom Activation System ----
    UFunction* m_funcActivateToolByName = nullptr;  // Activate tool by name (for broom)

    // ---- Page Override ----
    bool m_overridePageEnabled = false;
    std::wstring m_overridePageName;

    // ========================================================================
    // [8.2] Constructor / Destructor
    // ========================================================================

public:
    SpellCasterMod() : CppUserModBase()
    {
        ModName = STR("SpellCaster");
        ModVersion = STR("1.1");
        ModDescription = STR("Programmatic spell casting via shared memory");
        ModAuthors = STR("Cornebidouil");
    }

    ~SpellCasterMod() override
    {
        CleanupSharedMemory();
    }

    // ========================================================================
    // [8.3] Spell Unlock System
    // ========================================================================

private:
    /**
     * Initialize the spell unlock checking system.
     * Caches function pointer and interface CDO for IsUnlocked calls.
     * @return true if initialization succeeded
     */
    bool InitializeUnlockSystem()
    {
        if (m_unlockSystemReady)
            return true;

        // Find IsUnlocked function on SpellManagerBPInterface
        m_funcIsUnlocked = UObjectGlobals::StaticFindObject<UFunction*>(
            nullptr, nullptr,
            STR("/Script/Phoenix.SpellManagerBPInterface:IsUnlocked")
        );

        if (!m_funcIsUnlocked)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] Failed to find IsUnlocked function\n"));
            return false;
        }

        // Find the Interface CDO (required for calling static BP interface functions)
        m_interfaceCDO = UObjectGlobals::StaticFindObject<UObject*>(
            nullptr, nullptr,
            STR("/Script/Phoenix.Default__SpellManagerBPInterface")
        );

        if (!m_interfaceCDO)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] Failed to find SpellManagerBPInterface CDO\n"));
            return false;
        }

        m_unlockSystemReady = true;
        Output::send<LogLevel::Default>(STR("[SpellCaster] Unlock system initialized\n"));
        return true;
    }

public:
    /**
     * Check if a spell is unlocked in the player's spell book.
     * @param lockName Full lock name (e.g., "Spell_Lumos")
     * @return true if the spell is unlocked
     */
    bool IsSpellUnlocked(const wchar_t* lockName)
    {
        if (!m_unlockSystemReady && !InitializeUnlockSystem())
            return false;

        // Prepare parameter buffer: FName(8 bytes) + bool ReturnValue(1 byte)
        alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[Config::ParamSize::IS_UNLOCKED] = {};
        *reinterpret_cast<FName*>(&params[0]) = FName(lockName, FNAME_Add);

        m_interfaceCDO->ProcessEvent(m_funcIsUnlocked, params);

        return params[8] != 0;
    }

    /**
     * Check if a spell is unlocked using just the spell name.
     * Automatically prepends "Spell_" to the name.
     * @param spellName Spell name without prefix (e.g., "Lumos")
     * @return true if the spell is unlocked
     */
    bool IsSpellUnlockedByName(const wchar_t* spellName)
    {
        std::wstring fullName = STR("Spell_");
        fullName += spellName;
        return IsSpellUnlocked(fullName.c_str());
    }

    // ========================================================================
    // [8.4] Spell Casting System
    // ========================================================================

private:
    /**
     * Initialize the spell casting system.
     * Finds WandTool instance and caches function pointers.
     * @return true if initialization succeeded
     */
    bool InitializeCastingSystem()
    {
        // WandTool can be recreated, always refresh instance
        m_wandTool = UObjectGlobals::FindFirstOf(STR("WandTool"));

        if (!m_wandTool)
        {
            Output::send<LogLevel::Warning>(STR("[SpellCaster] WandTool not found (player may not be in game)\n"));
            m_castingSystemReady = false;
            return false;
        }

        // Only find functions once
        if (!m_castingSystemReady)
        {
            // WandTool functions
            m_funcCancelCurrentSpell = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.WandTool:CancelCurrentSpell"));

            m_funcActivateSpellTool = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.WandTool:ActivateSpellTool"));

            m_funcCastActiveSpell = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.WandTool:CastActiveSpell"));

            m_funcIsSpellToolAvailable = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.WandTool:IsSpellToolAvailable"));

            m_funcGetSpellTool = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.WandTool:GetSpellTool"));

            // LumosSpellTool::IsLumosActive (called on instance from GetSpellTool)
            m_funcIsLumosActive = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.LumosSpellTool:IsLumosActive"));

            if (!m_funcActivateSpellTool || !m_funcCastActiveSpell)
            {
                Output::send<LogLevel::Error>(STR("[SpellCaster] Failed to find required WandTool functions\n"));
                return false;
            }

            if (!m_funcGetSpellTool)
                Output::send<LogLevel::Warning>(STR("[SpellCaster] GetSpellTool not found\n"));

            if (!m_funcIsLumosActive)
                Output::send<LogLevel::Warning>(STR("[SpellCaster] IsLumosActive not found - Lumos toggle may not work\n"));

            m_castingSystemReady = true;
            Output::send<LogLevel::Default>(STR("[SpellCaster] Casting system initialized\n"));
        }

        return true;
    }

    /**
     * Get or cache a SpellToolRecord by its asset path.
     * @param recordPath Full asset path to the SpellToolRecord
     * @return The UObject pointer or nullptr if not found
     */
    UObject* GetSpellToolRecord(const wchar_t* recordPath)
    {
        std::wstring pathStr(recordPath);

        auto it = m_spellRecordCache.find(pathStr);
        if (it != m_spellRecordCache.end())
            return it->second;

        UObject* record = UObjectGlobals::StaticFindObject<UObject*>(nullptr, nullptr, recordPath);

        if (record)
            m_spellRecordCache[pathStr] = record;

        return record;
    }

    /**
     * Check if a spell tool is currently available for casting.
     * @param spellRecord The SpellToolRecord object
     * @return true if the spell can be cast right now
     */
    bool IsSpellToolAvailable(UObject* spellRecord)
    {
        if (!m_wandTool)
        {
            Output::send<LogLevel::Warning>(STR("[SpellCaster] IsSpellToolAvailable: WandTool is null\n"));
            return false;
        }

        if (!m_funcIsSpellToolAvailable)
        {
            Output::send<LogLevel::Warning>(STR("[SpellCaster] IsSpellToolAvailable: Function pointer is null\n"));
            return false;
        }

        if (!spellRecord)
        {
            Output::send<LogLevel::Warning>(STR("[SpellCaster] IsSpellToolAvailable: SpellRecord is null\n"));
            return false;
        }

        // Parameter layout: UObject*(8) + bImmediate(1) + ReturnValue(1)
        alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[Config::ParamSize::IS_SPELL_AVAILABLE] = {};
        *reinterpret_cast<UObject**>(&params[0]) = spellRecord;
        params[8] = 0;  // bImmediate = false

        m_wandTool->ProcessEvent(m_funcIsSpellToolAvailable, params);

        bool isAvailable = params[9] != 0;

        // Debug logging
        Output::send<LogLevel::Default>(STR("[SpellCaster] IsSpellToolAvailable result: {} for record: {}\n"),
            isAvailable ? STR("AVAILABLE") : STR("BUSY"),
            spellRecord->GetFullName());

        return isAvailable;
    }

    /**
     * Get the SpellTool instance for a given SpellToolRecord.
     * @param spellRecord The SpellToolRecord to get the tool for
     * @return The SpellTool instance or nullptr
     */
    UObject* GetSpellTool(UObject* spellRecord)
    {
        if (!m_wandTool || !m_funcGetSpellTool || !spellRecord)
            return nullptr;

        // GetSpellTool(SpellToolRecord) -> SpellTool*
        // Parameter layout: UObject* input (8) + UObject* return (8)
        alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[16] = {};
        *reinterpret_cast<UObject**>(&params[0]) = spellRecord;

        m_wandTool->ProcessEvent(m_funcGetSpellTool, params);

        return *reinterpret_cast<UObject**>(&params[8]);
    }

    /**
     * Check if Lumos spell is currently active.
     * Gets the LumosSpellTool via WandTool:GetSpellTool and calls IsLumosActive.
     * @return true if Lumos light is on
     */
    bool IsLumosActive()
    {
        if (!m_wandTool || !m_funcGetSpellTool || !m_funcIsLumosActive)
            return false;

        // Get Lumos spell record
        UObject* lumosRecord = GetSpellToolRecord(
            STR("/Game/Gameplay/ToolSet/Spells/Lumos/DA_LumosSpellRecord.DA_LumosSpellRecord"));

        if (!lumosRecord)
            return false;

        // Get the LumosSpellTool instance
        UObject* lumosSpellTool = GetSpellTool(lumosRecord);
        if (!lumosSpellTool)
            return false;

        // Call IsLumosActive() on the LumosSpellTool instance
        // IsLumosActive() -> bool (no parameters, just return value)
        alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[8] = {};

        lumosSpellTool->ProcessEvent(m_funcIsLumosActive, params);

        return params[0] != 0;
    }

    /**
     * Cancel the currently active spell via WandTool.
     */
    void CancelCurrentSpell()
    {
        if (m_wandTool && m_funcCancelCurrentSpell)
            m_wandTool->ProcessEvent(m_funcCancelCurrentSpell, nullptr);
    }

    /**
     * Execute the spell casting sequence via WandTool.
     * @param spellRecord The SpellToolRecord to cast
     * @return true if casting sequence completed
     */
    bool ExecuteSpellCast(UObject* spellRecord)
    {
        if (!m_wandTool || !spellRecord)
            return false;

        // Cancel any current spell
        if (m_funcCancelCurrentSpell)
            m_wandTool->ProcessEvent(m_funcCancelCurrentSpell, nullptr);

        // Activate the spell tool
        if (m_funcActivateSpellTool)
        {
            alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[Config::ParamSize::ACTIVATE_SPELL_TOOL] = {};
            *reinterpret_cast<UObject**>(&params[0]) = spellRecord;
            params[8] = 0;  // bForceSpell = false

            m_wandTool->ProcessEvent(m_funcActivateSpellTool, params);
        }

        // Cast the active spell
        if (m_funcCastActiveSpell)
        {
            alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[Config::ParamSize::CAST_ACTIVE_SPELL] = {};
            m_wandTool->ProcessEvent(m_funcCastActiveSpell, params);
        }

        return true;
    }

public:
    /**
     * Cast a spell by name with full validation.
     *
     * @param spellName Spell name from database (e.g., "Lumos", "Incendio")
     * @return SpellResult indicating the outcome
     */
    SpellResult CastSpellByName(const wchar_t* spellName)
    {
        Output::send<LogLevel::Default>(STR("[SpellCaster] === Attempting to cast: {} ===\n"), spellName);

        // Look up spell in database
        const SpellInfo* spell = SpellDatabase::FindByName(spellName);
        if (!spell)
        {
            Output::send<LogLevel::Warning>(STR("[SpellCaster] Unknown spell '{}' - discarding\n"), spellName);
            return SpellResult::Unknown;
        }

        Output::send<LogLevel::Default>(STR("[SpellCaster] Found spell record: {}\n"), spell->recordPath);

        // No unlock check here: IsSpellUnlocked exists but is deliberately not applied before
        // casting (see docs/known-issues.md). The game decides what an unlearned spell does.

        // Initialize casting system
        if (!InitializeCastingSystem())
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] Failed to initialize casting system\n"));
            return SpellResult::Failed;
        }

        // Get spell record
        UObject* record = GetSpellToolRecord(spell->recordPath);
        if (!record)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] SpellToolRecord not found: {}\n"), spell->recordPath);
            return SpellResult::Failed;
        }

        Output::send<LogLevel::Default>(STR("[SpellCaster] SpellToolRecord loaded successfully\n"));

        // Check if this spell requires a target (contextual spells)
        bool isContextualSpell = spell->requiresTarget;

        // Check if Lumos is active (common blocker for non-contextual spells)
        if (!isContextualSpell && m_funcIsLumosActive && IsLumosActive())
        {
            Output::send<LogLevel::Default>(STR("[SpellCaster] Lumos is active, attempting to cancel...\n"));
            CancelCurrentSpell();
        }

        // For contextual spells (Transformation, Conjuration, etc.), skip availability check
        // and let the game handle target detection
        if (isContextualSpell)
        {
            Output::send<LogLevel::Default>(STR("[SpellCaster] {} is a contextual spell - letting game handle targeting\n"), spellName);
        }
        else
        {
            // Check availability for non-contextual spells
            if (!IsSpellToolAvailable(record))
            {
                Output::send<LogLevel::Warning>(STR("[SpellCaster] {} not available on first check, attempting to cancel active spells...\n"), spellName);

                // Force cancel any active spell
                CancelCurrentSpell();

                // Check again after cancel
                if (!IsSpellToolAvailable(record))
                {
                    Output::send<LogLevel::Warning>(STR("[SpellCaster] {} still not available after cancel (spell tool busy)\n"), spellName);
                    return SpellResult::Busy;
                }

                Output::send<LogLevel::Default>(STR("[SpellCaster] {} now available after cancel\n"), spellName);
            }
        }

        // Execute cast
        Output::send<LogLevel::Default>(STR("[SpellCaster] Executing spell cast...\n"));
        if (ExecuteSpellCast(record))
        {
            Output::send<LogLevel::Default>(STR("[SpellCaster] ✓ Successfully cast: {}\n"), spellName);
            return SpellResult::Cast;
        }

        Output::send<LogLevel::Error>(STR("[SpellCaster] ✗ Failed to execute cast\n"));
        return SpellResult::Failed;
    }

    // ========================================================================
    // [8.5] Shared Memory System
    // ========================================================================

private:
    /**
     * Initialize shared memory connection for receiving external spell commands.
     * @return true if connection established
     */
    bool InitializeSharedMemory()
    {
        if (m_sharedMemoryConnected)
            return true;

        // Try to open existing shared memory
        m_hSharedMemory = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, Config::SHARED_MEMORY_NAME);

        if (!m_hSharedMemory || m_hSharedMemory == INVALID_HANDLE_VALUE)
        {
            // Create new shared memory
            m_hSharedMemory = CreateFileMappingA(
                INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                0, sizeof(SharedMemory::Layout), Config::SHARED_MEMORY_NAME
            );

            if (!m_hSharedMemory || m_hSharedMemory == INVALID_HANDLE_VALUE)
            {
                Output::send<LogLevel::Error>(STR("[SpellCaster] Failed to create shared memory\n"));
                return false;
            }
        }

        // Map view
        m_pSharedMemory = static_cast<SharedMemory::Layout*>(
            MapViewOfFile(m_hSharedMemory, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedMemory::Layout))
        );

        if (!m_pSharedMemory)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] Failed to map shared memory\n"));
            CloseHandle(m_hSharedMemory);
            m_hSharedMemory = INVALID_HANDLE_VALUE;
            return false;
        }

        m_sharedMemoryConnected = true;
        Output::send<LogLevel::Default>(STR("[SpellCaster] Shared memory connected\n"));
        return true;
    }

    /**
     * Clean up shared memory resources.
     */
    void CleanupSharedMemory()
    {
        if (m_pSharedMemory)
        {
            UnmapViewOfFile(m_pSharedMemory);
            m_pSharedMemory = nullptr;
        }

        if (m_hSharedMemory != INVALID_HANDLE_VALUE)
        {
            CloseHandle(m_hSharedMemory);
            m_hSharedMemory = INVALID_HANDLE_VALUE;
        }

        m_sharedMemoryConnected = false;
    }

    /**
     * Read pending spell commands from shared memory into the queue.
     */
    void ReadSpellCommands()
    {
        if (!m_pSharedMemory)
            return;

        // Update heartbeat periodically
        if (++m_heartbeatCounter >= Config::HEARTBEAT_UPDATE_INTERVAL)
        {
            m_pSharedMemory->control.consumerHeartbeat.fetch_add(1, std::memory_order_relaxed);
            m_heartbeatCounter = 0;
        }

        // Read all pending commands
        while (m_pSharedMemory->HasPendingCommands())
        {
            uint32_t readPos = m_pSharedMemory->control.readIndex.load(std::memory_order_relaxed);
            uint32_t bufferIndex = readPos & Config::COMMAND_BUFFER_MASK;

            SharedMemory::SpellCommand cmd = m_pSharedMemory->commandBuffer[bufferIndex];
            m_pSharedMemory->control.readIndex.store(readPos + 1, std::memory_order_release);

            // Ensure null termination and queue if valid
            cmd.spellName[Config::SPELL_NAME_MAX_LENGTH] = '\0';
            if (cmd.spellName[0] != '\0')
            {
                m_spellQueue.push(std::string(cmd.spellName));
            }
        }
    }

    /**
     * Process one spell from the queue.
     *
     * Behavior by result:
     *   - Cast/NotUnlocked/Unknown/Failed: Remove from queue (processed or discarded)
     *   - Busy: Keep in queue for retry next frame
     *
     * @return true if spell was processed, false if busy (will retry)
     */
    bool ProcessSpellQueue()
    {
        if (m_spellQueue.empty())
            return false;

        const std::string& spellName = m_spellQueue.front();
        std::wstring wideSpellName(spellName.begin(), spellName.end());

        // Try special spell handling first (extensibility point)
        if (TryHandleSpecialSpell(wideSpellName.c_str()))
        {
            m_spellQueue.pop();
            return true;
        }

        // Normal spell casting
        SpellResult result = CastSpellByName(wideSpellName.c_str());

        // Only retry if spell tool is busy, discard in all other cases
        if (result == SpellResult::Busy)
            return false;  // Keep in queue for retry

        // Remove from queue: cast successfully, not unlocked, unknown, or failed
        m_spellQueue.pop();
        return true;
    }

    // ========================================================================
    // [8.6] Special Spell Handling (Extensibility Point)
    // ========================================================================

    /**
     * Handle special spells that require custom logic.
     *
     * Currently implemented:
     *   - Lumos: Only cast if not already active
     *   - Nox: Only cancel Lumos if it is active
     *   - Finite: Cancel any active spell
     *
     * @param spellName The spell command received
     * @return true if handled (skip normal casting), false to continue normal flow
     */
    bool TryHandleSpecialSpell(const wchar_t* spellName)
    {
        // Ensure casting system is ready for state checks
        if (!InitializeCastingSystem())
            return false;

        // ---- Lumos: Only cast if NOT already active ----
        if (_wcsicmp(spellName, STR("Lumos")) == 0)
        {
            if (IsLumosActive())
            {
                Output::send<LogLevel::Default>(STR("[SpellCaster] Lumos already active - ignoring\n"));
                return true;  // Handled (do nothing)
            }
            // Not active - let normal casting proceed
            return false;
        }

        // ---- Nox: Only cancel if Lumos IS active ----
        if (_wcsicmp(spellName, STR("Nox")) == 0)
        {
            if (IsLumosActive())
            {
                CancelCurrentSpell();
                Output::send<LogLevel::Default>(STR("[SpellCaster] Nox - cancelled Lumos\n"));
            }
            else
            {
                Output::send<LogLevel::Default>(STR("[SpellCaster] Nox - Lumos not active, ignoring\n"));
            }
            return true;  // Always handled (don't try to cast Nox as a spell)
        }

        // ---- Finite: Cancel any active spell AND close menu ----
        if (_wcsicmp(spellName, STR("Finite")) == 0)
        {
            // Cancel general active spells (Lumos, Wingardium, etc.)
            CancelCurrentSpell();

            // Special handling for Disillusionment - requires IsSpellToolAvailable call to cancel
            const SpellInfo* invisicaSpell = SpellDatabase::FindByName(STR("Invisica"));
            if (invisicaSpell)
            {
                UObject* disillusionmentRecord = GetSpellToolRecord(invisicaSpell->recordPath);
                if (disillusionmentRecord)
                {
                    IsSpellToolAvailable(disillusionmentRecord);
                }
            }

            // Also close the Field Guide menu if open
            CloseFieldGuideMenu();

            Output::send<LogLevel::Default>(STR("[SpellCaster] Finite - cancelled active spell and closed menu\n"));
            return true;  // Always handled
        }

        // ---- Menu Access Spells (Apperta*) ----
        if (_wcsnicmp(spellName, STR("Apperta"), 7) == 0)
        {
            // Map spell names to menu actions
            if (_wcsicmp(spellName, STR("AppertaCodex")) == 0 || _wcsicmp(spellName, STR("AppertaSacculus")) == 0)
            {
                Output::send<LogLevel::Default>(STR("[SpellCaster] {} - Opening Inventory\n"), spellName);
                OpenOrNavigateToPage(EUMGInputAction::UMGInventoryScreenToggle);
                return true;
            }
            else if (_wcsicmp(spellName, STR("AppertaMeritas")) == 0 || _wcsicmp(spellName, STR("AppertaVestiarium")) == 0)
            {
                Output::send<LogLevel::Default>(STR("[SpellCaster] {} - Opening Character\n"), spellName);
                OpenOrNavigateToPage(EUMGInputAction::UMGCharacterScreenToggle);
                return true;
            }
            else if (_wcsicmp(spellName, STR("AppertaFalcultates")) == 0)
            {
                Output::send<LogLevel::Default>(STR("[SpellCaster] {} - Opening Talents\n"), spellName);
                OpenOrNavigateToPage(EUMGInputAction::UMGTalentsScreenToggle);
                return true;
            }
            else if (_wcsicmp(spellName, STR("AppertaQuaestiones")) == 0)
            {
                Output::send<LogLevel::Default>(STR("[SpellCaster] {} - Opening Quests\n"), spellName);
                OpenOrNavigateToPage(EUMGInputAction::UMGQuestScreenToggle);
                return true;
            }
            else if (_wcsicmp(spellName, STR("AppertaMappa")) == 0)
            {
                Output::send<LogLevel::Default>(STR("[SpellCaster] {} - Opening Map\n"), spellName);
                OpenOrNavigateToPage(EUMGInputAction::UMGMapScreenToggle);
                return true;
            }
            else if (_wcsicmp(spellName, STR("AppertaLiterae")) == 0)
            {
                Output::send<LogLevel::Default>(STR("[SpellCaster] {} - Opening Mail\n"), spellName);
                OpenOrNavigateToPage(EUMGInputAction::UMGOwlMailScreenToggle);
                return true;
            }
            else if (_wcsicmp(spellName, STR("AppertaCompendium")) == 0)
            {
                Output::send<LogLevel::Default>(STR("[SpellCaster] {} - Opening Collections\n"), spellName);
                OpenOrNavigateToPage(EUMGInputAction::UMGCompendiumScreenToggle);
                return true;
            }
            else if (_wcsicmp(spellName, STR("AppertaIncantatem")) == 0)
            {
                Output::send<LogLevel::Default>(STR("[SpellCaster] {} - Opening Studies\n"), spellName);
                OpenOrNavigateToPage(EUMGInputAction::UMGChallengeScreenToggle);
                return true;
            }
            else if (_wcsicmp(spellName, STR("AppertaConfiguratio")) == 0)
            {
                Output::send<LogLevel::Default>(STR("[SpellCaster] {} - Opening Settings\n"), spellName);
                OpenOrNavigateToPage(EUMGInputAction::UMGSettingsScreenToggle);
                return true;
            }
            else
            {
                Output::send<LogLevel::Warning>(STR("[SpellCaster] Unknown menu spell: {}\n"), spellName);
                return true;  // Still handled, just unknown
            }
        }

        // ---- AppareVestigium: Show objective path (V key) ----
        if (_wcsicmp(spellName, STR("AppareVestigium")) == 0)
        {
            Output::send<LogLevel::Default>(STR("[SpellCaster] AppareVestigium - Showing objective path\n"));
            ToggleObjectivePath();
            return true;
        }

        // ---- Accio Broom Spells: Summon/activate broomstick ----
        if (_wcsicmp(spellName, STR("AccioBroomstick")) == 0 ||
            _wcsicmp(spellName, STR("AccioBalais")) == 0 ||
            _wcsicmp(spellName, STR("AccioFirebolt")) == 0 ||
            _wcsicmp(spellName, STR("AccioEclairDeFeu")) == 0)
        {
            Output::send<LogLevel::Default>(STR("[SpellCaster] {} - Summoning broomstick\n"), spellName);
            ActivateBroom();
            return true;
        }

        // ---- Accio Mount Spells: Summon/activate creature mount ----
        if (_wcsicmp(spellName, STR("AccioMount")) == 0 ||
            _wcsicmp(spellName, STR("AccioMonture")) == 0 ||
            _wcsicmp(spellName, STR("AccioHippogriff")) == 0 ||
            _wcsicmp(spellName, STR("AccioHippogriffe")) == 0 ||
            _wcsicmp(spellName, STR("AccioGraphorn")) == 0 ||
            _wcsicmp(spellName, STR("AccioSombral")) == 0 ||
            _wcsicmp(spellName, STR("AccioThestral")) == 0)
        {
            Output::send<LogLevel::Default>(STR("[SpellCaster] {} - Summoning mount\n"), spellName);
            ActivateMount();
            return true;
        }

        return false;  // Not a special spell, use normal casting
    }

    // ========================================================================
    // [8.7] UI System
    // ========================================================================

    /**
     * Initialize the UI system.
     * Finds UIManager instance and caches function pointers.
     * @return true if initialization succeeded
     */
    bool InitializeUISystem()
    {
        // UIManager can be recreated, always refresh instance
        m_uiManager = UObjectGlobals::FindFirstOf(STR("UIManager"));

        if (!m_uiManager)
        {
            Output::send<LogLevel::Warning>(STR("[SpellCaster] UIManager not found\n"));
            m_uiSystemReady = false;
            return false;
        }

        // Only find functions once
        if (!m_uiSystemReady)
        {
            m_funcToggleMenuTab = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.UIManager:ToggleMenuTab"));

            m_funcFieldGuideMenuStart = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.UIManager:FieldGuideMenuStart"));

            m_funcExitFieldGuide = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.UIManager:ExitFieldGuideWithReason"));

            // GetFieldGuideWidget returns the current field guide menu instance
            m_funcGetFieldGuideWidget = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.UIManager:GetFieldGuideWidget"));

            // ChangeMenuPage is on UFieldGuideMenu (native function)
            m_funcChangeMenuPage = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.FieldGuideMenu:ChangeMenuPage"));

            // SetActiveMenu to navigate to a specific menu
            m_funcSetActiveMenu = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.UIManager:SetActiveMenu"));

            // OnInputAction on UMGInputManager - simulates input
            m_funcOnInputAction = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/UMGFocus.UMGInputManager:OnInputAction"));

            // LoadFieldGuideScreen - pre-loads/creates the field guide widget
            m_funcLoadFieldGuideScreen = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.UIManager:LoadFieldGuideScreen"));

            // TogglePathActive - toggle objective path display (V key)
            m_funcTogglePathActive = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.UIManager:TogglePathActive"));

            // ShowPathSelection functions - triggered by V key press/release
            m_funcShowPathSelectionPressed = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.UIManager:ShowPathSelectionPressed"));

            m_funcShowPathSelectionReleased = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Phoenix.UIManager:ShowPathSelectionReleased"));

            // ActivateToolByName - for activating tools like broom
            m_funcActivateToolByName = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Script/Toolset.ToolSetComponent:ActivateToolByName"));

            if (!m_funcFieldGuideMenuStart)
            {
                Output::send<LogLevel::Error>(STR("[SpellCaster] FieldGuideMenuStart function not found\n"));
                return false;
            }

            Output::send<LogLevel::Default>(STR("[SpellCaster] UI functions found:\n"));
            Output::send<LogLevel::Default>(STR("  ToggleMenuTab: {}\n"), m_funcToggleMenuTab ? STR("YES") : STR("NO"));
            Output::send<LogLevel::Default>(STR("  FieldGuideMenuStart: {}\n"), m_funcFieldGuideMenuStart ? STR("YES") : STR("NO"));
            Output::send<LogLevel::Default>(STR("  ExitFieldGuideWithReason: {}\n"), m_funcExitFieldGuide ? STR("YES") : STR("NO"));
            Output::send<LogLevel::Default>(STR("  GetFieldGuideWidget: {}\n"), m_funcGetFieldGuideWidget ? STR("YES") : STR("NO"));
            Output::send<LogLevel::Default>(STR("  ChangeMenuPage: {}\n"), m_funcChangeMenuPage ? STR("YES") : STR("NO"));
            Output::send<LogLevel::Default>(STR("  SetActiveMenu: {}\n"), m_funcSetActiveMenu ? STR("YES") : STR("NO"));
            Output::send<LogLevel::Default>(STR("  OnInputAction: {}\n"), m_funcOnInputAction ? STR("YES") : STR("NO"));
            Output::send<LogLevel::Default>(STR("  LoadFieldGuideScreen: {}\n"), m_funcLoadFieldGuideScreen ? STR("YES") : STR("NO"));
            Output::send<LogLevel::Default>(STR("  TogglePathActive: {}\n"), m_funcTogglePathActive ? STR("YES") : STR("NO"));
            Output::send<LogLevel::Default>(STR("  ShowPathSelectionPressed: {}\n"), m_funcShowPathSelectionPressed ? STR("YES") : STR("NO"));
            Output::send<LogLevel::Default>(STR("  ShowPathSelectionReleased: {}\n"), m_funcShowPathSelectionReleased ? STR("YES") : STR("NO"));
            Output::send<LogLevel::Default>(STR("  ActivateToolByName: {}\n"), m_funcActivateToolByName ? STR("YES") : STR("NO"));

            m_uiSystemReady = true;
            Output::send<LogLevel::Default>(STR("[SpellCaster] UI system initialized\n"));
        }

        return true;
    }

    /**
     * Toggle a menu page using UIManager::ToggleMenuTab
     * @param page The menu page to toggle
     * @param shouldShow true to show, false to hide
     */
    void ToggleMenuPage(EPauseMenuPage page, bool shouldShow)
    {
        if (!m_uiManager || !m_funcToggleMenuTab)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] UI system not ready\n"));
            return;
        }

        // ToggleMenuTab(EPauseMenuPage MenuPage, bool ShouldShow)
        // Parameter layout: uint8 (1 byte) + bool (1 byte) = 2 bytes, padded to 4 or 8
        alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[8] = {};
        params[0] = static_cast<uint8_t>(page);
        params[1] = shouldShow ? 1 : 0;

        Output::send<LogLevel::Default>(STR("[SpellCaster] Calling ToggleMenuTab(page={}, show={})\n"),
            static_cast<int>(page), shouldShow);

        m_uiManager->ProcessEvent(m_funcToggleMenuTab, params);

        Output::send<LogLevel::Default>(STR("[SpellCaster] ToggleMenuTab called\n"));
    }

    /**
     * Open the Field Guide menu using FieldGuideMenuStart
     * @param season The season to use (affects menu visuals)
     */
    void OpenFieldGuideMenu(ESeasonEnum season = ESeasonEnum::SEASON_FALL)
    {
        if (!m_uiManager || !m_funcFieldGuideMenuStart)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] UI system not ready for FieldGuideMenuStart\n"));
            return;
        }

        // FieldGuideMenuStart(ESeasonEnum PrePauseSeason)
        // Parameter layout: uint8 (1 byte)
        alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[8] = {};
        params[0] = static_cast<uint8_t>(season);

        Output::send<LogLevel::Default>(STR("[SpellCaster] Calling FieldGuideMenuStart(season={})\n"),
            static_cast<int>(season));

        m_uiManager->ProcessEvent(m_funcFieldGuideMenuStart, params);

        Output::send<LogLevel::Default>(STR("[SpellCaster] FieldGuideMenuStart called\n"));
    }

    /**
     * Close the Field Guide menu
     */
    void CloseFieldGuideMenu()
    {
        if (!m_uiManager || !m_funcExitFieldGuide)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] UI system not ready for ExitFieldGuide\n"));
            return;
        }

        // ExitFieldGuideWithReason(EFieldGuideExitReasons Reason, bool SkipFadeScreen, int32 CharacterID, FString Filename, FString FastTravelName)
        // This is a complex signature - for now use a large buffer and set minimal params
        alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[128] = {};
        // Reason = 0, SkipFadeScreen = false, rest = defaults

        Output::send<LogLevel::Default>(STR("[SpellCaster] Calling ExitFieldGuideWithReason\n"));

        m_uiManager->ProcessEvent(m_funcExitFieldGuide, params);

        Output::send<LogLevel::Default>(STR("[SpellCaster] ExitFieldGuideWithReason called\n"));
    }

    /**
     * Show the objective path (V key functionality)
     * Simulates pressing and releasing the V key to show the path temporarily
     */
    void ToggleObjectivePath()
    {
        if (!InitializeUISystem())
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] Failed to initialize UI system\n"));
            return;
        }

        if (!m_funcShowPathSelectionPressed || !m_funcShowPathSelectionReleased)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] ShowPathSelection functions not found\n"));
            return;
        }

        // Simulate V key press and release
        alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[8] = {};

        Output::send<LogLevel::Default>(STR("[SpellCaster] Calling ShowPathSelectionPressed\n"));
        m_uiManager->ProcessEvent(m_funcShowPathSelectionPressed, params);

        Output::send<LogLevel::Default>(STR("[SpellCaster] Calling ShowPathSelectionReleased\n"));
        m_uiManager->ProcessEvent(m_funcShowPathSelectionReleased, params);

        Output::send<LogLevel::Default>(STR("[SpellCaster] Path display triggered\n"));
    }

    /**
     * Generic function to activate an inventory tool (broom or mount)
     * Uses Biped_Player::LoadInventoryItemByName
     * @param toolType The type of tool to search for (e.g., "Broom", "Mount")
     * @param secondaryKeyword Optional secondary keyword to filter (e.g., "Flying" for brooms)
     * @return true if successfully activated, false otherwise
     */
    bool ActivateInventoryTool(const wchar_t* toolType, const wchar_t* secondaryKeyword = nullptr)
    {
        // Get the player pawn
        UObject* player = UObjectGlobals::FindFirstOf(STR("BP_Biped_Player_C"));
        if (!player)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] Player not found - cannot activate {}\n"), toolType);
            return false;
        }

        // Look up LoadInventoryItemByName function on player
        UFunction* funcLoadInventoryItemByName = UObjectGlobals::StaticFindObject<UFunction*>(
            nullptr, nullptr, STR("/Script/Phoenix.Biped_Player:LoadInventoryItemByName"));

        if (funcLoadInventoryItemByName)
        {
            // Try LoadInventoryItemByName approach first
            Output::send<LogLevel::Default>(STR("[SpellCaster] Trying LoadInventoryItemByName approach\n"));

            // Find a tool record name
            std::vector<UObject*> allToolRecords;
            UObjectGlobals::FindAllOf(STR("InventoryItemToolRecord"), allToolRecords);

            const wchar_t* targetToolName = nullptr;
            for (auto* record : allToolRecords)
            {
                if (!record)
                    continue;

                auto recordName = record->GetName();
                if (recordName.find(toolType) != std::wstring::npos)
                {
                    if (secondaryKeyword)
                    {
                        if (recordName.find(secondaryKeyword) != std::wstring::npos)
                        {
                            targetToolName = recordName.c_str();
                            Output::send<LogLevel::Default>(STR("[SpellCaster] Found tool: {}\n"), targetToolName);
                            break;
                        }
                    }
                    else
                    {
                        targetToolName = recordName.c_str();
                        Output::send<LogLevel::Default>(STR("[SpellCaster] Found tool: {}\n"), targetToolName);
                        break;
                    }
                }
            }

            if (targetToolName)
            {
                // LoadInventoryItemByName(FName ItemName)
                alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[16] = {};
                auto fname = FName(targetToolName, FNAME_Add);
                std::memcpy(&params[0], &fname, sizeof(FName));

                Output::send<LogLevel::Default>(STR("[SpellCaster] Calling LoadInventoryItemByName('{}')\n"), targetToolName);
                player->ProcessEvent(funcLoadInventoryItemByName, params);
                Output::send<LogLevel::Default>(STR("[SpellCaster] LoadInventoryItemByName called\n"));
                return true;
            }
        }

        // Find the ToolSetComponent - try BOTH regular and Inventory
        UObject* toolSetComponent = nullptr;
        UObject* inventoryToolSetComponent = nullptr;
        std::vector<UObject*> allComponents;
        UObjectGlobals::FindAllOf(STR("ToolSetComponent"), allComponents);

        for (auto* component : allComponents)
        {
            if (component && component->GetOuterPrivate() == player)
            {
                auto compName = component->GetName();
                if (compName.find(STR("Inventory")) != std::wstring::npos)
                {
                    inventoryToolSetComponent = component;
                }
                else if (compName == STR("ToolSetComponent"))
                {
                    toolSetComponent = component;
                }
            }
        }

        // Try regular ToolSetComponent first (brooms/mounts might be here)
        if (!toolSetComponent)
        {
            toolSetComponent = inventoryToolSetComponent;
        }

        if (!toolSetComponent)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] No ToolSetComponent found on player\n"));
            return false;
        }

        Output::send<LogLevel::Default>(STR("[SpellCaster] Using ToolSetComponent: {}\n"), toolSetComponent->GetName());

        // Look up ActivateTool function
        UFunction* funcActivateTool = UObjectGlobals::StaticFindObject<UFunction*>(
            nullptr, nullptr, STR("/Script/Toolset.ToolSetComponent:ActivateTool"));

        if (!funcActivateTool)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] ActivateTool function not found\n"));
            return false;
        }

        // Find the tool record from all available records
        std::vector<UObject*> allToolRecords;
        UObjectGlobals::FindAllOf(STR("InventoryItemToolRecord"), allToolRecords);

        if (allToolRecords.empty())
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] No InventoryItemToolRecords found\n"));
            return false;
        }

        UObject* targetRecord = nullptr;
        for (auto* record : allToolRecords)
        {
            if (!record)
                continue;

            auto recordName = record->GetName();

            // Check for primary keyword
            if (recordName.find(toolType) != std::wstring::npos)
            {
                // If secondary keyword provided, check it too
                if (secondaryKeyword)
                {
                    if (recordName.find(secondaryKeyword) != std::wstring::npos)
                    {
                        targetRecord = record;
                        Output::send<LogLevel::Default>(STR("[SpellCaster] Found {} record: {}\n"), toolType, recordName);
                        break;
                    }
                }
                else
                {
                    targetRecord = record;
                    Output::send<LogLevel::Default>(STR("[SpellCaster] Found {} record: {}\n"), toolType, recordName);
                    break;
                }
            }
        }

        if (!targetRecord)
        {
            Output::send<LogLevel::Warning>(STR("[SpellCaster] No {} ToolRecord found in inventory - you may need to unlock/obtain it first\n"), toolType);
            return false;
        }

        // ActivateTool(UToolRecord* ToolRecord)
        alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[16] = {};
        std::memcpy(&params[0], &targetRecord, sizeof(void*));

        Output::send<LogLevel::Default>(STR("[SpellCaster] Calling ActivateTool with {} record\n"), toolType);

        toolSetComponent->ProcessEvent(funcActivateTool, params);

        Output::send<LogLevel::Default>(STR("[SpellCaster] {} activated successfully\n"), toolType);
        return true;
    }

    /**
     * Activate/summon the broom using UI_BP_GadgetWheel::ResolveSelectedMountBroom
     * This simulates clicking the broom button in the gadget wheel
     */
    void ActivateBroom()
    {
        Output::send<LogLevel::Default>(STR("[SpellCaster] Activating broom via GadgetWheel::ResolveSelectedMountBroom\n"));

        // Find the gadget wheel widget
        UObject* gadgetWheel = UObjectGlobals::FindFirstOf(STR("UI_BP_GadgetWheel_C"));
        if (!gadgetWheel)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] GadgetWheel widget not found (not opened yet?)\n"));
            Output::send<LogLevel::Warning>(STR("[SpellCaster] Trying fallback: FindSlottedBroomMountItem approach\n"));

            // Fallback: Use FindSlottedBroomMountItem to get the item name, then activate it
            UFunction* funcFind = UObjectGlobals::StaticFindObject<UFunction*>(
                nullptr, nullptr, STR("/Game/UI/Menus/GadgetWheel/UI_BP_GadgetWheel.UI_BP_GadgetWheel_C:FindSlottedBroomMountItem"));

            if (funcFind)
            {
                // This approach needs the widget instance, so still need gadget wheel
                Output::send<LogLevel::Error>(STR("[SpellCaster] Cannot use fallback without widget instance\n"));
            }

            Output::send<LogLevel::Warning>(STR("[SpellCaster] Try opening gadget wheel first, or use alternative approach\n"));
            return;
        }

        Output::send<LogLevel::Default>(STR("[SpellCaster] Found GadgetWheel: {}\n"), gadgetWheel->GetFullName());

        // First, try to populate the item arrays to ensure slots are filled
        UFunction* funcPopulate = UObjectGlobals::StaticFindObject<UFunction*>(
            nullptr, nullptr, STR("/Game/UI/GadgetWheel/UI_BP_GadgetWheel.UI_BP_GadgetWheel_C:Populate Item Arrays"));

        if (funcPopulate)
        {
            Output::send<LogLevel::Default>(STR("[SpellCaster] Calling Populate Item Arrays...\n"));
            gadgetWheel->ProcessEvent(funcPopulate, nullptr);
        }

        // Also try SetStandaloneButtonData
        UFunction* funcSetButton = UObjectGlobals::StaticFindObject<UFunction*>(
            nullptr, nullptr, STR("/Game/UI/GadgetWheel/UI_BP_GadgetWheel.UI_BP_GadgetWheel_C:SetStandaloneButtonData"));

        if (funcSetButton)
        {
            Output::send<LogLevel::Default>(STR("[SpellCaster] Calling SetStandaloneButtonData...\n"));
            gadgetWheel->ProcessEvent(funcSetButton, nullptr);
        }

        // Find the ResolveSelectedMountBroom function using StaticFindObject with full path
        // Path discovered via UE4SS Live View: /Game/UI/GadgetWheel/UI_BP_GadgetWheel.UI_BP_GadgetWheel_C:ResolveSelectedMountBroom
        UFunction* funcResolve = UObjectGlobals::StaticFindObject<UFunction*>(
            nullptr, nullptr, STR("/Game/UI/GadgetWheel/UI_BP_GadgetWheel.UI_BP_GadgetWheel_C:ResolveSelectedMountBroom"));

        if (!funcResolve)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] ResolveSelectedMountBroom function not found via StaticFindObject\n"));
            return;
        }

        Output::send<LogLevel::Default>(STR("[SpellCaster] Found ResolveSelectedMountBroom function: {}\n"), funcResolve->GetFullName());

        // Get the FlyingMountButton from the widget (at offset 0x0398 according to header dump)
        // This is hacky but necessary since we can't access fields directly
        UObject** flyingMountButtonPtr = reinterpret_cast<UObject**>(reinterpret_cast<uint8_t*>(gadgetWheel) + 0x0398);
        UObject* flyingMountButton = *flyingMountButtonPtr;

        if (!flyingMountButton)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] FlyingMountButton not found in widget\n"));
            return;
        }

        Output::send<LogLevel::Default>(STR("[SpellCaster] Found FlyingMountButton: {}\n"), flyingMountButton->GetFullName());

        // Try to simulate clicking the button by calling its OnButtonClicked event handler
        UFunction* funcButtonClick = UObjectGlobals::StaticFindObject<UFunction*>(
            nullptr, nullptr, STR("/Game/UI/GadgetWheel/UI_BP_GadgetWheel_StandaloneItem.UI_BP_GadgetWheel_StandaloneItem_C:BndEvt__UI_BP_GadgetWheel_StandaloneItem_BoundingBox_K2Node_ComponentBoundEvent_2_OnButtonClickedEvent__DelegateSignature"));

        if (funcButtonClick)
        {
            Output::send<LogLevel::Default>(STR("[SpellCaster] Simulating button click...\n"));
            flyingMountButton->ProcessEvent(funcButtonClick, nullptr);
            Output::send<LogLevel::Default>(STR("[SpellCaster] Button click event triggered!\n"));
            return; // Exit early - button click should handle everything
        }
        else
        {
            Output::send<LogLevel::Warning>(STR("[SpellCaster] Button click function not found, falling back to ResolveSelectedMountBroom\n"));
        }

        // Fallback: Call ResolveSelectedMountBroom(SelectedItem, HolderID, Success)
        // Parameters: UUI_BP_GadgetWheel_StandaloneItem_C* SelectedItem, FName HolderID, bool& Success
        struct Params
        {
            UObject* SelectedItem;
            FName HolderID;
            bool Success;
        };

        Params params;
        params.SelectedItem = flyingMountButton;
        params.HolderID = FName(STR("Holder_FlyingMount"), FNAME_Add);
        params.Success = false;

        Output::send<LogLevel::Default>(STR("[SpellCaster] Calling ResolveSelectedMountBroom...\n"));
        gadgetWheel->ProcessEvent(funcResolve, &params);

        if (params.Success)
        {
            Output::send<LogLevel::Default>(STR("[SpellCaster] ResolveSelectedMountBroom returned Success=true\n"));

            // Check if BroomItemTool is now available
            UObject* broomTool = UObjectGlobals::FindFirstOf(STR("BroomItemTool"));
            if (broomTool)
            {
                Output::send<LogLevel::Default>(STR("[SpellCaster] BroomItemTool found after call: {}\n"), broomTool->GetFullName());

                // Try calling SpawnAndMountBroom on it
                UFunction* funcSpawn = UObjectGlobals::StaticFindObject<UFunction*>(
                    nullptr, nullptr, STR("/Script/Phoenix.BroomItemTool:SpawnAndMountBroom"));

                if (funcSpawn)
                {
                    Output::send<LogLevel::Default>(STR("[SpellCaster] Calling SpawnAndMountBroom on BroomItemTool...\n"));
                    broomTool->ProcessEvent(funcSpawn, nullptr);
                    Output::send<LogLevel::Default>(STR("[SpellCaster] SpawnAndMountBroom called!\n"));
                }
            }
            else
            {
                Output::send<LogLevel::Warning>(STR("[SpellCaster] BroomItemTool still not found after ResolveSelectedMountBroom\n"));

                // Try to find the player and check CurrentTool
                UObject* player = UObjectGlobals::FindFirstOf(STR("BP_Biped_Player_C"));
                if (player)
                {
                    Output::send<LogLevel::Default>(STR("[SpellCaster] Checking player's tool status...\n"));

                    // Try to find ToolSetComponent
                    UObject* toolSet = UObjectGlobals::FindFirstOf(STR("ToolSetComponent"));
                    if (toolSet)
                    {
                        Output::send<LogLevel::Default>(STR("[SpellCaster] ToolSetComponent found: {}\n"), toolSet->GetFullName());

                        // Try calling ActivateTool on ToolSetComponent
                        UFunction* funcActivate = UObjectGlobals::StaticFindObject<UFunction*>(
                            nullptr, nullptr, STR("/Script/Phoenix.ToolSetComponent:ActivateTool"));

                        if (funcActivate)
                        {
                            Output::send<LogLevel::Default>(STR("[SpellCaster] Calling ToolSetComponent::ActivateTool...\n"));
                            toolSet->ProcessEvent(funcActivate, nullptr);
                            Output::send<LogLevel::Default>(STR("[SpellCaster] ActivateTool called!\n"));
                        }
                    }
                }
            }
        }
        else
        {
            Output::send<LogLevel::Warning>(STR("[SpellCaster] ResolveSelectedMountBroom returned Success=false (broom locked or unavailable?)\n"));
        }
    }

    /**
     * Activate/summon a creature mount (Hippogriff, Graphorn, Thestral, etc.)
     */
    void ActivateMount()
    {
        // Try to find any mount - prefer Hippogriff first, then others
        if (!ActivateInventoryTool(STR("Hippogriff"), STR("Mount")))
        {
            if (!ActivateInventoryTool(STR("Graphorn"), STR("Mount")))
            {
                if (!ActivateInventoryTool(STR("Thestral"), STR("Mount")))
                {
                    Output::send<LogLevel::Warning>(STR("[SpellCaster] No mount found - you may need to unlock/tame a creature first\n"));
                }
            }
        }
    }

    /**
     * Get the current field guide widget via UIManager::GetFieldGuideWidget
     * @return The field guide widget or nullptr if not available
     */
    UObject* GetFieldGuideWidget()
    {
        if (!m_uiManager)
            return nullptr;

        // Try function call first
        if (m_funcGetFieldGuideWidget)
        {
            alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[16] = {};
            m_uiManager->ProcessEvent(m_funcGetFieldGuideWidget, params);
            UObject* result = *reinterpret_cast<UObject**>(&params[0]);
            if (result)
                return result;
        }

        // Fallback: Direct memory access at offset 0x0498 (from Phoenix.hpp)
        // class UFieldGuideMenu* FieldGuideWidget; // 0x0498 (size: 0x8)
        constexpr size_t FIELD_GUIDE_WIDGET_OFFSET = 0x0498;
        UObject* directAccess = *reinterpret_cast<UObject**>(
            reinterpret_cast<uint8_t*>(m_uiManager) + FIELD_GUIDE_WIDGET_OFFSET);

        if (directAccess)
        {
            Output::send<LogLevel::Default>(STR("[SpellCaster] Got FieldGuideWidget via direct memory access at offset 0x0498\n"));
        }

        return directAccess;
    }

    /**
     * Navigate to a specific menu page using ChangeMenuPage on the field guide widget.
     * This should be called after the menu is open.
     * @param action The EUMGInputAction for the target page (e.g., UMGInventoryScreenToggle)
     */
    void NavigateToMenuPage(EUMGInputAction action)
    {
        if (!m_funcChangeMenuPage)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] ChangeMenuPage function not found\n"));
            return;
        }

        // Get the field guide widget via UIManager
        UObject* fieldGuideWidget = GetFieldGuideWidget();

        // Fallback: try to find it directly with various class names
        if (!fieldGuideWidget)
        {
            Output::send<LogLevel::Warning>(STR("[SpellCaster] GetFieldGuideWidget returned null, trying FindFirstOf...\n"));

            // Try various possible class names
            const wchar_t* classNames[] = {
                STR("UI_BP_FieldGuide_C"),
                STR("FieldGuideMenu"),
                STR("UFieldGuideMenu"),
                STR("UI_BP_FieldGuide"),
            };

            for (const auto& className : classNames)
            {
                fieldGuideWidget = UObjectGlobals::FindFirstOf(className);
                if (fieldGuideWidget)
                {
                    Output::send<LogLevel::Default>(STR("[SpellCaster] Found widget via FindFirstOf('{}')\n"), className);
                    break;
                }
            }
        }

        if (!fieldGuideWidget)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] Field guide widget not found with any class name\n"));

            // Debug: Try to find Screen objects to see what's available
            UObject* screen = UObjectGlobals::FindFirstOf(STR("Screen"));
            if (screen)
            {
                Output::send<LogLevel::Default>(STR("[SpellCaster] DEBUG: Found a Screen object at {}\n"), (void*)screen);
                if (screen->GetClassPrivate())
                {
                    Output::send<LogLevel::Default>(STR("[SpellCaster] DEBUG: Screen class name: {}\n"),
                        screen->GetClassPrivate()->GetName().c_str());
                }
            }
            return;
        }

        Output::send<LogLevel::Default>(STR("[SpellCaster] Found field guide widget at {}, calling ChangeMenuPage({})\n"),
            (void*)fieldGuideWidget, static_cast<int>(action));

        // ChangeMenuPage(EUMGInputAction MenuPage)
        // Parameter layout: uint8 (1 byte)
        alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[8] = {};
        params[0] = static_cast<uint8_t>(action);

        fieldGuideWidget->ProcessEvent(m_funcChangeMenuPage, params);

        Output::send<LogLevel::Default>(STR("[SpellCaster] ChangeMenuPage called\n"));
    }

    /**
     * Navigate to a specific menu using SetActiveMenu on UIManager.
     * @param menuName The menu name (e.g., "Inventory", "Talents")
     * @param loadToSubPage Whether to load to a sub-page
     * @param lockToMenu Whether to lock to this menu
     */
    void SetActiveMenu(const wchar_t* menuName, bool loadToSubPage = false, bool lockToMenu = false)
    {
        if (!m_uiManager || !m_funcSetActiveMenu)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] SetActiveMenu not available\n"));
            return;
        }

        Output::send<LogLevel::Default>(STR("[SpellCaster] Calling SetActiveMenu('{}', {}, {})\n"),
            menuName, loadToSubPage, lockToMenu);

        // SetActiveMenu(FName MenuToLoad, bool LoadToSubPage, bool LockToMenu)
        // FName is 8 bytes, bools are 1 byte each
        alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[16] = {};
        *reinterpret_cast<FName*>(&params[0]) = FName(menuName, FNAME_Add);
        params[8] = loadToSubPage ? 1 : 0;
        params[9] = lockToMenu ? 1 : 0;

        m_uiManager->ProcessEvent(m_funcSetActiveMenu, params);

        Output::send<LogLevel::Default>(STR("[SpellCaster] SetActiveMenu called\n"));
    }

    /**
     * Simulate an input action via UMGInputManager.
     * @param action The EUMGInputAction to simulate
     * @param pressed true for key press, false for release
     */
    void SimulateInputAction(EUMGInputAction action, bool pressed = true)
    {
        if (!m_funcOnInputAction)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] OnInputAction function not found\n"));
            return;
        }

        // Find UMGInputManager
        UObject* inputManager = UObjectGlobals::FindFirstOf(STR("UMGInputManager"));
        if (!inputManager)
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] UMGInputManager not found\n"));
            return;
        }

        Output::send<LogLevel::Default>(STR("[SpellCaster] Calling OnInputAction({}, {})\n"),
            static_cast<int>(action), pressed ? STR("Pressed") : STR("Released"));

        // OnInputAction(EUMGInputAction InputAction, TEnumAsByte<EInputEvent> InputEvent)
        // EInputEvent: IE_Pressed = 0, IE_Released = 1
        alignas(Config::PARAM_BUFFER_ALIGNMENT) uint8_t params[8] = {};
        params[0] = static_cast<uint8_t>(action);
        params[1] = pressed ? 0 : 1;  // IE_Pressed = 0, IE_Released = 1

        inputManager->ProcessEvent(m_funcOnInputAction, params);

        Output::send<LogLevel::Default>(STR("[SpellCaster] OnInputAction called\n"));
    }

    /**
     * Convert EUMGInputAction to page name string
     */
    const wchar_t* GetPageNameFromAction(EUMGInputAction action)
    {
        switch (action)
        {
            case EUMGInputAction::UMGInventoryScreenToggle: return STR("Inventory");
            case EUMGInputAction::UMGCharacterScreenToggle: return STR("Character");
            case EUMGInputAction::UMGTalentsScreenToggle: return STR("Talents");
            case EUMGInputAction::UMGQuestScreenToggle: return STR("MissionLog");      // FIXED: was "Quests"
            case EUMGInputAction::UMGMapScreenToggle: return STR("Map");
            case EUMGInputAction::UMGOwlMailScreenToggle: return STR("OwlMail");
            case EUMGInputAction::UMGCompendiumScreenToggle: return STR("Compendium"); // FIXED: was "Collections"
            case EUMGInputAction::UMGChallengeScreenToggle: return STR("Studies");     // FIXED: was "Challenges"
            case EUMGInputAction::UMGSettingsScreenToggle: return STR("Settings");
            default: return STR("Inventory");
        }
    }

    /**
     * Smart menu navigation: Opens menu to page if closed, or navigates if already open.
     * This is the recommended function to use for menu spells.
     *
     * @param targetPage The EUMGInputAction for the target page
     */
    void OpenOrNavigateToPage(EUMGInputAction targetPage)
    {
        if (!InitializeUISystem())
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] Failed to initialize UI system\n"));
            return;
        }

        // Check if menu is already open
        UObject* fieldGuideWidget = GetFieldGuideWidget();

        if (fieldGuideWidget)
        {
            // Menu is open - simulate key press (like pressing Y/U/I/O/P)
            // This is more reliable than calling ChangeMenuPage directly
            const wchar_t* pageName = GetPageNameFromAction(targetPage);
            Output::send<LogLevel::Default>(STR("[SpellCaster] Menu already open - simulating input for '{}'\n"), pageName);
            SimulateInputAction(targetPage, true);   // Key press
            SimulateInputAction(targetPage, false);  // Key release
        }
        else
        {
            // Menu is closed - open it to the specified page
            const wchar_t* pageName = GetPageNameFromAction(targetPage);
            Output::send<LogLevel::Default>(STR("[SpellCaster] Menu closed - opening to '{}'\n"), pageName);
            OpenFieldGuideToPage(targetPage);
        }
    }

    /**
     * Open the Field Guide menu directly to a specific page.
     * Uses ProcessEvent hook to intercept and override FieldGuide_StartOnPage parameter.
     *
     * @param targetPage The EUMGInputAction for the target page (e.g., UMGInventoryScreenToggle)
     */
    void OpenFieldGuideToPage(EUMGInputAction targetPage)
    {
        if (!InitializeUISystem())
        {
            Output::send<LogLevel::Error>(STR("[SpellCaster] Failed to initialize UI system\n"));
            return;
        }

        const wchar_t* pageName = GetPageNameFromAction(targetPage);
        Output::send<LogLevel::Default>(STR("[SpellCaster] Opening Field Guide to page '{}'\n"), pageName);

        // Step 1: Set up page override for the hook
        m_overridePageEnabled = true;
        m_overridePageName = pageName;
        Output::send<LogLevel::Default>(STR("[SpellCaster] Step 1 - Page override armed for '{}'\n"), pageName);

        // Step 2: Open the Field Guide menu (hook will intercept FieldGuide_StartOnPage)
        Output::send<LogLevel::Default>(STR("[SpellCaster] Step 2 - Calling FieldGuideMenuStart\n"));
        OpenFieldGuideMenu(ESeasonEnum::SEASON_FALL);

        Output::send<LogLevel::Default>(STR("[SpellCaster] OpenFieldGuideToPage complete\n"));
    }

    // ========================================================================
    // [8.8] Lifecycle Hooks
    // ========================================================================

    // Static pointer for hook callback access
    static SpellCasterMod* s_instance;

    /**
     * ProcessEvent pre-callback. Its only job is to redirect FieldGuide_StartOnPage
     * to the page requested by a menu spell (see OpenFieldGuideToPage), one shot.
     */
    static void ProcessEventHook(UObject* Context, UFunction* Function, void* Parms)
    {
        if (!s_instance || !Context || !Function || !Parms)
            return;

        if (!s_instance->m_overridePageEnabled || s_instance->m_overridePageName.empty())
            return;

        if (Function->GetName() != STR("FieldGuide_StartOnPage"))
            return;

        // FieldGuide_StartOnPage(FString pageName): the FString is the first parameter
        FString* pageNameParam = reinterpret_cast<FString*>(Parms);
        Output::send<LogLevel::Default>(STR("[SpellCaster] FieldGuide_StartOnPage('{}') redirected to '{}'\n"),
            pageNameParam->GetCharArray(), s_instance->m_overridePageName.c_str());

        *pageNameParam = FString(s_instance->m_overridePageName.c_str());

        s_instance->m_overridePageEnabled = false;
        s_instance->m_overridePageName.clear();
    }

public:
    void on_unreal_init() override
    {
        Output::send<LogLevel::Default>(STR("[SpellCaster] Initializing...\n"));
        s_instance = this;
        InitializeSharedMemory();

        // Needed by the menu spells: lets OpenFieldGuideToPage choose the page the Field Guide opens on
        Unreal::Hook::RegisterProcessEventPreCallback(&ProcessEventHook);
        Output::send<LogLevel::Default>(STR("[SpellCaster] Ready\n"));
    }

    void on_update() override
    {
        ReadSpellCommands();
        ProcessSpellQueue();
    }
};

// Static instance for hook callback
SpellCasterMod* SpellCasterMod::s_instance = nullptr;

// ============================================================================
// [9] DLL Entry Points
// ============================================================================

#define SPELL_CASTER_MOD_API __declspec(dllexport)

extern "C"
{
    SPELL_CASTER_MOD_API CppUserModBase* start_mod()
    {
        return new SpellCasterMod();
    }

    SPELL_CASTER_MOD_API void uninstall_mod(CppUserModBase* mod)
    {
        delete mod;
    }
}
