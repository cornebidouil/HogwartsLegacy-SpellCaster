# SpellCaster UE4SS mod

A C++ mod for [UE4SS](https://github.com/UE4SS-RE/RE-UE4SS) that casts Hogwarts
Legacy spells on request. It reads spell names from a shared-memory queue filled
by the desktop application (or by any other program, see `tools/spell_sender.py`)
and casts them through the game's own `WandTool`. Because it drives the game
logic directly, it does not depend on the player's spell bar, controller layout
or key bindings.

The mod is installed into the game as `Mods/SpellCaster/dlls/main.dll`, registers
with UE4SS as `SpellCaster`, and prefixes its log lines with `[SpellCaster]`.
It has no key bindings of its own: everything is driven through the shared
memory channel.

## What it does

| Feature | How |
|---------|-----|
| Cast any known spell | Looks up the `SpellToolRecord` asset, calls `ActivateSpellTool` then `CastActiveSpell` on the player's `WandTool` |
| Unlock status | Not enforced today: the query through `SpellManagerBPInterface::IsUnlocked` is implemented but disabled before casting (see [docs/known-issues.md](../docs/known-issues.md)) |
| Retry when the wand is busy | Commands whose spell tool is not yet available stay queued and are retried on the next frame |
| Toggle Lumos / Nox | `Lumos` is ignored while active; `Nox` cancels the active Lumos |
| `Finite` | Cancels the active spell (including Disillusionment) and closes the Field Guide |
| Open menus by voice | `Apperta*` spells open the Field Guide directly on a page (see table below) |
| Show the objective path | `AppareVestigium` |
| Summon the broom or a mount | `AccioBroomstick` / `AccioBalais` / `AccioFirebolt` / `AccioEclairDeFeu`, and `AccioMount` / `AccioMonture` / `AccioHippogriff` / `AccioHippogriffe` / `AccioGraphorn` / `AccioThestral` / `AccioSombral` |

### Menu spells

| Spell name | Field Guide page |
|------------|------------------|
| `AppertaCodex`, `AppertaSacculus` | Inventory |
| `AppertaMeritas`, `AppertaVestiarium` | Character |
| `AppertaFalcultates` | Talents |
| `AppertaQuaestiones` | Quests (`MissionLog`) |
| `AppertaMappa` | Map |
| `AppertaLiterae` | Owl mail |
| `AppertaCompendium` | Collections (`Compendium`) |
| `AppertaIncantatem` | Challenges (`Studies`) |
| `AppertaConfiguratio` | Settings |
| `Finite` | Closes the menu |

### Spell database

`SpellDatabase::SPELLS` in `SpellCasterMod/dllmain.cpp` lists every castable
spell with its display name, the path of its `SpellToolRecord` asset, the lock
name used for the unlock check, and whether it needs a target. It covers the
base game spells and a number of spells added by community mods (SpellsEnhanced,
HRBSpellPack, HermitHollow, WFM_SpellMod_01, SpM_Test-01). Spells from a mod that
is not installed are reported as unknown and discarded.

Adding a spell is a one-line change: append a row to that table. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for how to find the asset path and lock name.

## Talking to the mod

Commands arrive through a Windows file mapping named `SpellCasterSharedMemory`.
The layout, the producer and consumer rules and the name normalisation are
documented in [docs/ipc-protocol.md](../docs/ipc-protocol.md).

`tools/spell_sender.py` is a reference producer written in Python. It needs no
microphone and no build of the desktop application, which makes it the quickest
way to test the mod:

```powershell
python tools\spell_sender.py Lumos            # cast one spell
python tools\spell_sender.py Lumos Incendio   # cast a sequence
python tools\spell_sender.py                  # interactive prompt, type "list" for the spell names
```

The game must be running with a save loaded; the mod creates the shared memory
once Unreal has initialised.

## Diagnostics

Every command the mod receives is logged to the UE4SS console and to
`UE4SS_Logs\UE4SS.log` with its outcome: cast, not unlocked, unknown spell,
busy (retried on the next frame) or failed. `tools\spell_sender.py` shows the
producer and consumer heartbeats of the shared memory, which tells you whether
the mod is running before you look at the log.

## Building

Prerequisites: Visual Studio 2022 with the C++ workload, CMake 3.22 or newer, Git.

1. Clone RE-UE4SS, with its submodules, next to this file. The mod was last built
   against commit `c1d91cf1c344dd0a20557a0544c2c3014ee525ee` (6 January 2026):

   ```powershell
   git clone --recursive https://github.com/UE4SS-RE/RE-UE4SS.git RE-UE4SS
   git -C RE-UE4SS checkout c1d91cf1c344dd0a20557a0544c2c3014ee525ee
   git -C RE-UE4SS submodule update --init --recursive
   ```

2. Build. The script configures CMake, builds the `SpellCasterMod` target in the
   `Game__Shipping__Win64` configuration and, if you pass the game folder, deploys
   the result:

   ```powershell
   .\build.ps1
   .\build.ps1 -GameDir "D:\SteamLibrary\steamapps\common\Hogwarts Legacy\Phoenix\Binaries\Win64"
   ```

   The same thing by hand:

   ```powershell
   cmake -S . -B build -G "Visual Studio 17 2022"
   cmake --build build --config Game__Shipping__Win64 --target SpellCasterMod
   ```

   The first configuration downloads and builds UE4SS itself, which takes a while.
   `build\Game__Shipping__Win64\bin\UE4SS.dll` from that build is the UE4SS
   binary bundled with the application releases.

3. Install manually if you did not use `-GameDir`. The expected layout under
   `Hogwarts Legacy\Phoenix\Binaries\Win64` is:

   ```
   UE4SS.dll
   dwmapi.dll
   UE4SS-settings.ini
   Mods\
     mods.txt
     Keybinds\Scripts\main.lua
     SpellCaster\
       enabled.txt
       dlls\main.dll        <- SpellCasterMod.dll renamed
   ```

   The desktop application performs this installation automatically at start-up
   from its `UE4SS Plugin` folder.

Debug builds of UE4SS mods are known to be unstable; build `Game__Shipping__Win64`
unless you are debugging UE4SS itself.

## Extending

- **New spell**: add a row to `SpellDatabase::SPELLS`.
- **New action that is not a spell** (a menu, a tool, a toggle): add a branch to
  `TryHandleSpecialSpell`. It runs before the normal casting path and returns
  `true` when it consumed the command.
- **New Field Guide page**: add the `EUMGInputAction` value and its page name in
  `GetPageNameFromAction`.
