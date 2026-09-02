# UEVR SpellCaster plugin

A [UEVR](https://github.com/praydog/UEVR) plugin for playing Hogwarts Legacy in
VR with voice-cast spells. It is the VR counterpart of the virtual-controller
path: the desktop application publishes button sequences into shared memory,
and this plugin merges them into the XInput state UEVR hands to the game, on
top of the physical controller.

With the UE4SS mod in place this plugin is optional even in VR, since the mod
casts spells directly. The plugin remains useful for actions that are still
mapped to controller buttons.

## Protocol

Commands are 16-byte `VRCommand` records read from a lock-free ring buffer in
shared memory. The layout, command types and liveness rules are documented in
[docs/ipc-protocol.md](../docs/ipc-protocol.md). The plugin keeps its own copy
of the structure in `UEVRSpellCasterPluginSharedMemory.cpp`; it must stay
identical to `app/Whisper-Spell/SharedMemoryStructure.h`.

## Building

Open `UEVRSpellCasterPlugin.sln` in Visual Studio 2022 and build
**Release | x64**. The UEVR API headers are vendored under
`UEVRSpellCasterPlugin/uevr/`, so there is nothing else to install. The output
is `UEVRSpellCasterPlugin.dll`.

## Installing

When VR output is selected, the desktop application copies
`UEVR Plugin\UEVRSpellCasterPlugin.dll` from its own folder into the game's
UEVR profile and refreshes it when the file version increases (see
`Keybinder.cpp`). To install by hand, copy the DLL into the `plugins` folder of
the game's UEVR profile under `%APPDATA%\UnrealVRMod\`. Start the game through
UEVR, then start the desktop application. The UEVR console logs the connection
state and every command received.

Antivirus software sometimes flags freshly built UEVR plugins because they hook
into the game process; add an exclusion for the plugins folder if that happens.
