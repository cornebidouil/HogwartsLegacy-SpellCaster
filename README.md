# Hogwarts Legacy SpellCaster

Cast spells in Hogwarts Legacy with your voice. Say *"Incendio"* and the spell
is cast, whatever is on your spell bar.

SpellCaster listens to your microphone, recognises the incantation with a
speech model fine-tuned on the spell vocabulary, and casts the spell inside the
game through a [UE4SS](https://github.com/UE4SS-RE/RE-UE4SS) mod. Everything
runs locally on your machine.

*Version française : [README-FR.md](README-FR.md).*

## How it works

```
 microphone ─► desktop application ─► shared memory ─► UE4SS mod ─► game
              (VAD + Whisper or Moonshine)             (casts via WandTool)
```

The desktop application does the listening and the recognition. It forwards
each recognised spell name to the mod, which casts it through the game's own
spell system. Because the mod drives
the game logic directly, there is no key binding or controller layout to
configure.

Two older output paths are still available for setups without the mod: a
virtual Xbox controller (ViGEm) that presses the buttons of your spell bar, and
a [UEVR](https://github.com/praydog/UEVR) plugin for VR play.

## For players

1. Download the latest `HogwartsLegacy-SpellCaster-win-x64-<version>.zip` from
   the [Releases](https://github.com/pierre-cheneau/HogwartsLegacy-SpellCaster/releases)
   page and unzip it anywhere.
2. Run `HogwartsLegacy-SpellCaster.exe`. On first start it asks which microphone
   to use, whether you want to contribute recordings to improve the models, and
   installs UE4SS and the SpellCaster mod into your game folder. Steam
   installations are found automatically; otherwise set `game_path` in
   `config.ini` to your `Hogwarts Legacy\Phoenix\Binaries\Win64` folder.
3. Start the game, load a save, and speak.

Requirements: Windows 10 or 11, 64-bit, a microphone. Recognition runs on the
CPU by default; a Vulkan-capable GPU speeds up the Whisper engine. Choose the
engine with `engine=whisper` or `engine=moonshine` in `config.ini`.

The `spellbook` folder in the release shows every recognised phrase with its
pronunciation. If something does not work, the console output of the
application and `UE4SS_Logs\UE4SS.log` in the game folder tell most of the
story; bring them to the Discord linked below.

## Spells

Everything the mod can cast is listed in `ue4ss-mod/SpellCasterMod/dllmain.cpp`.
In short:

| Group | Spells |
|-------|--------|
| Control | Accio, Levioso, Depulso, Descendo, Flipendo, Glacius, Arresto Momentum |
| Damage | Incendio, Confringo, Diffindo, Bombarda |
| Combat | Stupefy, Expelliarmus, Protego, Oppugno |
| Utility | Lumos, Nox, Reparo, Revelio, Invisica (Disillusionment), Wingardium Leviosa |
| Transfiguration | Transfigura Verto, Conjuration, Vanishment |
| Unforgivable | Avada Kedavra, Crucio, Imperio |
| Special | Smash (ancient magic), Stealth Takedown, Confundo, Episkey |
| Actions | Finite, Appare Vestigium, Accio Broomstick / Balais, Accio Mount / Hippogriff / Graphorn / Thestral |
| Menus | Apperta Codex, Meritas, Falcultates, Quaestiones, Mappa, Literae, Compendium, Incantatem, Configuratio |
| Community spell packs | SpellsEnhanced, HRBSpellPack, HermitHollow and others, when installed |

## Repository layout

| Folder | Content |
|--------|---------|
| `app/` | Desktop application (Visual Studio 2022, C++20): audio capture, VAD, Whisper and Moonshine engines, UE4SS installer, crowdsourcing client |
| `ue4ss-mod/` | UE4SS C++ mod that casts the spells, plus a Python tool to drive it without a microphone |
| `uevr-plugin/` | UEVR plugin for the VR controller path |
| `docs/` | Inter-process protocol, audio debug mode, known issues |

- Building any of the three: [BUILDING.md](BUILDING.md)
- Adding spells or otherwise contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- How the application talks to the mod: [docs/ipc-protocol.md](docs/ipc-protocol.md)
- What is known to need work: [docs/known-issues.md](docs/known-issues.md)
- Licences of everything bundled: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)

## Releases

| Version | Date | Highlights |
|---------|------|-----------|
| 1.8.0 beta | 2026-01-26 | Moonshine engine, UE4SS mod with direct casting, crowdsourcing |
| 1.7.0 | 2025-09-05 | VR support through UEVR |
| 1.6.x | 2025-06 to 2025-08 | Audio device selection, resampling |
| 1.3 to 1.5 | 2025-02 to 2025-03 | Configuration file, spellbook, Vulkan and OpenVINO backends |
| 1.1.0 | 2024-09-09 | Gamepad support, all graphics cards |
| 1.0.0 | 2024-08-22 | First release |

## Community and support

Created by **Cornebidouil**. Questions, feedback and recordings for the models
are all welcome:

- Discord: <https://discord.gg/zE4NRsTGdw>
- Training portal (help improve recognition): <http://hogwartslegacyspellcaster.xyz>
- GitHub: <https://github.com/pierre-cheneau>

If you enjoy the project and want to support its development:
ETH `0x1F61fa7923d5E914A5Fdf36B584a1336fde20721`

## Licence

MIT, see [LICENSE](LICENSE). Hogwarts Legacy is a trademark of Warner Bros.
Entertainment Inc.; this is an independent fan project.
