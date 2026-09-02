# Contributing

Thank you for looking into this. The project is small enough that most changes
are one file, and the two halves (speech recognition on one side, game control
on the other) can be worked on independently.

## Where things live

| I want to... | Look at |
|--------------|---------|
| Add or fix a spell the mod can cast | `ue4ss-mod/SpellCasterMod/dllmain.cpp`, table `SpellDatabase::SPELLS` |
| Add an action that is not a spell (menu, mount, toggle) | same file, `TryHandleSpecialSpell` |
| Change how spell names travel between processes | `docs/ipc-protocol.md`, then both sides listed there |
| Change audio capture, VAD or the recognition engines | `app/Whisper-Spell/Audio.h`, `SileroVAD.h`, `WhisperTranscriber.*`, `MoonshineTranscriber.h` |
| Change how a transcription becomes a spell name | `app/Whisper-Spell/Keybinder.cpp`, `SpellTransmitter.cpp` |
| Change the UE4SS auto-installer | `app/Whisper-Spell/UE4SSInstaller.cpp` |
| Change the VR path | `app/Whisper-Spell/VRSharedMemoryServer.cpp`, `uevr-plugin/` |
| Fix something from the known list | `docs/known-issues.md` |

## Adding a spell to the mod

1. Find the spell's `SpellToolRecord` asset path and its lock name. With UE4SS
   installed, open the Live View (enable `GuiConsoleEnabled` in
   `UE4SS-settings.ini`), search for `SpellRecord` and copy the full object
   path, for example
   `/Game/Gameplay/ToolSet/Spells/Glacius/DA_GlaciusSpellRecord.DA_GlaciusSpellRecord`.
   Lock names follow the pattern `Spell_<Name>`. The unlock check is currently
   disabled before casting (see `docs/known-issues.md`), so a wrong lock name
   has no visible effect today; fill it in anyway so the check can be turned
   back on without revisiting every row.
2. Append a row to `SpellDatabase::SPELLS`:

   ```cpp
   { STR("Glacius"), STR("/Game/.../DA_GlaciusSpellRecord.DA_GlaciusSpellRecord"), STR("Spell_Glacius"), false },
   ```

   The first field is the name the application sends: PascalCase, no spaces or
   accents, at most 27 characters. Set the last field to `true` if the spell
   only works with a target under the reticle.
3. Rebuild, deploy, load a save and test without the microphone:

   ```powershell
   python ue4ss-mod\tools\spell_sender.py Glacius
   ```

4. If the spell comes from a community mod, put it under that mod's heading in
   the table and mention the mod in your pull request.

Note that the desktop application only sends names it recognises. A brand new
incantation also needs the speech models to know the phrase, which is a
separate piece of work done through the training portal; open an issue to
discuss it before recording.

## Working on the application

Build instructions are in [BUILDING.md](BUILDING.md). Keep these in mind:

- `Release | x64` is the only complete configuration.
- The transmitter, the VR server and the mod duplicate the shared-memory
  structures on purpose (no shared header across three build systems). If you
  touch one, update all of them and the protocol document.
- Do not commit binaries, models or your `Dependencies.local.props`.
- Small self-contained tests live in `app/tests/`; each file says how to
  build and run it from a Visual Studio developer prompt. Run the relevant
  one after touching the code it covers (for example `SecretStoreTest.cpp`
  for the password protection).

## Pull requests

- One topic per pull request, with a short description of what changed and how
  it was tested (game version, engine used, spell names tried).
- Follow the surrounding code style: four-space indentation, braces on their
  own line in the mod, existing naming conventions.
- Update the relevant README or document in `docs/` when behaviour changes.
- New third-party code needs a licence compatible with MIT and an entry in
  `THIRD_PARTY_LICENSES.md`.

## Reporting problems

Open a GitHub issue with the application's console output, the
`UE4SS_Logs\UE4SS.log` file if the mod is involved, and the version of the
release you used. The community Discord linked from the README is the best
place for quick questions.
