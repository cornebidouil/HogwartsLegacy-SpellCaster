# Known issues and planned clean-ups

Things that are known to be imperfect in the current code. Contributions on any
of these are welcome; open an issue first if you plan a large change so the work
is not duplicated.

## Security

- **Crowdsourcing account password.** The password is stored in `config.ini`
  encrypted with the Windows Data Protection API for the current Windows user
  (`password_protected=`, see `SecretStore.h`), and files written by earlier
  versions with a clear-text `password=` line are rewritten on first load.
  Two limitations remain: the sync endpoint needs the credentials on every
  call because its token only lives one hour, so the password must be kept
  somewhere; and a config file moved to another machine or user account
  cannot be decrypted, in which case the application asks for the password
  again. A long-lived per-device token issued by the server would remove the
  need to keep the password at all.

## UE4SS mod

- **The unlock check is disabled.** `CastSpellByName` no longer calls
  `IsSpellUnlocked` before casting; the call was commented out upstream and
  the removal of dead code made that explicit. The unlock subsystem
  (`InitializeUnlockSystem`, `IsSpellUnlocked`) is still compiled and works,
  but nothing calls it, and `SpellResult::NotUnlocked` is never returned.
  Either the game's own refusal to activate an unlearned spell tool is being
  relied on, in which case such a spell may stay in the queue as "busy", or
  the check should be re-enabled once the lock names of community spells
  have been verified. This needs a test in game to settle.
- The shared-memory layout has no version field (see `docs/ipc-protocol.md`).
- `Bombarda` is mapped to the Expulso record because the game has no separate
  Bombarda `SpellToolRecord`; verify against the current game build.

## Desktop application

- The `Debug|x64` configuration is a reduced build without the Moonshine
  engine and links Release third-party libraries. Only `Release|x64` is a
  complete build.
- The CUDA Toolkit and the ROCm SDK are required at build time only to detect
  GPUs (`cuda_tools.h`, `Tools.cpp`). Loading the runtime DLLs dynamically would
  make both optional for contributors.
- The project compiles against whisper.cpp 1.6.2 headers while releases ship
  the whisper.cpp 1.7.4 runtime DLLs. It works because only the C API is used,
  but the two should be aligned.
- ONNX Runtime headers are taken from two places (the 1.18.1 package for the
  Silero VAD wrapper and the copy bundled with Moonshine). One should go.
- The `Keybinds` mod and `UE4SS-settings.ini` shipped in `UE4SS Plugin` are the
  stock UE4SS files with the GUI console disabled; they should be refreshed
  whenever the bundled UE4SS is upgraded.

## Documentation

- The recognised vocabulary (the phrases the speech models were fine-tuned on)
  is only documented as images in the release `spellbook` folder. A text list
  next to the mod's spell table would let contributors see both sides at once.
