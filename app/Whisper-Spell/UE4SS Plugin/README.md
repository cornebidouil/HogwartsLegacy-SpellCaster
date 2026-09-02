# UE4SS Plugin payload

At start-up the application copies this folder into the game's
`Phoenix\Binaries\Win64` directory (see `UE4SSInstaller.cpp`). The text files
are versioned here; the binaries are not, and must be added before running or
packaging the application:

| File | Origin |
|------|--------|
| `UE4SS.dll`, `dwmapi.dll` | Built from [RE-UE4SS](https://github.com/UE4SS-RE/RE-UE4SS) together with the mod (see `ue4ss-mod/README.md`), or taken from a UE4SS release |
| `Mods/SpellCaster/dlls/main.dll` | Built from `ue4ss-mod/` |

The installer creates `Mods/SpellCaster/enabled.txt` itself.
