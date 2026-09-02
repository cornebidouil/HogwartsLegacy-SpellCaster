# Building from source

The repository contains three independently built components:

| Component | Folder | Toolchain | Output |
|-----------|--------|-----------|--------|
| Desktop application | `app/` | Visual Studio 2022 (v143), C++20 | `HogwartsLegacy-SpellCaster.exe` |
| UE4SS mod | `ue4ss-mod/` | CMake + Visual Studio 2022 | `SpellCasterMod.dll`, installed as `main.dll` |
| UEVR plugin (VR only) | `uevr-plugin/` | Visual Studio 2022 (v143), C++20 | `UEVRSpellCasterPlugin.dll` |

All three target Windows x64 only.

## Desktop application

### Dependencies

Third-party SDK locations are read from `app/Dependencies.props`. Copy
`app/Dependencies.local.props.example` to `app/Dependencies.local.props` (ignored
by git) and point each property at your own copy. Without a local file, the
defaults expect every dependency to be unpacked under `<repo>/external/`.

| Dependency | Property | Last built with | Notes |
|------------|----------|-----------------|-------|
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | `WhisperCppDir`, `WhisperCppLib` | 1.6.2 headers; releases ship 1.7.4 runtime DLLs (BLAS, Vulkan and OpenVINO backends) | Build with CMake, Release; `whisper.lib` expected in `build\Release\` |
| [Moonshine](https://github.com/usefulsensors/moonshine) C++ runtime | `MoonshineDir`, `MoonshineLibs` | moonshine-v2 | Build `core` with CMake, Release. Provides `moonshine`, `moonshine-utils`, `bin-tokenizer`, `ort-utils`, `ten_vad` and a bundled ONNX Runtime |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | `OnnxRuntimeDir`, `OnnxRuntimeLib` | 1.18.1 (prebuilt `onnxruntime-win-x64`) | Headers for the Silero VAD wrapper |
| [PortAudio](https://www.portaudio.com/) | `PortAudioDir`, `PortAudioLib` | pa_stable_v190700 | Build with CMake, Release; `portaudio_x64.lib` |
| [SDL2](https://www.libsdl.org/) | `SDL2Dir` | 2.30.7 (`SDL2-devel-VC`) | Gamepad enumeration |
| [ViGEmBus SDK](https://github.com/nefarius/ViGEmBus) | `ViGEmSdkDir` | ViGEmBus 1.22 `sdk` folder | `ViGEmClient.lib`; the ViGEmBus driver must be installed to run |
| [SoX Resampler](https://sourceforge.net/projects/soxr/) | `SoxrDir`, `SoxrLib` | 0.1.3 | Build with CMake, Release; `soxr.lib` in `build\src\Release\` |
| AMD ROCm SDK | `ROCmDir` | 6.1 | HIP and hipBLAS headers and import libraries, used for GPU detection |
| NVIDIA CUDA Toolkit | `SpellCasterCudaVersion` | 13.1 | The Visual Studio integration (`CUDA <version>.props`) supplies the include path used by `cuda_tools.h` |

Runtime-only components: the ViGEmBus driver, and optionally the
[HidHide](https://github.com/nefarius/HidHide) driver to hide the physical
controller from the game while the virtual one is active.

### Build

```powershell
copy app\Dependencies.local.props.example app\Dependencies.local.props
# edit app\Dependencies.local.props
msbuild app\Whisper-Spell.sln /p:Configuration=Release /p:Platform=x64
```

or open `app\Whisper-Spell.sln` in Visual Studio and build **Release | x64**.
The executable lands in `app\x64\Release\HogwartsLegacy-SpellCaster.exe`.

`Debug | x64` is a reduced configuration without the Moonshine engine that links
the Release third-party libraries. It exists for stepping through the audio and
input code, not for producing a usable binary.

### Assembling a release folder

The application looks for everything relative to its own folder. A release is
the following layout, which is what the zip files on the Releases page contain:

```
HogwartsLegacy-SpellCaster.exe
config.ini                          <- app/Whisper-Spell/config.ini (defaults)
whisper.dll, ggml*.dll, libopenblas.dll, libwinpthread-1.dll, vulkan-1.dll
onnxruntime.dll, openvino.dll, tbb12.dll
portaudio_x64.dll, soxr.dll
models\whisper\ggml-model.bin                       fine-tuned Whisper model
models\whisper\ggml-tiny-encoder-openvino.{bin,xml} OpenVINO encoder
models\moonshine\encoder_model.ort
models\moonshine\decoder_model_merged.ort
models\moonshine\tokenizer.bin
spellbook\*.png                     <- pronunciation guide shown to players
UE4SS Plugin\                       <- app/Whisper-Spell/UE4SS Plugin + binaries (see its README)
UEVR Plugin\UEVRSpellCasterPlugin.dll   optional, VR builds only (from uevr-plugin/)
```

The models are fine-tuned on the spell vocabulary and are distributed only
through GitHub Releases; they are not in this repository.

## UE4SS mod

See [ue4ss-mod/README.md](ue4ss-mod/README.md). In short:

```powershell
cd ue4ss-mod
git clone --recursive https://github.com/UE4SS-RE/RE-UE4SS.git RE-UE4SS
git -C RE-UE4SS checkout c1d91cf1c344dd0a20557a0544c2c3014ee525ee
git -C RE-UE4SS submodule update --init --recursive
.\build.ps1 -GameDir "<...>\Hogwarts Legacy\Phoenix\Binaries\Win64"
```

The build also produces the `UE4SS.dll` that the application bundles.

## UEVR plugin

See [uevr-plugin/README.md](uevr-plugin/README.md). Open
`uevr-plugin\UEVRSpellCasterPlugin.sln`, build **Release | x64**, and copy
`UEVRSpellCasterPlugin.dll` into the `plugins` folder of the game's UEVR
profile. The UEVR API headers are vendored, so the project has no external
dependency.
