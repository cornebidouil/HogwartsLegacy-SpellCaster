# Third-party software

This project is released under the MIT licence (see `LICENSE`). It builds on,
vendors or redistributes the following components. Each keeps its own licence.

## Vendored in this repository

| Component | Location | Licence |
|-----------|----------|---------|
| [inih](https://github.com/benhoyt/inih) | `app/Whisper-Spell/inih/` | BSD-3-Clause |
| [miniz](https://github.com/richgel999/miniz) | `app/Whisper-Spell/miniz/` | MIT |
| [UEVR](https://github.com/praydog/UEVR) plugin API headers | `uevr-plugin/UEVRSpellCasterPlugin/uevr/` | MIT |
| [UE4SS](https://github.com/UE4SS-RE/RE-UE4SS) default `Keybinds` mod and settings | `app/Whisper-Spell/UE4SS Plugin/` | MIT |

## Linked at build time

| Component | Licence |
|-----------|---------|
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and ggml | MIT |
| [Moonshine](https://github.com/usefulsensors/moonshine) C++ runtime | MIT |
| [TEN VAD](https://github.com/TEN-framework/ten-vad) | Apache-2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT |
| [Silero VAD](https://github.com/snakers4/silero-vad) model | MIT |
| [PortAudio](https://www.portaudio.com/) | PortAudio licence (MIT-style) |
| [SDL2](https://www.libsdl.org/) | zlib |
| [ViGEmBus SDK / ViGEmClient](https://github.com/nefarius/ViGEmBus) | BSD-3-Clause |
| [SoX Resampler library](https://sourceforge.net/projects/soxr/) | LGPL-2.1-or-later, used as a separate DLL |
| [UE4SS](https://github.com/UE4SS-RE/RE-UE4SS) | MIT |
| AMD ROCm HIP / hipBLAS headers | MIT |
| NVIDIA CUDA Toolkit headers | NVIDIA CUDA Toolkit EULA (build-time only, not redistributed) |

## Redistributed in release archives

| File(s) | Component | Licence |
|---------|-----------|---------|
| `whisper.dll`, `ggml*.dll` | whisper.cpp / ggml | MIT |
| `onnxruntime.dll` | ONNX Runtime | MIT |
| `openvino.dll`, `tbb12.dll` | [OpenVINO](https://github.com/openvinotoolkit/openvino), [oneTBB](https://github.com/uxlfoundation/oneTBB) | Apache-2.0 |
| `libopenblas.dll` | [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) | BSD-3-Clause |
| `libwinpthread-1.dll` | mingw-w64 winpthreads | MIT and BSD-3-Clause |
| `vulkan-1.dll` | [Vulkan Loader](https://github.com/KhronosGroup/Vulkan-Loader) | Apache-2.0 |
| `portaudio_x64.dll` | PortAudio | PortAudio licence |
| `soxr.dll` | SoX Resampler library | LGPL-2.1-or-later |
| `UE4SS.dll`, `dwmapi.dll` | UE4SS | MIT |
| `models/whisper/ggml-model.bin` | Fine-tuned from [OpenAI Whisper](https://github.com/openai/whisper) | MIT (base model) |
| `models/moonshine/*` | Fine-tuned from Moonshine | MIT (base model) |

Runtime drivers the user installs separately: [ViGEmBus](https://github.com/nefarius/ViGEmBus)
(BSD-3-Clause) and [HidHide](https://github.com/nefarius/HidHide) (MIT).

## Game content

The UE4SS mod refers to spell assets by their in-game object path only and
ships none of their content. Community spell packs referenced in its table are
the work of their respective authors: SpellsEnhanced (Khione), HRBSpellPack,
HermitHollow, WFM_SpellMod_01 and SpM_Test-01.

Hogwarts Legacy is a trademark of Warner Bros. Entertainment Inc. This project
is an independent fan tool and is not affiliated with or endorsed by Warner
Bros. Games, Avalanche Software or Portkey Games.
