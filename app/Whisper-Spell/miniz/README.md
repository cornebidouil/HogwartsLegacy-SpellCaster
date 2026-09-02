# miniz - Compression Library

This folder contains the [miniz](https://github.com/richgel999/miniz) compression library.

## About

miniz is a lossless, high performance data compression library in C that implements:
- zlib/deflate compression (RFC 1950, RFC 1951)
- ZIP archive reading/writing/appending
- PNG writing

**Version:** 3.1.0
**License:** MIT/Public Domain
**Repository:** https://github.com/richgel999/miniz

## Files

- `miniz.h` - Main header with zlib-style API declarations
- `miniz.c` - Core compression/decompression implementation
- `miniz_common.h` - Common type definitions and macros
- `miniz_export.h` - Export macros for static/dynamic linking
- `miniz_tdef.h/c` - Deflate (compression) implementation
- `miniz_tinfl.h/c` - Inflate (decompression) implementation
- `miniz_zip.h/c` - ZIP archive API (reading/writing archives)

## Usage

Include the main header in your code:
```cpp
#include "miniz/miniz.h"
```

The library is compiled as part of the Visual Studio project.

## Purpose in SpellCaster

miniz is used by the crowdsourcing system to create ZIP archives containing:
- Audio recordings (WAV files)
- Metadata (JSON files)

These archives are compressed and uploaded to the crowdsourcing server.
