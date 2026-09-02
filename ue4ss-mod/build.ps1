<#
.SYNOPSIS
    Configures and builds the SpellCaster UE4SS mod, optionally deploying it to the game.

.DESCRIPTION
    Runs CMake against this folder (RE-UE4SS must already be cloned next to this
    script, see README.md), builds the SpellCasterMod target and, when -GameDir is
    given, copies the result to <GameDir>\Mods\SpellCaster\dlls\main.dll and makes
    sure the mod is enabled.

.PARAMETER GameDir
    Path to "Hogwarts Legacy\Phoenix\Binaries\Win64". Optional.

.PARAMETER Configuration
    UE4SS build configuration. Default: Game__Shipping__Win64.

.PARAMETER Generator
    CMake generator. Default: "Visual Studio 17 2022".

.EXAMPLE
    .\build.ps1
    .\build.ps1 -GameDir "D:\SteamLibrary\steamapps\common\Hogwarts Legacy\Phoenix\Binaries\Win64"
#>
param(
    [string]$GameDir,
    [string]$Configuration = "Game__Shipping__Win64",
    [string]$Generator = "Visual Studio 17 2022"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$buildDir = Join-Path $root "build"

if (-not (Test-Path (Join-Path $root "RE-UE4SS\CMakeLists.txt"))) {
    throw "RE-UE4SS not found. Run: git clone --recursive https://github.com/UE4SS-RE/RE-UE4SS.git `"$root\RE-UE4SS`""
}

cmake -S $root -B $buildDir -G $Generator
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed" }

cmake --build $buildDir --config $Configuration --target SpellCasterMod
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

$dll = Get-ChildItem -Path $buildDir -Recurse -Filter "SpellCasterMod.dll" |
    Where-Object { $_.FullName -like "*$Configuration*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $dll) { throw "SpellCasterMod.dll not found under $buildDir" }
Write-Host "Built $($dll.FullName)"

if ($GameDir) {
    if (-not (Test-Path $GameDir)) { throw "Game directory not found: $GameDir" }
    $modDir = Join-Path $GameDir "Mods\SpellCaster"
    $dllDir = Join-Path $modDir "dlls"
    New-Item -ItemType Directory -Force -Path $dllDir | Out-Null
    Copy-Item $dll.FullName (Join-Path $dllDir "main.dll") -Force
    $enabled = Join-Path $modDir "enabled.txt"
    if (-not (Test-Path $enabled)) { New-Item -ItemType File -Path $enabled | Out-Null }
    Write-Host "Deployed to $dllDir\main.dll"
}
