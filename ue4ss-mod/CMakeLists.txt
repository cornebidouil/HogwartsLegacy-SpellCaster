cmake_minimum_required(VERSION 3.22)

# RE-UE4SS is built from source alongside the mod, following
# https://docs.ue4ss.com/dev/guides/creating-a-c++-mod.html
# The check runs before project() so a missing clone is reported before any
# compiler detection takes place.
set(RE_UE4SS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/RE-UE4SS")
if(NOT EXISTS "${RE_UE4SS_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "RE-UE4SS was not found in ${RE_UE4SS_DIR}.\n"
        "Clone it next to this file:\n"
        "  git clone --recursive https://github.com/UE4SS-RE/RE-UE4SS.git \"${RE_UE4SS_DIR}\"\n"
        "README.md lists the commit this mod was last built against.")
endif()

project(SpellCasterUE4SSMod LANGUAGES CXX)

add_subdirectory(RE-UE4SS)
add_subdirectory(SpellCasterMod)
