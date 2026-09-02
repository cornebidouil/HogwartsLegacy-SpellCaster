set(TARGET SpellCasterMod)
project(${TARGET} LANGUAGES CXX)

add_library(${TARGET} SHARED dllmain.cpp)
target_include_directories(${TARGET} PRIVATE .)
target_link_libraries(${TARGET} PUBLIC UE4SS)
