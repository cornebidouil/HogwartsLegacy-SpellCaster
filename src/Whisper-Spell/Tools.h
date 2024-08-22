#ifndef DEF_TOOLS
#define DEF_TOOLS

#include <string>
#include <functional>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <comdef.h>
#include <Wbemidl.h>
#pragma comment(lib, "wbemuuid.lib")

/**
 * @brief Calculates the hash value of a file.
 *
 * @param filepath The path to the file to hash.
 * @return std::size_t The hash value of the file.
 */
std::size_t hashFile(const std::string& filepath);

/**
 * @brief Checks if the system has an Nvidia GPU.
 *
 * @return bool True if an Nvidia GPU is detected, false otherwise.
 */
bool hasNvidiaGPU();

#endif