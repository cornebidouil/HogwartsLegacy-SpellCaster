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

std::size_t hashFile(const std::string& filepath);
bool hasNvidiaGPU();
#endif