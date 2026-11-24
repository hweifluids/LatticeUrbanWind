#pragma once

// Operating-system headers needed by the console and file helpers.
// On Windows the lean <windows.h> is used and the min/max macros are removed
// so that the min()/max() functions of the host code stay callable.

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif
