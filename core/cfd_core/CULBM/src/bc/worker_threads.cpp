#include "worker_threads.hpp"

#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace culbm::bc {

unsigned available_worker_threads() {
#ifdef _WIN32
    using GetActiveProcessorCountFn = DWORD(WINAPI*)(WORD);
    HMODULE kernel32 = GetModuleHandleA("kernel32.dll");
    if (kernel32 != nullptr) {
        auto get_active_processor_count =
            reinterpret_cast<GetActiveProcessorCountFn>(GetProcAddress(kernel32, "GetActiveProcessorCount"));
        if (get_active_processor_count != nullptr) {
            constexpr WORD all_processor_groups = 0xFFFFu;
            const DWORD win_hw = get_active_processor_count(all_processor_groups);
            if (win_hw > 0u) return static_cast<unsigned>(win_hw);
        }
    }
#endif
    const unsigned hw0 = std::thread::hardware_concurrency();
    return hw0 == 0u ? 4u : hw0;
}

} // namespace culbm::bc
