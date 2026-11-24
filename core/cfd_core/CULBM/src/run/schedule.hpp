#pragma once

// Step schedule of a case run: the run length (run_nstep, default 20001,
// plus the legacy research_output steps), the unsteady output interval, the
// mean-field window (the last purge_avg steps, sampled every
// purge_avg_stride-th step) and the probe window (probes_output, else the
// longer of purge_avg and research_output, else the whole run).

#include "../core/core.hpp"

namespace culbm::cases {
struct CaseSettings;
}

namespace culbm::run {

struct RunSchedule {
    static constexpr ulong default_run_steps = 20001ull;
    ulong base_run_steps = 0ull;
    ulong extra_run_steps = 0ull;
    ulong total_steps = 0ull;
    ulong unsteady_interval = 0ull; // 0: no unsteady output
    bool do_avg = false;
    ulong avg_window = 0ull;
    ulong avg_stride = 1ull;
    ulong avg_start_t = max_ulong; // first sampled step; max_ulong: none
    ulong probe_window = 0ull;
    ulong probe_start_t = max_ulong;
};

RunSchedule make_run_schedule(const cases::CaseSettings& deck, bool probes_requested);

} // namespace culbm::run
