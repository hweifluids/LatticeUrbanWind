#include "schedule.hpp"
#include "../case/settings.hpp"

namespace culbm::run {

RunSchedule make_run_schedule(const cases::CaseSettings& deck, const bool probes_requested) {
    RunSchedule s;
    s.base_run_steps = deck.run_nstep_override > 0ull ? deck.run_nstep_override : RunSchedule::default_run_steps;
    s.extra_run_steps = deck.research_output_steps > 0u ? (ulong)deck.research_output_steps : 0ull;
    s.total_steps = s.base_run_steps + s.extra_run_steps;
    s.unsteady_interval = deck.unsteady_output_interval > 0u ? (ulong)deck.unsteady_output_interval : 0ull;
    s.do_avg = deck.purge_avg_steps > 0u;
    s.avg_window = s.do_avg ? std::min((ulong)deck.purge_avg_steps, s.total_steps) : 0ull;
    s.avg_stride = std::max((ulong)1u, (ulong)deck.purge_avg_stride);
    s.avg_start_t = s.avg_window > 0ull ? (s.total_steps - s.avg_window + (ulong)1u) : max_ulong;
    if (probes_requested) {
        if (deck.probes_output_defined && deck.probes_output_steps > 0u) {
            s.probe_window = std::min((ulong)deck.probes_output_steps, s.total_steps);
        }
        else if (deck.purge_avg_steps > 0u || deck.research_output_steps > 0u) {
            s.probe_window = std::min((ulong)std::max(deck.purge_avg_steps, deck.research_output_steps), s.total_steps);
        }
        else {
            s.probe_window = s.total_steps;
        }
    }
    s.probe_start_t = s.probe_window > 0ull ? (s.total_steps - s.probe_window + 1ull) : max_ulong;
    return s;
}

} // namespace culbm::run
