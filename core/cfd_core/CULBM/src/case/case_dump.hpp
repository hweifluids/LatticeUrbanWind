#pragma once

// Stages of the parsed-configuration dump (config_dump.hpp): each writes the
// parameters the case setup has derived when it is reached.

#include <string>
#include <vector>

namespace culbm::bc {
struct ProfileInflow;
}

namespace culbm::cases {

struct CaseContext;
class ConfigDump;

void dump_settings_stage(const CaseContext& ctx);
void dump_domain_stage(const CaseContext& ctx);
void dump_inflow_stage(const CaseContext& ctx);
void dump_geometry_stage(const CaseContext& ctx);
// Per case: lattice units, transport, forces and damping layers.
void dump_case_physics(const CaseContext& ctx, ConfigDump& dump);
// Per case: step schedule, scalar outlet and VK inlet runtime configuration.
void dump_case_schedule(const CaseContext& ctx, ConfigDump& dump, const std::string& case_downstream_bc);
// Profile decks: the profile table and its ground.
void dump_profile_stage(const bc::ProfileInflow& profile, const std::vector<float>& profile_u_si,
                        unsigned total_cases, bool profile_single_case_standard_output);

} // namespace culbm::cases
