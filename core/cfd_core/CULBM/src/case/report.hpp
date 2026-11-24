#pragma once

// Console report of the case setup: framed section titles and key | value
// rows, and the report sections that describe the deck, the domain and the
// reference scales before the solver starts.

#include "../core/core.hpp"

namespace culbm::cases {

struct CaseContext;
struct CaseSettings;

string hr_plain();
void print_section_title(const string& title);
void print_kv_row(const string& key, const string& value);
// Rows printed while the status line is live (StatusPause).
void print_runtime_kv(const string& key, const string& value);
void print_runtime_section(const string& title);
// Fixed-point text with `prec` decimals.
std::string fmtf(float v, int prec = 4);
// Shortest decimal text of a batch value (3 decimals at most), for names.
std::string format_tag(float v);
// LUW GUI progress event (the [[LUW_PROGRESS]] line).
void emit_progress_stage(const std::string& stage, const std::string& label, const std::string& detail,
                         long long current = -1ll, long long total = -1ll, bool indeterminate = true);
// "[tke,ti,tls]" of the turbulence fields written to the mean-field VTK.
string avg_turbulence_output_desc(const CaseSettings& deck);

void print_setup_banner();
void print_parameter_information(const CaseContext& ctx);
void print_domain_report(const CaseContext& ctx);
void print_reference_report(const CaseContext& ctx);

} // namespace culbm::cases
