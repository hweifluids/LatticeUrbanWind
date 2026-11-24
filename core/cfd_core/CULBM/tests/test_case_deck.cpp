// Deck reading of the case setup (case/deck), the step schedule of a run
// (run/schedule) and the number formatting of the configuration dump
// (case/config_dump). No GPU.

#include "case/config_dump.hpp"
#include "case/deck.hpp"
#include "case/settings.hpp"
#include "run/schedule.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int failures = 0;

void check(const bool condition, const std::string& what) {
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << what << '\n';
}

void test_keys_and_values() {
    using namespace culbm::cases;
    check(deck_normalize_key("  Culbm-SGS  Enable ") == "culbm_sgs_enable", "key separators and case");
    check(deck_normalize_key("vk_inlet_enable") == "turb_inflow_enable", "key alias");
    check(deck_normalize_key("culbm_temperature_reference_k") == "culbm_boussinesq_reference_temperature_k", "key alias 2");
    check(deck_unquote("  \"  +y \"  ") == "+y", "unquote");
    bool flag = false;
    check(deck_try_parse_bool("'On'", flag) && flag, "bool token");
    check(deck_try_parse_bool("0.0", flag) && !flag, "bool number");
    check(!deck_try_parse_bool("maybe", flag), "bool rejects text");
    double value = 0.0;
    check(deck_try_parse_finite_double(" 1.5e-3 ", value) && value == 1.5e-3, "double");
    check(!deck_try_parse_finite_double("1.5 m", value), "double rejects trailing text");
    check(!deck_try_parse_finite_double("inf", value), "double rejects infinity");
    uint count = 0u;
    check(deck_try_parse_uint("42", count) && count == 42u, "uint");
    check(!deck_try_parse_uint("-1", count), "uint rejects negative");
    check(!deck_try_parse_uint("4294967296", count), "uint rejects overflow");
}

void test_entries() {
    using namespace culbm::cases;
    std::istringstream in(
        "casename = first\n"
        "// whole-line comment = ignored\n"
        "probes = [\"center\" N, 1:2] // trailing comment\n"
        "path = 'a//b' // quoted slashes stay\n"
        "no separator line\n"
        "casename = second\n"
        " = empty key\n");
    const ParsedDeck deck = read_deck_entries(in);
    check(deck.values.size() == 3u, "entry count");
    check(deck.values.at("casename") == "second", "later entry wins");
    check(deck.values.at("probes") == "[\"center\" N, 1:2]", "comment after value");
    check(deck.values.at("path") == "'a//b'", "comment marker inside quotes");
}

void test_text() {
    using namespace culbm::cases;
    check(format_decimal_trimmed(121.5) == "121.5", "trimmed decimal");
    check(format_decimal_trimmed(3.0) == "3", "trimmed integer");
    check(sanitize_filename_component("a b/c:d.") == "a_b_c_d", "file name component");
    check(to_upper_ascii("n2e") == "N2E", "upper case");
}

void test_schedule() {
    using culbm::cases::CaseSettings;
    using culbm::run::make_run_schedule;
    CaseSettings deck;
    culbm::run::RunSchedule s = make_run_schedule(deck, false);
    check(s.total_steps == 20001ull && s.avg_window == 0ull && s.avg_start_t == max_ulong, "default schedule");
    check(s.probe_window == 0ull && s.probe_start_t == max_ulong, "no probes");
    deck.run_nstep_override = 400ull;
    deck.research_output_steps = 100u;
    deck.purge_avg_steps = 50u;
    deck.purge_avg_stride = 0u;
    s = make_run_schedule(deck, true);
    check(s.total_steps == 500ull && s.extra_run_steps == 100ull, "run length");
    check(s.avg_window == 50ull && s.avg_start_t == 451ull && s.avg_stride == 1ull, "mean-field window");
    check(s.probe_window == 100ull && s.probe_start_t == 401ull, "probe window from research_output");
    deck.probes_output_defined = true;
    deck.probes_output_steps = 30u;
    s = make_run_schedule(deck, true);
    check(s.probe_window == 30ull && s.probe_start_t == 471ull, "probe window from probes_output");
    deck.purge_avg_steps = 9000u;
    s = make_run_schedule(deck, false);
    check(s.avg_window == 500ull && s.avg_start_t == 1ull, "window limited to the run");
}

void test_dump_text() {
    using culbm::cases::Checksum;
    using culbm::cases::ConfigDump;
    check(ConfigDump::text(0.1f) == "0.100000001", "float with 9 digits");
    check(ConfigDump::text(0.1) == "0.10000000000000001", "double with 17 digits");
    const float f = 1.0f / 3.0f;
    check(std::strtof(ConfigDump::text(f).c_str(), nullptr) == f, "float round trip");
    Checksum empty;
    check(empty.hex() == "cbf29ce484222325", "FNV-1a offset basis");
    Checksum a;
    a.bytes("a", 1u);
    check(a.hex() == "af63dc4c8601ec8c", "FNV-1a of 'a'");
}

} // namespace

int main() {
    test_keys_and_values();
    test_entries();
    test_text();
    test_schedule();
    test_dump_text();
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "case deck tests passed\n";
    return 0;
}
