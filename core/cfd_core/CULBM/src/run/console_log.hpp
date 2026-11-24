#pragma once

// Time stamps of the console report and the per-run console log file
// <project>/proj_temp/<YYYYMMDDhhmmss>_lbm.log.

#include "../core/core.hpp"

namespace culbm::cases {
struct CaseSettings;
}

namespace culbm::run {

string now_str();       // "YYYYMMDD hh:mm:ss"
string now_stamp_str(); // "YYYYMMDDhhmmss"
// Starts mirroring the console into a new log file; the path, or empty.
string init_console_log_file(const string& project_dir);
// Opens the case log in the project of `deck` and reports where it is.
void open_case_log(const cases::CaseSettings& deck);

} // namespace culbm::run
