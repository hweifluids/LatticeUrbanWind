#include "console_log.hpp"
#include "../case/settings.hpp"

#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace culbm::run {

// helper: return current local time in "YYYY-MM-DD hh:mm:ss" format
string now_str() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t tt = system_clock::to_time_t(now);
    std::tm tm{};
    #ifdef _WIN32
         if (localtime_s(&tm, &tt) != 0) {
             throw std::runtime_error("Failed to get local time.");
         }
    #else
         if (localtime_r(&tt, &tm) == nullptr) {
             throw std::runtime_error("Failed to get local time.");
         }
    #endif

     std::stringstream ss;
     ss << std::put_time(&tm, "%Y%m%d %T"); // YYYYMMDD hh:mm:ss
     return ss.str();
 }
string now_stamp_str() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t tt = system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    if (localtime_s(&tm, &tt) != 0) {
        throw std::runtime_error("Failed to get local time.");
    }
#else
    if (localtime_r(&tt, &tm) == nullptr) {
        throw std::runtime_error("Failed to get local time.");
    }
#endif
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y%m%d%H%M%S");
    return ss.str();
}
string init_console_log_file(const string& project_dir) {
    if (!culbm::mpi_is_root() || project_dir.empty()) return "";
    std::filesystem::path log_dir = std::filesystem::path(project_dir) / "proj_temp";
    std::error_code ec;
    std::filesystem::create_directories(log_dir, ec);
    if (ec) return "";
    const std::string log_name = now_stamp_str() + "_lbm.log";
    const std::filesystem::path log_path = log_dir / log_name;
    return set_console_log_file(log_path.string()) ? log_path.string() : "";
}

void open_case_log(const cases::CaseSettings& deck) {
    const std::string lbm_log_path = init_console_log_file(deck.parent);
    if (!lbm_log_path.empty()) {
        println("| Console log     | " + lbm_log_path + " |");
    }
    else {
        println("| WARNING: failed to create proj_temp log file.                               |");
    }
}

} // namespace culbm::run
