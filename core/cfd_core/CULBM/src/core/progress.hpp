#pragma once

// Machine-readable progress lines for the LUW GUI ("[[LUW_PROGRESS]]{json}"),
// emitted only on the MPI root and only when LUW_PROGRESS_MODE selects GUI mode.
// LatticeUrbanWind-owned code, carried over unchanged from the former
// solver utilities (it was written for LUW and has no upstream origin).

#include "console_log.hpp"
#include "text.hpp"
#include "../cuda_mpi.hpp"

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>

namespace culbm {
namespace core {

using std::string;

inline bool luw_progress_gui_mode() {
	static int cached = -1;
	if(cached < 0) {
		string mode = "";
#if defined(_WIN32)
		char* env = nullptr;
		size_t env_len = 0u;
		if(_dupenv_s(&env, &env_len, "LUW_PROGRESS_MODE") == 0 && env != nullptr) {
			mode = string(env);
			std::free(env);
		}
#else
		const char* env = std::getenv("LUW_PROGRESS_MODE");
		if(env != nullptr) mode = string(env);
#endif
		cached = (mode == "gui" || mode == "1" || mode == "true") ? 1 : 0;
	}
	return cached == 1;
}
inline string luw_progress_escape_json(const string& s) {
	string out;
	out.reserve(s.size() + 16u);
	for(size_t i = 0u; i < s.size(); i++) {
		const char c = s[i];
		switch(c) {
		case '\\': out += "\\\\"; break;
		case '"': out += "\\\""; break;
		case '\n': out += "\\n"; break;
		case '\r': out += "\\r"; break;
		case '\t': out += "\\t"; break;
		default: out.push_back(c); break;
		}
	}
	return out;
}
inline void luw_emit_progress(const string& stage,
							  const string& label,
							  const string& detail = "",
							  const long long current = -1ll,
							  const long long total = -1ll,
							  const bool indeterminate = true) {
	if(!culbm::mpi_is_root() || !luw_progress_gui_mode()) return;
	const string payload =
		"[[LUW_PROGRESS]]{\"stage\":\"" + luw_progress_escape_json(stage) +
		"\",\"label\":\"" + luw_progress_escape_json(label) +
		"\",\"detail\":\"" + luw_progress_escape_json(detail) +
		"\",\"current\":" + to_string(current) +
		",\"total\":" + to_string(total) +
		",\"indeterminate\":" + string(indeterminate ? "true" : "false") + "}";
	std::lock_guard<std::mutex> lock(console_io_mutex());
	std::cout << payload << "\n";
	std::cout.flush();
}

} // namespace core
} // namespace culbm
