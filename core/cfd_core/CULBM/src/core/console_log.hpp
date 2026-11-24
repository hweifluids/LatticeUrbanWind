#pragma once

// Console log file and the console I/O lock.
// LatticeUrbanWind-owned code, carried over unchanged from the former
// solver utilities (it was written for LUW and has no upstream origin).
// Every console write takes console_io_mutex(); a mirrored copy without ANSI
// escape sequences goes to the optional log file.

#include "types.hpp"

#include <fstream>
#include <mutex>
#include <string>

namespace culbm {
namespace core {

using std::string;

inline std::mutex& console_io_mutex() {
	static std::mutex m;
	return m;
}
inline std::ofstream& console_log_stream() {
	static std::ofstream f;
	return f;
}
inline string& console_log_path() {
	static string p;
	return p;
}
inline string strip_ansi_escape_codes(const string& s) {
	string out;
	out.reserve(s.size());
	for(size_t i=0u; i<s.size(); i++) {
		if((uchar)s[i]==27u && i+1u<s.size() && s[i+1u]=='[') {
			i += 2u;
			while(i<s.size() && !((s[i]>='@' && s[i]<='~') || s[i]=='\n' || s[i]=='\r')) i++;
			continue;
		}
		out.push_back(s[i]);
	}
	return out;
}
inline void write_console_log(const string& s) {
	auto& f = console_log_stream();
	if(!f.is_open()) return;
	f << strip_ansi_escape_codes(s);
	f.flush();
}
inline bool set_console_log_file(const string& path) {
	std::lock_guard<std::mutex> lock(console_io_mutex());
	auto& f = console_log_stream();
	if(f.is_open()) f.close();
	f.open(path, std::ios::out|std::ios::trunc);
	if(!f.is_open()) {
		console_log_path().clear();
		return false;
	}
	console_log_path() = path;
	return true;
}
inline void close_console_log_file() {
	std::lock_guard<std::mutex> lock(console_io_mutex());
	auto& f = console_log_stream();
	if(f.is_open()) f.close();
	console_log_path().clear();
}
inline string get_console_log_file() {
	std::lock_guard<std::mutex> lock(console_io_mutex());
	return console_log_path();
}

} // namespace core
} // namespace culbm
