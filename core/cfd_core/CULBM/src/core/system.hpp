#pragma once

// Timing, thread-parallel loops and file-system helpers of the host code.

#include "platform.hpp"
#include "types.hpp"
#include "../cuda_mpi.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace culbm {
namespace core {

// Wall-clock stopwatch; stop() returns the seconds since construction or start().
class Clock {
public:
	Clock() { start(); }
	void start() { origin = std::chrono::steady_clock::now(); }
	double stop() const { return std::chrono::duration<double>(std::chrono::steady_clock::now()-origin).count(); }
private:
	std::chrono::steady_clock::time_point origin;
};

// Sleeps for the given seconds, rounded to whole milliseconds; zero or less returns at once.
inline void sleep(const double seconds) {
	if(!(seconds>0.0)) return;
	const int milliseconds = (int)(1E3*seconds+0.5);
	std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

// ---- parallel_for --------------------------------------------------------------
// Static block schedule: worker t (0 <= t < threads) handles the contiguous
// index block [N*t/threads, N*(t+1)/threads) in increasing order, on its own
// thread and with its own copy of the loop body. Per-worker accumulators
// indexed by t therefore always see the same indices. threads==0 runs nothing.
namespace detail {
template<typename Body> inline void run_static_blocks(const ulong count, const uint threads, const Body& body) {
	std::vector<std::thread> workers;
	workers.reserve(threads);
	for(uint t=0u; t<threads; t++) {
		const ulong first = count*(ulong)t/(ulong)threads;
		const ulong last = count*(ulong)(t+1u)/(ulong)threads;
		workers.emplace_back([body, first, last, t]() {
			for(ulong n=first; n<last; n++) body(n, t);
		});
	}
	for(std::thread& worker : workers) worker.join();
}
inline uint default_thread_count() { return (uint)std::thread::hardware_concurrency(); }
} // namespace detail

inline void parallel_for(const uint N, const uint threads, std::function<void(uint, uint)> lambda) { // lambda(n, t)
	detail::run_static_blocks(N, threads, [lambda](const ulong n, const uint t) { lambda((uint)n, t); });
}
inline void parallel_for(const uint N, const uint threads, std::function<void(uint)> lambda) { // lambda(n)
	detail::run_static_blocks(N, threads, [lambda](const ulong n, const uint) { lambda((uint)n); });
}
inline void parallel_for(const uint N, std::function<void(uint)> lambda) {
	parallel_for(N, detail::default_thread_count(), lambda);
}
inline void parallel_for(const ulong N, const uint threads, std::function<void(ulong, uint)> lambda) { // lambda(n, t)
	detail::run_static_blocks(N, threads, [lambda](const ulong n, const uint t) { lambda(n, t); });
}
inline void parallel_for(const ulong N, const uint threads, std::function<void(ulong)> lambda) { // lambda(n)
	detail::run_static_blocks(N, threads, [lambda](const ulong n, const uint) { lambda(n); });
}
inline void parallel_for(const ulong N, std::function<void(ulong)> lambda) {
	parallel_for(N, detail::default_thread_count(), lambda);
}

// ---- paths and files ---------------------------------------------------------------
// Directory of the running executable with '/' separators and a trailing '/'.
// On Windows each UTF-16 unit of the module path is narrowed to one char.
inline std::string get_exe_path() {
	std::string path;
#if defined(_WIN32)
	wchar_t module_path[260] = {0};
	GetModuleFileNameW(nullptr, module_path, 260);
	for(uint i=0u; i<260u&&module_path[i]!=L'\0'; i++) {
		const char narrowed = (char)module_path[i];
		path.push_back(narrowed=='\\' ? '/' : narrowed);
	}
#elif defined(__APPLE__)
	uint size = 0u;
	_NSGetExecutablePath(nullptr, &size);
	path.resize(size+1u, '\0');
	_NSGetExecutablePath(path.data(), &size);
#else
	char link_target[260];
	const int size = (int)readlink("/proc/self/exe", link_target, sizeof(link_target));
	if(size>0) path.assign(link_target, (size_t)size);
#endif
	return path.substr(0u, path.rfind('/')+1u);
}
// Creates the directory part of `path` (everything before the last '/').
inline void create_folder(const std::string& path) {
	const size_t separator = path.rfind('/');
	if(separator==std::string::npos) return;
	const std::filesystem::path folder = path.substr(0u, separator);
	if(!std::filesystem::is_directory(folder)||!std::filesystem::exists(folder)) std::filesystem::create_directories(folder);
}
// Replaces a short alphanumeric (or matching) extension of `filename`, or
// appends `extension`; an empty extension only strips such a suffix.
inline std::string create_file_extension(const std::string& filename, const std::string& extension) {
	std::string stem = filename;
	const size_t dot = filename.find_last_of('.');
	const size_t separator = filename.find_last_of("/\\");
	if(dot!=std::string::npos&&(separator==std::string::npos||dot>separator)) {
		const std::string suffix = filename.substr(dot+1u);
		const std::string wanted = !extension.empty()&&extension[0]=='.' ? extension.substr(1u) : extension;
		const bool same_extension = !suffix.empty()&&suffix==wanted;
		bool short_alphanumeric = !suffix.empty()&&suffix.length()<=5u;
		for(const char c : suffix) short_alphanumeric = short_alphanumeric&&((c>='0'&&c<='9')||(c>='A'&&c<='Z')||(c>='a'&&c<='z'));
		if(same_extension||short_alphanumeric) stem = filename.substr(0u, dot);
	}
	if(extension.empty()) return stem;
	return extension[0]=='.' ? stem+extension : stem+"."+extension;
}
// Writes `content` as a text-mode file (MPI root only), creating its folder.
inline void write_file(const std::string& filename, const std::string& content="") {
	if(!culbm::mpi_is_root()) return;
	create_folder(filename);
	std::ofstream output(filename); // text mode: "\n" becomes the platform line end
	output.write(content.data(), (std::streamsize)content.length());
}

} // namespace core
} // namespace culbm
