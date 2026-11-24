#include "program.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

namespace culbm::gpu {
namespace {

string device_prelude() {
	string source;
#include "culbm_cuda_prelude.inc"
	return source;
}

bool environment_flag(const char* const name) {
	return std::getenv(name)!=nullptr;
}

bool cache_enabled() {
	const char* const value = std::getenv("CULBM_PTX_CACHE");
	if(value==nullptr) return true;
	for(const char* off : {"0", "off", "OFF", "false", "FALSE"}) {
		if(std::strcmp(value, off)==0) return false;
	}
	return true;
}

// 64-bit FNV-1a over NUL-terminated fields; the first field versions the cache.
class CacheKey {
public:
	void field(const string& text) {
		for(const char c : text) mix((unsigned char)c);
		mix(0u);
	}
	string hex() const {
		static const char digits[] = "0123456789abcdef";
		string text(16u, '0');
		for(int i=15; i>=0; --i) text[(std::size_t)(15-i)] = digits[(state_>>(4*i))&0xFu];
		return text;
	}

private:
	void mix(const unsigned char byte) {
		state_ ^= byte;
		state_ *= 1099511628211ull;
	}
	std::uint64_t state_ = 1469598103934665603ull;
};

std::filesystem::path cache_file(const string& source, const std::vector<string>& options) {
	CacheKey key;
	key.field("culbm-ptx-cache-v1");
	key.field(source);
	for(const string& option : options) key.field(option);
	return std::filesystem::path(get_exe_path())/"culbm-ptx-cache"/("kernel-"+key.hex()+".ptx");
}

// The returned text is NUL-terminated, as cuModuleLoadDataEx expects.
bool read_cached(const std::filesystem::path& path, std::vector<char>& ptx) {
	std::error_code error;
	if(!std::filesystem::is_regular_file(path, error) || error) return false;
	const std::uintmax_t size = std::filesystem::file_size(path, error);
	if(error || size==0u || size>=(std::uintmax_t)std::numeric_limits<std::size_t>::max()) return false;
	std::ifstream file(path, std::ios::binary);
	if(!file) return false;
	ptx.assign((std::size_t)size, '\0');
	if(!file.read(ptx.data(), (std::streamsize)size)) {
		ptx.clear();
		return false;
	}
	if(ptx.back()!='\0') ptx.push_back('\0');
	return true;
}

// Only the MPI root writes, and an existing entry is never replaced.
void write_cached(const std::filesystem::path& path, const std::vector<char>& ptx) {
	if(!culbm::mpi_is_root() || ptx.empty()) return;
	std::error_code error;
	if(std::filesystem::exists(path, error) && !error) return;
	std::filesystem::create_directories(path.parent_path(), error);
	if(error) return;
	std::ofstream file(path, std::ios::binary|std::ios::trunc);
	if(file) file.write(ptx.data(), (std::streamsize)ptx.size());
}

std::vector<char> compile(const string& source, const std::vector<string>& options) {
	nvrtcProgram program = nullptr;
	check(nvrtcCreateProgram(&program, source.c_str(), "culbm_kernel.cu", 0, nullptr, nullptr),
		"nvrtcCreateProgram");
	std::vector<const char*> arguments;
	for(const string& option : options) arguments.push_back(option.c_str());
	const nvrtcResult status = nvrtcCompileProgram(program, (int)arguments.size(), arguments.data());
	std::size_t log_size = 0u;
	nvrtcGetProgramLogSize(program, &log_size);
	if(status!=NVRTC_SUCCESS && log_size>1u) {
		string log(log_size, '\0');
		nvrtcGetProgramLog(program, &log[0]);
		const string log_path = get_exe_path()+"cuda-kernel-build.log";
		const string source_path = get_exe_path()+"cuda-kernel-source.cu";
		write_file(log_path, log);
		write_file(source_path, source);
		log_warning(log);
		fatal_error("CUDA kernel compilation failed. Full log: "+log_path+", native CUDA source: "+source_path);
	}
	check(status, "nvrtcCompileProgram");
	std::size_t ptx_size = 0u;
	check(nvrtcGetPTXSize(program, &ptx_size), "nvrtcGetPTXSize");
	std::vector<char> ptx(ptx_size);
	check(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX");
	check(nvrtcDestroyProgram(&program), "nvrtcDestroyProgram");
	return ptx;
}

} // namespace

CUmodule load_kernel_module(const DeviceProperties& properties, const string& native_source) {
	const string source = device_prelude()+"\n"+native_source;
	const string id = to_string(properties.ordinal);
	if(environment_flag("CULBM_DUMP_SOURCE"))
		write_file(get_exe_path()+"cuda-kernel-device-"+id+".cu", source);
	// The option list is part of the cache key: keep it stable.
	const std::vector<string> options = {
		"--std=c++17",
		"--gpu-architecture=compute_"+to_string(properties.compute_capability),
#ifdef CULBM_INCLUDE_DIR
		string("--include-path=")+CULBM_INCLUDE_DIR,
#endif
		"--device-as-default-execution-space",
		"--fmad=true",
		"--ftz=false",
		"--prec-div=false",
		"--prec-sqrt=false",
		"--restrict",
		"--diag-suppress=177"
	};
	const bool use_cache = cache_enabled();
	const std::filesystem::path cached = use_cache ? cache_file(source, options) : std::filesystem::path();
	CUmodule module = nullptr;
	std::vector<char> ptx;
	if(use_cache && read_cached(cached, ptx)) {
		if(cuModuleLoadDataEx(&module, ptx.data(), 0u, nullptr, nullptr)==CUDA_SUCCESS) {
			log_info("GPU "+id+" kernels loaded from the PTX cache.");
		} else {
			module = nullptr;
			ptx.clear();
			log_warning("Cached PTX for GPU "+id+" could not be loaded; compiling the kernels again.");
		}
	}
	if(module==nullptr) {
		ptx = compile(source, options);
		if(use_cache) write_cached(cached, ptx);
		check(cuModuleLoadDataEx(&module, ptx.data(), 0u, nullptr, nullptr), "cuModuleLoadDataEx");
		log_info("GPU "+id+" kernels compiled by NVRTC for sm_"+to_string(properties.compute_capability)+".");
	}
	if(environment_flag("CULBM_DUMP_PTX"))
		write_file(get_exe_path()+"cuda-kernel-device-"+id+".ptx", string(ptx.data(), ptx.empty() ? 0u : ptx.size()-1u));
	return module;
}

} // namespace culbm::gpu
