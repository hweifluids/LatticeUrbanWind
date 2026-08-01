#include "cuda.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>

namespace {

std::mutex cuda_init_mutex;
bool cuda_initialized = false;
std::mutex cuda_peer_mutex;
std::map<std::pair<uint, uint>, bool> cuda_peer_access;

std::uint32_t transfer_channel_index(const CudaTransferChannel channel) {
	const std::uint32_t index = (std::uint32_t)channel;
	if(index>=cuda_transfer_channel_count)
		print_error("Invalid CUDA transfer channel.");
	return index;
}

CudaTransferState& transfer_state(
	const std::shared_ptr<CudaDeviceState>& state,
	const CudaTransferChannel channel) {
	return state->transfer[transfer_channel_index(channel)];
}

bool p2p_disabled() {
	const char* const value = std::getenv("CULBM_DISABLE_P2P");
	return value != nullptr && string(value) != "0";
}

bool ptx_cache_disabled() {
	const char* const value = std::getenv("CULBM_PTX_CACHE");
	return value != nullptr && (
		std::strcmp(value, "0")==0 ||
		std::strcmp(value, "off")==0 ||
		std::strcmp(value, "OFF")==0 ||
		std::strcmp(value, "false")==0 ||
		std::strcmp(value, "FALSE")==0);
}

std::uint64_t ptx_cache_hash_append(
	const std::uint64_t seed, const char* const text, const std::size_t length) {
	std::uint64_t hash = seed;
	for(std::size_t i=0u; i<length; ++i) {
		hash ^= static_cast<unsigned char>(text[i]);
		hash *= 1099511628211ull;
	}
	return hash;
}

std::string ptx_cache_key(const string& source,
	const vector<string>& option_strings) {
	std::uint64_t hash = 1469598103934665603ull;
	const auto append = [&](const string& value) {
		hash = ptx_cache_hash_append(hash, value.data(), value.size());
		const char separator = '\0';
		hash = ptx_cache_hash_append(hash, &separator, 1u);
	};
	append("culbm-ptx-cache-v1");
	append(source);
	for(const string& option : option_strings) append(option);
	std::ostringstream stream;
	stream << std::hex << std::setw(16) << std::setfill('0') << hash;
	return stream.str();
}

std::filesystem::path ptx_cache_path(const string& key) {
	return std::filesystem::path(get_exe_path()) / "culbm-ptx-cache" /
		("kernel-"+key+".ptx");
}

bool load_cached_ptx(const std::filesystem::path& path, vector<char>& ptx) {
	std::error_code error;
	if(!std::filesystem::is_regular_file(path, error) || error) return false;
	const std::uintmax_t bytes = std::filesystem::file_size(path, error);
	if(error || bytes==0u ||
	   bytes>=static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()))
		return false;
	std::ifstream file(path, std::ios::in|std::ios::binary);
	if(!file.is_open()) return false;
	ptx.resize(static_cast<std::size_t>(bytes));
	file.read(ptx.data(), static_cast<std::streamsize>(bytes));
	if(!file || ptx.empty()) {
		ptx.clear();
		return false;
	}
	if(ptx.back()!='\0') ptx.push_back('\0');
	return true;
}

void store_cached_ptx(const std::filesystem::path& path,
	const vector<char>& ptx) {
	if(!culbm::mpi_is_root() || ptx.empty()) return;
	std::error_code error;
	if(std::filesystem::exists(path, error) && !error) return;
	std::filesystem::create_directories(path.parent_path(), error);
	if(error) return;
	std::ofstream file(path, std::ios::out|std::ios::binary|std::ios::trunc);
	if(!file.is_open()) return;
	file.write(ptx.data(), static_cast<std::streamsize>(ptx.size()));
}

void initialize_cuda_once() {
	std::lock_guard<std::mutex> guard(cuda_init_mutex);
	if(cuda_initialized) return;
	cuda_check(cuInit(0u), "cuInit");
	cuda_initialized = true;
}

string cuda_error_text(const CUresult result) {
	const char* name = nullptr;
	const char* description = nullptr;
	cuGetErrorName(result, &name);
	cuGetErrorString(result, &description);
	return string(name ? name : "CUDA_ERROR_UNKNOWN")+" ("+
		string(description ? description : "no description")+")";
}

bool enable_peer_access(
	const std::shared_ptr<CudaDeviceState>& destination,
	const std::shared_ptr<CudaDeviceState>& source) {
	if(destination->context == source->context) return true;
	const std::pair<uint, uint> key(destination->ordinal, source->ordinal);
	std::lock_guard<std::mutex> guard(cuda_peer_mutex);
	const auto cached = cuda_peer_access.find(key);
	if(cached != cuda_peer_access.end()) return cached->second;

	bool enabled = false;
	string fallback_reason = "";
	if(p2p_disabled()) {
		fallback_reason = "CULBM_DISABLE_P2P is set";
	} else {
		int can_access = 0;
		const CUresult query_result = cuDeviceCanAccessPeer(
			&can_access, destination->device, source->device);
		if(query_result == CUDA_SUCCESS && can_access != 0) {
			cuda_set_current(destination->context);
			const CUresult enable_result =
				cuCtxEnablePeerAccess(source->context, 0u);
			enabled = enable_result == CUDA_SUCCESS ||
				enable_result == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
			if(!enabled) fallback_reason = cuda_error_text(enable_result);
		} else if(query_result != CUDA_SUCCESS) {
			fallback_reason = cuda_error_text(query_result);
		} else {
			fallback_reason = "CUDA driver reports no peer access";
		}
	}
	cuda_peer_access.emplace(key, enabled);
	print_info(
		"CUDA transfer GPU "+to_string(source->ordinal)+" -> GPU "+
		to_string(destination->ordinal)+" uses "+
		(enabled
			? "direct P2P."
			: "pinned-host fallback ("+fallback_reason+")."));
	return enabled;
}

string cuda_kernel_prelude() {
	return R"CUDA(
#include <cuda_runtime.h>
#include <cuda_fp16.h>

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long ulong;

#define cuda_threads_per_block 64u
#define CUDA_LOCAL_MEM_FENCE 0
#define CUDA_GLOBAL_MEM_FENCE 0

#define cuda_global_thread_id(dim) ((ulong)blockIdx.x*(ulong)blockDim.x+(ulong)threadIdx.x)
#define cuda_global_thread_count(dim) ((ulong)gridDim.x*(ulong)blockDim.x)
#define cuda_local_thread_id(dim) ((uint)threadIdx.x)
#define cuda_local_thread_count(dim) ((uint)blockDim.x)
#define cuda_block_count(dim) ((uint)gridDim.x)
#define cuda_block_id(dim) ((uint)blockIdx.x)
#define cuda_global_offset(dim) 0u
#define cuda_block_barrier(x) __syncthreads()
#define cuda_memory_fence(x) __threadfence_block()
#define cuda_read_memory_fence(x) __threadfence_block()
#define cuda_write_memory_fence(x) __threadfence_block()

__device__ __forceinline__ float2 cuda_float2(float v) {
	return make_float2(v, v);
}
__device__ __forceinline__ float2 cuda_float2(float x, float y) {
	return make_float2(x, y);
}
__device__ __forceinline__ float3 cuda_float3(float v) {
	return make_float3(v, v, v);
}
__device__ __forceinline__ float3 cuda_float3(float x, float y, float z) {
	return make_float3(x, y, z);
}
__device__ __forceinline__ uint3 cuda_uint3(uint v) {
	return make_uint3(v, v, v);
}
__device__ __forceinline__ uint3 cuda_uint3(uint x, uint y, uint z) {
	return make_uint3(x, y, z);
}
__device__ __forceinline__ int3 cuda_int3(int v) {
	return make_int3(v, v, v);
}
__device__ __forceinline__ int3 cuda_int3(int x, int y, int z) {
	return make_int3(x, y, z);
}
__device__ __forceinline__ uchar4 cuda_uchar4(uchar v) {
	return make_uchar4(v, v, v, v);
}
__device__ __forceinline__ uchar4 cuda_uchar4(
	uchar x, uchar y, uchar z, uchar w) {
	return make_uchar4(x, y, z, w);
}

__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
	return cuda_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__device__ __forceinline__ float3 operator-(float3 a, float3 b) {
	return cuda_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__device__ __forceinline__ float3 operator-(float3 a) {
	return cuda_float3(-a.x, -a.y, -a.z);
}
__device__ __forceinline__ float3 operator*(float3 a, float3 b) {
	return cuda_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}
__device__ __forceinline__ float3 operator*(float3 a, float b) {
	return cuda_float3(a.x*b, a.y*b, a.z*b);
}
__device__ __forceinline__ float3 operator*(float a, float3 b) {
	return b*a;
}
__device__ __forceinline__ float3 operator/(float3 a, float b) {
	return cuda_float3(a.x/b, a.y/b, a.z/b);
}
__device__ __forceinline__ float3& operator+=(float3& a, float3 b) {
	a.x+=b.x; a.y+=b.y; a.z+=b.z; return a;
}
__device__ __forceinline__ float3& operator-=(float3& a, float3 b) {
	a.x-=b.x; a.y-=b.y; a.z-=b.z; return a;
}
__device__ __forceinline__ float3& operator*=(float3& a, float b) {
	a.x*=b; a.y*=b; a.z*=b; return a;
}
__device__ __forceinline__ float3& operator/=(float3& a, float b) {
	a.x/=b; a.y/=b; a.z/=b; return a;
}

__device__ __forceinline__ uint3 operator+(uint3 a, uint3 b) {
	return cuda_uint3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__device__ __forceinline__ uint3 operator-(uint3 a, uint3 b) {
	return cuda_uint3(a.x-b.x, a.y-b.y, a.z-b.z);
}

template<typename T>
__device__ __forceinline__ T cuda_min(T a, T b) {
	return a < b ? a : b;
}
template<typename T>
__device__ __forceinline__ T cuda_max(T a, T b) {
	return a > b ? a : b;
}
template<typename T>
__device__ __forceinline__ T cuda_clamp(T x, T lo, T hi) {
	return cuda_min(cuda_max(x, lo), hi);
}
__device__ __forceinline__ float3 cuda_clamp(
	float3 x, float lo, float hi) {
	return cuda_float3(
		cuda_clamp(x.x, lo, hi),
		cuda_clamp(x.y, lo, hi),
		cuda_clamp(x.z, lo, hi));
}

__device__ __forceinline__ float cuda_fma(float a, float b, float c) {
	return fmaf(a, b, c);
}
__device__ __forceinline__ float3 cuda_fma(
	float a, float3 b, float3 c) {
	return cuda_float3(
		fmaf(a, b.x, c.x),
		fmaf(a, b.y, c.y),
		fmaf(a, b.z, c.z));
}
__device__ __forceinline__ float3 cuda_fma(
	float3 a, float b, float3 c) {
	return cuda_float3(
		fmaf(a.x, b, c.x),
		fmaf(a.y, b, c.y),
		fmaf(a.z, b, c.z));
}
__device__ __forceinline__ float3 cuda_fma(
	float3 a, float3 b, float3 c) {
	return cuda_float3(
		fmaf(a.x, b.x, c.x),
		fmaf(a.y, b.y, c.y),
		fmaf(a.z, b.z, c.z));
}

__device__ __forceinline__ float dot(float2 a, float2 b) {
	return fmaf(a.x, b.x, a.y*b.y);
}
__device__ __forceinline__ float dot(float3 a, float3 b) {
	return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z*b.z));
}
__device__ __forceinline__ float3 cross(float3 a, float3 b) {
	return cuda_float3(
		fmaf(a.y, b.z, -a.z*b.y),
		fmaf(a.z, b.x, -a.x*b.z),
		fmaf(a.x, b.y, -a.y*b.x));
}
__device__ __forceinline__ float length(float2 v) {
	return sqrtf(dot(v, v));
}
__device__ __forceinline__ float length(float3 v) {
	return sqrtf(dot(v, v));
}
__device__ __forceinline__ float distance(float3 a, float3 b) {
	return length(b-a);
}
__device__ __forceinline__ float3 normalize(float3 v) {
	const float inv = rsqrtf(dot(v, v));
	return v*inv;
}
__device__ __forceinline__ float sign(float v) {
	return (float)((v>0.0f)-(v<0.0f));
}
__device__ __forceinline__ float3 floor(float3 v) {
	return cuda_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
__device__ __forceinline__ float acospi(float v) {
	return acosf(v)*0.31830988618379067154f;
}

__device__ __forceinline__ int cuda_as_int(float value) {
	return __float_as_int(value);
}
__device__ __forceinline__ float cuda_as_float(int value) {
	return __int_as_float(value);
}
__device__ __forceinline__ float cuda_as_float(uint value) {
	return __uint_as_float(value);
}
__device__ __forceinline__ uint cuda_as_uint(float value) {
	return __float_as_uint(value);
}
__device__ __forceinline__ uchar4 cuda_as_uchar4(int value) {
	union Bits { int i; uchar4 v; } bits;
	bits.i = value;
	return bits.v;
}
__device__ __forceinline__ int cuda_as_int(uchar4 value) {
	union Bits { int i; uchar4 v; } bits;
	bits.v = value;
	return bits.i;
}

__device__ __forceinline__ float cuda_vload_half(
	size_t offset, const half* pointer) {
	return __half2float(pointer[offset]);
}
__device__ __forceinline__ void cuda_vstore_half_rn(
	float value, size_t offset, half* pointer) {
	pointer[offset] = __float2half_rn(value);
}

__device__ __forceinline__ int cuda_atomic_max(
	volatile int* address, int value) {
	return atomicMax((int*)address, value);
}
__device__ __forceinline__ uint cuda_atomic_add(
	volatile uint* address, uint value) {
	return atomicAdd((uint*)address, value);
}
__device__ __forceinline__ float cuda_atomic_xchg(
	volatile float* address, float value) {
	return atomicExch((float*)address, value);
}

__device__ __forceinline__ float cuda_fast_rsqrt(float value) {
	return rsqrtf(value);
}
__device__ __forceinline__ float cuda_fast_sqrt(float value) {
	return sqrtf(value);
}
__device__ __forceinline__ float cuda_rsqrt(float value) {
	return rsqrtf(value);
}
__device__ __forceinline__ float cuda_pown(float value, int exponent) {
	return powf(value, (float)exponent);
}
)CUDA";
}

} // namespace

void cuda_check(const CUresult result, const char* operation) {
	if(result == CUDA_SUCCESS) return;
	print_error(string(operation)+" failed: "+cuda_error_text(result)+".");
}

void nvrtc_check(const nvrtcResult result, const char* operation) {
	if(result == NVRTC_SUCCESS) return;
	print_error(string(operation)+" failed: "+nvrtcGetErrorString(result)+".");
}

void cuda_set_current(const CUcontext context) {
	cuda_check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

Device_Info::Device_Info(const CUdevice device, const uint ordinal) {
	cu_device = device;
	id = ordinal;
	char device_name[256] = {};
	cuda_check(cuDeviceGetName(device_name, (int)sizeof(device_name), device),
	           "cuDeviceGetName");
	name = trim(device_name);

	size_t total_memory = 0u;
	cuda_check(cuDeviceTotalMem(&total_memory, device), "cuDeviceTotalMem");
	memory = (uint)(total_memory/1048576ull);
	max_global_buffer = memory;

	int value = 0;
	cuda_check(cuDeviceGetAttribute(&value,
		CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
		"cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)");
	compute_units = (uint)value;
	cuda_check(cuDeviceGetAttribute(&value,
		CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device),
		"cuDeviceGetAttribute(CLOCK_RATE)");
	clock_frequency = (uint)(value/1000);
	cuda_check(cuDeviceGetAttribute(&value,
		CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device),
		"cuDeviceGetAttribute(L2_CACHE_SIZE)");
	global_cache = (uint)(value/1024);
	cuda_check(cuDeviceGetAttribute(&value,
		CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device),
		"cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK)");
	local_cache = (uint)(value/1024);

	int major = 0, minor = 0;
	cuda_check(cuDeviceGetAttribute(&major,
		CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
		"cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)");
	cuda_check(cuDeviceGetAttribute(&minor,
		CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
		"cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)");
	nvidia_compute_capability = (uint)(major*10+minor);

	const uint cores_per_sm =
		nvidia_compute_capability < 30u ? 32u :
		nvidia_compute_capability < 50u ? 192u :
		((nvidia_compute_capability>=70u&&nvidia_compute_capability<=80u)||
		 nvidia_compute_capability==60u) ? 64u : 128u;
	cores = compute_units*cores_per_sm;
	tflops = 2.0e-6f*(float)cores*(float)clock_frequency;
	is_dp4a_capable = nvidia_compute_capability>=61u ? 1u : 0u;

	int driver = 0;
	cuda_check(cuDriverGetVersion(&driver), "cuDriverGetVersion");
	driver_version = to_string((uint)(driver/1000))+"."+
		to_string((uint)((driver%1000)/10));
}

vector<Device_Info> get_devices(const bool print_info) {
	initialize_cuda_once();
	int count = 0;
	cuda_check(cuDeviceGetCount(&count), "cuDeviceGetCount");
	if(count <= 0) {
		print_error("No CUDA devices are available. Install an NVIDIA CUDA driver.");
	}
	vector<Device_Info> devices;
	devices.reserve((size_t)count);
	for(int ordinal=0; ordinal<count; ++ordinal) {
		CUdevice device = 0;
		cuda_check(cuDeviceGet(&device, ordinal), "cuDeviceGet");
		devices.emplace_back(device, (uint)ordinal);
	}
	if(print_info) {
		println("|-----------------.-----------------------------------------------------------|");
		for(uint i=0u; i<(uint)devices.size(); ++i)
			println("| CUDA Device "+alignr(3u, i)+" | "+
				alignl(58u, devices[i].name)+" |");
		println("|-----------------'-----------------------------------------------------------|");
	}
	return devices;
}

CudaDeviceState::~CudaDeviceState() {
	if(context == nullptr) return;
	cuCtxSetCurrent(context);
	if(module != nullptr) cuModuleUnload(module);
	for(CudaTransferState& channel : transfer) {
		if(channel.ready != nullptr) cuEventDestroy(channel.ready);
		if(channel.complete != nullptr) cuEventDestroy(channel.complete);
		if(channel.consumed != nullptr) cuEventDestroy(channel.consumed);
		if(channel.outbound_stream != nullptr)
			cuStreamDestroy(channel.outbound_stream);
		if(channel.transfer_stream != nullptr)
			cuStreamDestroy(channel.transfer_stream);
	}
	if(stream != nullptr) cuStreamDestroy(stream);
	cuDevicePrimaryCtxRelease(device);
}

Device::Device(const Device_Info& selected_info,
               const string& native_kernel_source) {
	initialize_cuda_once();
	info = selected_info;
	print_device_info(info);

	state = std::make_shared<CudaDeviceState>();
	state->device = info.cu_device;
	state->ordinal = info.id;
	cuda_check(cuDevicePrimaryCtxRetain(&state->context, state->device),
	           "cuDevicePrimaryCtxRetain");
	cuda_set_current(state->context);
	cuda_check(cuStreamCreate(&state->stream, CU_STREAM_NON_BLOCKING),
	           "cuStreamCreate");
	for(CudaTransferState& channel : state->transfer) {
		cuda_check(cuStreamCreate(
			&channel.transfer_stream, CU_STREAM_NON_BLOCKING),
			"cuStreamCreate(transfer)");
		cuda_check(cuStreamCreate(
			&channel.outbound_stream, CU_STREAM_NON_BLOCKING),
			"cuStreamCreate(outbound)");
		cuda_check(cuEventCreate(
			&channel.ready, CU_EVENT_DISABLE_TIMING),
			"cuEventCreate(transfer_ready)");
		cuda_check(cuEventCreate(
			&channel.complete, CU_EVENT_DISABLE_TIMING),
			"cuEventCreate(transfer_complete)");
		cuda_check(cuEventCreate(
			&channel.consumed, CU_EVENT_DISABLE_TIMING),
			"cuEventCreate(transfer_consumed)");
		cuda_check(cuEventRecord(
			channel.consumed, state->stream),
			"cuEventRecord(initial transfer_consumed)");
	}

	const string cuda_source = cuda_kernel_prelude()+"\n"+native_kernel_source;
	if(std::getenv("CULBM_DUMP_SOURCE") != nullptr) {
		write_file(
			get_exe_path()+"cuda-kernel-device-"+to_string(info.id)+".cu",
			cuda_source);
	}
	const string architecture = "--gpu-architecture=compute_"+
		to_string(info.nvidia_compute_capability);
	const vector<string> option_strings = {
		"--std=c++17",
		architecture,
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
	vector<const char*> options;
	options.reserve(option_strings.size());
	for(const string& option : option_strings) options.push_back(option.c_str());
	vector<char> ptx;
	const bool use_ptx_cache = !ptx_cache_disabled();
	const std::filesystem::path cache_file = use_ptx_cache
		? ptx_cache_path(ptx_cache_key(cuda_source, option_strings))
		: std::filesystem::path();
	bool loaded_cached_ptx = false;
	if(use_ptx_cache && load_cached_ptx(cache_file, ptx)) {
		const CUresult cache_load = cuModuleLoadDataEx(
			&state->module, ptx.data(), 0u, nullptr, nullptr);
		if(cache_load==CUDA_SUCCESS) {
			loaded_cached_ptx = true;
			print_info("CUDA PTX cache hit for GPU "+to_string(info.id)+".");
		}
		else {
			state->module = nullptr;
			ptx.clear();
			print_warning("CUDA PTX cache entry could not be loaded; recompiling kernels.");
		}
	}
	if(!loaded_cached_ptx) {
		nvrtcProgram program = nullptr;
		nvrtc_check(nvrtcCreateProgram(
			&program, cuda_source.c_str(), "culbm_kernel.cu",
			0, nullptr, nullptr), "nvrtcCreateProgram");

		const nvrtcResult compile_result = nvrtcCompileProgram(
			program, (int)options.size(), options.data());
		size_t log_size = 0u;
		nvrtcGetProgramLogSize(program, &log_size);
		if(log_size > 1u) {
			string log(log_size, '\0');
			nvrtcGetProgramLog(program, &log[0]);
			if(compile_result != NVRTC_SUCCESS) {
				const string log_path = get_exe_path()+"cuda-kernel-build.log";
				const string source_path = get_exe_path()+"cuda-kernel-source.cu";
				write_file(log_path, log);
				write_file(source_path, cuda_source);
				print_warning(log);
				print_error("CUDA kernel compilation failed. Full log: "+log_path+
					", native CUDA source: "+source_path);
			}
		}
		nvrtc_check(compile_result, "nvrtcCompileProgram");

		size_t ptx_size = 0u;
		nvrtc_check(nvrtcGetPTXSize(program, &ptx_size), "nvrtcGetPTXSize");
		ptx.resize(ptx_size);
		nvrtc_check(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX");
		nvrtc_check(nvrtcDestroyProgram(&program), "nvrtcDestroyProgram");
		if(use_ptx_cache) store_cached_ptx(cache_file, ptx);
		cuda_check(cuModuleLoadDataEx(
			&state->module, ptx.data(), 0u, nullptr, nullptr),
			"cuModuleLoadDataEx");
		print_info("CUDA C++ code successfully compiled.");
	}
	if(std::getenv("CULBM_DUMP_PTX") != nullptr) {
		write_file(
			get_exe_path()+"cuda-kernel-device-"+to_string(info.id)+".ptx",
			string(ptx.data(), ptx.size() > 0u ? ptx.size()-1u : 0u));
	}
	exists = true;
}

void Device::finish_queue() const {
	if(!state) return;
	cuda_set_current(state->context);
	cuda_check(cuStreamSynchronize(state->stream), "cuStreamSynchronize");
}

size_t Device::get_free_memory_bytes() const {
	if(!state) return 0u;
	cuda_set_current(state->context);
	size_t free_bytes = 0u, total_bytes = 0u;
	cuda_check(cuMemGetInfo(&free_bytes, &total_bytes), "cuMemGetInfo");
	return free_bytes;
}

size_t Device::get_total_memory_bytes() const {
	if(!state) return 0u;
	cuda_set_current(state->context);
	size_t free_bytes = 0u, total_bytes = 0u;
	cuda_check(cuMemGetInfo(&free_bytes, &total_bytes), "cuMemGetInfo");
	return total_bytes;
}

void Device::record_transfer_ready(
	const CudaTransferChannel channel) const {
	if(!state) return;
	cuda_set_current(state->context);
	CudaTransferState& transfer = transfer_state(state, channel);
	cuda_check(
		cuEventRecord(transfer.ready, state->stream),
		"cuEventRecord(transfer_ready)");
}

void Device::begin_transfer_batch(
	const CudaTransferChannel channel) const {
	if(!state) return;
	cuda_set_current(state->context);
	CudaTransferState& transfer = transfer_state(state, channel);
	cuda_check(
		cuStreamWaitEvent(
			transfer.transfer_stream, transfer.consumed, 0u),
		"cuStreamWaitEvent(transfer_consumed)");
}

void Device::complete_transfer_batch(
	const CudaTransferChannel channel) const {
	if(!state) return;
	cuda_set_current(state->context);
	CudaTransferState& transfer = transfer_state(state, channel);
	cuda_check(
		cuEventRecord(transfer.complete, transfer.transfer_stream),
		"cuEventRecord(transfer_complete)");
	cuda_check(
		cuStreamWaitEvent(state->stream, transfer.complete, 0u),
		"cuStreamWaitEvent(transfer_complete)");
}

void Device::record_transfer_consumed(
	const CudaTransferChannel channel) const {
	if(!state) return;
	cuda_set_current(state->context);
	CudaTransferState& transfer = transfer_state(state, channel);
	cuda_check(
		cuEventRecord(transfer.consumed, state->stream),
		"cuEventRecord(transfer_consumed)");
}

void Device::wait_for_transfer_batch(
	const Device& destination,
	const CudaTransferChannel channel) const {
	if(!state || !destination.state) return;
	cuda_set_current(state->context);
	CudaTransferState& destination_transfer =
		transfer_state(destination.state, channel);
	cuda_check(
		cuStreamWaitEvent(
			state->stream, destination_transfer.complete, 0u),
		"cuStreamWaitEvent(outbound transfer_complete)");
}

void cuda_enqueue_transfer_copy(
	const std::shared_ptr<CudaDeviceState>& destination,
	const CUdeviceptr destination_buffer,
	const std::shared_ptr<CudaDeviceState>& source,
	const CUdeviceptr source_buffer,
	void* const host_staging_buffer,
	const bool host_staging_buffer_pinned,
	const CUevent host_staging_ready,
	const size_t bytes,
	const CudaTransferChannel channel) {
	if(bytes == 0u) return;
	if(!destination || !source)
		print_error("CUDA inter-device transfer has no device state.");
	CudaTransferState& destination_transfer =
		transfer_state(destination, channel);
	CudaTransferState& source_transfer =
		transfer_state(source, channel);

	if(destination->context == source->context) {
		cuda_set_current(destination->context);
		cuda_check(
			cuStreamWaitEvent(
				destination_transfer.transfer_stream,
				source_transfer.ready, 0u),
			"cuStreamWaitEvent(transfer_ready)");
		cuda_check(
			cuMemcpyDtoDAsync(
				destination_buffer, source_buffer, bytes,
				destination_transfer.transfer_stream),
			"cuMemcpyDtoDAsync(halo)");
		return;
	}

	if(enable_peer_access(destination, source)) {
		cuda_set_current(destination->context);
		cuda_check(
			cuStreamWaitEvent(
				destination_transfer.transfer_stream,
				source_transfer.ready, 0u),
			"cuStreamWaitEvent(transfer_ready)");
		const CUresult result = cuMemcpyPeerAsync(
			destination_buffer, destination->context,
			source_buffer, source->context,
			bytes, destination_transfer.transfer_stream);
		if(result == CUDA_SUCCESS) return;
		print_warning(
			"CUDA P2P halo copy failed for GPU "+
			to_string(source->ordinal)+" -> GPU "+
			to_string(destination->ordinal)+" ("+
			cuda_error_text(result)+"); using host fallback.");
	}

	if(host_staging_buffer == nullptr)
		print_error("CUDA host fallback has no staging buffer.");
	if(host_staging_ready == nullptr)
		print_error("CUDA host fallback has no staging event.");
	if(host_staging_buffer_pinned) {
		cuda_set_current(source->context);
		cuda_check(
			cuStreamWaitEvent(
				source_transfer.outbound_stream,
				source_transfer.ready, 0u),
			"cuStreamWaitEvent(outbound transfer_ready)");
		cuda_check(
			cuMemcpyDtoHAsync(
				host_staging_buffer, source_buffer, bytes,
				source_transfer.outbound_stream),
			"cuMemcpyDtoHAsync(halo fallback)");
		cuda_check(
			cuEventRecord(
				host_staging_ready, source_transfer.outbound_stream),
			"cuEventRecord(host_staging_ready)");
		cuda_set_current(destination->context);
		cuda_check(
			cuStreamWaitEvent(
				destination_transfer.transfer_stream,
				host_staging_ready, 0u),
			"cuStreamWaitEvent(host_staging_ready)");
		cuda_check(
			cuMemcpyHtoDAsync(
				destination_buffer, host_staging_buffer, bytes,
				destination_transfer.transfer_stream),
			"cuMemcpyHtoDAsync(halo fallback)");
		return;
	}

	cuda_set_current(source->context);
	cuda_check(
		cuEventSynchronize(source_transfer.ready),
		"cuEventSynchronize(transfer_ready)");
	cuda_check(
		cuMemcpyDtoH(host_staging_buffer, source_buffer, bytes),
		"cuMemcpyDtoH(halo fallback)");
	cuda_set_current(destination->context);
	cuda_check(
		cuStreamSynchronize(destination_transfer.transfer_stream),
		"cuStreamSynchronize(halo fallback)");
	cuda_check(
		cuMemcpyHtoD(destination_buffer, host_staging_buffer, bytes),
		"cuMemcpyHtoD(halo fallback)");
}

void Kernel::initialize(const Device& device, const string& kernel_name) {
	if(!device.is_initialized())
		print_error("No CUDA Device selected. Call Device constructor.");
	name = kernel_name;
	launch_operation = "cuLaunchKernel("+name+")";
	state = device.get_cuda_state();
	cuda_set_current(state->context);
	const CUresult result = cuModuleGetFunction(
		&function, state->module, name.c_str());
	if(result != CUDA_SUCCESS) {
		print_error("There is no CUDA kernel with name \""+name+
			"(...)\" in the translated CULBM kernel source.");
	}
	if(name=="stream_collide") {
		int suggested_block_size = (int)WORKGROUP_SIZE;
		const char* const override_value =
			std::getenv("CULBM_STREAM_BLOCK_SIZE");
		if(override_value != nullptr) {
			char* end = nullptr;
			const unsigned long parsed =
				std::strtoul(override_value, &end, 10);
			if(end==override_value || *end!='\0' ||
			   parsed<32ul || parsed>1024ul || parsed%32ul!=0ul) {
				print_error(
					"CULBM_STREAM_BLOCK_SIZE must be a "
					"multiple of 32 in [32, 1024].");
			}
			suggested_block_size = (int)parsed;
		}
		preferred_workgroup_size = (ulong)suggested_block_size;
		print_info(
			"CUDA stream_collide block size: "+
			to_string((uint)preferred_workgroup_size)+" threads.");
	}
}

Kernel& Kernel::enqueue_run(const uint times,
                            const vector<Event>*, Event*) {
	if(N == 0ull || times == 0u) return *this;
	cuda_set_current(state->context);
	const ulong blocks_64 = (N+workgroup_size-1ull)/workgroup_size;
	if(blocks_64 > (ulong)0x7FFFFFFFu)
		print_error("CUDA launch grid is too large for kernel \""+name+"\".");
	const uint blocks = (uint)blocks_64;
	const uint threads = (uint)workgroup_size;

	if(parameter_pointers_dirty) {
		parameter_pointers.clear();
		parameter_pointers.reserve(number_of_parameters);
		for(uint i=0u; i<number_of_parameters; ++i) {
			if(i >= arguments.size() || arguments[i].bytes.empty())
				print_error("CUDA kernel \""+name+
					"\" has an unset argument at index "+to_string(i)+".");
			parameter_pointers.push_back(arguments[i].bytes.data());
		}
		parameter_pointers_dirty = false;
	}

	for(uint i=0u; i<times; ++i) {
		cuda_check(cuLaunchKernel(
			function,
			blocks, 1u, 1u,
			threads, 1u, 1u,
			0u, state->stream,
			parameter_pointers.data(), nullptr),
			launch_operation.c_str());
	}
	return *this;
}

Kernel& Kernel::finish_queue() {
	if(state) {
		cuda_set_current(state->context);
		cuda_check(cuStreamSynchronize(state->stream),
		           ("cuStreamSynchronize("+name+")").c_str());
	}
	return *this;
}
