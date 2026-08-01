#pragma once

#define WORKGROUP_SIZE 64

#ifdef pi
#pragma push_macro("pi")
#undef pi
#define CULBM_RESTORE_PI
#endif
#include <cuda.h>
#include <nvrtc.h>
#ifdef CULBM_RESTORE_PI
#pragma pop_macro("pi")
#undef CULBM_RESTORE_PI
#endif

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "cuda_mpi.hpp"
#include "cuda_topology.hpp"
#include "utilities.hpp"

// Event parameters are optional queue plumbing. The headless CULBM execution
// path does not construct explicit events, so this lightweight type keeps the
// public API compact without exposing CUDA implementation details.
struct Event {};

string get_cuda_kernel_source();

struct Device_Info {
	CUdevice cu_device = 0;
	uint id = 0u;
	string name = "", vendor = "NVIDIA";
	string driver_version = "", cuda_c_version = "CUDA";
	uint memory = 0u;
	uint memory_used = 0u;
	uint global_cache = 0u, local_cache = 0u;
	uint max_global_buffer = 0u, max_constant_buffer = 64u;
	uint compute_units = 0u;
	uint clock_frequency = 0u;
	bool is_cpu = false, is_gpu = true, uses_ram = false;
	uint is_fp64_capable = 1u, is_fp32_capable = 1u, is_fp16_capable = 2u;
	uint is_int64_capable = 1u, is_int32_capable = 1u;
	uint is_int16_capable = 1u, is_int8_capable = 1u;
	uint is_dp4a_capable = 1u;
	uint cores = 0u;
	float tflops = 0.0f;
	uint nvidia_compute_capability = 0u;
	bool patch_intel_gpu_above_4gb = false;
	bool patch_nvidia_fp16 = true;
	bool patch_legacy_gpu_fma = false;

	Device_Info() = default;
	Device_Info(CUdevice device, uint ordinal);
};

enum class CudaTransferChannel : std::uint32_t {
	simulation = 0u,
	graphics = 1u
};

constexpr std::uint32_t cuda_transfer_channel_count = 2u;

struct CudaTransferState {
	CUstream transfer_stream = nullptr;
	CUstream outbound_stream = nullptr;
	CUevent ready = nullptr;
	CUevent complete = nullptr;
	CUevent consumed = nullptr;
};

void cuda_check(CUresult result, const char* operation);
void nvrtc_check(nvrtcResult result, const char* operation);
void cuda_set_current(CUcontext context);

inline void print_device_info(const Device_Info& d) {
#if defined(_WIN32)
	const string os = "Windows";
#elif defined(__linux__)
	const string os = "Linux";
#else
	const string os = "unknown operating system";
#endif
	println("|-----------------.-----------------------------------------------------------|");
	println("| Device ID      | "+alignl(58, to_string(d.id))+" |");
	println("| Device Name    | "+alignl(58, d.name)+" |");
	println("| Device Vendor  | "+alignl(58, d.vendor)+" |");
	println("| CUDA Driver    | "+alignl(58, d.driver_version+" ("+os+")")+" |");
	println("| CUDA Backend   | "+alignl(58, "NVRTC, compute capability "+
		to_string(d.nvidia_compute_capability/10u)+"."+
		to_string(d.nvidia_compute_capability%10u))+" |");
	println("| Compute Units  | "+alignl(58, to_string(d.compute_units)+" at "+
		to_string(d.clock_frequency)+" MHz ("+to_string(d.cores)+" cores, "+
		to_string(d.tflops, 3)+" TFLOPs/s)")+" |");
	println("| Memory, Cache  | "+alignl(58, to_string(d.memory)+" MB VRAM, "+
		to_string(d.global_cache)+" KB L2 / "+to_string(d.local_cache)+
		" KB shared per block")+" |");
	println("|-----------------'-----------------------------------------------------------|");
}

vector<Device_Info> get_devices(bool print_info = true);

inline Device_Info select_device_with_most_flops(
	const vector<Device_Info>& devices = get_devices()) {
	float best_value = -1.0f;
	uint best_i = 0u;
	for(uint i = 0u; i < (uint)devices.size(); ++i) {
		if(devices[i].tflops > best_value) {
			best_value = devices[i].tflops;
			best_i = i;
		}
	}
	return devices[best_i];
}

inline Device_Info select_device_with_most_memory(
	const vector<Device_Info>& devices = get_devices()) {
	uint best_value = 0u;
	uint best_i = 0u;
	for(uint i = 0u; i < (uint)devices.size(); ++i) {
		if(devices[i].memory > best_value) {
			best_value = devices[i].memory;
			best_i = i;
		}
	}
	return devices[best_i];
}

inline Device_Info select_device_with_id(
	const uint id, const vector<Device_Info>& devices = get_devices()) {
	if(id < (uint)devices.size()) return devices[id];
	print_error("Your selected CUDA Device ID ("+to_string(id)+") is wrong.");
	return devices[0];
}

struct CudaDeviceState {
	CUdevice device = 0;
	CUcontext context = nullptr;
	CUstream stream = nullptr;
	CudaTransferState transfer[cuda_transfer_channel_count];
	CUmodule module = nullptr;
	uint ordinal = 0u;

	~CudaDeviceState();
};

void cuda_enqueue_transfer_copy(
	const std::shared_ptr<CudaDeviceState>& destination,
	const CUdeviceptr destination_buffer,
	const std::shared_ptr<CudaDeviceState>& source,
	const CUdeviceptr source_buffer,
	void* const host_staging_buffer,
	const bool host_staging_buffer_pinned,
	const CUevent host_staging_ready,
	const size_t bytes,
	const CudaTransferChannel channel);

class Device {
private:
	std::shared_ptr<CudaDeviceState> state;
	bool exists = false;

public:
	Device_Info info;

	Device(const Device_Info& info,
	       const string& cuda_kernel_source = get_cuda_kernel_source());
	Device() = default;

	void barrier(const vector<Event>* = nullptr, Event* = nullptr) {
		finish_queue();
	}
	void finish_queue() const;
	void record_transfer_ready(
		CudaTransferChannel channel = CudaTransferChannel::simulation) const;
	void begin_transfer_batch(
		CudaTransferChannel channel = CudaTransferChannel::simulation) const;
	void complete_transfer_batch(
		CudaTransferChannel channel = CudaTransferChannel::simulation) const;
	void record_transfer_consumed(
		CudaTransferChannel channel = CudaTransferChannel::simulation) const;
	void wait_for_transfer_batch(
		const Device& destination,
		CudaTransferChannel channel = CudaTransferChannel::simulation) const;
	size_t get_free_memory_bytes() const;
	size_t get_total_memory_bytes() const;
	CUcontext get_cuda_context() const { return state->context; }
	CUstream get_cuda_stream() const { return state->stream; }
	CUmodule get_cuda_module() const { return state->module; }
	std::shared_ptr<CudaDeviceState> get_cuda_state() const { return state; }
	bool is_initialized() const { return exists; }
};

template<typename T>
class Memory {
private:
	ulong N = 0ull;
	uint d = 1u;
	bool host_buffer_exists = false;
	bool device_buffer_exists = false;
	bool external_host_buffer = false;
	bool host_buffer_pinned = false;
	mutable CUevent host_staging_ready = nullptr;
	T* host_buffer = nullptr;
	T* host_buffer_unaligned = nullptr;
	CUdeviceptr device_buffer = 0;
	Device* device = nullptr;
	std::shared_ptr<CudaDeviceState> state;

	void initialize_auxiliary_pointers() {
		x = s0 = host_buffer;
		if(d > 0x4u) s4 = host_buffer+N*0x4ull;
		if(d > 0x8u) s8 = host_buffer+N*0x8ull;
		if(d > 0xCu) sC = host_buffer+N*0xCull;
		if(d > 0x1u) y = s1 = host_buffer+N;
		if(d > 0x5u) s5 = host_buffer+N*0x5ull;
		if(d > 0x9u) s9 = host_buffer+N*0x9ull;
		if(d > 0xDu) sD = host_buffer+N*0xDull;
		if(d > 0x2u) z = s2 = host_buffer+N*0x2ull;
		if(d > 0x6u) s6 = host_buffer+N*0x6ull;
		if(d > 0xAu) sA = host_buffer+N*0xAull;
		if(d > 0xEu) sE = host_buffer+N*0xEull;
		if(d > 0x3u) w = s3 = host_buffer+N*0x3ull;
		if(d > 0x7u) s7 = host_buffer+N*0x7ull;
		if(d > 0xBu) sB = host_buffer+N*0xBull;
		if(d > 0xFu) sF = host_buffer+N*0xFull;
	}

	void clear_auxiliary_pointers() {
		x = y = z = w = nullptr;
		s0 = s1 = s2 = s3 = s4 = s5 = s6 = s7 = nullptr;
		s8 = s9 = sA = sB = sC = sD = sE = sF = nullptr;
	}

	void allocate_host_buffer(Device& linked_device,
	                          const bool allocate_host,
	                          const bool prefer_pinned) {
		if(!allocate_host) return;
		const size_t bytes = (size_t)capacity();
		if(prefer_pinned) {
			cuda_set_current(linked_device.get_cuda_context());
			void* ptr = nullptr;
			if(cuMemHostAlloc(&ptr, bytes, CU_MEMHOSTALLOC_PORTABLE)==CUDA_SUCCESS) {
				host_buffer = static_cast<T*>(ptr);
				host_buffer_unaligned = host_buffer;
				host_buffer_pinned = true;
			}
		}
		if(host_buffer == nullptr) {
#if defined(_WIN32)
			host_buffer = static_cast<T*>(_aligned_malloc(bytes, 64u));
#else
			const size_t padded = (bytes+63u)&~size_t(63u);
			host_buffer = static_cast<T*>(std::aligned_alloc(64u, padded));
#endif
			if(host_buffer == nullptr) print_error("CUDA host buffer allocation failed.");
			host_buffer_unaligned = host_buffer;
		}
		host_buffer_exists = true;
		initialize_auxiliary_pointers();
	}

	void allocate_device_buffer(Device& linked_device,
	                            const bool allocate_device) {
		device = &linked_device;
		state = linked_device.get_cuda_state();
		if(!allocate_device) return;
		const ulong capacity_bytes = capacity();
		const ulong allocation_mb =
			capacity_bytes/1048576ull+
			(capacity_bytes%1048576ull!=0ull);
		const uint allocation_mb_u = (uint)min(
			(ulong)max_uint, allocation_mb);
		linked_device.info.memory_used =
			allocation_mb_u>max_uint-linked_device.info.memory_used
				? max_uint
				: linked_device.info.memory_used+allocation_mb_u;
		if(linked_device.info.memory_used > linked_device.info.memory) {
			print_error("Device \""+linked_device.info.name+
				"\" does not have enough memory. Allocating another "+
				to_string(allocation_mb_u)+
				" MB would use a total of "+
				to_string(linked_device.info.memory_used)+" MB / "+
				to_string(linked_device.info.memory)+" MB.");
		}
		cuda_set_current(state->context);
		size_t free_bytes = 0u, total_bytes = 0u;
		cuda_check(
			cuMemGetInfo(&free_bytes, &total_bytes),
			"cuMemGetInfo before allocation");
		const size_t reserve_bytes = min(
			(size_t)(256ull*1048576ull), total_bytes/10u);
		if(capacity_bytes > free_bytes ||
		   free_bytes-(size_t)capacity_bytes < reserve_bytes) {
			print_error(
				"Device \""+linked_device.info.name+
				"\" has insufficient free VRAM for a "+
				to_string(allocation_mb_u)+
				" MB allocation while retaining the CUDA runtime reserve. "+
				"Free="+to_string((ulong)(free_bytes/1048576ull))+
				" MB, reserve="+
				to_string((ulong)(reserve_bytes/1048576ull))+" MB.");
		}
		cuda_check(cuMemAlloc(&device_buffer, (size_t)capacity_bytes),
		           "cuMemAlloc");
		device_buffer_exists = true;
	}

	void copy_from_device(const ulong offset, const ulong length,
	                      const bool blocking) {
		if(!host_buffer_exists || !device_buffer_exists || length == 0ull) return;
		cuda_set_current(state->context);
		const size_t bytes = (size_t)(length*sizeof(T));
		void* target = static_cast<void*>(host_buffer+offset);
		const CUdeviceptr source = device_buffer+offset*sizeof(T);
		// Pageable buffers cannot provide reliable asynchronous transfer
		// semantics.  The simulation stream is CU_STREAM_NON_BLOCKING, so a
		// synchronous default-stream copy alone does not order itself against
		// preceding/following kernels.  Explicit barriers make the fallback a
		// genuinely blocking operation.  Pinned buffers honor the caller's
		// asynchronous choice and stay on the owning stream.
		if(blocking || !host_buffer_pinned) {
			cuda_check(
				cuStreamSynchronize(state->stream),
				"cuStreamSynchronize before cuMemcpyDtoH");
			cuda_check(cuMemcpyDtoH(target, source, bytes), "cuMemcpyDtoH");
			cuda_check(
				cuCtxSynchronize(),
				"cuCtxSynchronize after cuMemcpyDtoH");
		} else {
			cuda_check(cuMemcpyDtoHAsync(target, source, bytes, state->stream),
			           "cuMemcpyDtoHAsync");
		}
	}

	void copy_to_device(const ulong offset, const ulong length,
	                    const bool blocking) {
		if(!host_buffer_exists || !device_buffer_exists || length == 0ull) return;
		cuda_set_current(state->context);
		const size_t bytes = (size_t)(length*sizeof(T));
		const void* source = static_cast<const void*>(host_buffer+offset);
		const CUdeviceptr target = device_buffer+offset*sizeof(T);
		if(blocking || !host_buffer_pinned) {
			cuda_check(
				cuStreamSynchronize(state->stream),
				"cuStreamSynchronize before cuMemcpyHtoD");
			cuda_check(cuMemcpyHtoD(target, source, bytes), "cuMemcpyHtoD");
			cuda_check(
				cuCtxSynchronize(),
				"cuCtxSynchronize after cuMemcpyHtoD");
		} else {
			cuda_check(cuMemcpyHtoDAsync(target, source, bytes, state->stream),
			           "cuMemcpyHtoDAsync");
		}
	}

public:
	T *x=nullptr, *y=nullptr, *z=nullptr, *w=nullptr;
	T *s0=nullptr, *s1=nullptr, *s2=nullptr, *s3=nullptr;
	T *s4=nullptr, *s5=nullptr, *s6=nullptr, *s7=nullptr;
	T *s8=nullptr, *s9=nullptr, *sA=nullptr, *sB=nullptr;
	T *sC=nullptr, *sD=nullptr, *sE=nullptr, *sF=nullptr;

	Memory(Device& linked_device, const ulong length,
	       const uint dimensions=1u, const bool allocate_host=true,
	       const bool allocate_device=true, const T value=(T)0,
	       const bool allow_zero_copy=true) {
		if(!linked_device.is_initialized())
			print_error("No CUDA Device selected. Call Device constructor.");
		if(length*(ulong)dimensions == 0ull)
			print_error("Memory size must be larger than 0.");
		N = length;
		d = dimensions;
		// Transfer buffers explicitly disable zero-copy in CULBM. They are
		// small and latency-sensitive, so pin exactly those host allocations.
		allocate_host_buffer(linked_device, allocate_host, !allow_zero_copy);
		allocate_device_buffer(linked_device, allocate_device);
		reset(value);
	}

	Memory(Device& linked_device, const ulong length, const uint dimensions,
	       T* const external_buffer, const bool allocate_device=true,
	       const bool = true) {
		if(!linked_device.is_initialized())
			print_error("No CUDA Device selected. Call Device constructor.");
		if(length*(ulong)dimensions == 0ull)
			print_error("Memory size must be larger than 0.");
		N = length;
		d = dimensions;
		host_buffer = external_buffer;
		host_buffer_unaligned = external_buffer;
		host_buffer_exists = true;
		external_host_buffer = true;
		initialize_auxiliary_pointers();
		allocate_device_buffer(linked_device, allocate_device);
		write_to_device();
	}

	Memory() = default;
	Memory(const Memory&) = delete;
	Memory& operator=(const Memory&) = delete;

	~Memory() { delete_buffers(); }

	Memory(Memory&& other) noexcept { *this = std::move(other); }

	Memory& operator=(Memory&& other) noexcept {
		if(this == &other) return *this;
		delete_buffers();
		N = other.N;
		d = other.d;
		host_buffer_exists = other.host_buffer_exists;
		device_buffer_exists = other.device_buffer_exists;
		external_host_buffer = other.external_host_buffer;
		host_buffer_pinned = other.host_buffer_pinned;
		host_staging_ready = other.host_staging_ready;
		host_buffer = other.host_buffer;
		host_buffer_unaligned = other.host_buffer_unaligned;
		device_buffer = other.device_buffer;
		device = other.device;
		state = std::move(other.state);
		if(host_buffer_exists) initialize_auxiliary_pointers();

		other.N = 0ull;
		other.d = 1u;
		other.host_buffer_exists = false;
		other.device_buffer_exists = false;
		other.external_host_buffer = false;
		other.host_buffer_pinned = false;
		other.host_staging_ready = nullptr;
		other.host_buffer = nullptr;
		other.host_buffer_unaligned = nullptr;
		other.device_buffer = 0;
		other.device = nullptr;
		other.clear_auxiliary_pointers();
		return *this;
	}

	T* const exchange_host_buffer(T* const replacement) {
		T* const old = host_buffer;
		host_buffer = replacement;
		initialize_auxiliary_pointers();
		return old;
	}

	T* const exchange_host_buffer_unaligned(T* const replacement) {
		T* const old = host_buffer_unaligned;
		host_buffer_unaligned = replacement;
		return old;
	}

	void add_host_buffer() {
		if(!host_buffer_exists && device_buffer_exists) {
			allocate_host_buffer(*device, true, false);
			read_from_device();
		} else if(!device_buffer_exists) {
			print_error("There is no existing device buffer, so can't add host buffer.");
		}
	}

	void add_device_buffer(const bool = true) {
		if(!device_buffer_exists && host_buffer_exists) {
			allocate_device_buffer(*device, true);
			write_to_device();
		} else if(!host_buffer_exists) {
			print_error("There is no existing host buffer, so can't add device buffer.");
		}
	}

	void delete_host_buffer() {
		if(host_staging_ready != nullptr) {
			if(state) cuda_set_current(state->context);
			cuda_check(
				cuEventDestroy(host_staging_ready),
				"cuEventDestroy(host_staging_ready)");
			host_staging_ready = nullptr;
		}
		if(!host_buffer_exists) return;
		if(!external_host_buffer && host_buffer_unaligned != nullptr) {
			if(host_buffer_pinned) {
				if(state) cuda_set_current(state->context);
				cuda_check(cuMemFreeHost(host_buffer_unaligned), "cuMemFreeHost");
			} else {
#if defined(_WIN32)
				_aligned_free(host_buffer_unaligned);
#else
				std::free(host_buffer_unaligned);
#endif
			}
		}
		host_buffer_exists = false;
		host_buffer = nullptr;
		host_buffer_unaligned = nullptr;
		host_buffer_pinned = false;
		clear_auxiliary_pointers();
		if(!device_buffer_exists) {
			N = 0ull;
			d = 1u;
		}
	}

	void delete_device_buffer() {
		if(!device_buffer_exists) return;
		if(device != nullptr) {
			const ulong capacity_bytes = capacity();
			const ulong allocation_mb =
				capacity_bytes/1048576ull+
				(capacity_bytes%1048576ull!=0ull);
			const uint allocation_mb_u = (uint)min(
				(ulong)max_uint, allocation_mb);
			device->info.memory_used =
				allocation_mb_u>device->info.memory_used
					? 0u
					: device->info.memory_used-allocation_mb_u;
		}
		if(state) {
			cuda_set_current(state->context);
			cuda_check(cuMemFree(device_buffer), "cuMemFree");
		}
		device_buffer_exists = false;
		device_buffer = 0;
		if(!host_buffer_exists) {
			N = 0ull;
			d = 1u;
		}
	}

	void delete_buffers() {
		delete_device_buffer();
		delete_host_buffer();
	}

	void reset(const T value=(T)0) {
		if(host_buffer_exists) std::fill(host_buffer, host_buffer+range(), value);
		if(host_buffer_exists) write_to_device();
		else if(device_buffer_exists && value == (T)0) {
			cuda_set_current(state->context);
			cuda_check(cuMemsetD8(device_buffer, 0u, (size_t)capacity()),
			           "cuMemsetD8");
		}
	}

	const ulong length() const { return N; }
	const uint dimensions() const { return d; }
	const ulong range() const { return N*(ulong)d; }
	const ulong capacity() const { return range()*sizeof(T); }
	bool is_host_buffer_pinned() const { return host_buffer_pinned; }
	T* const data() { return host_buffer; }
	const T* const data() const { return host_buffer; }
	T* const operator()() { return host_buffer; }
	const T* const operator()() const { return host_buffer; }
	T& operator[](const ulong i) { return host_buffer[i]; }
	const T& operator[](const ulong i) const { return host_buffer[i]; }
	const T operator()(const ulong i) const { return host_buffer[i]; }
	const T operator()(const ulong i, const uint dimension) const {
		return host_buffer[i+(ulong)dimension*N];
	}

	void read_from_device(const bool blocking=true,
	                      const vector<Event>* = nullptr, Event* = nullptr) {
		copy_from_device(0ull, range(), blocking);
	}

	void write_to_device(const bool blocking=true,
	                     const vector<Event>* = nullptr, Event* = nullptr) {
		copy_to_device(0ull, range(), blocking);
	}

	void read_from_device(const ulong offset, const ulong length,
	                      const bool blocking=true,
	                      const vector<Event>* = nullptr, Event* = nullptr) {
		const ulong safe_offset = min(offset, range());
		const ulong safe_length = min(length, range()-safe_offset);
		copy_from_device(safe_offset, safe_length, blocking);
	}

	void write_to_device(const ulong offset, const ulong length,
	                     const bool blocking=true,
	                     const vector<Event>* = nullptr, Event* = nullptr) {
		const ulong safe_offset = min(offset, range());
		const ulong safe_length = min(length, range()-safe_offset);
		copy_to_device(safe_offset, safe_length, blocking);
	}

	void read_from_device_1d(const ulong x0, const ulong x1,
	                         const int dimension=-1, const bool blocking=true,
	                         const vector<Event>* = nullptr, Event* = nullptr) {
		const uint i0=(uint)max(0, dimension), i1=dimension<0 ? d : i0+1u;
		for(uint i=i0; i<i1; ++i)
			read_from_device((ulong)i*N+x0, x1-x0, false);
		if(blocking) finish_queue();
	}

	void write_to_device_1d(const ulong x0, const ulong x1,
	                        const int dimension=-1, const bool blocking=true,
	                        const vector<Event>* = nullptr, Event* = nullptr) {
		const uint i0=(uint)max(0, dimension), i1=dimension<0 ? d : i0+1u;
		for(uint i=i0; i<i1; ++i)
			write_to_device((ulong)i*N+x0, x1-x0, false);
		if(blocking) finish_queue();
	}

	void read_from_device_2d(const ulong x0, const ulong x1,
	                         const ulong y0, const ulong y1,
	                         const ulong Nx, const ulong,
	                         const int dimension=-1, const bool blocking=true,
	                         const vector<Event>* = nullptr, Event* = nullptr) {
		for(ulong yv=y0; yv<y1; ++yv)
			read_from_device_1d(x0+yv*Nx, x1+yv*Nx, dimension, false);
		if(blocking) finish_queue();
	}

	void write_to_device_2d(const ulong x0, const ulong x1,
	                        const ulong y0, const ulong y1,
	                        const ulong Nx, const ulong,
	                        const int dimension=-1, const bool blocking=true,
	                        const vector<Event>* = nullptr, Event* = nullptr) {
		for(ulong yv=y0; yv<y1; ++yv)
			write_to_device_1d(x0+yv*Nx, x1+yv*Nx, dimension, false);
		if(blocking) finish_queue();
	}

	// A single z plane is strided by the local x pitch but otherwise
	// rectangular.  CUDA's 2D DMA path replaces one Driver call per row with
	// one call per field component.  Nested-grid interfaces use this for their
	// hot plane transfers; keeping it here also preserves the normal Memory
	// ownership and pageable-memory fallback semantics.
	void read_from_device_2d_pitched(const ulong x0, const ulong x1,
	                                 const ulong y0, const ulong y1,
	                                 const ulong z,
	                                 const ulong Nx, const ulong Ny,
	                                 const int dimension=-1,
	                                 const bool blocking=true,
	                                 const vector<Event>* = nullptr,
	                                 Event* = nullptr) {
		if(!host_buffer_exists || !device_buffer_exists ||
		   x1 <= x0 || y1 <= y0) return;
		const uint i0=(uint)max(0, dimension), i1=dimension<0 ? d : i0+1u;
		const ulong base=x0+(y0+z*Ny)*Nx;
		const size_t pitch=(size_t)Nx*sizeof(T);
		const size_t width=(size_t)(x1-x0)*sizeof(T);
		const size_t height=(size_t)(y1-y0);
		cuda_set_current(state->context);
		const bool synchronous = blocking || !host_buffer_pinned;
		if(synchronous) {
			cuda_check(
				cuStreamSynchronize(state->stream),
				"cuStreamSynchronize before cuMemcpy2D DtoH");
		}
		for(uint i=i0; i<i1; ++i) {
			const ulong offset=(ulong)i*N+base;
			CUDA_MEMCPY2D copy{};
			copy.srcMemoryType=CU_MEMORYTYPE_DEVICE;
			copy.srcDevice=device_buffer+(CUdeviceptr)(offset*sizeof(T));
			copy.srcPitch=pitch;
			copy.dstMemoryType=CU_MEMORYTYPE_HOST;
			copy.dstHost=static_cast<void*>(host_buffer+offset);
			copy.dstPitch=pitch;
			copy.WidthInBytes=width;
			copy.Height=height;
			if(synchronous) {
				cuda_check(cuMemcpy2D(&copy), "cuMemcpy2D");
			} else {
				cuda_check(cuMemcpy2DAsync(&copy, state->stream), "cuMemcpy2DAsync");
			}
		}
		if(synchronous) {
			cuda_check(
				cuCtxSynchronize(),
				"cuCtxSynchronize after cuMemcpy2D DtoH");
		}
	}

	void write_to_device_2d_pitched(const ulong x0, const ulong x1,
	                                const ulong y0, const ulong y1,
	                                const ulong z,
	                                const ulong Nx, const ulong Ny,
	                                const int dimension=-1,
	                                const bool blocking=true,
	                                const vector<Event>* = nullptr,
	                                Event* = nullptr) {
		if(!host_buffer_exists || !device_buffer_exists ||
		   x1 <= x0 || y1 <= y0) return;
		const uint i0=(uint)max(0, dimension), i1=dimension<0 ? d : i0+1u;
		const ulong base=x0+(y0+z*Ny)*Nx;
		const size_t pitch=(size_t)Nx*sizeof(T);
		const size_t width=(size_t)(x1-x0)*sizeof(T);
		const size_t height=(size_t)(y1-y0);
		cuda_set_current(state->context);
		const bool synchronous = blocking || !host_buffer_pinned;
		if(synchronous) {
			cuda_check(
				cuStreamSynchronize(state->stream),
				"cuStreamSynchronize before cuMemcpy2D HtoD");
		}
		for(uint i=i0; i<i1; ++i) {
			const ulong offset=(ulong)i*N+base;
			CUDA_MEMCPY2D copy{};
			copy.srcMemoryType=CU_MEMORYTYPE_HOST;
			copy.srcHost=static_cast<const void*>(host_buffer+offset);
			copy.srcPitch=pitch;
			copy.dstMemoryType=CU_MEMORYTYPE_DEVICE;
			copy.dstDevice=device_buffer+(CUdeviceptr)(offset*sizeof(T));
			copy.dstPitch=pitch;
			copy.WidthInBytes=width;
			copy.Height=height;
			if(synchronous) {
				cuda_check(cuMemcpy2D(&copy), "cuMemcpy2D");
			} else {
				cuda_check(cuMemcpy2DAsync(&copy, state->stream), "cuMemcpy2DAsync");
			}
		}
		if(synchronous) {
			cuda_check(
				cuCtxSynchronize(),
				"cuCtxSynchronize after cuMemcpy2D HtoD");
		}
	}

	void read_from_device_3d(const ulong x0, const ulong x1,
	                         const ulong y0, const ulong y1,
	                         const ulong z0, const ulong z1,
	                         const ulong Nx, const ulong Ny, const ulong,
	                         const int dimension=-1, const bool blocking=true,
	                         const vector<Event>* = nullptr, Event* = nullptr) {
		for(ulong zv=z0; zv<z1; ++zv)
			for(ulong yv=y0; yv<y1; ++yv) {
				const ulong base=(yv+zv*Ny)*Nx;
				read_from_device_1d(x0+base, x1+base, dimension, false);
			}
		if(blocking) finish_queue();
	}

	void write_to_device_3d(const ulong x0, const ulong x1,
	                        const ulong y0, const ulong y1,
	                        const ulong z0, const ulong z1,
	                        const ulong Nx, const ulong Ny, const ulong,
	                        const int dimension=-1, const bool blocking=true,
	                        const vector<Event>* = nullptr, Event* = nullptr) {
		for(ulong zv=z0; zv<z1; ++zv)
			for(ulong yv=y0; yv<y1; ++yv) {
				const ulong base=(yv+zv*Ny)*Nx;
				write_to_device_1d(x0+base, x1+base, dimension, false);
			}
		if(blocking) finish_queue();
	}

	void enqueue_read_from_device(const vector<Event>* wait=nullptr,
	                              Event* returned=nullptr) {
		read_from_device(false, wait, returned);
	}
	void enqueue_write_to_device(const vector<Event>* wait=nullptr,
	                             Event* returned=nullptr) {
		write_to_device(false, wait, returned);
	}
	void enqueue_read_from_device(const ulong offset, const ulong length,
	                              const vector<Event>* wait=nullptr,
	                              Event* returned=nullptr) {
		read_from_device(offset, length, false, wait, returned);
	}
	void enqueue_write_to_device(const ulong offset, const ulong length,
	                             const vector<Event>* wait=nullptr,
	                             Event* returned=nullptr) {
		write_to_device(offset, length, false, wait, returned);
	}
	void enqueue_copy_from_device_buffer(
		const Memory<T>& source, const ulong offset, const ulong length,
		const CudaTransferChannel channel =
			CudaTransferChannel::simulation) {
		const ulong safe_offset = min(offset, min(range(), source.range()));
		const ulong safe_length =
			min(length, min(range(), source.range())-safe_offset);
		if(safe_length == 0ull) return;
		if(!device_buffer_exists || !source.device_buffer_exists)
			print_error("CUDA inter-device transfer requires device buffers.");
		if(!source.host_buffer_exists)
			print_error("CUDA host fallback requires a source staging buffer.");
		if(source.host_staging_ready == nullptr) {
			cuda_set_current(source.state->context);
			cuda_check(
				cuEventCreate(
					&source.host_staging_ready,
					CU_EVENT_DISABLE_TIMING),
				"cuEventCreate(host_staging_ready)");
		}
		cuda_enqueue_transfer_copy(
			state, device_buffer+safe_offset*sizeof(T),
			source.state, source.device_buffer+safe_offset*sizeof(T),
			static_cast<void*>(source.host_buffer+safe_offset),
			source.host_buffer_pinned,
			source.host_staging_ready,
			(size_t)(safe_length*sizeof(T)),
			channel);
	}
	void finish_queue() {
		if(state) {
			cuda_set_current(state->context);
			cuda_check(cuStreamSynchronize(state->stream), "cuStreamSynchronize");
		}
	}
	CUdeviceptr get_cl_buffer() const { return device_buffer; }
	CUdeviceptr get_cuda_buffer() const { return device_buffer; }
};

class Kernel {
private:
	struct Argument {
		vector<unsigned char> bytes;
	};

	ulong N = 0ull;
	uint number_of_parameters = 0u;
	string name = "";
	CUfunction function = nullptr;
	std::shared_ptr<CudaDeviceState> state;
	vector<Argument> arguments;
	vector<void*> parameter_pointers;
	bool parameter_pointers_dirty = true;
	string launch_operation = "";
	ulong workgroup_size = WORKGROUP_SIZE;
	ulong preferred_workgroup_size = WORKGROUP_SIZE;

	template<typename T>
	void store_argument(const uint position, const T& value) {
		if(arguments.size() <= position) {
			arguments.resize(position+1u);
			parameter_pointers_dirty = true;
		}
		vector<unsigned char>& bytes = arguments[position].bytes;
		const unsigned char* const previous_data = bytes.data();
		bytes.resize(sizeof(T));
		if(bytes.data()!=previous_data) parameter_pointers_dirty = true;
		std::memcpy(bytes.data(), &value, sizeof(T));
		if(number_of_parameters < position+1u) {
			number_of_parameters = position+1u;
			parameter_pointers_dirty = true;
		}
	}

	template<typename T>
	void link_parameter(const uint position, const Memory<T>& memory) {
		const CUdeviceptr pointer = memory.get_cuda_buffer();
		store_argument(position, pointer);
	}

	template<typename T>
	void link_parameter(const uint position, const T& constant) {
		static_assert(std::is_trivially_copyable<T>::value,
		              "CUDA kernel arguments must be trivially copyable.");
		store_argument(position, constant);
	}

	void link_parameters(const uint starting_position) {
		if(number_of_parameters < starting_position) {
			number_of_parameters = starting_position;
			parameter_pointers_dirty = true;
		}
	}

	template<class T, class... U>
	void link_parameters(const uint starting_position, const T& parameter,
	                     const U&... parameters) {
		link_parameter(starting_position, parameter);
		link_parameters(starting_position+1u, parameters...);
	}

	void initialize(const Device& device, const string& kernel_name);

public:
	template<class... T>
	Kernel(const Device& device, const ulong range, const string& kernel_name,
	       const T&... parameters) {
		initialize(device, kernel_name);
		link_parameters(0u, parameters...);
		set_ranges(range);
	}

	template<class... T>
	Kernel(const Device& device, const ulong range, const uint group_size,
	       const string& kernel_name, const T&... parameters) {
		initialize(device, kernel_name);
		link_parameters(0u, parameters...);
		set_ranges(range, group_size);
	}

	Kernel() = default;

	Kernel& set_ranges(const ulong range,
	                   const ulong group_size=(ulong)WORKGROUP_SIZE) {
		N = range;
		workgroup_size =
			group_size==(ulong)WORKGROUP_SIZE
			? preferred_workgroup_size : group_size;
		return *this;
	}
	const ulong range() const { return N; }
	uint get_number_of_parameters() const { return number_of_parameters; }

	template<class... T>
	Kernel& add_parameters(const T&... parameters) {
		link_parameters(number_of_parameters, parameters...);
		return *this;
	}

	template<class... T>
	Kernel& set_parameters(const uint starting_position,
	                       const T&... parameters) {
		link_parameters(starting_position, parameters...);
		return *this;
	}

	Kernel& enqueue_run(const uint times=1u,
	                    const vector<Event>* = nullptr, Event* = nullptr);
	Kernel& run(const uint times=1u,
	            const vector<Event>* wait=nullptr, Event* returned=nullptr) {
		enqueue_run(times, wait, returned);
		return finish_queue();
	}
	Kernel& operator()(const uint times=1u,
	                   const vector<Event>* wait=nullptr,
	                   Event* returned=nullptr) {
		return run(times, wait, returned);
	}
	Kernel& finish_queue();
};
