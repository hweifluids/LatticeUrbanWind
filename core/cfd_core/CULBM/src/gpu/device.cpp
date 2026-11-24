#include "device.hpp"

#include "program.hpp"
#include "vram.hpp"

#include <algorithm>

namespace culbm::gpu {
namespace {

int attribute(const CUdevice handle, const CUdevice_attribute which, const char* const what) {
	int value = 0;
	check(cuDeviceGetAttribute(&value, which, handle), what);
	return value;
}

// FP32 lanes per streaming multiprocessor, by compute capability.
uint fp32_lanes_per_multiprocessor(const uint compute_capability) {
	if(compute_capability<30u) return 32u;
	if(compute_capability<50u) return 192u;
	if(compute_capability==60u || (compute_capability>=70u && compute_capability<=80u)) return 64u;
	return 128u;
}

DeviceProperties query(const CUdevice handle, const uint ordinal) {
	DeviceProperties p;
	p.ordinal = ordinal;
	p.handle = handle;
	char name[256] = {};
	check(cuDeviceGetName(name, (int)sizeof(name), handle), "cuDeviceGetName");
	p.name = trim(name);
	std::size_t vram = 0u;
	check(cuDeviceTotalMem(&vram, handle), "cuDeviceTotalMem");
	p.memory_mb = (uint)(vram/mib_bytes);
	p.multiprocessors = (uint)attribute(handle, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, "multiprocessor count");
	p.clock_mhz = (uint)(attribute(handle, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, "clock rate")/1000);
	p.l2_kb = (uint)(attribute(handle, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, "L2 size")/1024);
	p.shared_kb = (uint)(attribute(handle, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, "shared memory per block")/1024);
	const int major = attribute(handle, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "compute capability");
	const int minor = attribute(handle, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, "compute capability");
	p.compute_capability = (uint)(10*major+minor);
	p.fp32_lanes = p.multiprocessors*fp32_lanes_per_multiprocessor(p.compute_capability);
	p.fp32_tflops = 2.0e-6f*(float)p.fp32_lanes*(float)p.clock_mhz;
	int driver = 0;
	check(cuDriverGetVersion(&driver), "cuDriverGetVersion");
	p.driver_version = to_string((uint)(driver/1000))+"."+to_string((uint)(driver%1000/10));
	return p;
}

string report_row(const string& label, const string& value) {
	return "| "+alignl(15u, label)+" | "+alignl(57u, value)+" |";
}

} // namespace

std::vector<DeviceProperties> list_devices() {
	ensure_driver();
	int count = 0;
	check(cuDeviceGetCount(&count), "cuDeviceGetCount");
	if(count<=0) fatal_error("No CUDA devices are available. Install an NVIDIA CUDA driver.");
	std::vector<DeviceProperties> devices;
	devices.reserve((std::size_t)count);
	for(int ordinal=0; ordinal<count; ++ordinal) {
		CUdevice handle = 0;
		check(cuDeviceGet(&handle, ordinal), "cuDeviceGet");
		devices.push_back(query(handle, (uint)ordinal));
	}
	return devices;
}

void print_device_report(const DeviceProperties& p) {
#if defined(_WIN32)
	const char* const platform = "Windows";
#elif defined(__linux__)
	const char* const platform = "Linux";
#else
	const char* const platform = "unknown OS";
#endif
	println("|-----------------.-----------------------------------------------------------|");
	println(report_row("GPU "+to_string(p.ordinal), p.name));
	println(report_row("Architecture", "sm_"+to_string(p.compute_capability)+", "+to_string(p.multiprocessors)+
		" SMs, "+to_string(p.fp32_lanes)+" FP32 lanes at "+to_string(p.clock_mhz)+" MHz"));
	println(report_row("Peak FP32", to_string(p.fp32_tflops, 2u)+" TFLOP/s"));
	println(report_row("Memory", to_string(p.memory_mb)+" MiB VRAM, "+to_string(p.l2_kb)+" KiB L2, "+
		to_string(p.shared_kb)+" KiB shared per block"));
	println(report_row("Driver", "CUDA "+p.driver_version+" on "+platform+", kernels built by NVRTC"));
	println("|-----------------'-----------------------------------------------------------|");
}

std::uint64_t runtime_reserve_bytes(const std::uint64_t total_bytes) {
	return std::min<std::uint64_t>(256u*mib_bytes, total_bytes/10u);
}

DeviceState::~DeviceState() {
	if(context==nullptr) return;
	cuCtxSetCurrent(context);
	if(module!=nullptr) cuModuleUnload(module);
	if(halo.ready!=nullptr) cuEventDestroy(halo.ready);
	if(halo.complete!=nullptr) cuEventDestroy(halo.complete);
	if(halo.consumed!=nullptr) cuEventDestroy(halo.consumed);
	if(pace_events[0]!=nullptr) cuEventDestroy(pace_events[0]);
	if(pace_events[1]!=nullptr) cuEventDestroy(pace_events[1]);
	if(halo.outbound_stream!=nullptr) cuStreamDestroy(halo.outbound_stream);
	if(halo.copy_stream!=nullptr) cuStreamDestroy(halo.copy_stream);
	if(stream!=nullptr) cuStreamDestroy(stream);
	cuDevicePrimaryCtxRelease(properties.handle);
}

CUdeviceptr DeviceState::allocate(const std::uint64_t bytes) {
	const std::uint64_t mib = mib_ceil(bytes);
	const std::uint64_t booked = booked_mib.fetch_add(mib)+mib;
	if(booked>(std::uint64_t)properties.memory_mb) {
		fatal_error("GPU "+to_string(ordinal())+" ("+properties.name+") cannot hold another "+
			to_string(mib)+" MiB: the run would book "+to_string(booked)+" of "+
			to_string(properties.memory_mb)+" MiB.");
	}
	make_current(context);
	std::size_t free = 0u, total = 0u;
	check(cuMemGetInfo(&free, &total), "cuMemGetInfo");
	const std::uint64_t reserve = runtime_reserve_bytes(total);
	if(bytes>free || free-bytes<reserve) {
		fatal_error("GPU "+to_string(ordinal())+" ("+properties.name+") lacks free VRAM for a "+
			to_string(mib)+" MiB buffer: "+to_string((ulong)(free/mib_bytes))+" MiB free, "+
			to_string((ulong)(reserve/mib_bytes))+" MiB must stay reserved for the CUDA runtime.");
	}
	CUdeviceptr address = 0;
	check(cuMemAlloc(&address, (std::size_t)bytes), "cuMemAlloc");
	allocated_bytes.fetch_add(bytes);
	return address;
}

void DeviceState::release(const CUdeviceptr address, const std::uint64_t bytes) {
	const std::uint64_t mib = mib_ceil(bytes);
	std::uint64_t booked = booked_mib.load();
	while(!booked_mib.compare_exchange_weak(booked, booked>mib ? booked-mib : 0u)) {}
	allocated_bytes.fetch_sub(std::min(bytes, allocated_bytes.load()));
	make_current(context);
	check(cuMemFree(address), "cuMemFree");
}

void DeviceState::synchronize() const {
	make_current(context);
	check(cuStreamSynchronize(stream), "cuStreamSynchronize");
	pace_armed[0] = false;
	pace_armed[1] = false;
}

void DeviceState::record_pace() const {
	make_current(context);
	check(cuEventRecord(pace_events[pace_slot], stream), "cuEventRecord(pace)");
	pace_armed[pace_slot] = true;
}

void DeviceState::wait_previous_pace() const {
	const uint previous = pace_slot ^ 1u;
	if(pace_armed[previous]) {
		make_current(context);
		check(cuEventSynchronize(pace_events[previous]), "cuEventSynchronize(pace)");
		pace_armed[previous] = false;
	}
	pace_slot = previous;
}

Device::Device(const DeviceProperties& properties, const string& kernel_source) {
	ensure_driver();
	print_device_report(properties);
	state_ = std::make_shared<DeviceState>();
	state_->properties = properties;
	check(cuDevicePrimaryCtxRetain(&state_->context, properties.handle), "cuDevicePrimaryCtxRetain");
	make_current(state_->context);
	check(cuStreamCreate(&state_->stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate(compute)");
	TransferLane& lane = state_->halo;
	check(cuStreamCreate(&lane.copy_stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate(halo copy)");
	check(cuStreamCreate(&lane.outbound_stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate(halo outbound)");
	check(cuEventCreate(&lane.ready, CU_EVENT_DISABLE_TIMING), "cuEventCreate(halo ready)");
	check(cuEventCreate(&lane.complete, CU_EVENT_DISABLE_TIMING), "cuEventCreate(halo complete)");
	check(cuEventCreate(&lane.consumed, CU_EVENT_DISABLE_TIMING), "cuEventCreate(halo consumed)");
	check(cuEventCreate(&state_->pace_events[0], CU_EVENT_DISABLE_TIMING), "cuEventCreate(pace 0)");
	check(cuEventCreate(&state_->pace_events[1], CU_EVENT_DISABLE_TIMING), "cuEventCreate(pace 1)");
	// The receive buffers start out free.
	check(cuEventRecord(lane.consumed, state_->stream), "cuEventRecord(halo consumed)");
	state_->module = load_kernel_module(properties, kernel_source);
}

void Device::synchronize() const {
	if(state_) state_->synchronize();
}

void Device::record_pace() const {
	if(state_) state_->record_pace();
}

void Device::wait_previous_pace() const {
	if(state_) state_->wait_previous_pace();
}

std::size_t Device::free_memory_bytes() const {
	if(!state_) return 0u;
	make_current(state_->context);
	std::size_t free = 0u, total = 0u;
	check(cuMemGetInfo(&free, &total), "cuMemGetInfo");
	return free;
}

std::size_t Device::total_memory_bytes() const {
	if(!state_) return 0u;
	make_current(state_->context);
	std::size_t free = 0u, total = 0u;
	check(cuMemGetInfo(&free, &total), "cuMemGetInfo");
	return total;
}

std::uint64_t Device::booked_mib() const {
	return state_ ? state_->booked_mib.load() : 0u;
}

std::uint64_t Device::allocated_bytes() const {
	return state_ ? state_->allocated_bytes.load() : 0u;
}

bool Device::optional_allocation_fits(const std::uint64_t mib, const std::uint64_t later_mib) const {
	constexpr std::uint64_t limit = ~std::uint64_t(0u)/mib_bytes;
	if(mib>limit || later_mib>limit) return false;
	const std::uint64_t free = (std::uint64_t)free_memory_bytes();
	const std::uint64_t reserve = runtime_reserve_bytes((std::uint64_t)total_memory_bytes());
	const std::uint64_t now = mib*mib_bytes, later = later_mib*mib_bytes;
	if(now>free || later>free-now) return false;
	return free-now-later>=reserve;
}

void Device::halo_sources_ready() const {
	if(!state_) return;
	make_current(state_->context);
	check(cuEventRecord(state_->halo.ready, state_->stream), "cuEventRecord(halo ready)");
}

void Device::halo_copies_begin() const {
	if(!state_) return;
	make_current(state_->context);
	TransferLane& lane = state_->halo;
	check(cuStreamWaitEvent(lane.copy_stream, lane.consumed, 0u), "cuStreamWaitEvent(halo consumed)");
}

void Device::halo_copies_end() const {
	if(!state_) return;
	make_current(state_->context);
	TransferLane& lane = state_->halo;
	check(cuEventRecord(lane.complete, lane.copy_stream), "cuEventRecord(halo complete)");
	check(cuStreamWaitEvent(state_->stream, lane.complete, 0u), "cuStreamWaitEvent(halo complete)");
}

void Device::halo_buffers_consumed() const {
	if(!state_) return;
	make_current(state_->context);
	check(cuEventRecord(state_->halo.consumed, state_->stream), "cuEventRecord(halo consumed)");
}

void Device::wait_for_peer_copies(const Device& peer) const {
	if(!state_ || !peer.state_) return;
	make_current(state_->context);
	check(cuStreamWaitEvent(state_->stream, peer.state_->halo.complete, 0u),
		"cuStreamWaitEvent(peer halo complete)");
}

} // namespace culbm::gpu
