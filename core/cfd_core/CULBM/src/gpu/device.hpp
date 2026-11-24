#pragma once

// Opened CUDA devices: properties, primary context, compute and halo streams,
// the NVRTC kernel module and the allocation bookkeeping of one device.

#include "driver.hpp"
#include "../core/core.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace culbm::gpu {

// Streams and events of the halo exchange of one device.
struct TransferLane {
	CUstream copy_stream = nullptr;     // copies into this device's receive buffers
	CUstream outbound_stream = nullptr; // staging reads of the pinned-host fallback
	CUevent ready = nullptr;            // send buffers filled on the compute stream
	CUevent complete = nullptr;         // all copies into this device are done
	CUevent consumed = nullptr;         // receive buffers read by the insert kernels
};

// Properties of a CUDA device as reported by the driver.
struct DeviceProperties {
	uint ordinal = 0u;
	CUdevice handle = 0;
	string name;
	uint memory_mb = 0u;          // installed VRAM in MiB
	uint multiprocessors = 0u;
	uint clock_mhz = 0u;
	uint l2_kb = 0u;
	uint shared_kb = 0u;          // shared memory per block
	uint compute_capability = 0u; // 10*major+minor
	uint fp32_lanes = 0u;         // CUDA cores
	float fp32_tflops = 0.0f;     // peak FP32 rate (FMA counted as two operations)
	string driver_version;
};

// All visible CUDA devices in driver order; aborts when there is none.
std::vector<DeviceProperties> list_devices();
void print_device_report(const DeviceProperties& properties);

// Driver objects of an opened device. Shared by all Device handles and every
// buffer allocated on it, so it outlives the last allocation.
struct DeviceState {
	DeviceProperties properties;
	CUcontext context = nullptr;
	CUstream stream = nullptr; // compute stream (non-blocking)
	TransferLane halo;
	CUevent pace_events[2] = {nullptr, nullptr};
	mutable uint pace_slot = 0u;
	mutable bool pace_armed[2] = {false, false};
	CUmodule module = nullptr;
	std::atomic<std::uint64_t> booked_mib{0u};      // per-allocation MiB rounding
	std::atomic<std::uint64_t> allocated_bytes{0u}; // exact live device bytes

	DeviceState() = default;
	DeviceState(const DeviceState&) = delete;
	DeviceState& operator=(const DeviceState&) = delete;
	~DeviceState();

	uint ordinal() const { return properties.ordinal; }
	CUdeviceptr allocate(std::uint64_t bytes);
	void release(CUdeviceptr address, std::uint64_t bytes);
	void synchronize() const;
	void record_pace() const;
	void wait_previous_pace() const;
};

// Free VRAM the runtime keeps back from allocations: 256 MiB or a tenth of the
// device, whichever is smaller.
std::uint64_t runtime_reserve_bytes(std::uint64_t total_bytes);

class Device {
public:
	Device() = default;
	// Opens the device and builds `kernel_source` for it with NVRTC.
	Device(const DeviceProperties& properties, const string& kernel_source);

	bool ready() const { return state_!=nullptr; }
	const DeviceProperties& properties() const { return state_->properties; }
	uint ordinal() const { return state_->properties.ordinal; }

	// Wait until the compute stream is idle.
	void synchronize() const;
	// Two-slot event pacing: keeps at most one step queued ahead on the GPU.
	void record_pace() const;
	void wait_previous_pace() const;

	std::size_t free_memory_bytes() const;
	std::size_t total_memory_bytes() const;
	std::uint64_t booked_mib() const;
	std::uint64_t allocated_bytes() const;
	// Whether an optional allocation of `mib` still leaves room for `later_mib`
	// of required allocations and the runtime reserve.
	bool optional_allocation_fits(std::uint64_t mib, std::uint64_t later_mib) const;

	// Halo exchange protocol:
	//   halo_sources_ready    after the extract kernels on the compute stream
	//   halo_copies_begin     copy stream waits until the receive buffers are free
	//   (copy_halo calls)
	//   halo_copies_end       compute stream waits for the copies into this device
	//   halo_buffers_consumed after the insert kernels
	//   wait_for_peer_copies  compute stream waits for copies a peer reads from us
	void halo_sources_ready() const;
	void halo_copies_begin() const;
	void halo_copies_end() const;
	void halo_buffers_consumed() const;
	void wait_for_peer_copies(const Device& peer) const;

	CUcontext context() const { return state_->context; }
	CUstream stream() const { return state_->stream; }
	CUstream copy_stream() const { return state_->halo.copy_stream; }
	CUstream outbound_stream() const { return state_->halo.outbound_stream; }
	CUmodule module() const { return state_->module; }
	const std::shared_ptr<DeviceState>& state() const { return state_; }

private:
	std::shared_ptr<DeviceState> state_;
};

} // namespace culbm::gpu
