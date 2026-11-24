#pragma once

// Configured launches of NVRTC kernels.

#include "buffer.hpp"

#include <type_traits>
#include <vector>

namespace culbm::gpu {

// A kernel of a device module together with its 1D extent, block size and
// argument slots. Arguments are stored by value when they are set; a buffer
// argument stores the buffer's device address. A launch is queued on the
// device's compute stream.
//
//   Launch step(device, "stream_collide", cells);
//   step.push(fi, rho, u, flags, t);   // append arguments
//   step.bind(4u, t+1u).submit();      // overwrite slot 4, queue one launch
class Launch {
public:
	Launch() = default;
	Launch(const Device& device, const string& kernel, std::uint64_t threads=0u);
	Launch(const Launch&) = delete;
	Launch& operator=(const Launch&) = delete;
	Launch(Launch&&) = default;
	Launch& operator=(Launch&&) = default;

	// Number of threads (one per work item); 0 makes submit() a no-op.
	Launch& over(const std::uint64_t threads) & {
		threads_ = threads;
		return *this;
	}
	Launch&& over(const std::uint64_t threads) && { return std::move(over(threads)); }
	Launch& block(uint threads_per_block) &;
	Launch&& block(const uint threads_per_block) && { return std::move(block(threads_per_block)); }

	// Append arguments. On a temporary the launch is passed on, so
	//   kernel = Launch(device, "name", n).push(a, b);
	// configures and stores a launch in one expression.
	template<class... A>
	Launch& push(const A&... values) & {
		(store(slot_count(), values), ...);
		return *this;
	}
	template<class... A>
	Launch&& push(const A&... values) && { return std::move(push(values...)); }
	template<class... A>
	Launch& bind(const uint first_slot, const A&... values) {
		uint slot = first_slot;
		(store(slot++, values), ...);
		return *this;
	}

	uint slot_count() const { return (uint)slots_.size(); }
	std::uint64_t threads() const { return threads_; }
	uint threads_per_block() const { return block_; }

	// Queue `repeat` launches on the compute stream.
	void submit(uint repeat=1u);
	// Queue `repeat` launches on an explicit stream of the same device.
	void submit_on(CUstream stream, uint repeat=1u);
	// Queue one launch and wait for the compute stream.
	void submit_and_wait();

private:
	struct Slot {
		alignas(16) unsigned char bytes[16];
		bool set;
	};

	template<typename T>
	static CUdeviceptr argument(const Buffer<T>& buffer) { return buffer.address(); }
	template<typename T>
	static CUdeviceptr argument(const MirroredBuffer<T>& buffer) { return buffer.address(); }
	template<typename A>
	static const A& argument(const A& value) {
		static_assert(std::is_trivially_copyable<A>::value, "kernel arguments are copied bytewise");
		return value;
	}
	template<class A>
	void store(const uint slot, const A& value) {
		const auto& stored = argument(value);
		write(slot, &stored, sizeof(stored));
	}
	void write(uint slot, const void* data, std::size_t size);

	std::shared_ptr<DeviceState> owner_;
	CUfunction function_ = nullptr;
	string name_;
	std::uint64_t threads_ = 0u;
	uint block_ = default_block_size;
	std::vector<Slot> slots_;
	std::vector<void*> pointers_; // slot addresses handed to cuLaunchKernel
	bool pointers_stale_ = true;
};

} // namespace culbm::gpu
