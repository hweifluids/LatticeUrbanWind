#include "buffer.hpp"

#include <cstdlib>
#include <map>
#include <mutex>
#include <new>

namespace culbm::gpu {
namespace {

constexpr std::align_val_t host_alignment{64u};

// Peer access is decided once per (target, source) GPU pair and process.
std::mutex peer_mutex;
std::map<std::pair<uint, uint>, bool> peer_routes;

bool peer_access_disabled() {
	const char* const value = std::getenv("CULBM_DISABLE_P2P");
	return value!=nullptr && string(value)!="0";
}

bool peer_route(DeviceState& target, DeviceState& source) {
	if(target.context==source.context) return true;
	const std::pair<uint, uint> key(target.ordinal(), source.ordinal());
	std::lock_guard<std::mutex> lock(peer_mutex);
	const auto known = peer_routes.find(key);
	if(known!=peer_routes.end()) return known->second;
	bool direct = false;
	string reason;
	if(peer_access_disabled()) {
		reason = "CULBM_DISABLE_P2P is set";
	} else {
		int possible = 0;
		const CUresult query = cuDeviceCanAccessPeer(&possible, target.properties.handle, source.properties.handle);
		if(query!=CUDA_SUCCESS) {
			reason = describe(query);
		} else if(possible==0) {
			reason = "CUDA driver reports no peer access";
		} else {
			make_current(target.context);
			const CUresult enabled = cuCtxEnablePeerAccess(source.context, 0u);
			direct = enabled==CUDA_SUCCESS || enabled==CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
			if(!direct) reason = describe(enabled);
		}
	}
	peer_routes.emplace(key, direct);
	log_info("Halo copies GPU "+to_string(source.ordinal())+" -> GPU "+to_string(target.ordinal())+
		(direct ? ": direct peer access." : ": staged through host memory ("+reason+")."));
	return direct;
}

void copy_planes(DeviceState& owner, const detail::PlaneCopy* const planes, const unsigned count,
	const std::size_t pitch, const std::size_t row_bytes, const std::size_t rows,
	const bool queued, const bool download) {
	make_current(owner.context);
	if(!queued) check(cuStreamSynchronize(owner.stream), "cuStreamSynchronize(plane copy)");
	for(unsigned c=0u; c<count; ++c) {
		CUDA_MEMCPY2D copy{};
		if(download) {
			copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
			copy.srcDevice = planes[c].device;
			copy.dstMemoryType = CU_MEMORYTYPE_HOST;
			copy.dstHost = planes[c].host;
		} else {
			copy.srcMemoryType = CU_MEMORYTYPE_HOST;
			copy.srcHost = planes[c].host;
			copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
			copy.dstDevice = planes[c].device;
		}
		copy.srcPitch = pitch;
		copy.dstPitch = pitch;
		copy.WidthInBytes = row_bytes;
		copy.Height = rows;
		if(queued) check(cuMemcpy2DAsync(&copy, owner.stream), "cuMemcpy2DAsync");
		else check(cuMemcpy2D(&copy), "cuMemcpy2D");
	}
	if(!queued) check(cuCtxSynchronize(), "cuCtxSynchronize(plane copy)");
}

} // namespace

namespace detail {

void* host_allocate(DeviceState& owner, const std::size_t bytes, HostMemory& kind) {
	if(kind==HostMemory::pinned) {
		make_current(owner.context);
		void* data = nullptr;
		if(cuMemHostAlloc(&data, bytes, CU_MEMHOSTALLOC_PORTABLE)==CUDA_SUCCESS) return data;
		kind = HostMemory::pageable;
	}
	void* const data = ::operator new(bytes, host_alignment, std::nothrow);
	if(data==nullptr) fatal_error("CUDA host mirror allocation failed.");
	return data;
}

void host_free(DeviceState* const owner, void* const data, const HostMemory kind) {
	if(kind==HostMemory::pinned) {
		if(owner!=nullptr) make_current(owner->context);
		check(cuMemFreeHost(data), "cuMemFreeHost");
	} else {
		::operator delete(data, host_alignment);
	}
}

CUevent create_event(DeviceState& owner) {
	make_current(owner.context);
	CUevent event = nullptr;
	check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "cuEventCreate(staging)");
	return event;
}

void destroy_event(DeviceState* const owner, const CUevent event) {
	if(owner!=nullptr) make_current(owner->context);
	check(cuEventDestroy(event), "cuEventDestroy(staging)");
}

void zero_device(DeviceState& owner, const CUdeviceptr address, const std::uint64_t bytes) {
	make_current(owner.context);
	check(cuMemsetD8(address, 0u, (std::size_t)bytes), "cuMemsetD8");
}

void zero_device_async(DeviceState& owner, const CUdeviceptr address, const std::uint64_t bytes) {
	make_current(owner.context);
	check(cuMemsetD8Async(address, 0u, (std::size_t)bytes, owner.stream), "cuMemsetD8Async");
}

void to_host(DeviceState& owner, void* const target, const CUdeviceptr source,
	const std::size_t bytes, const bool queued) {
	make_current(owner.context);
	if(queued) {
		check(cuMemcpyDtoHAsync(target, source, bytes, owner.stream), "cuMemcpyDtoHAsync");
		return;
	}
	// The compute stream does not synchronise with the legacy stream of the
	// blocking copy, hence the explicit drains on both sides.
	check(cuStreamSynchronize(owner.stream), "cuStreamSynchronize(download)");
	check(cuMemcpyDtoH(target, source, bytes), "cuMemcpyDtoH");
	check(cuCtxSynchronize(), "cuCtxSynchronize(download)");
}

void to_device(DeviceState& owner, const CUdeviceptr target, const void* const source,
	const std::size_t bytes, const bool queued) {
	make_current(owner.context);
	if(queued) {
		check(cuMemcpyHtoDAsync(target, source, bytes, owner.stream), "cuMemcpyHtoDAsync");
		return;
	}
	check(cuStreamSynchronize(owner.stream), "cuStreamSynchronize(upload)");
	check(cuMemcpyHtoD(target, source, bytes), "cuMemcpyHtoD");
	check(cuCtxSynchronize(), "cuCtxSynchronize(upload)");
}

void planes_to_host(DeviceState& owner, const PlaneCopy* const planes, const unsigned count,
	const std::size_t pitch, const std::size_t row_bytes, const std::size_t rows, const bool queued) {
	copy_planes(owner, planes, count, pitch, row_bytes, rows, queued, true);
}

void planes_to_device(DeviceState& owner, const PlaneCopy* const planes, const unsigned count,
	const std::size_t pitch, const std::size_t row_bytes, const std::size_t rows, const bool queued) {
	copy_planes(owner, planes, count, pitch, row_bytes, rows, queued, false);
}

void peer_copy(DeviceState& target, const CUdeviceptr target_address,
	DeviceState& source, const CUdeviceptr source_address,
	void* const staging, const bool staging_pinned, const CUevent staging_ready,
	const std::size_t bytes) {
	if(bytes==0u) return;
	TransferLane& in = target.halo;
	TransferLane& out = source.halo;

	// Same context, or direct peer access: one device-side copy on the
	// target's copy stream once the source halo is ready.
	const bool same_context = target.context==source.context;
	if(same_context || peer_route(target, source)) {
		make_current(target.context);
		check(cuStreamWaitEvent(in.copy_stream, out.ready, 0u), "cuStreamWaitEvent(halo ready)");
		if(same_context) {
			check(cuMemcpyDtoDAsync(target_address, source_address, bytes, in.copy_stream),
				"cuMemcpyDtoDAsync(halo)");
			return;
		}
		const CUresult copied = cuMemcpyPeerAsync(target_address, target.context,
			source_address, source.context, bytes, in.copy_stream);
		if(copied==CUDA_SUCCESS) return;
		log_warning("Peer halo copy GPU "+to_string(source.ordinal())+" -> GPU "+to_string(target.ordinal())+
			" failed ("+describe(copied)+"); staging through host memory.");
	}

	if(staging==nullptr) fatal_error("Host-staged halo copy has no staging buffer.");
	if(staging_ready==nullptr) fatal_error("Host-staged halo copy has no staging event.");
	if(staging_pinned) {
		// Asynchronous: source outbound stream reads into pinned memory, the
		// target copy stream writes once that read has finished.
		make_current(source.context);
		check(cuStreamWaitEvent(out.outbound_stream, out.ready, 0u), "cuStreamWaitEvent(halo ready, outbound)");
		check(cuMemcpyDtoHAsync(staging, source_address, bytes, out.outbound_stream), "cuMemcpyDtoHAsync(halo staging)");
		check(cuEventRecord(staging_ready, out.outbound_stream), "cuEventRecord(halo staging)");
		make_current(target.context);
		check(cuStreamWaitEvent(in.copy_stream, staging_ready, 0u), "cuStreamWaitEvent(halo staging)");
		check(cuMemcpyHtoDAsync(target_address, staging, bytes, in.copy_stream), "cuMemcpyHtoDAsync(halo staging)");
		return;
	}
	// Pageable staging can only be copied synchronously.
	make_current(source.context);
	check(cuEventSynchronize(out.ready), "cuEventSynchronize(halo ready)");
	check(cuMemcpyDtoH(staging, source_address, bytes), "cuMemcpyDtoH(halo staging)");
	make_current(target.context);
	check(cuStreamSynchronize(in.copy_stream), "cuStreamSynchronize(halo staging)");
	check(cuMemcpyHtoD(target_address, staging, bytes), "cuMemcpyHtoD(halo staging)");
}

} // namespace detail
} // namespace culbm::gpu
