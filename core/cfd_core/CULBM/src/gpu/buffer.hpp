#pragma once

// Typed device buffers and explicit host mirrors.
//
//   Buffer<T>         device memory only; `count` elements per component,
//                     components stored one after another (structure of arrays)
//   HostMirror<T>     host memory for a buffer's contents, pinned or pageable
//   MirroredBuffer<T> a Buffer with a HostMirror and the transfers between them;
//                     the mirror can also be attached on first host use
//
// Transfers named *_async are queued on the owning device's compute stream when
// the mirror is pinned. Pageable mirrors always copy synchronously: the compute
// stream is drained first and the context afterwards, so the copy is ordered
// against all kernels around it.

#include "device.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

namespace culbm::gpu {

enum class HostMemory : unsigned char {
	pageable,
	pinned // falls back to pageable when page-locking fails
};

// Initial contents of a new device-only buffer.
enum class Init : unsigned char {
	zero,
	none
};

namespace detail {

// Untyped primitives behind the templates below (buffer.cpp). Sizes in bytes.
void* host_allocate(DeviceState& owner, std::size_t bytes, HostMemory& kind);
void host_free(DeviceState* owner, void* data, HostMemory kind);
CUevent create_event(DeviceState& owner);
void destroy_event(DeviceState* owner, CUevent event);
void zero_device(DeviceState& owner, CUdeviceptr address, std::uint64_t bytes);
void zero_device_async(DeviceState& owner, CUdeviceptr address, std::uint64_t bytes);
void to_host(DeviceState& owner, void* target, CUdeviceptr source, std::size_t bytes, bool queued);
void to_device(DeviceState& owner, CUdeviceptr target, const void* source, std::size_t bytes, bool queued);

// One pitched rectangle per component; all share pitch, row width and height.
struct PlaneCopy {
	CUdeviceptr device = 0;
	void* host = nullptr;
};
void planes_to_host(DeviceState& owner, const PlaneCopy* planes, unsigned count,
	std::size_t pitch, std::size_t row_bytes, std::size_t rows, bool queued);
void planes_to_device(DeviceState& owner, const PlaneCopy* planes, unsigned count,
	std::size_t pitch, std::size_t row_bytes, std::size_t rows, bool queued);

void peer_copy(DeviceState& target, CUdeviceptr target_address,
	DeviceState& source, CUdeviceptr source_address,
	void* staging, bool staging_pinned, CUevent staging_ready,
	std::size_t bytes);

} // namespace detail

template<typename T>
class HostMirror {
	static_assert(std::is_trivially_copyable<T>::value, "host mirrors hold plain data");

public:
	HostMirror() = default;
	HostMirror(const Device& device, const std::uint64_t elements, const HostMemory kind)
		: HostMirror(device.state(), elements, kind) {}
	HostMirror(std::shared_ptr<DeviceState> owner, const std::uint64_t elements, const HostMemory kind)
		: owner_(std::move(owner)), size_(elements), kind_(kind) {
		data_ = static_cast<T*>(detail::host_allocate(*owner_, (std::size_t)(elements*sizeof(T)), kind_));
	}
	~HostMirror() { release(); }
	HostMirror(const HostMirror&) = delete;
	HostMirror& operator=(const HostMirror&) = delete;
	HostMirror(HostMirror&& other) noexcept { take(other); }
	HostMirror& operator=(HostMirror&& other) noexcept {
		if(this!=&other) {
			release();
			take(other);
		}
		return *this;
	}

	T* data() { return data_; }
	const T* data() const { return data_; }
	T& operator[](const std::uint64_t i) { return data_[i]; }
	const T& operator[](const std::uint64_t i) const { return data_[i]; }
	std::uint64_t size() const { return size_; }
	bool empty() const { return data_==nullptr; }
	bool pinned() const { return data_!=nullptr && kind_==HostMemory::pinned; }

	// Event recorded after a staging read of the pinned halo fallback.
	CUevent staging_event() {
		if(staging_==nullptr) staging_ = detail::create_event(*owner_);
		return staging_;
	}

	void release() {
		if(staging_!=nullptr) detail::destroy_event(owner_.get(), staging_);
		if(data_!=nullptr) detail::host_free(owner_.get(), data_, kind_);
		staging_ = nullptr;
		data_ = nullptr;
		size_ = 0u;
	}

private:
	void take(HostMirror& other) {
		owner_ = std::move(other.owner_);
		data_ = std::exchange(other.data_, nullptr);
		size_ = std::exchange(other.size_, 0u);
		kind_ = other.kind_;
		staging_ = std::exchange(other.staging_, nullptr);
	}

	std::shared_ptr<DeviceState> owner_;
	T* data_ = nullptr;
	std::uint64_t size_ = 0u;
	HostMemory kind_ = HostMemory::pageable;
	CUevent staging_ = nullptr;
};

template<typename T>
class Buffer {
	static_assert(std::is_trivially_copyable<T>::value, "device buffers hold plain data");

public:
	Buffer() = default;
	Buffer(const Device& device, const std::uint64_t count, const uint components=1u, const Init init=Init::zero)
		: owner_(device.state()), count_(count), components_(components) {
		if(!owner_) fatal_error("A CUDA buffer needs an opened device.");
		if(size()==0u) fatal_error("A CUDA buffer needs at least one element.");
		address_ = owner_->allocate(bytes());
		if(init==Init::zero) detail::zero_device(*owner_, address_, bytes());
	}
	~Buffer() { release(); }
	Buffer(const Buffer&) = delete;
	Buffer& operator=(const Buffer&) = delete;
	Buffer(Buffer&& other) noexcept { take(other); }
	Buffer& operator=(Buffer&& other) noexcept {
		if(this!=&other) {
			release();
			take(other);
		}
		return *this;
	}

	std::uint64_t count() const { return count_; } // elements per component
	uint components() const { return components_; }
	std::uint64_t size() const { return count_*(std::uint64_t)components_; }
	std::uint64_t bytes() const { return size()*sizeof(T); }
	bool allocated() const { return address_!=0; }
	CUdeviceptr address() const { return address_; }
	CUdeviceptr address_of(const std::uint64_t element) const { return address_+(CUdeviceptr)(element*sizeof(T)); }
	DeviceState& owner() const { return *owner_; }
	const std::shared_ptr<DeviceState>& shared_owner() const { return owner_; }

	// Synchronous upload of size() elements from ordinary host memory.
	void copy_from(const T* const host) {
		if(allocated()) detail::to_device(*owner_, address_, host, (std::size_t)bytes(), false);
	}
	// Queue an asynchronous zero-fill of the device allocation on the compute stream.
	void zero_async() {
		if(allocated()) detail::zero_device_async(*owner_, address_, bytes());
	}

	void release() {
		if(address_!=0) owner_->release(address_, bytes());
		address_ = 0;
		count_ = 0u;
		components_ = 1u;
	}

private:
	void take(Buffer& other) {
		owner_ = std::move(other.owner_);
		address_ = std::exchange(other.address_, 0);
		count_ = std::exchange(other.count_, 0u);
		components_ = std::exchange(other.components_, 1u);
	}

	std::shared_ptr<DeviceState> owner_;
	CUdeviceptr address_ = 0;
	std::uint64_t count_ = 0u;
	uint components_ = 1u;
};

// Selects the MirroredBuffer constructor that leaves the host mirror out until
// attach_host() is called.
struct HostOnDemand {
	HostMemory kind = HostMemory::pageable;
};

template<typename T>
class MirroredBuffer {
public:
	MirroredBuffer() = default;
	// Both sides start out holding `value`.
	MirroredBuffer(const Device& device, const std::uint64_t count, const uint components=1u,
		const HostMemory host=HostMemory::pageable, const T value=T())
		: host_(device, count*(std::uint64_t)components, host),
		  buffer_(device, count, components, Init::none) {
		fill(value);
	}
	// Zeroed device buffer without a host mirror yet.
	MirroredBuffer(const Device& device, const std::uint64_t count, const uint components, const HostOnDemand later)
		: buffer_(device, count, components, Init::zero), later_(later.kind) {}

	// Create the host mirror of an on-demand buffer. It starts zeroed, which is
	// what the device holds as long as nothing was uploaded through a mirror.
	void attach_host() {
		if(has_host() || !buffer_.allocated()) return;
		host_ = HostMirror<T>(buffer_.shared_owner(), size(), later_);
		std::fill(host_.data(), host_.data()+size(), T());
	}

	Buffer<T>& buffer() { return buffer_; }
	const Buffer<T>& buffer() const { return buffer_; }
	HostMirror<T>& host() { return host_; }
	const HostMirror<T>& host() const { return host_; }

	std::uint64_t count() const { return buffer_.count(); }
	uint components() const { return buffer_.components(); }
	std::uint64_t size() const { return buffer_.size(); }
	std::uint64_t bytes() const { return buffer_.bytes(); }
	CUdeviceptr address() const { return buffer_.address(); }

	// Host side.
	bool has_host() const { return !host_.empty(); }
	bool host_pinned() const { return host_.pinned(); }
	T* data() { return host_.data(); }
	const T* data() const { return host_.data(); }
	T& operator[](const std::uint64_t i) { return host_[i]; }
	const T& operator[](const std::uint64_t i) const { return host_[i]; }
	T* component(const uint c) { return host_.data()+(std::uint64_t)c*count(); }
	const T* component(const uint c) const { return host_.data()+(std::uint64_t)c*count(); }
	// Drop the mirror; the device contents stay.
	void release_host() { host_.release(); }

	// Whole buffer.
	void download() { to_host(0u, size(), false); }
	void download_async() { to_host(0u, size(), true); }
	void upload() { to_device(0u, size(), false); }
	void upload_async() { to_device(0u, size(), true); }
	// Elements [first, first+elements) of the flat array, clipped to the buffer.
	void download(const std::uint64_t first, const std::uint64_t elements) { to_host(first, elements, false); }
	void download_async(const std::uint64_t first, const std::uint64_t elements) { to_host(first, elements, true); }
	void upload(const std::uint64_t first, const std::uint64_t elements) { to_device(first, elements, false); }
	void upload_async(const std::uint64_t first, const std::uint64_t elements) { to_device(first, elements, true); }

	// The rectangle [x0,x1) x [y0,y1) of layer z of an nx x ny x nz lattice, every component.
	void download_plane_async(const std::uint64_t x0, const std::uint64_t x1, const std::uint64_t y0,
		const std::uint64_t y1, const std::uint64_t z, const std::uint64_t nx, const std::uint64_t ny) {
		plane(x0, x1, y0, y1, z, nx, ny, true);
	}
	void upload_plane_async(const std::uint64_t x0, const std::uint64_t x1, const std::uint64_t y0,
		const std::uint64_t y1, const std::uint64_t z, const std::uint64_t nx, const std::uint64_t ny) {
		plane(x0, x1, y0, y1, z, nx, ny, false);
	}

	// Set every host element to `value` and upload the buffer.
	void fill(const T value) {
		if(!has_host()) return;
		std::fill(host_.data(), host_.data()+size(), value);
		upload();
	}
	// Queue an asynchronous zero-fill of the device buffer on the compute stream.
	void zero_device_async() {
		buffer_.zero_async();
	}

	// Wait for the owning device's compute stream.
	void synchronize() const {
		if(buffer_.allocated()) buffer_.owner().synchronize();
	}

private:
	void to_host(const std::uint64_t first, const std::uint64_t elements, const bool queued) {
		const std::uint64_t begin = std::min(first, size());
		const std::uint64_t n = std::min(elements, size()-begin);
		if(!has_host() || !buffer_.allocated() || n==0u) return;
		detail::to_host(buffer_.owner(), host_.data()+begin, buffer_.address_of(begin),
			(std::size_t)(n*sizeof(T)), queued && host_.pinned());
	}
	void to_device(const std::uint64_t first, const std::uint64_t elements, const bool queued) {
		const std::uint64_t begin = std::min(first, size());
		const std::uint64_t n = std::min(elements, size()-begin);
		if(!has_host() || !buffer_.allocated() || n==0u) return;
		detail::to_device(buffer_.owner(), buffer_.address_of(begin), host_.data()+begin,
			(std::size_t)(n*sizeof(T)), queued && host_.pinned());
	}
	void plane(const std::uint64_t x0, const std::uint64_t x1, const std::uint64_t y0, const std::uint64_t y1,
		const std::uint64_t z, const std::uint64_t nx, const std::uint64_t ny, const bool download) {
		if(!has_host() || !buffer_.allocated() || x1<=x0 || y1<=y0) return;
		std::vector<detail::PlaneCopy> planes(components());
		const std::uint64_t corner = x0+(y0+z*ny)*nx;
		for(uint c=0u; c<components(); ++c) {
			const std::uint64_t first = (std::uint64_t)c*count()+corner;
			planes[c].device = buffer_.address_of(first);
			planes[c].host = host_.data()+first;
		}
		const std::size_t pitch = (std::size_t)(nx*sizeof(T));
		const std::size_t row = (std::size_t)((x1-x0)*sizeof(T));
		const std::size_t rows = (std::size_t)(y1-y0);
		if(download) detail::planes_to_host(buffer_.owner(), planes.data(), components(), pitch, row, rows, host_.pinned());
		else detail::planes_to_device(buffer_.owner(), planes.data(), components(), pitch, row, rows, host_.pinned());
	}

	// Host before device: allocation order of the mirror and its buffer.
	HostMirror<T> host_;
	Buffer<T> buffer_;
	HostMemory later_ = HostMemory::pageable; // kind of an on-demand mirror
};

// Queue a halo copy of elements [first, first+elements) from `source` on its
// device into `target` on another (or the same) device, on the target's halo
// copy stream. Uses direct peer access when available, otherwise stages
// through the source's host mirror.
template<typename T>
void copy_halo(Buffer<T>& target, MirroredBuffer<T>& source, const std::uint64_t first,
	const std::uint64_t elements) {
	const std::uint64_t limit = std::min(target.size(), source.size());
	const std::uint64_t begin = std::min(first, limit);
	const std::uint64_t n = std::min(elements, limit-begin);
	if(n==0u) return;
	if(!target.allocated() || !source.buffer().allocated())
		fatal_error("CUDA halo copies need device buffers on both sides.");
	if(!source.has_host())
		fatal_error("CUDA halo copies need a host mirror on the source buffer for staging.");
	detail::peer_copy(target.owner(), target.address_of(begin),
		source.buffer().owner(), source.buffer().address_of(begin),
		source.host().data()+begin, source.host().pinned(), source.host().staging_event(),
		(std::size_t)(n*sizeof(T)));
}

} // namespace culbm::gpu
