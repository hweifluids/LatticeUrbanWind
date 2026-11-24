#include "vram.hpp"

#include <limits>

namespace culbm::gpu {
namespace {

constexpr std::uint64_t saturated = std::numeric_limits<std::uint64_t>::max();

std::uint64_t saturating_add(const std::uint64_t a, const std::uint64_t b, bool& valid) {
	if(b>saturated-a) {
		valid = false;
		return saturated;
	}
	return a+b;
}

} // namespace

AllocationTally& AllocationTally::add(const std::uint64_t elements,
	const std::uint64_t bytes_per_element, const std::uint64_t copies) {
	if(elements==0u || copies==0u) return *this;
	if(bytes_per_element>saturated/elements) {
		valid_ = false;
		mib_ = bytes_ = saturated;
		return *this;
	}
	const std::uint64_t one = elements*bytes_per_element;
	for(std::uint64_t copy=0u; copy<copies; ++copy) {
		mib_ = saturating_add(mib_, mib_ceil(one), valid_);
		bytes_ = saturating_add(bytes_, one, valid_);
	}
	return *this;
}

} // namespace culbm::gpu
