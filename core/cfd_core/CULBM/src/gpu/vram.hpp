#pragma once

// Device-memory bookkeeping: the MiB rounding the allocator books with and a
// tally for planned allocations. The per-cell field table that plans a lattice
// lives with the solver (solver/field_table.hpp).

#include <cstdint>

namespace culbm::gpu {

constexpr std::uint64_t mib_bytes = 1048576u;

// Every allocation is booked in whole MiB, rounded up on its own.
constexpr std::uint64_t mib_ceil(const std::uint64_t bytes) {
	return bytes/mib_bytes+(bytes%mib_bytes!=0u ? 1u : 0u);
}

// Sum of planned allocations with per-allocation MiB rounding. An overflowing
// term saturates the tally and marks it invalid.
class AllocationTally {
public:
	// `copies` allocations of `elements` x `bytes_per_element`; empty ones are free.
	AllocationTally& add(std::uint64_t elements, std::uint64_t bytes_per_element,
		std::uint64_t copies = 1u);
	std::uint64_t mib() const { return mib_; }
	std::uint64_t bytes() const { return bytes_; }
	bool valid() const { return valid_; }

private:
	std::uint64_t mib_ = 0u;
	std::uint64_t bytes_ = 0u;
	bool valid_ = true;
};

} // namespace culbm::gpu
