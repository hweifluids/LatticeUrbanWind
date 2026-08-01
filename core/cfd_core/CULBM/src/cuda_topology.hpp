#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace culbm {

constexpr std::uint32_t positive_domain_neighbor(
	const std::uint32_t domain,
	const std::uint32_t domains_x,
	const std::uint32_t domains_y,
	const std::uint32_t domains_z,
	const std::uint32_t direction) {
	const std::uint32_t xy = domains_x*domains_y;
	const std::uint32_t x = (domain%xy)%domains_x;
	const std::uint32_t y = (domain%xy)/domains_x;
	const std::uint32_t z = domain/xy;
	const std::uint32_t xp =
		direction==0u ? (x+1u)%domains_x : x;
	const std::uint32_t yp =
		direction==1u ? (y+1u)%domains_y : y;
	const std::uint32_t zp =
		direction==2u ? (z+1u)%domains_z : z;
	return xp+(yp+zp*domains_y)*domains_x;
}

constexpr std::uint32_t negative_domain_neighbor(
	const std::uint32_t domain,
	const std::uint32_t domains_x,
	const std::uint32_t domains_y,
	const std::uint32_t domains_z,
	const std::uint32_t direction) {
	const std::uint32_t xy = domains_x*domains_y;
	const std::uint32_t x = (domain%xy)%domains_x;
	const std::uint32_t y = (domain%xy)/domains_x;
	const std::uint32_t z = domain/xy;
	const std::uint32_t xm =
		direction==0u ? (x+domains_x-1u)%domains_x : x;
	const std::uint32_t ym =
		direction==1u ? (y+domains_y-1u)%domains_y : y;
	const std::uint32_t zm =
		direction==2u ? (z+domains_z-1u)%domains_z : z;
    return xm+(ym+zm*domains_y)*domains_x;
}

// Score an assignment of CUDA devices to Cartesian domains. Each positive
// edge represents one direction's bidirectional halo route; a peer link is
// useful only when CUDA reports access in both directions. Face weights let
// callers favor the largest halo slabs without coupling this header to CUDA.
inline std::uint64_t peer_assignment_score(
    const std::vector<unsigned char>& bidirectional_peer_access,
    const std::uint32_t device_count,
    const std::vector<std::uint32_t>& domain_to_device,
    const std::uint32_t domains_x,
    const std::uint32_t domains_y,
    const std::uint32_t domains_z,
    const std::array<std::uint64_t, 3u>& face_weights) {
    const std::uint32_t domain_count = domains_x*domains_y*domains_z;
    if (domains_x == 0u || domains_y == 0u || domains_z == 0u ||
        domain_to_device.size() != domain_count ||
        bidirectional_peer_access.size() !=
            static_cast<std::size_t>(device_count)*device_count) return 0ull;
    const std::uint32_t counts[3] = { domains_x, domains_y, domains_z };
    std::uint64_t score = 0ull;
    for (std::uint32_t domain = 0u; domain < domain_count; ++domain) {
        const std::uint32_t source_device = domain_to_device[domain];
        if (source_device >= device_count) return 0ull;
        for (std::uint32_t direction = 0u; direction < 3u; ++direction) {
            if (counts[direction] <= 1u || face_weights[direction] == 0ull)
                continue;
            const std::uint32_t neighbor = positive_domain_neighbor(
                domain, domains_x, domains_y, domains_z, direction);
            const std::uint32_t target_device = domain_to_device[neighbor];
            if (target_device >= device_count) return 0ull;
            const std::size_t access_index =
                static_cast<std::size_t>(source_device)*device_count+
                target_device;
            if (bidirectional_peer_access[access_index] != 0u)
                score += face_weights[direction];
        }
    }
    return score;
}

// A deterministic hill climb is sufficient for the small number of GPUs used
// by one process. It only accepts strict improvements, so equal topology
// scores retain the selected-device order exactly.
inline bool optimize_peer_device_assignment(
    const std::vector<unsigned char>& bidirectional_peer_access,
    const std::uint32_t device_count,
    std::vector<std::uint32_t>& domain_to_device,
    const std::uint32_t domains_x,
    const std::uint32_t domains_y,
    const std::uint32_t domains_z,
    const std::array<std::uint64_t, 3u>& face_weights,
    std::uint64_t* const final_score = nullptr) {
    std::uint64_t current = peer_assignment_score(
        bidirectional_peer_access, device_count, domain_to_device,
        domains_x, domains_y, domains_z, face_weights);
    bool changed = false;
    for (;;) {
        std::uint64_t best = current;
        std::uint32_t best_left = 0u;
        std::uint32_t best_right = 0u;
        for (std::uint32_t left = 0u;
             left < static_cast<std::uint32_t>(domain_to_device.size());
             ++left) {
            for (std::uint32_t right = left+1u;
                 right < static_cast<std::uint32_t>(domain_to_device.size());
                 ++right) {
                std::swap(domain_to_device[left], domain_to_device[right]);
                const std::uint64_t candidate = peer_assignment_score(
                    bidirectional_peer_access, device_count, domain_to_device,
                    domains_x, domains_y, domains_z, face_weights);
                std::swap(domain_to_device[left], domain_to_device[right]);
                if (candidate > best) {
                    best = candidate;
                    best_left = left;
                    best_right = right;
                }
            }
        }
        if (best == current) break;
        std::swap(domain_to_device[best_left], domain_to_device[best_right]);
        current = best;
        changed = true;
    }
    if (final_score != nullptr) *final_score = current;
    return changed;
}

} // namespace culbm
