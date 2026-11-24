#include "cuda_topology.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
	for(std::uint32_t domains_x=1u; domains_x<=6u; ++domains_x) {
		for(std::uint32_t domains_y=1u; domains_y<=6u; ++domains_y) {
			for(std::uint32_t domains_z=1u; domains_z<=6u; ++domains_z) {
				const std::uint32_t count =
					domains_x*domains_y*domains_z;
				const std::uint32_t dimensions[3] = {
					domains_x, domains_y, domains_z
				};
				for(std::uint32_t direction=0u;
				    direction<3u; ++direction) {
					std::vector<std::uint32_t> incoming(count, 0u);
					for(std::uint32_t domain=0u;
					    domain<count; ++domain) {
						std::uint32_t current = domain;
						for(std::uint32_t hop=0u;
						    hop<dimensions[direction]; ++hop) {
							current =
								culbm::
									positive_domain_neighbor(
										current,
										domains_x,
										domains_y,
										domains_z,
										direction);
							if(current>=count) {
								std::cerr
									<< "out-of-range neighbor\n";
								return 1;
							}
						}
						if(current!=domain) {
							std::cerr
								<< "periodic ring did not close\n";
							return 1;
						}
						const std::uint32_t neighbor =
							culbm::
								positive_domain_neighbor(
									domain,
									domains_x,
									domains_y,
									domains_z,
									direction);
						++incoming[neighbor];
						const std::uint32_t back =
							culbm::
								negative_domain_neighbor(
									neighbor,
									domains_x,
									domains_y,
									domains_z,
									direction);
						if(back!=domain) {
							std::cerr
								<< "positive/negative maps disagree\n";
							return 1;
						}
					}
					for(const std::uint32_t edges : incoming) {
						if(edges!=1u) {
							std::cerr
								<< "neighbor map is not bijective\n";
							return 1;
						}
					}
				}
			}
		}
	}

	// Device ordering is unchanged without a strictly better peer route.
	const std::array<std::uint64_t, 3u> x_face_weight = { 1ull, 0ull, 0ull };
	std::vector<unsigned char> no_peer(16u, 0u);
	std::vector<std::uint32_t> unchanged = { 0u, 2u, 1u, 3u };
	if(culbm::optimize_peer_device_assignment(
		   no_peer, 4u, unchanged, 4u, 1u, 1u, x_face_weight) ||
	   unchanged != std::vector<std::uint32_t>({ 0u, 2u, 1u, 3u })) {
		std::cerr << "peer optimizer changed a tie assignment\n";
		return 2;
	}

	// Two direct-P2P pairs should be placed on adjacent positions of a
	// four-domain ring when that is a strict improvement over the initial map.
	std::vector<unsigned char> peer(16u, 0u);
	auto connect = [&](const std::uint32_t left, const std::uint32_t right) {
		peer[(std::size_t)left*4u+right] = 1u;
		peer[(std::size_t)right*4u+left] = 1u;
	};
	connect(0u, 1u);
	connect(2u, 3u);
	std::vector<std::uint32_t> assignment = { 0u, 2u, 1u, 3u };
	const std::uint64_t initial_score = culbm::peer_assignment_score(
		peer, 4u, assignment, 4u, 1u, 1u, x_face_weight);
	std::uint64_t optimized_score = 0ull;
	if(!culbm::optimize_peer_device_assignment(
		   peer, 4u, assignment, 4u, 1u, 1u, x_face_weight,
		   &optimized_score) || optimized_score <= initial_score) {
		std::cerr << "peer optimizer did not improve a routable topology\n";
		return 3;
	}
	std::cout
		<< "Validated periodic Cartesian domain maps and P2P-aware "
		<< "device ordering for Dx,Dy,Dz in [1,6] (up to 216 GPUs).\n";
	return 0;
}
