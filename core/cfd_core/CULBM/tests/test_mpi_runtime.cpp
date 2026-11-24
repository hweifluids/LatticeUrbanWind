#include "cuda_mpi.hpp"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
	culbm::mpi_initialize(&argc, &argv);
	const int rank = culbm::mpi_rank();
	const int size = culbm::mpi_size();
	if(size!=2) {
		if(rank==0) std::cerr << "expected exactly two MPI ranks\n";
		return 1;
	}
	if(!culbm::mpi_is_distributed()) return 2;
	if(culbm::mpi_local_rank()<0 ||
	   culbm::mpi_local_rank()>=culbm::mpi_local_size())
		return 3;

	const std::uint64_t local =
		0x1020304050607080ull+(std::uint64_t)rank;
	std::vector<std::uint64_t> gathered((std::size_t)size, 0ull);
	culbm::mpi_allgather_equal_bytes(
		&local, sizeof(local), gathered.data());
	for(int source=0; source<size; ++source) {
		if(gathered[(std::size_t)source] !=
		   0x1020304050607080ull+(std::uint64_t)source)
			return 4;
	}

	const int positive = (rank+1)%size;
	const int negative = (rank+size-1)%size;
	std::uint64_t received = 0ull;
	culbm::mpi_sendrecv_bytes(
		&local, sizeof(local), positive,
		&received, sizeof(received), negative, 77);
	if(received != 0x1020304050607080ull+(std::uint64_t)negative)
		return 5;

	const std::uint64_t send_positive =
		0x5162738495A6B7C8ull+(std::uint64_t)(2*rank);
	const std::uint64_t send_negative = send_positive+1ull;
	std::uint64_t receive_from_positive = 0ull;
	std::uint64_t receive_from_negative = 0ull;
	culbm::mpi_exchange_halo_pair_bytes(
		&send_positive, &send_negative, sizeof(send_positive),
		positive, negative, &receive_from_positive, &receive_from_negative,
		78, 79);
	const std::uint64_t neighbor_positive =
		0x5162738495A6B7C8ull+(std::uint64_t)(2*positive);
	const std::uint64_t neighbor_negative =
		0x5162738495A6B7C8ull+(std::uint64_t)(2*negative)+1ull;
	if(receive_from_positive!=neighbor_negative ||
	   receive_from_negative!=neighbor_positive)
		return 15;

	receive_from_positive = 0ull;
	receive_from_negative = 0ull;
	culbm::mpi_exchange_graphics_halo_pair_bytes(
		&send_positive, &send_negative, sizeof(send_positive),
		positive, negative, &receive_from_positive, &receive_from_negative,
		80, 81);
	if(receive_from_positive!=neighbor_negative ||
	   receive_from_negative!=neighbor_positive)
		return 16;

	float float_sum[3] = {
		(float)(rank+1),
		(float)(2*(rank+1)),
		(float)(3*(rank+1))
	};
	culbm::mpi_allreduce_sum_float(float_sum, 3u);
	if(float_sum[0]!=3.0f || float_sum[1]!=6.0f ||
	   float_sum[2]!=9.0f)
		return 6;

	std::uint64_t integer_sum = (std::uint64_t)(rank+1);
	culbm::mpi_allreduce_sum_uint64(&integer_sum, 1u);
	if(integer_sum!=3ull) return 7;

	std::vector<std::uint64_t> root_gather;
	if(rank==0) root_gather.resize((std::size_t)size);
	culbm::mpi_gather_equal_bytes(
		&local, sizeof(local),
		rank==0 ? root_gather.data() : nullptr);
	if(rank==0) {
		for(int source=0; source<size; ++source) {
			if(root_gather[(std::size_t)source] !=
			   0x1020304050607080ull+(std::uint64_t)source)
				return 8;
		}
	}

	std::uint64_t graphics_state =
		rank==0 ? 0x8877665544332211ull : 0ull;
	culbm::mpi_broadcast_graphics_bytes(
		&graphics_state, sizeof(graphics_state));
	if(graphics_state!=0x8877665544332211ull) return 9;

	std::uint64_t simulation_state =
		rank==0 ? 0x1122334455667788ull : 0ull;
	culbm::mpi_broadcast_simulation_bytes(
		&simulation_state, sizeof(simulation_state));
	if(simulation_state!=0x1122334455667788ull) return 10;

	bool all_render_ready = true;
	if(!culbm::mpi_synchronize_graphics_control(
		   true, rank==0, 7ull, &all_render_ready) ||
	   all_render_ready)
		return 11;
	if(culbm::mpi_synchronize_graphics_control(
		   rank==0, true, 7ull, &all_render_ready) ||
	   !all_render_ready)
		return 12;
	if(!culbm::mpi_synchronize_graphics_control(
		   true, true, 7ull+(std::uint64_t)rank, &all_render_ready) ||
	   all_render_ready)
		return 13;
	if(!culbm::mpi_synchronize_graphics_control(
		   true, true, 9ull, &all_render_ready) ||
	   !all_render_ready)
		return 14;

	// Exercise actual simulation and graphics halo traffic concurrently, with
	// the same tag but different payload sizes and patterns. Separate
	// communicators must prevent cross-channel matching, and guard bytes catch
	// any overwrite beyond the requested receive range.
	constexpr std::size_t guard = 16u;
	constexpr std::size_t physics_bytes = 37u;
	constexpr std::size_t graphics_bytes = 73u;
	constexpr int rounds = 64;
	std::atomic_int ready = 0;
	std::atomic_int concurrent_failure = 0;
	std::vector<std::uint64_t> concurrent_gather;
	if(rank==0) concurrent_gather.resize((std::size_t)size);
	std::vector<unsigned char> physics_send(physics_bytes);
	std::vector<unsigned char> physics_receive(
		physics_bytes+2u*guard, 0xA5u);
	std::vector<unsigned char> graphics_send(graphics_bytes);
	std::vector<unsigned char> graphics_receive(
		graphics_bytes+2u*guard, 0x5Au);
	auto pattern = [](const int channel, const int source_rank,
	                  const int round, const std::size_t index) {
		return (unsigned char)(
			17*channel+53*source_rank+29*round+7*(int)index);
	};
	auto rendezvous = [&]() {
		ready.fetch_add(1, std::memory_order_acq_rel);
		while(ready.load(std::memory_order_acquire)<2)
			std::this_thread::yield();
	};
	auto physics_work = [&]() {
		rendezvous();
		for(int round=0; round<rounds; ++round) {
			for(std::size_t i=0u; i<physics_bytes; ++i)
				physics_send[i] = pattern(1, rank, round, i);
			culbm::mpi_sendrecv_bytes(
				physics_send.data(), physics_bytes, positive,
				physics_receive.data()+guard, physics_bytes,
				negative, 88);
			for(std::size_t i=0u; i<physics_bytes; ++i) {
				if(physics_receive[guard+i] !=
				   pattern(1, negative, round, i))
					concurrent_failure.store(1);
			}
			for(std::size_t i=0u; i<guard; ++i) {
				if(physics_receive[i]!=0xA5u ||
				   physics_receive[guard+physics_bytes+i]!=0xA5u)
					concurrent_failure.store(2);
			}
		}
	};
	auto graphics_work = [&]() {
		rendezvous();
		for(int round=0; round<rounds; ++round) {
			for(std::size_t i=0u; i<graphics_bytes; ++i)
				graphics_send[i] = pattern(2, rank, round, i);
			culbm::mpi_sendrecv_graphics_bytes(
				graphics_send.data(), graphics_bytes, positive,
				graphics_receive.data()+guard, graphics_bytes,
				negative, 88);
			for(std::size_t i=0u; i<graphics_bytes; ++i) {
				if(graphics_receive[guard+i] !=
				   pattern(2, negative, round, i))
					concurrent_failure.store(3);
			}
			for(std::size_t i=0u; i<guard; ++i) {
				if(graphics_receive[i]!=0x5Au ||
				   graphics_receive[guard+graphics_bytes+i]!=0x5Au)
					concurrent_failure.store(4);
			}
		}
		culbm::mpi_gather_equal_bytes(
			&local, sizeof(local),
			rank==0 ? concurrent_gather.data() : nullptr);
	};
	std::thread physics_thread;
	std::thread graphics_thread;
	if(rank==0) {
		physics_thread = std::thread(physics_work);
		graphics_thread = std::thread(graphics_work);
	} else {
		graphics_thread = std::thread(graphics_work);
		physics_thread = std::thread(physics_work);
	}
	physics_thread.join();
	graphics_thread.join();
	if(concurrent_failure.load()!=0)
		return 20+concurrent_failure.load();
	if(rank==0) {
		for(int source=0; source<size; ++source) {
			if(concurrent_gather[(std::size_t)source] !=
			   0x1020304050607080ull+(std::uint64_t)source)
				return 25;
		}
		std::cout << "MPI runtime checks passed on 2 ranks\n";
	}
	culbm::mpi_barrier();
	return 0;
}
