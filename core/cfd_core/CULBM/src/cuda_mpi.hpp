#ifndef CULBM_MPI_HPP
#define CULBM_MPI_HPP

#include <cstdint>

namespace culbm {

// MPI is initialized from the executable entry point before CULBM starts
// worker threads. A direct launch remains on the local multi-GPU path;
// launcher rank markers or CULBM_MPI=1 activate MPI.
void mpi_initialize(int* argc = nullptr, char*** argv = nullptr);
void mpi_barrier();
[[noreturn]] void mpi_abort(const char* message);

bool mpi_is_compiled();
bool mpi_is_initialized();
bool mpi_is_distributed();
bool mpi_is_root();

int mpi_rank();
int mpi_size();
int mpi_local_rank();
int mpi_local_size();

// In distributed mode CULBM uses one Cartesian domain and one GPU per rank.
// This validates the deck topology and aborts the whole MPI job on mismatch.
void mpi_validate_domain_count(std::uint32_t domain_count);

// Select a CUDA ordinal within the GPUs visible to this process.  A scheduler
// may expose one GPU per rank; otherwise the shared-memory local rank is used.
std::uint32_t mpi_cuda_device_ordinal(std::uint32_t visible_device_count);

// Equal-sized byte collectives deliberately use byte counts and chunking so
// field sizes are not constrained by MPI's signed-int element count.
void mpi_allgather_equal_bytes(
	const void* send_buffer,
	std::uint64_t bytes_per_rank,
	void* receive_buffer);
void mpi_gather_equal_bytes(
	const void* send_buffer,
	std::uint64_t bytes_per_rank,
	void* receive_buffer_on_root);
void mpi_broadcast_simulation_bytes(
	void* buffer,
	std::uint64_t bytes,
	int root_rank = 0);
void mpi_broadcast_graphics_bytes(
	void* buffer,
	std::uint64_t bytes,
	int root_rank = 0);

// Interactive graphics ranks enter one collective control point before every
// frame. A frame is rendered only while every rank still owns a live solver
// object; a stop request from any rank terminates all graphics loops together.
bool mpi_synchronize_graphics_control(
	bool local_running,
	bool local_render_ready,
	std::uint64_t local_session_id,
	bool* all_render_ready);

void mpi_sendrecv_bytes(
	const void* send_buffer,
	std::uint64_t send_bytes,
	int destination_rank,
	void* receive_buffer,
	std::uint64_t receive_bytes,
	int source_rank,
	int tag);

// Exchange the two directed halo slabs for one Cartesian face together.
// Payloads remain raw bytes; this only lets MPI make the opposing transfers
// progress concurrently instead of serializing two Sendrecv calls.
void mpi_exchange_halo_pair_bytes(
	const void* send_positive,
	const void* send_negative,
	std::uint64_t bytes,
	int positive_rank,
	int negative_rank,
	void* receive_from_positive,
	void* receive_from_negative,
	int positive_tag,
	int negative_tag);

// Graphics halo traffic has a separate communicator and call lock so it can
// progress independently from the simulation thread on every MPI rank.
void mpi_sendrecv_graphics_bytes(
	const void* send_buffer,
	std::uint64_t send_bytes,
	int destination_rank,
	void* receive_buffer,
	std::uint64_t receive_bytes,
	int source_rank,
	int tag);

void mpi_exchange_graphics_halo_pair_bytes(
	const void* send_positive,
	const void* send_negative,
	std::uint64_t bytes,
	int positive_rank,
	int negative_rank,
	void* receive_from_positive,
	void* receive_from_negative,
	int positive_tag,
	int negative_tag);

void mpi_allreduce_sum_float(float* values, std::uint32_t count);
void mpi_allreduce_sum_double(double* values, std::uint32_t count);
void mpi_allreduce_max_float(float* values, std::uint32_t count);
void mpi_allreduce_sum_uint64(std::uint64_t* values, std::uint32_t count);

} // namespace culbm

#endif // CULBM_MPI_HPP
