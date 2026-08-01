#include "cuda_mpi.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#ifdef CULBM_ENABLE_MPI
#include <mpi.h>
#endif

namespace culbm {
namespace {

constexpr std::uint64_t mpi_chunk_bytes = 64ull*1024ull*1024ull;

#ifdef CULBM_ENABLE_MPI
struct Mpi_Runtime {
	std::mutex mutex;
	std::recursive_mutex simulation_call_mutex;
	std::recursive_mutex graphics_call_mutex;
	std::atomic_bool active = false;
	std::atomic_bool initialization_attempted = false;
	bool initialized_here = false;
	int rank = 0;
	int size = 1;
	int local_rank = 0;
	int local_size = 1;
	MPI_Comm local_communicator = MPI_COMM_NULL;
	MPI_Comm simulation_communicator = MPI_COMM_NULL;
	MPI_Comm graphics_communicator = MPI_COMM_NULL;

	~Mpi_Runtime() {
		if(!active.load(std::memory_order_acquire)) return;
		std::lock_guard<std::recursive_mutex> simulation_lock(
			simulation_call_mutex);
		std::lock_guard<std::recursive_mutex> graphics_lock(
			graphics_call_mutex);
		int initialized = 0;
		int finalized = 0;
		MPI_Initialized(&initialized);
		if(initialized) MPI_Finalized(&finalized);
		if(!initialized || finalized) return;
		if(graphics_communicator!=MPI_COMM_NULL) {
			MPI_Comm_free(&graphics_communicator);
			graphics_communicator = MPI_COMM_NULL;
		}
		if(simulation_communicator!=MPI_COMM_NULL) {
			MPI_Comm_free(&simulation_communicator);
			simulation_communicator = MPI_COMM_NULL;
		}
		if(local_communicator!=MPI_COMM_NULL) {
			MPI_Comm_free(&local_communicator);
			local_communicator = MPI_COMM_NULL;
		}
		if(initialized_here) MPI_Finalize();
	}
};

bool environment_value_is_false(const char* const value) {
	return value!=nullptr && (
		std::strcmp(value, "0")==0 ||
		std::strcmp(value, "false")==0 ||
		std::strcmp(value, "FALSE")==0 ||
		std::strcmp(value, "off")==0 ||
		std::strcmp(value, "OFF")==0 ||
		std::strcmp(value, "no")==0 ||
		std::strcmp(value, "NO")==0);
}

bool mpi_launcher_detected() {
	const char* const launcher_markers[] = {
		"PMI_RANK",
		"PMIX_RANK",
		"OMPI_COMM_WORLD_RANK",
		"MV2_COMM_WORLD_RANK",
		"MPI_LOCALRANKID",
		"SLURM_PROCID",
		"ALPS_APP_PE",
		"PALS_RANKID"
	};
	for(const char* const marker : launcher_markers) {
		const char* const value = std::getenv(marker);
		if(value!=nullptr && *value!='\0') return true;
	}
	return false;
}

bool mpi_initialization_requested() {
	const char* const override_value = std::getenv("CULBM_MPI");
	if(override_value!=nullptr && *override_value!='\0')
		return !environment_value_is_false(override_value);
	return mpi_launcher_detected();
}

Mpi_Runtime& runtime() {
	static Mpi_Runtime value;
	return value;
}

[[noreturn]] void mpi_abort_with_message(const std::string& message) {
	Mpi_Runtime& state = runtime();
	std::fprintf(
		stderr, "CULBM MPI error on rank %d: %s\n",
		state.rank, message.c_str());
	std::fflush(stderr);
	MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	std::abort();
}

void mpi_check(const int result, const char* operation) {
	if(result==MPI_SUCCESS) return;
	char error[MPI_MAX_ERROR_STRING] = {};
	int length = 0;
	MPI_Error_string(result, error, &length);
	mpi_abort_with_message(
		std::string(operation)+" failed: "+
		std::string(error, (std::size_t)std::max(length, 0)));
}

void ensure_initialized() {
	if(!mpi_is_initialized()) mpi_initialize();
}

#else

void ensure_initialized() {}

#endif

std::uint32_t parse_device_override(const std::uint32_t count) {
	const char* text = std::getenv("CULBM_DEVICE");
	if(text==nullptr || *text=='\0') return UINT_MAX;
	errno = 0;
	char* end = nullptr;
	const unsigned long value = std::strtoul(text, &end, 10);
	if(errno!=0 || end==text || *end!='\0' || value>=count) {
		throw std::runtime_error(
			"CULBM_DEVICE must be a visible CUDA ordinal in [0, "+
			std::to_string(count==0u ? 0u : count-1u)+"]");
	}
	return (std::uint32_t)value;
}

} // namespace

void mpi_initialize(int* argc, char*** argv) {
#ifdef CULBM_ENABLE_MPI
	Mpi_Runtime& state = runtime();
	if(state.active.load(std::memory_order_acquire) ||
	   state.initialization_attempted.load(std::memory_order_acquire))
		return;
	std::lock_guard<std::mutex> lock(state.mutex);
	if(state.active.load(std::memory_order_acquire) ||
	   state.initialization_attempted.load(std::memory_order_relaxed))
		return;
	std::lock_guard<std::recursive_mutex> mpi_lock(
		state.simulation_call_mutex);
	int initialized = 0;
	mpi_check(MPI_Initialized(&initialized), "MPI_Initialized");
	const char* const override_value = std::getenv("CULBM_MPI");
	if(environment_value_is_false(override_value)) {
		if(initialized || mpi_launcher_detected()) {
			std::fprintf(
				stderr,
				"CULBM_MPI=0 cannot disable an initialized or "
				"launcher-managed MPI process.\n");
			std::fflush(stderr);
			std::exit(EXIT_FAILURE);
		}
		state.initialization_attempted.store(true, std::memory_order_release);
		return;
	}
	if(!initialized && !mpi_initialization_requested()) {
		// A normal direct launch keeps the local multi-GPU fast path and does
		// not start an otherwise unused MPI runtime. MPI launchers expose one
		// of the rank markers above; CULBM_MPI=1 is the portable override.
		state.initialization_attempted.store(true, std::memory_order_release);
		return;
	}
	if(!initialized) {
		int provided = MPI_THREAD_SINGLE;
		mpi_check(
			MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided),
			"MPI_Init_thread");
		state.initialized_here = true;
	}
	int provided = MPI_THREAD_SINGLE;
	mpi_check(MPI_Query_thread(&provided), "MPI_Query_thread");
	if(provided<MPI_THREAD_MULTIPLE)
		mpi_abort_with_message(
			"CULBM distributed graphics requires MPI_THREAD_MULTIPLE");
	mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &state.rank), "MPI_Comm_rank");
	mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &state.size), "MPI_Comm_size");
	mpi_check(
		MPI_Comm_dup(MPI_COMM_WORLD, &state.simulation_communicator),
		"MPI_Comm_dup(simulation)");
	mpi_check(
		MPI_Comm_dup(MPI_COMM_WORLD, &state.graphics_communicator),
		"MPI_Comm_dup(graphics)");
	if(state.local_communicator==MPI_COMM_NULL) {
		mpi_check(
			MPI_Comm_split_type(
				MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, state.rank,
				MPI_INFO_NULL, &state.local_communicator),
			"MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)");
		mpi_check(
			MPI_Comm_rank(state.local_communicator, &state.local_rank),
			"MPI_Comm_rank(local)");
		mpi_check(
			MPI_Comm_size(state.local_communicator, &state.local_size),
			"MPI_Comm_size(local)");
	}
	state.active.store(true, std::memory_order_release);
	state.initialization_attempted.store(true, std::memory_order_release);
#else
	(void)argc;
	(void)argv;
#endif
}

void mpi_barrier() {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	Mpi_Runtime& state = runtime();
	if(state.size>1) {
		std::lock_guard<std::recursive_mutex> mpi_lock(
			state.simulation_call_mutex);
		mpi_check(
			MPI_Barrier(state.simulation_communicator),
			"MPI_Barrier");
	}
#endif
}

[[noreturn]] void mpi_abort(const char* message) {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	if(mpi_is_initialized())
		mpi_abort_with_message(message==nullptr ? "unspecified error" : message);
	std::fprintf(
		stderr, "CULBM error: %s\n",
		message==nullptr ? "unspecified error" : message);
	std::fflush(stderr);
	std::exit(EXIT_FAILURE);
#else
	std::fprintf(
		stderr, "CULBM error: %s\n",
		message==nullptr ? "unspecified error" : message);
	std::fflush(stderr);
	std::exit(EXIT_FAILURE);
#endif
}

bool mpi_is_compiled() {
#ifdef CULBM_ENABLE_MPI
	return true;
#else
	return false;
#endif
}

bool mpi_is_initialized() {
#ifdef CULBM_ENABLE_MPI
	return runtime().active.load(std::memory_order_acquire);
#else
	return false;
#endif
}

bool mpi_is_distributed() {
#ifdef CULBM_ENABLE_MPI
	return mpi_is_initialized() && runtime().size>1;
#else
	return false;
#endif
}

bool mpi_is_root() {
#ifdef CULBM_ENABLE_MPI
	return !mpi_is_initialized() || runtime().rank==0;
#else
	return true;
#endif
}

int mpi_rank() {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	return runtime().rank;
#else
	return 0;
#endif
}

int mpi_size() {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	return runtime().size;
#else
	return 1;
#endif
}

int mpi_local_rank() {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	return runtime().local_rank;
#else
	return 0;
#endif
}

int mpi_local_size() {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	return runtime().local_size;
#else
	return 1;
#endif
}

void mpi_validate_domain_count(const std::uint32_t domain_count) {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	if(runtime().size<=1) return;
	if(domain_count!=(std::uint32_t)runtime().size) {
		mpi_abort_with_message(
			"deck has Dx*Dy*Dz="+std::to_string(domain_count)+
			", but the MPI job has "+std::to_string(runtime().size)+
			" ranks; distributed mode requires one domain per rank");
	}
#else
	(void)domain_count;
#endif
}

std::uint32_t mpi_cuda_device_ordinal(
	const std::uint32_t visible_device_count) {
	if(visible_device_count==0u) {
#ifdef CULBM_ENABLE_MPI
		if(mpi_is_distributed())
			mpi_abort_with_message("No visible CUDA devices");
#endif
		throw std::runtime_error("No visible CUDA devices");
	}
	std::uint32_t override = UINT_MAX;
	try {
		override = parse_device_override(visible_device_count);
	} catch(const std::exception& error) {
#ifdef CULBM_ENABLE_MPI
		if(mpi_is_distributed()) mpi_abort_with_message(error.what());
#else
		(void)error;
#endif
		throw;
	}
	if(override!=UINT_MAX) return override;
	if(visible_device_count==1u) return 0u;
	const int local = mpi_local_rank();
	if(local<0 || (std::uint32_t)local>=visible_device_count) {
#ifdef CULBM_ENABLE_MPI
		mpi_abort_with_message(
			"local MPI rank "+std::to_string(local)+
			" cannot be bound without oversubscription: only "+
			std::to_string(visible_device_count)+
			" CUDA devices are visible on this node; set CUDA_VISIBLE_DEVICES "
			"per rank or CULBM_DEVICE explicitly");
#else
		throw std::runtime_error("CUDA device ordinal is out of range");
#endif
	}
	return (std::uint32_t)local;
}

void mpi_allgather_equal_bytes(
	const void* send_buffer,
	const std::uint64_t bytes_per_rank,
	void* receive_buffer) {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	Mpi_Runtime& state = runtime();
	if(bytes_per_rank==0ull) return;
	if(state.size==1) {
		std::memcpy(receive_buffer, send_buffer, (std::size_t)bytes_per_rank);
		return;
	}
	std::lock_guard<std::recursive_mutex> mpi_lock(
		state.simulation_call_mutex);
	char* output = static_cast<char*>(receive_buffer);
	std::memcpy(
		output+(std::uint64_t)state.rank*bytes_per_rank,
		send_buffer, (std::size_t)bytes_per_rank);
	for(int root=0; root<state.size; ++root) {
		for(std::uint64_t offset=0ull; offset<bytes_per_rank;
		    offset+=mpi_chunk_bytes) {
			const int chunk = (int)std::min(
				mpi_chunk_bytes, bytes_per_rank-offset);
			mpi_check(
				MPI_Bcast(
					output+(std::uint64_t)root*bytes_per_rank+offset,
					chunk, MPI_BYTE, root, state.simulation_communicator),
				"MPI_Bcast(field gather)");
		}
	}
#else
	if(bytes_per_rank>0ull)
		std::memcpy(receive_buffer, send_buffer, (std::size_t)bytes_per_rank);
#endif
}

void mpi_gather_equal_bytes(
	const void* send_buffer,
	const std::uint64_t bytes_per_rank,
	void* receive_buffer_on_root) {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	Mpi_Runtime& state = runtime();
	if(bytes_per_rank==0ull) return;
	if(state.size==1) {
		if(receive_buffer_on_root!=nullptr)
			std::memcpy(
				receive_buffer_on_root, send_buffer,
				(std::size_t)bytes_per_rank);
		return;
	}
	std::lock_guard<std::recursive_mutex> mpi_lock(
		state.graphics_call_mutex);
	constexpr int gather_tag = 31017;
	if(state.rank==0) {
		if(receive_buffer_on_root==nullptr)
			mpi_abort_with_message(
				"root receive buffer is null in mpi_gather_equal_bytes");
		char* output = static_cast<char*>(receive_buffer_on_root);
		std::memcpy(output, send_buffer, (std::size_t)bytes_per_rank);
		for(int source=1; source<state.size; ++source) {
			for(std::uint64_t offset=0ull; offset<bytes_per_rank;
			    offset+=mpi_chunk_bytes) {
				const int chunk = (int)std::min(
					mpi_chunk_bytes, bytes_per_rank-offset);
				mpi_check(
					MPI_Recv(
						output+(std::uint64_t)source*bytes_per_rank+offset,
						chunk, MPI_BYTE, source, gather_tag,
						state.graphics_communicator, MPI_STATUS_IGNORE),
					"MPI_Recv(root gather)");
			}
		}
	} else {
		const char* input = static_cast<const char*>(send_buffer);
		for(std::uint64_t offset=0ull; offset<bytes_per_rank;
		    offset+=mpi_chunk_bytes) {
			const int chunk = (int)std::min(
				mpi_chunk_bytes, bytes_per_rank-offset);
			mpi_check(
				MPI_Send(
					input+offset, chunk, MPI_BYTE, 0, gather_tag,
					state.graphics_communicator),
				"MPI_Send(root gather)");
		}
	}
#else
	if(bytes_per_rank>0ull && receive_buffer_on_root!=nullptr)
		std::memcpy(
			receive_buffer_on_root, send_buffer,
			(std::size_t)bytes_per_rank);
#endif
}

void mpi_broadcast_simulation_bytes(
	void* buffer,
	const std::uint64_t bytes,
	const int root_rank) {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	Mpi_Runtime& state = runtime();
	if(bytes==0ull || state.size<=1) return;
	if(buffer==nullptr)
		mpi_abort_with_message(
			"buffer is null in mpi_broadcast_simulation_bytes");
	std::lock_guard<std::recursive_mutex> mpi_lock(
		state.simulation_call_mutex);
	char* data = static_cast<char*>(buffer);
	for(std::uint64_t offset=0ull; offset<bytes;
	    offset+=mpi_chunk_bytes) {
		const int chunk = (int)std::min(
			mpi_chunk_bytes, bytes-offset);
		mpi_check(
			MPI_Bcast(
				data+offset, chunk, MPI_BYTE, root_rank,
				state.simulation_communicator),
			"MPI_Bcast(simulation state)");
	}
#else
	(void)buffer;
	(void)bytes;
	(void)root_rank;
#endif
}

void mpi_broadcast_graphics_bytes(
	void* buffer,
	const std::uint64_t bytes,
	const int root_rank) {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	Mpi_Runtime& state = runtime();
	if(bytes==0ull || state.size<=1) return;
	if(buffer==nullptr)
		mpi_abort_with_message(
			"buffer is null in mpi_broadcast_graphics_bytes");
	std::lock_guard<std::recursive_mutex> mpi_lock(
		state.graphics_call_mutex);
	char* data = static_cast<char*>(buffer);
	for(std::uint64_t offset=0ull; offset<bytes;
	    offset+=mpi_chunk_bytes) {
		const int chunk = (int)std::min(
			mpi_chunk_bytes, bytes-offset);
		mpi_check(
			MPI_Bcast(
				data+offset, chunk, MPI_BYTE, root_rank,
				state.graphics_communicator),
			"MPI_Bcast(graphics state)");
	}
#else
	(void)buffer;
	(void)bytes;
	(void)root_rank;
#endif
}

bool mpi_synchronize_graphics_control(
	const bool local_running,
	const bool local_render_ready,
	const std::uint64_t local_session_id,
	bool* const all_render_ready) {
	if(all_render_ready==nullptr)
		throw std::runtime_error(
			"all_render_ready is null in "
			"mpi_synchronize_graphics_control");
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	Mpi_Runtime& state = runtime();
	if(state.size<=1) {
		*all_render_ready = local_render_ready;
		return local_running;
	}
	const unsigned int local[2] = {
		local_running ? 1u : 0u,
		local_render_ready ? 1u : 0u
	};
	unsigned int global[2] = {};
	unsigned long long session_min =
		(unsigned long long)local_session_id;
	unsigned long long session_max = session_min;
	std::lock_guard<std::recursive_mutex> mpi_lock(
		state.graphics_call_mutex);
	mpi_check(
		MPI_Allreduce(
			local, global, 2, MPI_UNSIGNED, MPI_MIN,
			state.graphics_communicator),
		"MPI_Allreduce(graphics control)");
	mpi_check(
		MPI_Allreduce(
			MPI_IN_PLACE, &session_min, 1, MPI_UNSIGNED_LONG_LONG,
			MPI_MIN, state.graphics_communicator),
		"MPI_Allreduce(graphics session minimum)");
	mpi_check(
		MPI_Allreduce(
			MPI_IN_PLACE, &session_max, 1, MPI_UNSIGNED_LONG_LONG,
			MPI_MAX, state.graphics_communicator),
		"MPI_Allreduce(graphics session maximum)");
	*all_render_ready =
		global[1]!=0u && session_min==session_max;
	return global[0]!=0u;
#else
	(void)local_session_id;
	*all_render_ready = local_render_ready;
	return local_running;
#endif
}

void mpi_sendrecv_bytes(
	const void* send_buffer,
	const std::uint64_t send_bytes,
	const int destination_rank,
	void* receive_buffer,
	const std::uint64_t receive_bytes,
	const int source_rank,
	const int tag) {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	if(send_bytes!=receive_bytes)
		throw std::runtime_error(
			"halo send/receive byte counts differ in mpi_sendrecv_bytes");
	if(!mpi_is_initialized() || runtime().size<=1) {
		if(send_bytes>0ull)
			std::memcpy(
				receive_buffer, send_buffer, (std::size_t)send_bytes);
		return;
	}
	Mpi_Runtime& state = runtime();
	std::lock_guard<std::recursive_mutex> mpi_lock(
		state.simulation_call_mutex);
	const char* send = static_cast<const char*>(send_buffer);
	char* receive = static_cast<char*>(receive_buffer);
	for(std::uint64_t offset=0ull; offset<send_bytes;
	    offset+=mpi_chunk_bytes) {
		const int chunk = (int)std::min(
			mpi_chunk_bytes, send_bytes-offset);
		mpi_check(
			MPI_Sendrecv(
				send+offset, chunk, MPI_BYTE, destination_rank, tag,
				receive+offset, chunk, MPI_BYTE, source_rank, tag,
				state.simulation_communicator, MPI_STATUS_IGNORE),
			"MPI_Sendrecv(halo)");
	}
#else
	(void)destination_rank;
	(void)source_rank;
	(void)tag;
	if(send_bytes!=receive_bytes)
		throw std::runtime_error(
			"halo send/receive byte counts differ");
	if(send_bytes>0ull)
		std::memcpy(receive_buffer, send_buffer, (std::size_t)send_bytes);
#endif
}

namespace {

void mpi_exchange_halo_pair_impl(
	const void* send_positive,
	const void* send_negative,
	const std::uint64_t bytes,
	const int positive_rank,
	const int negative_rank,
	void* receive_from_positive,
	void* receive_from_negative,
	const int positive_tag,
	const int negative_tag,
	const bool graphics_channel) {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	if(!mpi_is_initialized() || runtime().size<=1) {
		if(bytes>0ull) {
			std::memcpy(receive_from_negative, send_positive,
				(std::size_t)bytes);
			std::memcpy(receive_from_positive, send_negative,
				(std::size_t)bytes);
		}
		return;
	}
	Mpi_Runtime& state = runtime();
	std::recursive_mutex& call_mutex = graphics_channel
		? state.graphics_call_mutex : state.simulation_call_mutex;
	std::lock_guard<std::recursive_mutex> mpi_lock(call_mutex);
	const MPI_Comm communicator = graphics_channel
		? state.graphics_communicator : state.simulation_communicator;
	const char* const positive = static_cast<const char*>(send_positive);
	const char* const negative = static_cast<const char*>(send_negative);
	char* const from_positive = static_cast<char*>(receive_from_positive);
	char* const from_negative = static_cast<char*>(receive_from_negative);
	for(std::uint64_t offset=0ull; offset<bytes; offset+=mpi_chunk_bytes) {
		const int chunk = (int)std::min(mpi_chunk_bytes, bytes-offset);
		MPI_Request requests[4] = {};
		mpi_check(MPI_Irecv(
			from_negative+offset, chunk, MPI_BYTE, negative_rank, positive_tag,
			communicator, &requests[0]), "MPI_Irecv(halo positive payload)");
		mpi_check(MPI_Irecv(
			from_positive+offset, chunk, MPI_BYTE, positive_rank, negative_tag,
			communicator, &requests[1]), "MPI_Irecv(halo negative payload)");
		mpi_check(MPI_Isend(
			positive+offset, chunk, MPI_BYTE, positive_rank, positive_tag,
			communicator, &requests[2]), "MPI_Isend(halo positive payload)");
		mpi_check(MPI_Isend(
			negative+offset, chunk, MPI_BYTE, negative_rank, negative_tag,
			communicator, &requests[3]), "MPI_Isend(halo negative payload)");
		mpi_check(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE),
			"MPI_Waitall(halo pair)");
	}
#else
	(void)positive_rank;
	(void)negative_rank;
	(void)positive_tag;
	(void)negative_tag;
	(void)graphics_channel;
	if(bytes>0ull) {
		std::memcpy(receive_from_negative, send_positive, (std::size_t)bytes);
		std::memcpy(receive_from_positive, send_negative, (std::size_t)bytes);
	}
#endif
}

} // namespace

void mpi_exchange_halo_pair_bytes(
	const void* send_positive,
	const void* send_negative,
	const std::uint64_t bytes,
	const int positive_rank,
	const int negative_rank,
	void* receive_from_positive,
	void* receive_from_negative,
	const int positive_tag,
	const int negative_tag) {
	mpi_exchange_halo_pair_impl(
		send_positive, send_negative, bytes, positive_rank, negative_rank,
		receive_from_positive, receive_from_negative, positive_tag, negative_tag,
		false);
}

void mpi_sendrecv_graphics_bytes(
	const void* send_buffer,
	const std::uint64_t send_bytes,
	const int destination_rank,
	void* receive_buffer,
	const std::uint64_t receive_bytes,
	const int source_rank,
	const int tag) {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	if(send_bytes!=receive_bytes)
		throw std::runtime_error(
			"halo send/receive byte counts differ in "
			"mpi_sendrecv_graphics_bytes");
	if(!mpi_is_initialized() || runtime().size<=1) {
		if(send_bytes>0ull)
			std::memcpy(
				receive_buffer, send_buffer, (std::size_t)send_bytes);
		return;
	}
	Mpi_Runtime& state = runtime();
	std::lock_guard<std::recursive_mutex> mpi_lock(
		state.graphics_call_mutex);
	const char* send = static_cast<const char*>(send_buffer);
	char* receive = static_cast<char*>(receive_buffer);
	for(std::uint64_t offset=0ull; offset<send_bytes;
	    offset+=mpi_chunk_bytes) {
		const int chunk = (int)std::min(
			mpi_chunk_bytes, send_bytes-offset);
		mpi_check(
			MPI_Sendrecv(
				send+offset, chunk, MPI_BYTE, destination_rank, tag,
				receive+offset, chunk, MPI_BYTE, source_rank, tag,
				state.graphics_communicator, MPI_STATUS_IGNORE),
			"MPI_Sendrecv(graphics halo)");
	}
#else
	(void)destination_rank;
	(void)source_rank;
	(void)tag;
	if(send_bytes!=receive_bytes)
		throw std::runtime_error(
			"graphics halo send/receive byte counts differ");
	if(send_bytes>0ull)
		std::memcpy(receive_buffer, send_buffer, (std::size_t)send_bytes);
#endif
}

void mpi_exchange_graphics_halo_pair_bytes(
	const void* send_positive,
	const void* send_negative,
	const std::uint64_t bytes,
	const int positive_rank,
	const int negative_rank,
	void* receive_from_positive,
	void* receive_from_negative,
	const int positive_tag,
	const int negative_tag) {
	mpi_exchange_halo_pair_impl(
		send_positive, send_negative, bytes, positive_rank, negative_rank,
		receive_from_positive, receive_from_negative, positive_tag, negative_tag,
		true);
}

void mpi_allreduce_sum_float(
	float* values, const std::uint32_t count) {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	Mpi_Runtime& state = runtime();
	if(state.size<=1 || count==0u) return;
	std::lock_guard<std::recursive_mutex> mpi_lock(
		state.simulation_call_mutex);
	mpi_check(
		MPI_Allreduce(
			MPI_IN_PLACE, values, (int)count, MPI_FLOAT, MPI_SUM,
			state.simulation_communicator),
		"MPI_Allreduce(float sum)");
#else
	(void)values;
	(void)count;
#endif
}

void mpi_allreduce_sum_uint64(
	std::uint64_t* values, const std::uint32_t count) {
#ifdef CULBM_ENABLE_MPI
	ensure_initialized();
	Mpi_Runtime& state = runtime();
	if(state.size<=1 || count==0u) return;
	std::lock_guard<std::recursive_mutex> mpi_lock(
		state.simulation_call_mutex);
	mpi_check(
		MPI_Allreduce(
			MPI_IN_PLACE, values, (int)count, MPI_UNSIGNED_LONG_LONG,
			MPI_SUM, state.simulation_communicator),
		"MPI_Allreduce(uint64 sum)");
#else
	(void)values;
	(void)count;
#endif
}

} // namespace culbm
