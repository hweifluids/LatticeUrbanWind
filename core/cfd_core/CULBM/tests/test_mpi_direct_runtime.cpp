#include "cuda_mpi.hpp"

#include <cstdint>
#include <cstring>

int main(int argc, char** argv) {
	const bool expect_initialized =
		argc>1 && std::strcmp(argv[1], "--expect-initialized")==0;
	culbm::mpi_initialize(&argc, &argv);
	if(culbm::mpi_is_initialized()!=expect_initialized) return 1;
	if(culbm::mpi_is_distributed()) return 2;
	if(!culbm::mpi_is_root()) return 3;
	if(culbm::mpi_rank()!=0 || culbm::mpi_size()!=1) return 4;
	if(culbm::mpi_local_rank()!=0 || culbm::mpi_local_size()!=1) return 5;

	const std::uint64_t input = 0x1020304050607080ull;
	std::uint64_t gathered = 0ull;
	culbm::mpi_allgather_equal_bytes(
		&input, sizeof(input), &gathered);
	if(gathered!=input) return 6;

	std::uint64_t exchanged = 0ull;
	culbm::mpi_sendrecv_bytes(
		&input, sizeof(input), 0,
		&exchanged, sizeof(exchanged), 0, 77);
	if(exchanged!=input) return 7;

	const std::uint64_t positive = 0xA1A2A3A4A5A6A7A8ull;
	const std::uint64_t negative = 0xB1B2B3B4B5B6B7B8ull;
	std::uint64_t received_positive = 0ull;
	std::uint64_t received_negative = 0ull;
	culbm::mpi_exchange_halo_pair_bytes(
		&positive, &negative, sizeof(positive), 0, 0,
		&received_positive, &received_negative, 78, 79);
	if(received_positive!=negative || received_negative!=positive) return 8;

	float float_sum[2] = { 1.0f, 2.0f };
	culbm::mpi_allreduce_sum_float(float_sum, 2u);
	if(float_sum[0]!=1.0f || float_sum[1]!=2.0f) return 9;
	return 0;
}
