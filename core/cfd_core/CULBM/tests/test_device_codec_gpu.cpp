// GPU bit-exactness unit test for the Step 7a device common layer:
//   - decode_fp16c across all 65,536 16-bit codes
//   - encode_fp16c across signed zeros, subnormals, transitions, boundaries and random floats
//   - cell_coord / cell_linear / make_axis_offsets / neighbor_cell wrap-around on a non-power-of-two grid
//   - culbm_init_damping_tables bit-exactness against direct device sinf evaluation

#include "gpu/gpu.hpp"
#include "solver/kernel_set.hpp"
#include "solver/lattice_config.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

float coriolis_Omegax_lbmu = 0.0f;
float coriolis_Omegay_lbmu = 0.0f;
float coriolis_Omegaz_lbmu = 0.0f;
bool buffer_nudging_active = true;
int buffer_n_cells = 24;
float buffer_inv_tau_lbmu = 1.0f / 300.0f;
int buffer_nudge_vertical = 1;
int buffer_downstream_face_id = 2;
bool top_sponge_active = true;
int sponge_n_cells = 30;
float sponge_inv_tau_lbmu = 1.0f / 120.0f;
int sponge_ref_mode = 0;

namespace {

std::uint32_t f2u(const float v) {
	std::uint32_t u = 0u;
	std::memcpy(&u, &v, sizeof(u));
	return u;
}

float u2f(const std::uint32_t u) {
	float v = 0.0f;
	std::memcpy(&v, &u, sizeof(v));
	return v;
}

std::uint32_t reference_decode_bits(const std::uint16_t x) {
	const std::uint32_t e = (x & 0x7800u) >> 11;
	const std::uint32_t m = (x & 0x07FFu) << 12;
	const std::uint32_t v = f2u(static_cast<float>(m)) >> 23;
	return ((static_cast<std::uint32_t>(x & 0x8000u)) << 16) |
	       (e != 0u ? (((e + 112u) << 23) | m) : 0u) |
	       ((e == 0u && m != 0u) ? (((v - 37u) << 23) | ((m << (150u - v)) & 0x007FF000u)) : 0u);
}

std::uint16_t reference_encode(const float x) {
	const std::uint32_t b = f2u(x) + 0x00000800u;
	const std::uint32_t e = (b & 0x7F800000u) >> 23;
	const std::uint32_t m = b & 0x007FFFFFu;
	const std::uint32_t sign = (b & 0x80000000u) >> 16;
	const std::uint32_t norm = (e > 112u) ? ((((e - 112u) << 11) & 0x7800u) | (m >> 12)) : 0u;
	const std::uint32_t sub = (e < 113u && e > 100u) ? ((((0x007FF800u + m) >> (124u - e)) + 1u) >> 1) : 0u;
	return static_cast<std::uint16_t>(sign | norm | sub);
}

const char* const kTestKernels = R"CUDA(
extern "C" __global__ void test_decode_all(uint *out_bits) {
  const uint x = (uint)cuda_global_thread_id(0);
  if (x >= 65536u) return;
  out_bits[x] = __float_as_uint(decode_fp16c((ushort)x));
}

extern "C" __global__ void test_encode_batch(const uint *in_bits, ushort *out_codes, const uint count) {
  const uint i = (uint)cuda_global_thread_id(0);
  if (i >= count) return;
  out_codes[i] = encode_fp16c(__uint_as_float(in_bits[i]));
}

extern "C" __global__ void test_neighbors_grid(uint *errors) {
  const cell_idx_t n = (cell_idx_t)cuda_global_thread_id(0);
  if (n >= (cell_idx_t)kcfg::cells) return;
  const uint3 xyz = cell_coord(n);
  if (cell_linear(xyz) != n) {
    atomicAdd(errors, 1u);
    return;
  }
  const AxisOffsets o = make_axis_offsets(xyz);
  for (uint q = 0u; q < kcfg::q_mom; ++q) {
    const uint ex = (uint)(((int)xyz.x + vel_cx(q) + (int)kcfg::nx) % (int)kcfg::nx);
    const uint ey = (uint)(((int)xyz.y + vel_cy(q) + (int)kcfg::ny) % (int)kcfg::ny);
    const uint ez = (uint)(((int)xyz.z + vel_cz(q) + (int)kcfg::nz) % (int)kcfg::nz);
    const cell_idx_t expected = (cell_idx_t)ex + ((cell_idx_t)ey + (cell_idx_t)ez * (cell_idx_t)kcfg::ny) * (cell_idx_t)kcfg::nx;
    if (neighbor_cell(o, q) != expected) {
      atomicAdd(errors, 1u);
      return;
    }
  }
}

extern "C" __global__ void test_damping_tables(uint *errors) {
  const uint d = (uint)cuda_global_thread_id(0);
#ifdef BUFFER_NUDGING
  if (d <= kcfg::buffer_cells) {
    const float xi = 1.0f - (float)d / (float)kcfg::buffer_cells;
    float w = sinf(1.5707963267948966f * xi);
    w *= w;
    if (__float_as_uint(g_buffer_rate[d]) != __float_as_uint(w * kcfg::buffer_inv_tau))
      atomicAdd(errors, 1u);
  }
#endif
#ifdef TOP_SPONGE
  if (d < kcfg::sponge_cells) {
    const float xi = 1.0f - (float)d / (float)(kcfg::sponge_cells - 1u);
    float s = sinf(1.5707963267948966f * xi);
    if (__float_as_uint(g_sponge_rate[d]) != __float_as_uint(kcfg::sponge_inv_tau * s * s))
      atomicAdd(errors, 1u);
  }
#endif
}
)CUDA";

int run_for_lattice(const Momentum_Lattice lat, const gpu::DeviceProperties& props) {
	if (!set_momentum_lattice(lat) || !set_ddf_storage(DDF_Storage::FP16C) ||
	    !set_temperature_lattice(Temperature_Lattice::Q19)) {
		std::cerr << "FAIL: unable to select lattice\n";
		return 1;
	}
	const culbm::solver::Decomposition grid = culbm::solver::Decomposition::fitted(7u, 11u, 13u, 1u, 1u, 1u);
	const culbm::solver::Placement placement = grid.place(0u);
	const culbm::solver::FeatureSet features = culbm::solver::FeatureSet::current();
	culbm::solver::Transport transport;
	const string src = culbm::solver::kernel_source(placement, grid, features, transport) + "\n" + kTestKernels;
	gpu::Device dev(props, src);

	// 1. Decode all 65,536 FP16C codes
	{
		gpu::MirroredBuffer<uint> out(dev, 65536u);
		gpu::Launch(dev, "test_decode_all", 65536u).push(out).submit_and_wait();
		out.download();
		uint diffs = 0u;
		for (uint x = 0u; x < 65536u; ++x) {
			const std::uint32_t expected = reference_decode_bits(static_cast<std::uint16_t>(x));
			if (out[x] != expected) {
				if (diffs < 5u)
					std::cerr << "FAIL decode x=0x" << std::hex << x << " got=0x" << out[x]
					          << " exp=0x" << expected << std::dec << '\n';
				++diffs;
			}
		}
		if (diffs != 0u) return 1;
	}

	// 2. Encode test batch: subnormals, transitions, boundary ties, signed zeros, and 1M random floats
	{
		std::vector<uint> inputs;
		inputs.reserve(2500000u);
		// Exhaustive stride across e=99..114 (subnormal and transition region) + edge mantissas for all e=0..128
		for (uint sign = 0u; sign <= 1u; ++sign) {
			for (uint e = 0u; e <= 128u; ++e) {
				const uint step = (e >= 99u && e <= 114u) ? 31u : 131071u;
				for (uint m = 0u; m < 0x800000u; m += step)
					inputs.push_back((sign << 31) | (e << 23) | m);
				// Boundary/tie mantissas around 0x800 rounding point
				for (const uint m : {0x000000u, 0x0007FFu, 0x000800u, 0x000801u, 0x000FFFu,
				                     0x7FF000u, 0x7FF7FFu, 0x7FF800u, 0x7FF801u, 0x7FFFFFu})
					inputs.push_back((sign << 31) | (e << 23) | m);
			}
		}
		std::mt19937 rng(20261123u);
		std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
		for (uint i = 0u; i < 500000u; ++i) inputs.push_back(f2u(dist(rng)));

		const uint count = static_cast<uint>(inputs.size());
		gpu::MirroredBuffer<uint> d_in(dev, count);
		gpu::MirroredBuffer<ushort> d_out(dev, count);
		std::memcpy(d_in.data(), inputs.data(), (size_t)count * sizeof(uint));
		d_in.upload();
		gpu::Launch(dev, "test_encode_batch", count).push(d_in, d_out, count).submit_and_wait();
		d_out.download();

		uint diffs = 0u;
		for (uint i = 0u; i < count; ++i) {
			const std::uint16_t expected = reference_encode(u2f(inputs[i]));
			if (d_out[i] != expected) {
				if (diffs < 5u)
					std::cerr << "FAIL encode u=0x" << std::hex << inputs[i] << " got=0x" << d_out[i]
					          << " exp=0x" << expected << std::dec << '\n';
				++diffs;
			}
		}
		if (diffs != 0u) return 1;
	}

	// 3. Neighbor wrap-around across all cells and all directions (D3Q19 / D3Q27)
	{
		gpu::MirroredBuffer<uint> errs(dev, 1u);
		errs[0] = 0u;
		errs.upload();
		gpu::Launch(dev, "test_neighbors_grid", placement.cells()).push(errs).submit_and_wait();
		errs.download();
		if (errs[0] != 0u) {
			std::cerr << "FAIL neighbor_cell wrap-around errors=" << errs[0] << '\n';
			return 1;
		}
	}

	// 4. Damping tables bit-exactness
	{
		gpu::MirroredBuffer<uint> errs(dev, 1u);
		errs[0] = 0u;
		errs.upload();
		const ulong n_tbl = static_cast<ulong>(std::max(buffer_n_cells + 1, sponge_n_cells));
		gpu::Launch(dev, "culbm_init_damping_tables", n_tbl).submit();
		gpu::Launch(dev, "test_damping_tables", n_tbl).push(errs).submit_and_wait();
		errs.download();
		if (errs[0] != 0u) {
			std::cerr << "FAIL damping table errors=" << errs[0] << '\n';
			return 1;
		}
	}

	return 0;
}

} // namespace

int main() {
	try {
		if (cuInit(0u) != CUDA_SUCCESS) {
			std::cout << "SKIP: CUDA driver unavailable.\n";
			return 77;
		}
		int count = 0;
		if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0) {
			std::cout << "SKIP: no CUDA device available.\n";
			return 77;
		}
		const auto gpus = gpu::choose_devices(1u);
		if (run_for_lattice(Momentum_Lattice::Q19, gpus.front()) != 0) return 1;
		if (run_for_lattice(Momentum_Lattice::Q27, gpus.front()) != 0) return 1;
		std::cout << "Device codec (65536 decode + encode sweep), neighbor offsets and damping tables verified bit-exact.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "DEVICE_CODEC_TEST_ERROR: " << e.what() << '\n';
		return 2;
	}
}
