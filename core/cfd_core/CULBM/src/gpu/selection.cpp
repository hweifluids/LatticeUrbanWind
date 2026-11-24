#include "selection.hpp"

#include "../cuda_topology.hpp"

#include <array>
#include <cstdlib>
#include <cstring>

namespace culbm::gpu {
namespace {

// Strict "faster" order of the mixed-model fallback: peak FP32 rate, then
// installed memory, then the lower driver ordinal.
bool preferred(const DeviceProperties& a, const DeviceProperties& b) {
	if(a.fp32_tflops!=b.fp32_tflops) return a.fp32_tflops>b.fp32_tflops;
	if(a.memory_mb!=b.memory_mb) return a.memory_mb>b.memory_mb;
	return a.ordinal<b.ordinal;
}

bool keep_given_order() {
	const char* const p2p = std::getenv("CULBM_DISABLE_P2P");
	if(p2p!=nullptr && std::strcmp(p2p, "0")!=0) return true;
	const char* const value = std::getenv("CULBM_P2P_TOPOLOGY");
	if(value==nullptr) return false;
	for(const char* off : {"0", "off", "OFF", "false", "FALSE"}) {
		if(std::strcmp(value, off)==0) return true;
	}
	return false;
}

bool mutual_peers(const DeviceProperties& a, const DeviceProperties& b) {
	int ab = 0, ba = 0;
	return cuDeviceCanAccessPeer(&ab, a.handle, b.handle)==CUDA_SUCCESS &&
		cuDeviceCanAccessPeer(&ba, b.handle, a.handle)==CUDA_SUCCESS && ab!=0 && ba!=0;
}

} // namespace

std::vector<DeviceProperties> choose_devices(const uint domains) {
	const std::vector<DeviceProperties> visible = list_devices();
	if(domains>(uint)visible.size()) {
		for(const DeviceProperties& p : visible) {
			log_info("Visible GPU "+to_string(p.ordinal)+": "+p.name+", "+to_string(p.memory_mb)+" MiB.");
		}
		fatal_error("The decomposition needs "+to_string(domains)+" GPUs, but only "+
			to_string((uint)visible.size())+" are visible. Reduce Dx*Dy*Dz to at most "+
			to_string((uint)visible.size())+".");
	}
	// Models in order of first appearance.
	std::vector<std::vector<DeviceProperties>> models;
	for(const DeviceProperties& p : visible) {
		bool known = false;
		for(std::vector<DeviceProperties>& model : models) {
			if(model.front().name==p.name) {
				model.push_back(p);
				known = true;
			}
		}
		if(!known) models.push_back({p});
	}
	const std::vector<DeviceProperties>* fastest = nullptr;
	for(const std::vector<DeviceProperties>& model : models) {
		if((uint)model.size()<domains) continue;
		if(fastest==nullptr || model.front().fp32_tflops>fastest->front().fp32_tflops) fastest = &model;
	}
	if(fastest!=nullptr) return std::vector<DeviceProperties>(fastest->begin(), fastest->begin()+domains);

	log_warning("No GPU model is installed "+to_string(domains)+" times; combining the fastest distinct GPUs.");
	std::vector<DeviceProperties> chosen;
	std::vector<bool> taken(visible.size(), false);
	for(uint d=0u; d<domains; ++d) {
		std::size_t best = visible.size();
		for(std::size_t i=0u; i<visible.size(); ++i) {
			if(!taken[i] && (best==visible.size() || preferred(visible[i], visible[best]))) best = i;
		}
		taken[best] = true;
		chosen.push_back(visible[best]);
	}
	return chosen;
}

DeviceProperties rank_device() {
	const std::vector<DeviceProperties> visible = list_devices();
	return visible[culbm::mpi_cuda_device_ordinal((std::uint32_t)visible.size())];
}

void arrange_for_peer_halos(std::vector<DeviceProperties>& devices,
	const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz) {
	const uint n = (uint)devices.size();
	if(n<2u || keep_given_order()) return;
	std::vector<unsigned char> peers((std::size_t)n*n, 0u);
	for(uint a=0u; a<n; ++a) {
		for(uint b=0u; b<n; ++b) {
			peers[(std::size_t)a*n+b] = a==b || mutual_peers(devices[a], devices[b]) ? 1u : 0u;
		}
	}
	std::vector<std::uint32_t> placement(n);
	for(uint d=0u; d<n; ++d) placement[d] = d;
	// Halo slab sizes of the three split directions.
	const std::array<std::uint64_t, 3u> faces = {
		(std::uint64_t)(Ny/Dy)*(std::uint64_t)(Nz/Dz),
		(std::uint64_t)(Nx/Dx)*(std::uint64_t)(Nz/Dz),
		(std::uint64_t)(Nx/Dx)*(std::uint64_t)(Ny/Dy)
	};
	const std::uint64_t before = culbm::peer_assignment_score(peers, n, placement, Dx, Dy, Dz, faces);
	std::uint64_t after = before;
	if(!culbm::optimize_peer_device_assignment(peers, n, placement, Dx, Dy, Dz, faces, &after)) return;
	std::vector<DeviceProperties> arranged;
	string mapping;
	for(uint d=0u; d<n; ++d) {
		arranged.push_back(devices[placement[d]]);
		mapping += (d==0u ? "" : ", ")+string("domain ")+to_string(d)+" -> GPU "+to_string(arranged.back().ordinal);
	}
	devices.swap(arranged);
	log_info("GPU order changed for direct peer halos ("+mapping+"); peer-linked halo area "+
		to_string(before)+" -> "+to_string(after)+".");
}

} // namespace culbm::gpu
