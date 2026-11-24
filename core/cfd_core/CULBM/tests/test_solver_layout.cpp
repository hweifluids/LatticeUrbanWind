// Host-side layout of the solver (no GPU needed): the decomposition into
// subdomains and the ownership of every global cell, the halo faces and their
// routing between neighbours, the cell field table with its VRAM plan, and
// the order of the stages of a time step.
#include "solver/decomposition.hpp"
#include "solver/field_table.hpp"
#include "solver/step_pipeline.hpp"

#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace culbm::solver;

namespace {

int failures = 0;

void expect(const bool condition, const std::string& message) {
	if(condition) return;
	++failures;
	std::cerr << "FAIL: " << message << '\n';
}

std::string dims(const unsigned a, const unsigned b, const unsigned c) {
	return std::to_string(a)+"x"+std::to_string(b)+"x"+std::to_string(c);
}

void check_decomposition(const unsigned nx, const unsigned ny, const unsigned nz,
	const unsigned dx, const unsigned dy, const unsigned dz) {
	const std::string name = dims(nx, ny, nz)+" split "+dims(dx, dy, dz);
	const Decomposition grid(nx, ny, nz, dx, dy, dz);
	expect(grid.count()==dx*dy*dz && grid.cells()==(index_t)nx*ny*nz, name+": counts");
	// Every subdomain: extent = interior + one halo layer per side of a split
	// axis, origin = first interior cell minus that layer.
	for(unsigned id=0u; id<grid.count(); ++id) {
		const Placement p = grid.place(id);
		expect(p.id==id && p.cell[0]+(p.cell[1]+p.cell[2]*dy)*dx==id, name+": subdomain numbering");
		for(unsigned a=0u; a<3u; ++a) {
			const unsigned halo = grid.split(a)>1u ? 1u : 0u;
			expect(p.halo[a]==halo && p.extent[a]==grid.inner(a)+2u*halo, name+": extent along axis "+std::to_string(a));
			expect(p.origin[a]==(int)(p.cell[a]*grid.inner(a))-(int)halo, name+": origin along axis "+std::to_string(a));
		}
	}
	// Every global cell belongs to exactly one interior cell of one subdomain,
	// and that cell sits at the same global coordinate.
	std::set<std::pair<unsigned, index_t>> seen;
	for(unsigned z=0u; z<nz; ++z) for(unsigned y=0u; y<ny; ++y) for(unsigned x=0u; x<nx; ++x) {
		const Decomposition::Owner owner = grid.owner(x, y, z);
		const Placement p = grid.place(owner.id);
		const index_t lx = owner.local%p.extent[0];
		const index_t ly = (owner.local/p.extent[0])%p.extent[1];
		const index_t lz = owner.local/((index_t)p.extent[0]*p.extent[1]);
		const bool interior = lx>=p.halo[0] && lx<p.halo[0]+grid.inner(0u) && ly>=p.halo[1] &&
			ly<p.halo[1]+grid.inner(1u) && lz>=p.halo[2] && lz<p.halo[2]+grid.inner(2u);
		expect(interior, name+": cell owned by a halo layer");
		expect((int)lx+p.origin[0]==(int)x && (int)ly+p.origin[1]==(int)y && (int)lz+p.origin[2]==(int)z,
			name+": owner coordinate");
		expect(owner.local==p.local_index((unsigned)lx, (unsigned)ly, (unsigned)lz), name+": local index");
		expect(seen.insert({owner.id, owner.local}).second, name+": two cells share an owner cell");
	}
	// Halo routing is periodic; the face buffers hold the largest split face.
	index_t capacity = 0u;
	for(unsigned a=0u; a<3u; ++a) {
		for(unsigned id=0u; id<grid.count(); ++id) {
			const Placement p = grid.place(id);
			const unsigned up = grid.upper_neighbour(id, a), down = grid.lower_neighbour(id, a);
			const Placement u = grid.place(up), w = grid.place(down);
			expect(grid.lower_neighbour(up, a)==id && grid.upper_neighbour(down, a)==id, name+": neighbour symmetry");
			expect(u.cell[a]==(p.cell[a]+1u)%grid.split(a) && w.cell[a]==(p.cell[a]+grid.split(a)-1u)%grid.split(a),
				name+": periodic neighbours");
			for(unsigned b=0u; b<3u; ++b)
				if(b!=a) expect(u.cell[b]==p.cell[b] && w.cell[b]==p.cell[b], name+": neighbour off axis");
			expect(p.face_cells(a)==(index_t)p.extent[(a+1u)%3u]*p.extent[(a+2u)%3u], name+": face cells");
		}
		if(grid.split(a)>1u) capacity = std::max(capacity, grid.place(0u).face_cells(a));
	}
	expect(grid.halo_capacity()==capacity, name+": halo capacity");
}

void check_decompositions() {
	check_decomposition(8u, 6u, 4u, 1u, 1u, 1u);
	check_decomposition(8u, 6u, 4u, 2u, 1u, 1u);
	check_decomposition(8u, 6u, 4u, 2u, 3u, 1u);
	check_decomposition(6u, 4u, 8u, 1u, 2u, 4u);
	check_decomposition(6u, 6u, 6u, 3u, 3u, 3u);
	const Decomposition fitted = Decomposition::fitted(101u, 64u, 31u, 2u, 1u, 3u);
	expect(fitted.size(0u)==100u && fitted.size(1u)==64u && fitted.size(2u)==30u, "fitted lattice");
	expect(Decomposition(8u, 6u, 4u, 1u, 1u, 1u).halo_capacity()==0u, "an unsplit lattice has no halo buffers");
}

void check_field_table() {
	LatticeStorage q19; // D3Q19, FP16C, D3Q7
	const CellCost c19 = cell_cost(q19);
	expect(c19.device==85u && c19.host==21u && c19.traffic==143u, "D3Q19/FP16C/D3Q7 per-cell cost");
	const std::vector<CellField> table = cell_fields(q19);
	const FieldId order[] = {FieldId::ddf, FieldId::density, FieldId::velocity, FieldId::flags,
		FieldId::force, FieldId::thermal_ddf, FieldId::temperature};
	expect(table.size()==7u, "D3Q19 field count");
	for(std::size_t i=0u; i<table.size() && i<7u; ++i) expect(table[i].id==order[i], "field order");
	for(const CellField& f : table) {
		const bool device_only = f.id==FieldId::ddf || f.id==FieldId::thermal_ddf;
		expect(device_only ? f.host==HostCopy::none : f.id==FieldId::force ? f.host==HostCopy::on_demand : f.host==HostCopy::resident,
			std::string("host copy of ")+f.name);
	}
	expect(table[0].word_bytes==2u && table[0].components==19u, "FP16C DDF words");
	expect(halo_cell_bytes(q19)==17u, "D3Q19/FP16C halo bytes: the macro exchange is the largest");

	LatticeStorage q27;
	q27.velocity_set = 27u;
	q27.ddf_bytes = 4u;
	q27.face_ddfs = 9u;
	q27.temperature_set = 19u;
	q27.temperature_face_ddfs = 5u;
	const CellCost c27 = cell_cost(q27);
	expect(c27.device==217u && c27.host==21u && c27.traffic==419u, "D3Q27/FP32/D3Q19 per-cell cost");
	expect(cell_fields(q27)[0].components==54u, "FP32 DDFs take two 16-bit words");
	expect(halo_cell_bytes(q27)==56u, "D3Q27/FP32/D3Q19 halo bytes: coupled DDF exchange");

	LatticeStorage ibm;
	ibm.surface_ibm = true;
	const CellCost ci = cell_cost(ibm);
	expect(ci.device==98u && ci.traffic==144u && cell_fields(ibm).size()==9u, "SurfaceIBM fields");

	// One subdomain: every field is one allocation of N cells.
	const DomainFootprint one = domain_footprint(q19, 256u, 192u, 128u, 1u, 1u, 1u);
	expect(one.valid && one.bytes==256ull*192u*128u*85u && one.mib==510u, "256x192x128 footprint");
	// Two subdomains along x: halo layers on x and four face buffers.
	const DomainFootprint two = domain_footprint(q19, 384u, 256u, 128u, 2u, 1u, 1u);
	const std::uint64_t cells = 194ull*256u*128u;
	const std::uint64_t faces = 4ull*256u*128u*17u;
	expect(two.valid && two.bytes==cells*85u+faces, "two-subdomain footprint bytes");
}

void check_pipeline() {
	const auto names = [](const StepPipeline& p) {
		std::string text;
		for(const Stage s : p.stages()) text += std::string(text.empty() ? "" : " ")+stage_name(s);
		return text;
	};
	const StepPipeline single(PipelineConfig{1u, false, false});
	expect(names(single)=="thermal_outlet stream_collide device_barrier advance_clock",
		"single GPU stages: "+names(single));
	const StepPipeline local(PipelineConfig{2u, true, true});
	expect(names(local)=="thermal_outlet immersed_forcing stream_collide population_halo advance_clock",
		"two local GPUs with SurfaceIBM: "+names(local));
	const StepPipeline rank(PipelineConfig{1u, true, false});
	expect(names(rank)=="thermal_outlet stream_collide population_halo device_barrier advance_clock",
		"one subdomain of an MPI run: "+names(rank));
}

} // namespace

int main() {
	check_decompositions();
	check_field_table();
	check_pipeline();
	if(failures==0) std::cout << "Solver layout checks passed.\n";
	return failures==0 ? 0 : 1;
}
