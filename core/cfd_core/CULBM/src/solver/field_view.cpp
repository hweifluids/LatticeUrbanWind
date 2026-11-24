#include "field_view.hpp"
#include "../core/lattice_units.hpp"

#include <fstream>
#include <thread>

float3 vtk_origin_shift = float3(0.0f);

string default_filename(const string& path, const string& name, const string& extension, const ulong t) {
	// The step number is zero-padded to nine digits; longer numbers keep their
	// last nine digits.
	string step = to_string(t);
	step = step.length()<9u ? string(9u-step.length(), '0')+step : step.substr(step.length()-9u);
	const string folder = path.empty() ? get_exe_path()+"export/" : path;
	return folder+create_file_extension((name.empty() ? string("file") : name)+"-"+step, extension);
}

string default_filename(const string& name, const string& extension, const ulong t) {
	return default_filename(string(), name, extension, t);
}

namespace culbm::solver::detail {

RawVtkScale raw_vtk_scale(const string& field, const bool si_units) {
	RawVtkScale scale;
	if(!si_units) return scale;
	if(field=="rho") scale.factor = units.si_rho(1.0f);
	else if(field=="u") scale.factor = units.si_u(1.0f);
	else if(field=="F") scale.factor = units.si_F(1.0f);
	else if(field=="T") scale.temperature = true;
	return scale;
}

float si_temperature(const float lattice_temperature) {
	return units.si_T(lattice_temperature);
}

void write_raw_vtk(const string& path, const Decomposition& grid, const unsigned nz_written, const bool si_units,
	const char* const type, const unsigned components, const std::size_t value_bytes,
	const std::function<void(index_t first, index_t count, unsigned char* out)>& encode) {
	if(type==nullptr) fatal_error("The raw VTK writer does not support this field type.");
	const string target = create_file_extension(path, ".vtk");
	const float spacing = si_units ? units.si_x(1.0f) : 1.0f;
	// Cell centres of a box centred on the origin; the full solver height sets
	// the centre even when only the lowest layers are written.
	float3 origin = spacing*float3(0.5f-0.5f*(float)grid.size(0u), 0.5f-0.5f*(float)grid.size(1u), 0.5f-0.5f*(float)grid.size(2u));
	if(si_units) origin += vtk_origin_shift;
	const index_t points = (index_t)grid.size(0u)*grid.size(1u)*nz_written;
	const string title = target.substr(target.rfind('/')+1u);
	string header = "# vtk DataFile Version 3.0\nCULBM "+title+"\nBINARY\nDATASET STRUCTURED_POINTS\n";
	header += "DIMENSIONS "+to_string(grid.size(0u))+" "+to_string(grid.size(1u))+" "+to_string(nz_written)+"\n";
	header += "ORIGIN "+to_string(origin.x)+" "+to_string(origin.y)+" "+to_string(origin.z)+"\n";
	header += "SPACING "+to_string(spacing)+" "+to_string(spacing)+" "+to_string(spacing)+"\n";
	header += "POINT_DATA "+to_string(points)+"\n";
	header += "SCALARS data "+string(type)+" "+to_string(components)+"\nLOOKUP_TABLE default\n";
	// Values are converted in slices of 4 MiB per hardware thread.
	const std::size_t cell_bytes = (std::size_t)components*value_bytes;
	const index_t slice = std::max<index_t>(1u, (index_t)4u*1048576u*std::max(1u, std::thread::hardware_concurrency())/cell_bytes);
	std::vector<unsigned char> buffer((std::size_t)(std::min(slice, std::max<index_t>(points, 1u))*cell_bytes));
	create_folder(target);
	std::ofstream stream(target, std::ios::binary);
	stream.write(header.data(), (std::streamsize)header.size());
	for(index_t first=0u; first<points; first+=slice) {
		const index_t count = std::min(slice, points-first);
		encode(first, count, buffer.data());
		stream.write(reinterpret_cast<const char*>(buffer.data()), (std::streamsize)(count*cell_bytes));
	}
}

void report_saved(const string& filename) {
	const StatusPause pause;
	log_info("File \""+create_file_extension(filename, ".vtk")+"\" saved.");
}

} // namespace culbm::solver::detail
