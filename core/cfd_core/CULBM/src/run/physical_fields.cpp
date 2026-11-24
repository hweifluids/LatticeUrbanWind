#include "physical_fields.hpp"
#include "../case/report.hpp"
#include "../case/settings.hpp"
#include "../solver/field_view.hpp"

namespace culbm::run {

using cases::print_kv_row;

void write_physical_outputs(const cases::CaseSettings& deck, const culbm::PhysicalGridView& view,
                            const culbm::PhysicalAglSurfaceView& surface,
                            const std::string& results_vtk_dir, const std::string& results_netcdf_dir,
                            const std::string& name_prefix, const ulong step) {
    const std::string global_name = name_prefix + "_global";
    const std::string agl_name = name_prefix + "_agl_points";
    std::string output_error;
    if (deck.culbm_output_global_vtk) {
        const std::string path = default_filename(results_vtk_dir, global_name, ".vtk", step);
        if (culbm::write_global_structured_vtk(path, view, output_error)) {
            print_kv_row("Physical VTK", path + " saved");
        }
        else {
            println("| ERROR: physical global VTK export failed: " + output_error + " |");
        }
    }
    if (deck.culbm_output_global_vtkhdf) {
        const std::string path = default_filename(results_vtk_dir, global_name, ".vtkhdf", step);
        output_error.clear();
        if (culbm::write_global_structured_vtkhdf(path, view, output_error)) {
            print_kv_row("Physical VTKHDF", path + " saved");
        }
        else {
            println("| ERROR: physical global VTKHDF export failed: " + output_error + " |");
        }
    }
    if (deck.culbm_output_global_netcdf) {
        const std::string path = default_filename(results_netcdf_dir, global_name, ".nc", step);
        output_error.clear();
        if (culbm::write_global_structured_netcdf(path, view, output_error)) {
            print_kv_row("Physical NetCDF", path + " saved");
        }
        else {
            println("| ERROR: physical global NetCDF export failed: " + output_error + " |");
        }
    }
    if (deck.culbm_output_agl_vtk) {
        const std::string path = default_filename(results_vtk_dir, agl_name, ".vtk", step);
        output_error.clear();
        if (culbm::write_agl_points_vtk(path, view, surface, output_error)) {
            print_kv_row("AGL VTK", path + " saved");
        }
        else {
            println("| ERROR: AGL VTK export failed: " + output_error + " |");
        }
    }
    if (deck.culbm_output_agl_netcdf) {
        const std::string path = default_filename(results_netcdf_dir, agl_name, ".nc", step);
        output_error.clear();
        if (culbm::write_agl_points_netcdf(path, view, surface, output_error)) {
            print_kv_row("AGL NetCDF", path + " saved");
        }
        else {
            println("| ERROR: AGL NetCDF export failed: " + output_error + " |");
        }
    }
}

} // namespace culbm::run
