#pragma once

// Physical outputs of a case (culbm_output_*): the final fields on the grid
// above the output datum as a global structured VTK, VTKHDF or NetCDF file
// and as terrain-relative (AGL) point sets in VTK or NetCDF.

#include "../physical_output.hpp"
#include "../core/core.hpp"

namespace culbm::cases {
struct CaseSettings;
}

namespace culbm::run {

// Writes every requested format of one field view and reports each file;
// names are <name_prefix>_global / <name_prefix>_agl_points at `step`.
void write_physical_outputs(const cases::CaseSettings& deck, const culbm::PhysicalGridView& view,
                            const culbm::PhysicalAglSurfaceView& surface,
                            const std::string& results_vtk_dir, const std::string& results_netcdf_dir,
                            const std::string& name_prefix, ulong step);

} // namespace culbm::run
