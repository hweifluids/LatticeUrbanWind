#include "luw/command_registry.hpp"

#include "luw/accelerator_report_command.hpp"
#include "luw/cleanluw_command.hpp"
#include "luw/buildingscale_command.hpp"
#include "luw/detect_opencl_command.hpp"
#include "luw/inspect_commands.hpp"
#include "luw/luwbc_command.hpp"
#include "luw/luwcut_command.hpp"
#include "luw/luwvox_command.hpp"
#include "luw/makeluw_command.hpp"
#include "luw/runluw_command.hpp"
#include "luw/shptester_command.hpp"
#include "luw/validate_commands.hpp"
#include "luw/visdem_command.hpp"
#include "luw/vtk_post_commands.hpp"

#if __has_include("commands/deck_command.hpp")
#include "commands/deck_command.hpp"
#define LUW_HAS_DECK_COMMAND 1
#else
#define LUW_HAS_DECK_COMMAND 0
#endif

#include <ostream>
#include <string>
#include <utility>

namespace luw::cli {

namespace {

CommandHandler unavailable(std::string name) {
    return [command = std::move(name)](const CommandArguments&,
                                       std::ostream&,
                                       std::ostream& error) {
        error << "ERROR: Native command '" << command
              << "' is not available in this build.\n";
        return 1;
    };
}

} // namespace

CommandRegistry makeCommandRegistry() {
    CommandRegistry registry;

#if LUW_HAS_DECK_COMMAND
    registry.add({"deck", "Inspect or edit a LUW deck", "luw deck <operation> [arguments]",
                  DeckArgumentPolicy::None, runDeckCommand});
#else
    registry.add({"deck", "Inspect or edit a LUW deck", "luw deck <operation> [arguments]",
                  DeckArgumentPolicy::None, unavailable("deck")});
#endif

    registry.add({"accelerator_report", "Print native CUDA/OpenCL diagnostics as JSON",
                  "accelerator_report [--json] [--startup-report PATH] [--prepare-cuda-runtime]",
                  DeckArgumentPolicy::None, run_accelerator_report});
    registry.add({"detect_opencl", "Report native OpenCL platforms and devices",
                  "detect_opencl",
                  DeckArgumentPolicy::None, run_detect_opencl});
    registry.add({"cdfinspect", "Inspect wind-climate inputs", "cdfinspect <path-to-conf.luw>",
                  DeckArgumentPolicy::AutoWhenMissing, run_cdfinspect});
    registry.add({"shpinspect", "Inspect building footprints", "shpinspect <path-to-conf.luw>",
                  DeckArgumentPolicy::AutoWhenMissing, run_shpinspect});
    registry.add({"shptester", "Diagnose degenerate polygon features",
                  "shptester <path_to_shp>",
                  DeckArgumentPolicy::None, run_shptester});
    registry.add({"visdem", "Visualize a DEM GeoTIFF",
                  "visdem [path_to_tif]",
                  DeckArgumentPolicy::None, run_visdem});
    registry.add({"luwval", "Validate a case before CFD", "luwval <path-to-conf.luw>",
                  DeckArgumentPolicy::Required, run_luwval, true});
    registry.add({"dgprepare", "Prepare a diagnostic-geometry case",
                  "dgprepare <path-to-conf.luwdg-or-conf.luwpf>",
                  DeckArgumentPolicy::None, run_dgprepare, true});
    registry.add({"makeluw", "Run the six-stage LUW preparation workflow", "makeluw <path-to-conf.luw>",
                  DeckArgumentPolicy::Required, run_makeluw, true});
    registry.add({"runluw", "Run the deck-selected CFD solver",
                  "runluw <path-to-deck> [solver-arguments]",
                  DeckArgumentPolicy::None, run_runluw});

    registry.add({"luwbc", "Generate boundary conditions",
                  "luwbc <path-to-conf.luw> [--elevation-scale N] [--write-vtk]",
                  DeckArgumentPolicy::Required, run_luwbc, true});
    registry.add({"luwcut", "Crop the geometry domain",
                  "luwcut <path-to-conf.luw> [--preview] [--validate]",
                  DeckArgumentPolicy::Required, run_luwcut, true});
    registry.add({"luwvox", "Generate the voxel domain",
                  "luwvox <path-to-conf.luw> [terrain options]",
                  DeckArgumentPolicy::None, run_luwvox, true});
    registry.add({"visluw", "Visualize LUW results",
                  "visluw <path-to-conf.luw> [--lon-min N --lon-max N --lat-min N --lat-max N] [--layers N] [--export-nc]",
                  DeckArgumentPolicy::None, run_visluw});
    registry.add({"transluw", "Translate and merge raw VTK results",
                  "transluw <vtk-or-conf> [--tailname NAME] [--absonly]",
                  DeckArgumentPolicy::AutoWhenMissing, run_transluw});
    registry.add({"vtk2nc", "Convert VTK results to NetCDF",
                  "vtk2nc <path-to-conf.luw> [--output PATH]",
                  DeckArgumentPolicy::Required, run_vtk2nc});
    registry.add({"vtk2nc_new", "Convert VTK results to Cartesian lon/lat NetCDF",
                  "vtk2nc_new <path-to-conf.luw>",
                  DeckArgumentPolicy::None, run_vtk2nc_new});
    registry.add({"cleanluw", "Remove generated case products", "cleanluw <path-to-conf.luw>",
                  DeckArgumentPolicy::Required, run_cleanluw});
    registry.add({"season_average", "Generate seasonal averages",
                  "season_average <path-to-conf.luw> [--vtk-dir DIR] [--output-spacing M [M M]] [--skip-figures]",
                  DeckArgumentPolicy::Required, run_season_average});
    registry.add({"les_spectra", "Analyze LES spectra",
                  "les_spectra [options] [vtk_path]",
                  DeckArgumentPolicy::None, run_les_spectra});
    registry.add({"tke_viz", "Visualize turbulent kinetic energy",
                  "tke_viz [options] [vtk_file]",
                  DeckArgumentPolicy::None, run_tke_viz});
    registry.add({"cut_vis", "Crop visualization products",
                  "cut_vis [options] [vtk_file]",
                  DeckArgumentPolicy::None, run_cut_vis});
    registry.add({"batch_tke_geo_viz", "Batch crop and visualize georeferenced VTK files",
                  "batch_tke_geo_viz <config_path> [--crop-z-max N]",
                  DeckArgumentPolicy::None, run_batch_tke_geo_viz});
    registry.add({"buildingscale", "Analyze building scales",
                  "buildingscale <shp_path> [--out OUT]",
                  DeckArgumentPolicy::None, run_buildingscale});
    registry.add({"vtk_avg_to_utm_asl_nc", "Convert averaged VTK fields",
                  "vtk_avg_to_utm_asl_nc [--case-dir DIR | --root DIR]",
                  DeckArgumentPolicy::None, run_vtk_avg_to_utm_asl_nc});

    return registry;
}

} // namespace luw::cli
