#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace luw::cli {

int run_vtk2nc(const std::vector<std::string>& arguments,
               std::ostream& output,
               std::ostream& error);

int run_vtk2nc_new(const std::vector<std::string>& arguments,
                   std::ostream& output,
                   std::ostream& error);

int run_visluw(const std::vector<std::string>& arguments,
               std::ostream& output,
               std::ostream& error);

int run_transluw(const std::vector<std::string>& arguments,
                 std::ostream& output,
                 std::ostream& error);

// tke_viz.py and cut_vis.py in the legacy distribution are byte-for-byte
// aliases.  Both public commands intentionally share this native handler.
int run_tke_viz(const std::vector<std::string>& arguments,
                std::ostream& output,
                std::ostream& error);

int run_cut_vis(const std::vector<std::string>& arguments,
                std::ostream& output,
                std::ostream& error);

int run_batch_tke_geo_viz(const std::vector<std::string>& arguments,
                          std::ostream& output,
                          std::ostream& error);

int run_season_average(const std::vector<std::string>& arguments,
                       std::ostream& output,
                       std::ostream& error);

int run_les_spectra(const std::vector<std::string>& arguments,
                    std::ostream& output,
                    std::ostream& error);

int run_vtk_avg_to_utm_asl_nc(const std::vector<std::string>& arguments,
                              std::ostream& output,
                              std::ostream& error);

} // namespace luw::cli
