#pragma once
#include "setup.hpp"
#include <functional>
#include <string>

void apply_flux_correction(LBM& lbm,
                           const std::string& downstream_bc,
                           const std::function<float3(const float3&)>& inlet_eval,
                           bool show_report,
                           double* avg_delta_mps,
                           double* net_before,
                           double* net_after);
