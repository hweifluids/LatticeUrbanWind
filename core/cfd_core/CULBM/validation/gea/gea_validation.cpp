#include "lbm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// The production executable defines these controls in setup.cpp.  This
// validation runner intentionally bypasses the urban-wind deck frontend.
float coriolis_Omegax_lbmu = 0.0f;
float coriolis_Omegay_lbmu = 0.0f;
float coriolis_Omegaz_lbmu = 0.0f;
bool buffer_nudging_active = false;
int buffer_n_cells = 1;
float buffer_inv_tau_lbmu = 0.0f;
int buffer_nudge_vertical = 0;
int buffer_downstream_face_id = 0;
bool top_sponge_active = false;
int sponge_n_cells = 1;
float sponge_inv_tau_lbmu = 0.0f;
int sponge_ref_mode = 0;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kGravity = 9.81;
constexpr double kReferenceTheta = 300.0;
constexpr double kGasConstant = 287.0;
constexpr double kGroundPressure = 100000.0;
constexpr uint kFreeSlipFaces = 1u | 2u | 16u | 32u;
constexpr double kGeaSmagorinskyCsSquared = 0.453989945;
constexpr double kDryAirMolecularNu = 1.48E-5;

enum class CaseKind {
    RisingThermal,
    DensityCurrent
};

enum class ViscosityMode {
    DynamicAtGroundDensity,
    NumericAsKinematic
};

struct Config {
    CaseKind kind = CaseKind::RisingThermal;
    double spacing = 125.0;
    double dt = 0.1;
    double end_time = -1.0;
    double sample_seconds = 5.0;
    double reference_density =
        kGroundPressure/(kGasConstant*kReferenceTheta);
    double theta_scale = -1.0;
    double bubble_center_x = 0.0;
    Temperature_Lattice thermal_lattice = Temperature_Lattice::Q19;
    ViscosityMode viscosity_mode =
        ViscosityMode::DynamicAtGroundDensity;
#ifdef CULBM_GEA_SGS_DEFAULT
    bool subgrid_enabled = true;
    double subgrid_cs_squared = kGeaSmagorinskyCsSquared;
    bool subgrid_scalar_coupling = true;
    double subgrid_turbulent_prandtl = 1.0;
    double base_nu_m2_s = kDryAirMolecularNu;
    double base_alpha_m2_s = kDryAirMolecularNu;
    double minimum_momentum_tau = 0.5001;
    // Preserve the requested alpha/nu ratio when the momentum floor is used.
    // The resolved thermal tau depends on the selected scalar lattice cs^2.
    double minimum_thermal_tau = 0.5;
    bool derive_minimum_thermal_tau = true;
#else
    bool subgrid_enabled = false;
    double subgrid_cs_squared = 0.03002109145;
    bool subgrid_scalar_coupling = false;
    double subgrid_turbulent_prandtl = 1.0;
    double base_nu_m2_s = std::numeric_limits<double>::quiet_NaN();
    double base_alpha_m2_s = std::numeric_limits<double>::quiet_NaN();
    // A value of 0.5 disables flooring and preserves the published AV runner.
    double minimum_momentum_tau = 0.5;
    double minimum_thermal_tau = 0.5;
    bool derive_minimum_thermal_tau = false;
#endif
    uint ny = 4u;
    uint dx = 1u;
    uint dz = 1u;
    bool write_slices = true;
    bool write_vtk = false;
    std::filesystem::path output;
};

struct TransportResolution {
    double requested_nu_m2_s = 0.0;
    double requested_nu_lbm = 0.0;
    double requested_momentum_tau = 0.5;
    double effective_nu_m2_s = 0.0;
    double effective_nu_lbm = 0.0;
    double effective_momentum_tau = 0.5;
    bool momentum_floor_applied = false;
    double requested_alpha_m2_s = 0.0;
    double requested_alpha_lbm = 0.0;
    double requested_thermal_tau = 0.5;
    double effective_alpha_m2_s = 0.0;
    double effective_alpha_lbm = 0.0;
    double effective_thermal_tau = 0.5;
    double minimum_thermal_tau = 0.5;
    bool minimum_thermal_tau_derived = false;
    bool thermal_floor_applied = false;
};

struct CaseSpec {
    std::string name;
    double length_x = 0.0;
    double length_z = 0.0;
    double end_time = 0.0;
    double artificial_dynamic_viscosity = 0.0;
    double theta_scale = 1.0;
    std::vector<double> snapshot_times;
};

struct Metrics {
    double u_min = std::numeric_limits<double>::infinity();
    double u_max = -std::numeric_limits<double>::infinity();
    double w_min = std::numeric_limits<double>::infinity();
    double w_max = -std::numeric_limits<double>::infinity();
    double theta_min = std::numeric_limits<double>::infinity();
    double theta_max = -std::numeric_limits<double>::infinity();
    double rho_min = std::numeric_limits<double>::infinity();
    double rho_max = -std::numeric_limits<double>::infinity();
    double rho_mean = 0.0;
    double theta_mean = 0.0;
    double theta_l2 = 0.0;
    double max_speed = 0.0;
    double max_mach = 0.0;
    double max_y_temperature_spread = 0.0;
    double front_interpolated = std::numeric_limits<double>::quiet_NaN();
    double front_bracket_left = std::numeric_limits<double>::quiet_NaN();
    double front_bracket_right = std::numeric_limits<double>::quiet_NaN();
    double subgrid_mean_nu_t_lbm = 0.0;
    double subgrid_max_nu_t_lbm = 0.0;
    double subgrid_mean_rho_nu_t_lbm = 0.0;
    double subgrid_max_rho_nu_t_lbm = 0.0;
    double subgrid_mean_nu_t_m2_s = 0.0;
    double subgrid_max_nu_t_m2_s = 0.0;
    double subgrid_mean_mu_t_pa_s = 0.0;
    double subgrid_max_mu_t_pa_s = 0.0;
    std::uint64_t subgrid_count = 0ull;
    bool subgrid_statistics_valid = false;
    bool finite = true;
    bool positive_density = true;
};

CaseSpec case_spec(const Config& config) {
    if(config.kind==CaseKind::RisingThermal) {
        return {
            "rising_thermal",
            5000.0,
            10000.0,
            1020.0,
            15.0,
            config.theta_scale>0.0 ? config.theta_scale : 2.0,
            {0.0, 300.0, 600.0, 900.0, 1020.0}
        };
    }
    return {
        "density_current",
        25600.0,
        6400.0,
        900.0,
        75.0,
        config.theta_scale>0.0 ? config.theta_scale : 15.0,
        {0.0, 300.0, 600.0, 750.0, 900.0}
    };
}

std::string lowercase(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](const unsigned char c) {
            return (char)std::tolower(c);
        });
    return value;
}

double parse_double(const std::string& text, const std::string& option) {
    std::size_t consumed = 0u;
    const double value = std::stod(text, &consumed);
    if(consumed!=text.size() || !std::isfinite(value)) {
        throw std::invalid_argument(
            "Invalid numeric value for "+option+": "+text);
    }
    return value;
}

uint parse_uint(const std::string& text, const std::string& option) {
    std::size_t consumed = 0u;
    const unsigned long value = std::stoul(text, &consumed);
    if(consumed!=text.size() ||
       value>(unsigned long)std::numeric_limits<uint>::max()) {
        throw std::invalid_argument(
            "Invalid integer value for "+option+": "+text);
    }
    return (uint)value;
}

bool parse_on_off(const std::string& text, const std::string& option) {
    const std::string value = lowercase(text);
    if(value=="on" || value=="true" || value=="1") return true;
    if(value=="off" || value=="false" || value=="0") return false;
    throw std::invalid_argument(
        option+" must be on or off, got: "+text);
}

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable
        << " --case rising-thermal|density-current --h METRES"
           " --output DIR [options]\n"
           "\nOptions:\n"
           "  --dt SECONDS                 Physical time step (default 0.1)\n"
           "  --end-time SECONDS           Override benchmark final time\n"
           "  --sample-seconds SECONDS     Metrics cadence (default 5)\n"
           "  --thermal D3Q7|D3Q19         Scalar lattice (default D3Q19)\n"
           "  --viscosity dynamic|numeric  mu/rho_ground or numeric nu\n"
           "  --rho-ref KG_M3              Reference density for dynamic mode\n"
           "  --sgs on|off                 Enable Smagorinsky SGS\n"
           "  --sgs-cs VALUE               Smagorinsky Cs (mutually exclusive with Cs^2)\n"
           "  --sgs-cs-squared VALUE       Smagorinsky Cs^2 (GEA: 0.453989945)\n"
           "  --sgs-scalar on|off          Couple nu_t/Pr_t to scalar diffusion\n"
           "  --sgs-pr-t VALUE             Positive turbulent Prandtl number\n"
           "  --base-nu-m2-s VALUE         Requested base kinematic viscosity\n"
           "  --base-alpha-m2-s VALUE      Requested base scalar diffusivity\n"
           "  --min-momentum-tau VALUE     Minimum base tau (>=0.5; 0.5 disables)\n"
           "  --min-thermal-tau VALUE      Explicit base tau_T floor (0.5 disables)\n"
           "  --theta-scale K              Affine scalar perturbation scale\n"
           "  --bubble-center-x METRES     Default 0 (GEA half-domain case)\n"
           "  --ny CELLS                   Periodic thin direction (default 4)\n"
           "  --dx DOMAINS --dz DOMAINS    GPU decomposition (default 1,1)\n"
           "  --no-slices                  Do not write 2D snapshot CSV files\n"
           "  --write-vtk                  Also write native VTK snapshots\n";
}

Config parse_arguments(const int argc, const char* const argv[]) {
    Config config;
    bool has_case = false;
    bool has_spacing = false;
    bool has_subgrid_cs = false;
    bool has_subgrid_cs_squared = false;
    for(int i=1; i<argc; ++i) {
        const std::string option = argv[i];
        const auto require_value = [&]() -> std::string {
            if(i+1>=argc) {
                throw std::invalid_argument(
                    "Missing value after "+option);
            }
            return argv[++i];
        };
        if(option=="--help" || option=="-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if(option=="--case") {
            const std::string value = lowercase(require_value());
            if(value=="rising-thermal" || value=="rising_thermal" ||
               value=="bubble") {
                config.kind = CaseKind::RisingThermal;
            } else if(value=="density-current" ||
                      value=="density_current" ||
                      value=="density") {
                config.kind = CaseKind::DensityCurrent;
            } else {
                throw std::invalid_argument("Unknown case: "+value);
            }
            has_case = true;
        } else if(option=="--h" || option=="--spacing") {
            config.spacing = parse_double(require_value(), option);
            has_spacing = true;
        } else if(option=="--dt") {
            config.dt = parse_double(require_value(), option);
        } else if(option=="--end-time") {
            config.end_time = parse_double(require_value(), option);
        } else if(option=="--sample-seconds") {
            config.sample_seconds =
                parse_double(require_value(), option);
        } else if(option=="--rho-ref") {
            config.reference_density =
                parse_double(require_value(), option);
        } else if(option=="--sgs") {
            config.subgrid_enabled =
                parse_on_off(require_value(), option);
        } else if(option=="--sgs-cs") {
            if(has_subgrid_cs_squared) {
                throw std::invalid_argument(
                    "--sgs-cs and --sgs-cs-squared are mutually exclusive");
            }
            const double cs = parse_double(require_value(), option);
            if(cs<0.0) {
                throw std::invalid_argument("--sgs-cs must be non-negative");
            }
            config.subgrid_cs_squared = cs*cs;
            has_subgrid_cs = true;
        } else if(option=="--sgs-cs-squared") {
            if(has_subgrid_cs) {
                throw std::invalid_argument(
                    "--sgs-cs and --sgs-cs-squared are mutually exclusive");
            }
            config.subgrid_cs_squared = parse_double(require_value(), option);
            if(config.subgrid_cs_squared<0.0) {
                throw std::invalid_argument(
                    "--sgs-cs-squared must be non-negative");
            }
            has_subgrid_cs_squared = true;
        } else if(option=="--sgs-scalar") {
            config.subgrid_scalar_coupling =
                parse_on_off(require_value(), option);
        } else if(option=="--sgs-pr-t") {
            config.subgrid_turbulent_prandtl =
                parse_double(require_value(), option);
        } else if(option=="--base-nu-m2-s") {
            config.base_nu_m2_s = parse_double(require_value(), option);
        } else if(option=="--base-alpha-m2-s") {
            config.base_alpha_m2_s = parse_double(require_value(), option);
        } else if(option=="--min-momentum-tau") {
            config.minimum_momentum_tau =
                parse_double(require_value(), option);
        } else if(option=="--min-thermal-tau") {
            config.minimum_thermal_tau =
                parse_double(require_value(), option);
            config.derive_minimum_thermal_tau = false;
        } else if(option=="--theta-scale") {
            config.theta_scale =
                parse_double(require_value(), option);
        } else if(option=="--bubble-center-x") {
            config.bubble_center_x =
                parse_double(require_value(), option);
        } else if(option=="--thermal") {
            const std::string value = lowercase(require_value());
            if(value=="d3q7" || value=="q7") {
                config.thermal_lattice = Temperature_Lattice::Q7;
            } else if(value=="d3q19" || value=="q19") {
                config.thermal_lattice = Temperature_Lattice::Q19;
            } else {
                throw std::invalid_argument(
                    "Unknown thermal lattice: "+value);
            }
        } else if(option=="--viscosity") {
            const std::string value = lowercase(require_value());
            if(value=="dynamic" || value=="mu-over-rho") {
                config.viscosity_mode =
                    ViscosityMode::DynamicAtGroundDensity;
            } else if(value=="numeric" || value=="kinematic") {
                config.viscosity_mode =
                    ViscosityMode::NumericAsKinematic;
            } else {
                throw std::invalid_argument(
                    "Unknown viscosity mode: "+value);
            }
        } else if(option=="--ny") {
            config.ny = parse_uint(require_value(), option);
        } else if(option=="--dx") {
            config.dx = parse_uint(require_value(), option);
        } else if(option=="--dz") {
            config.dz = parse_uint(require_value(), option);
        } else if(option=="--output") {
            config.output = require_value();
        } else if(option=="--no-slices") {
            config.write_slices = false;
        } else if(option=="--write-vtk") {
            config.write_vtk = true;
        } else {
            throw std::invalid_argument("Unknown option: "+option);
        }
    }

    if(!has_case || !has_spacing || config.output.empty()) {
        print_usage(argv[0]);
        throw std::invalid_argument(
            "--case, --h, and --output are required");
    }
    if(config.spacing<=0.0 || config.dt<=0.0 ||
       config.sample_seconds<=0.0 ||
       config.reference_density<=0.0 ||
       config.ny==0u || config.dx==0u || config.dz==0u) {
        throw std::invalid_argument(
            "Spacing, dt, sampling, density, and grid counts must be positive");
    }
    if(!std::isfinite(config.subgrid_cs_squared) ||
       !(config.subgrid_cs_squared>=0.0) ||
       !std::isfinite(config.subgrid_turbulent_prandtl) ||
       !(config.subgrid_turbulent_prandtl>0.0) ||
       !std::isfinite(config.minimum_momentum_tau) ||
       !(config.minimum_momentum_tau>=0.5) ||
       !std::isfinite(config.minimum_thermal_tau) ||
       !(config.minimum_thermal_tau>=0.5) ||
       (!std::isnan(config.base_nu_m2_s) &&
        (!std::isfinite(config.base_nu_m2_s) ||
         !(config.base_nu_m2_s>0.0))) ||
       (!std::isnan(config.base_alpha_m2_s) &&
        (!std::isfinite(config.base_alpha_m2_s) ||
         !(config.base_alpha_m2_s>=0.0)))) {
        throw std::invalid_argument(
            "SGS coefficients/base transport must be finite and non-negative; "
            "Pr_t must be positive and tau floors must be at least 0.5");
    }
    return config;
}

uint physical_cells(const double length, const double spacing) {
    const double cells = length/spacing;
    const long long rounded = std::llround(cells);
    if(rounded<=0 ||
       std::fabs(cells-(double)rounded)>1.0E-8*
                                        std::max(1.0, std::fabs(cells))) {
        throw std::invalid_argument(
            "Domain length is not an integer multiple of h");
    }
    if((unsigned long long)rounded>
       (unsigned long long)std::numeric_limits<uint>::max()-2ull) {
        throw std::invalid_argument("Grid dimension exceeds uint range");
    }
    return (uint)rounded;
}

double cell_coordinate(
    const CaseKind kind,
    const uint index,
    const double spacing) {
    // The checked-in GEA bubble meshes place the first cell centre at
    // x=z=0 by using block bounds [-h/2,L-h/2].  The density-current
    // meshes use the conventional [0,L] bounds and half-cell centres.
    return kind==CaseKind::RisingThermal
        ? ((double)index-1.0)*spacing
        : ((double)index-0.5)*spacing;
}

const char* grid_registration_name(const CaseKind kind) {
    return kind==CaseKind::RisingThermal
        ? "GEA cell-centred coordinates (bounds -h/2 to L-h/2)"
        : "GEA wall-centred bounds (0 to L)";
}

const char* collision_operator_name() {
#ifdef TRT
    return "TRT";
#else
    return "SRT";
#endif
}

TransportResolution resolve_transport(
    const Config& config,
    const CaseSpec& spec) {
    TransportResolution result;
    const double legacy_nu =
        config.viscosity_mode==ViscosityMode::DynamicAtGroundDensity
        ? spec.artificial_dynamic_viscosity/config.reference_density
        : spec.artificial_dynamic_viscosity;
    result.requested_nu_m2_s = std::isfinite(config.base_nu_m2_s)
        ? config.base_nu_m2_s : legacy_nu;
    result.requested_alpha_m2_s =
        std::isfinite(config.base_alpha_m2_s)
        ? config.base_alpha_m2_s : result.requested_nu_m2_s;

    const double lbm_scale =
        config.dt/(config.spacing*config.spacing);
    result.requested_nu_lbm = result.requested_nu_m2_s*lbm_scale;
    result.requested_momentum_tau =
        0.5+3.0*result.requested_nu_lbm;
    float momentum_floor = (float)config.minimum_momentum_tau;
    if((double)momentum_floor<config.minimum_momentum_tau) {
        momentum_floor = std::nextafter(
            momentum_floor, std::numeric_limits<float>::infinity());
    }
    float effective_nu = (float)result.requested_nu_lbm;
    float effective_tau = 0.5f+3.0f*effective_nu;
    result.momentum_floor_applied =
        (double)effective_tau<config.minimum_momentum_tau;
    if(result.momentum_floor_applied) {
        effective_nu = (momentum_floor-0.5f)/3.0f;
        effective_tau = 0.5f+3.0f*effective_nu;
        while(effective_tau<momentum_floor) {
            effective_nu = std::nextafter(
                effective_nu, std::numeric_limits<float>::infinity());
            effective_tau = 0.5f+3.0f*effective_nu;
        }
    }
    if(!(std::isfinite(effective_nu) &&
         std::isfinite(effective_tau) && effective_tau>0.5f)) {
        throw std::invalid_argument(
            "Momentum transport/floor cannot be represented in FP32");
    }
    result.effective_nu_lbm = (double)effective_nu;
    result.effective_momentum_tau = (double)effective_tau;
    result.effective_nu_m2_s =
        result.effective_nu_lbm/lbm_scale;

    const double thermal_cs2 =
        config.thermal_lattice==Temperature_Lattice::Q19
        ? 1.0/3.0 : 1.0/4.0;
    result.requested_alpha_lbm =
        result.requested_alpha_m2_s*lbm_scale;
    result.requested_thermal_tau =
        0.5+result.requested_alpha_lbm/thermal_cs2;
    const float inverse_thermal_cs2 =
        config.thermal_lattice==Temperature_Lattice::Q19 ? 3.0f : 4.0f;
    const float thermal_cs2_float = 1.0f/inverse_thermal_cs2;
    float derived_alpha_floor = 0.0f;
    float thermal_floor = 0.5f;
    if(config.derive_minimum_thermal_tau) {
        float momentum_nu_floor = (momentum_floor-0.5f)/3.0f;
        float momentum_tau_at_floor =
            0.5f+3.0f*momentum_nu_floor;
        while(momentum_tau_at_floor<momentum_floor) {
            momentum_nu_floor = std::nextafter(
                momentum_nu_floor,
                std::numeric_limits<float>::infinity());
            momentum_tau_at_floor =
                0.5f+3.0f*momentum_nu_floor;
        }
        const double requested_alpha_over_nu =
            result.requested_alpha_lbm/result.requested_nu_lbm;
        const double derived_alpha_floor_double =
            (double)momentum_nu_floor*requested_alpha_over_nu;
        derived_alpha_floor = (float)derived_alpha_floor_double;
        if((double)derived_alpha_floor<derived_alpha_floor_double) {
            derived_alpha_floor = std::nextafter(
                derived_alpha_floor,
                std::numeric_limits<float>::infinity());
        }
        thermal_floor =
            0.5f+derived_alpha_floor*inverse_thermal_cs2;
        result.minimum_thermal_tau_derived = true;
    } else {
        thermal_floor = (float)config.minimum_thermal_tau;
        if((double)thermal_floor<config.minimum_thermal_tau) {
            thermal_floor = std::nextafter(
                thermal_floor, std::numeric_limits<float>::infinity());
        }
    }
    if(!(std::isfinite(thermal_floor) && thermal_floor>=0.5f &&
         std::isfinite(derived_alpha_floor))) {
        throw std::invalid_argument(
            "Thermal transport floor cannot be represented in FP32");
    }
    result.minimum_thermal_tau = (double)thermal_floor;
    float effective_alpha = (float)result.requested_alpha_lbm;
    float effective_tau_t =
        0.5f+effective_alpha*inverse_thermal_cs2;
    result.thermal_floor_applied =
        effective_tau_t<thermal_floor;
    if(result.thermal_floor_applied) {
        effective_alpha = config.derive_minimum_thermal_tau
            ? derived_alpha_floor
            : (thermal_floor-0.5f)*thermal_cs2_float;
        effective_tau_t =
            0.5f+effective_alpha*inverse_thermal_cs2;
        while(effective_tau_t<thermal_floor) {
            effective_alpha = std::nextafter(
                effective_alpha,
                std::numeric_limits<float>::infinity());
            effective_tau_t =
                0.5f+effective_alpha*inverse_thermal_cs2;
        }
    }
    if(!(std::isfinite(effective_alpha) &&
         std::isfinite(effective_tau_t) && effective_tau_t>=0.5f)) {
        throw std::invalid_argument(
            "Thermal transport/floor cannot be represented in FP32");
    }
    result.effective_alpha_lbm = (double)effective_alpha;
    result.effective_thermal_tau = (double)effective_tau_t;
    result.effective_alpha_m2_s =
        result.effective_alpha_lbm/lbm_scale;
    return result;
}

double initial_theta(
    const Config& config, const double x, const double z) {
    if(config.kind==CaseKind::RisingThermal) {
        const double radius =
            std::hypot(x-config.bubble_center_x, z-2000.0);
        return radius<=2000.0
            ? kReferenceTheta+2.0*(1.0-radius/2000.0)
            : kReferenceTheta;
    }
    const double radius = std::sqrt(
        (x/4000.0)*(x/4000.0)+
        ((z-3000.0)/2000.0)*((z-3000.0)/2000.0));
    return radius<=1.0
        ? kReferenceTheta-7.5*(1.0+std::cos(kPi*radius))
        : kReferenceTheta;
}

std::string time_token(const double time) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(1) << time;
    std::string token = stream.str();
    std::replace(token.begin(), token.end(), '.', 'p');
    return token;
}

void read_fields(LBM& lbm) {
    lbm.external_finish();
    lbm.rho.read_from_device();
    lbm.u.read_from_device();
    lbm.T.read_from_device();
}

Metrics evaluate_metrics(
    LBM& lbm,
    const Config& config,
    const CaseSpec& spec,
    const uint nx,
    const uint nz) {
    Metrics metrics;
    const double velocity_factor = config.spacing/config.dt;
    const double inverse_sound_speed = std::sqrt(3.0);
    std::uint64_t count = 0ull;
    for(uint z=1u; z<nz-1u; ++z) {
        for(uint y=0u; y<config.ny; ++y) {
            for(uint x=1u; x<nx-1u; ++x) {
                const ulong n = lbm.index(x, y, z);
                const double ux_lbm = (double)lbm.u.x[n];
                const double uy_lbm = (double)lbm.u.y[n];
                const double uz_lbm = (double)lbm.u.z[n];
                const double ux = ux_lbm*velocity_factor;
                const double uz = uz_lbm*velocity_factor;
                const double theta =
                    ((double)lbm.T[n]-1.0)*spec.theta_scale;
                const double rho = (double)lbm.rho[n];
                const double speed_lbm =
                    std::sqrt(ux_lbm*ux_lbm+
                              uy_lbm*uy_lbm+
                              uz_lbm*uz_lbm);
                metrics.u_min = std::min(metrics.u_min, ux);
                metrics.u_max = std::max(metrics.u_max, ux);
                metrics.w_min = std::min(metrics.w_min, uz);
                metrics.w_max = std::max(metrics.w_max, uz);
                metrics.theta_min = std::min(metrics.theta_min, theta);
                metrics.theta_max = std::max(metrics.theta_max, theta);
                metrics.rho_min = std::min(metrics.rho_min, rho);
                metrics.rho_max = std::max(metrics.rho_max, rho);
                metrics.rho_mean += rho;
                metrics.theta_mean += theta;
                metrics.theta_l2 += theta*theta;
                metrics.max_speed = std::max(
                    metrics.max_speed, speed_lbm*velocity_factor);
                metrics.max_mach = std::max(
                    metrics.max_mach,
                    speed_lbm*inverse_sound_speed);
                metrics.finite =
                    metrics.finite &&
                    std::isfinite(ux_lbm) &&
                    std::isfinite(uy_lbm) &&
                    std::isfinite(uz_lbm) &&
                    std::isfinite(theta) &&
                    std::isfinite(rho);
                metrics.positive_density =
                    metrics.positive_density && rho>0.0;
                ++count;
            }
        }
    }
    metrics.rho_mean /= (double)count;
    metrics.theta_mean /= (double)count;
    metrics.theta_l2 = std::sqrt(metrics.theta_l2/(double)count);

    for(uint z=1u; z<nz-1u; ++z) {
        for(uint x=1u; x<nx-1u; ++x) {
            double minimum = std::numeric_limits<double>::infinity();
            double maximum = -std::numeric_limits<double>::infinity();
            for(uint y=0u; y<config.ny; ++y) {
                const double theta =
                    ((double)lbm.T[lbm.index(x, y, z)]-1.0)*
                    spec.theta_scale;
                minimum = std::min(minimum, theta);
                maximum = std::max(maximum, theta);
            }
            metrics.max_y_temperature_spread = std::max(
                metrics.max_y_temperature_spread, maximum-minimum);
        }
    }

    if(config.kind==CaseKind::DensityCurrent) {
        constexpr double threshold = -1.0;
        for(uint x=1u; x+1u<nx-1u; ++x) {
            double left = 0.0;
            double right = 0.0;
            for(uint y=0u; y<config.ny; ++y) {
                left +=
                    ((double)lbm.T[lbm.index(x, y, 1u)]-1.0)*
                    spec.theta_scale;
                right +=
                    ((double)lbm.T[lbm.index(x+1u, y, 1u)]-1.0)*
                    spec.theta_scale;
            }
            left /= (double)config.ny;
            right /= (double)config.ny;
            if(left<=threshold && right>threshold) {
                const double x_left =
                    ((double)x-0.5)*config.spacing;
                const double x_right =
                    ((double)x+0.5)*config.spacing;
                const double denominator = right-left;
                metrics.front_bracket_left = x_left;
                metrics.front_bracket_right = x_right;
                metrics.front_interpolated =
                    std::fabs(denominator)>1.0E-14
                    ? x_left+
                      (threshold-left)/denominator*
                      (x_right-x_left)
                    : 0.5*(x_left+x_right);
            }
        }
    }
    return metrics;
}

void sample_subgrid_statistics(
    LBM& lbm,
    const Config& config,
    Metrics& metrics) {
    if(!config.subgrid_enabled) return;
    const Subgrid_Statistics statistics = lbm.subgrid_statistics();
    metrics.subgrid_statistics_valid = statistics.valid;
    metrics.subgrid_count = statistics.count;
    if(!statistics.valid) return;
    const double diffusivity_scale =
        config.spacing*config.spacing/config.dt;
    metrics.subgrid_mean_nu_t_lbm = statistics.mean_nu_t;
    metrics.subgrid_max_nu_t_lbm = (double)statistics.max_nu_t;
    metrics.subgrid_mean_rho_nu_t_lbm = statistics.mean_rho_nu_t;
    metrics.subgrid_max_rho_nu_t_lbm =
        (double)statistics.max_rho_nu_t;
    metrics.subgrid_mean_nu_t_m2_s =
        statistics.mean_nu_t*diffusivity_scale;
    metrics.subgrid_max_nu_t_m2_s =
        (double)statistics.max_nu_t*diffusivity_scale;
    metrics.subgrid_mean_mu_t_pa_s =
        config.reference_density*statistics.mean_rho_nu_t*
        diffusivity_scale;
    metrics.subgrid_max_mu_t_pa_s =
        config.reference_density*(double)statistics.max_rho_nu_t*
        diffusivity_scale;
}

void write_metrics_header(std::ofstream& output) {
    output
        << "step,time_s,u_min_m_s,u_max_m_s,w_min_m_s,w_max_m_s,"
           "theta_prime_min_K,theta_prime_max_K,rho_min_lbm,"
           "rho_max_lbm,rho_mean_lbm,theta_prime_mean_K,"
           "theta_prime_rms_K,max_speed_m_s,max_mach,"
           "max_y_theta_spread_K,front_x_m,front_bracket_left_m,"
           "front_bracket_right_m,nu_t_mean_lbm,nu_t_max_lbm,"
           "rho_nu_t_mean_lbm,rho_nu_t_max_lbm,nu_t_mean_m2_s,"
           "nu_t_max_m2_s,mu_t_mean_Pa_s,mu_t_max_Pa_s,"
           "subgrid_count,subgrid_statistics_valid,finite,positive_density\n";
}

void write_metrics_row(
    std::ofstream& output,
    const std::uint64_t step,
    const double time,
    const Metrics& metrics) {
    output << std::setprecision(12)
           << step << ',' << time << ','
           << metrics.u_min << ',' << metrics.u_max << ','
           << metrics.w_min << ',' << metrics.w_max << ','
           << metrics.theta_min << ',' << metrics.theta_max << ','
           << metrics.rho_min << ',' << metrics.rho_max << ','
           << metrics.rho_mean << ',' << metrics.theta_mean << ','
           << metrics.theta_l2 << ',' << metrics.max_speed << ','
           << metrics.max_mach << ','
           << metrics.max_y_temperature_spread << ','
           << metrics.front_interpolated << ','
           << metrics.front_bracket_left << ','
           << metrics.front_bracket_right << ','
           << metrics.subgrid_mean_nu_t_lbm << ','
           << metrics.subgrid_max_nu_t_lbm << ','
           << metrics.subgrid_mean_rho_nu_t_lbm << ','
           << metrics.subgrid_max_rho_nu_t_lbm << ','
           << metrics.subgrid_mean_nu_t_m2_s << ','
           << metrics.subgrid_max_nu_t_m2_s << ','
           << metrics.subgrid_mean_mu_t_pa_s << ','
           << metrics.subgrid_max_mu_t_pa_s << ','
           << metrics.subgrid_count << ','
           << (metrics.subgrid_statistics_valid ? 1 : 0) << ','
           << (metrics.finite ? 1 : 0) << ','
           << (metrics.positive_density ? 1 : 0) << '\n';
    output.flush();
}

void print_progress(
    const std::uint64_t step,
    const std::uint64_t total_steps,
    const double time,
    const Metrics& metrics,
    const std::chrono::steady_clock::time_point start) {
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-start).count();
    const double rate = elapsed>0.0 ? (double)step/elapsed : 0.0;
    const double remaining =
        rate>0.0 ? (double)(total_steps-step)/rate : 0.0;
    std::cout << std::fixed << std::setprecision(4)
              << "GEA_PROGRESS step=" << step << '/' << total_steps
              << " time_s=" << time
              << " w=[" << metrics.w_min << ',' << metrics.w_max << ']'
              << " theta=[" << metrics.theta_min << ','
              << metrics.theta_max << ']'
              << " rho=[" << metrics.rho_min << ','
              << metrics.rho_max << ']'
              << " max_Ma=" << metrics.max_mach;
    if(std::isfinite(metrics.front_interpolated)) {
        std::cout << " front_m=" << metrics.front_interpolated;
    }
    if(metrics.subgrid_statistics_valid) {
        std::cout << " nu_t_mean_m2_s="
                  << metrics.subgrid_mean_nu_t_m2_s
                  << " nu_t_max_m2_s="
                  << metrics.subgrid_max_nu_t_m2_s
                  << " mu_t_mean_Pa_s="
                  << metrics.subgrid_mean_mu_t_pa_s
                  << " mu_t_max_Pa_s="
                  << metrics.subgrid_max_mu_t_pa_s
                  << " sgs_cells=" << metrics.subgrid_count;
    }
    std::cout << " steps_per_s=" << rate
              << " eta_s=" << remaining << '\n';
}

void write_slice(
    const std::filesystem::path& filename,
    LBM& lbm,
    const Config& config,
    const CaseSpec& spec,
    const uint nx,
    const uint nz) {
    std::ofstream output(filename);
    if(!output) {
        throw std::runtime_error(
            "Cannot write slice: "+filename.string());
    }
    output << "x_m,z_m,u_m_s,w_m_s,theta_prime_K,rho_lbm\n";
    output << std::setprecision(10);
    const double velocity_factor = config.spacing/config.dt;
    for(uint z=1u; z<nz-1u; ++z) {
        const double z_si =
            cell_coordinate(config.kind, z, config.spacing);
        for(uint x=1u; x<nx-1u; ++x) {
            const double x_si =
                cell_coordinate(config.kind, x, config.spacing);
            double ux = 0.0;
            double uz = 0.0;
            double theta = 0.0;
            double rho = 0.0;
            for(uint y=0u; y<config.ny; ++y) {
                const ulong n = lbm.index(x, y, z);
                ux += (double)lbm.u.x[n]*velocity_factor;
                uz += (double)lbm.u.z[n]*velocity_factor;
                theta += ((double)lbm.T[n]-1.0)*spec.theta_scale;
                rho += (double)lbm.rho[n];
            }
            const double inverse_y = 1.0/(double)config.ny;
            output << x_si << ',' << z_si << ','
                   << ux*inverse_y << ',' << uz*inverse_y << ','
                   << theta*inverse_y << ',' << rho*inverse_y << '\n';
        }
    }
}

void write_profile_1200(
    const std::filesystem::path& filename,
    LBM& lbm,
    const Config& config,
    const CaseSpec& spec,
    const uint nx,
    const uint nz) {
    const double lattice_z = 1200.0/config.spacing+0.5;
    uint lower = (uint)std::floor(lattice_z);
    lower = std::max(1u, std::min(lower, nz-2u));
    const uint upper = std::min(lower+1u, nz-2u);
    const double lower_z = ((double)lower-0.5)*config.spacing;
    const double upper_z = ((double)upper-0.5)*config.spacing;
    const double weight =
        upper==lower ? 0.0 : (1200.0-lower_z)/(upper_z-lower_z);

    std::ofstream output(filename);
    if(!output) {
        throw std::runtime_error(
            "Cannot write profile: "+filename.string());
    }
    output << "x_m,theta_prime_K,z_target_m,z_lower_m,z_upper_m\n";
    output << std::setprecision(10);
    for(uint x=1u; x<nx-1u; ++x) {
        double lower_theta = 0.0;
        double upper_theta = 0.0;
        for(uint y=0u; y<config.ny; ++y) {
            lower_theta +=
                ((double)lbm.T[lbm.index(x, y, lower)]-1.0)*
                spec.theta_scale;
            upper_theta +=
                ((double)lbm.T[lbm.index(x, y, upper)]-1.0)*
                spec.theta_scale;
        }
        lower_theta /= (double)config.ny;
        upper_theta /= (double)config.ny;
        const double theta =
            lower_theta+weight*(upper_theta-lower_theta);
        output << ((double)x-0.5)*config.spacing << ','
               << theta << ",1200," << lower_z << ',' << upper_z
               << '\n';
    }
}

void write_vtk_snapshot(
    const std::filesystem::path& directory,
    LBM& lbm) {
    std::filesystem::create_directories(directory);
    std::string path = directory.string();
    if(!path.empty() && path.back()!='/' && path.back()!='\\') {
        path.push_back(std::filesystem::path::preferred_separator);
    }
    lbm.u.write_host_to_vtk(path, true, false);
    lbm.T.write_host_to_vtk(path, true, false);
    lbm.rho.write_host_to_vtk(path, true, false);
}

void write_metadata(
    const std::filesystem::path& filename,
    const Config& config,
    const CaseSpec& spec,
    const uint nx,
    const uint nz,
    const std::uint64_t steps,
    const TransportResolution& transport,
    const double beta_lbm,
    const double gravity_lbm) {
    std::ofstream output(filename);
    if(!output) {
        throw std::runtime_error(
            "Cannot write metadata: "+filename.string());
    }
    output << std::setprecision(12)
           << "{\n"
           << "  \"case\": \"" << spec.name << "\",\n"
           << "  \"model\": \"CULBM weakly-compressible Boussinesq analogue\",\n"
           << "  \"momentum_lattice\": \"D3Q19\",\n"
           << "  \"momentum_collision\": \""
           << collision_operator_name() << "\",\n"
           << "  \"thermal_lattice\": \""
           << (config.thermal_lattice==Temperature_Lattice::Q19
               ? "D3Q19" : "D3Q7") << "\",\n"
           << "  \"precision\": \"FP32 DDF\",\n"
           << "  \"subgrid_model\": \""
           << (config.subgrid_enabled
               ? "Smagorinsky-Lilly" : "disabled") << "\",\n"
           << "  \"subgrid_enabled\": "
           << (config.subgrid_enabled ? "true" : "false") << ",\n"
           << "  \"subgrid_scope\": \""
           << (!config.subgrid_enabled
               ? "none"
               : config.subgrid_scalar_coupling
                 ? "momentum and thermal scalar"
                 : "momentum only") << "\",\n"
           << "  \"subgrid_cs\": "
           << std::sqrt(config.subgrid_cs_squared) << ",\n"
           << "  \"subgrid_cs_squared\": "
           << config.subgrid_cs_squared << ",\n"
           << "  \"subgrid_kernel_prefactor\": "
           << 18.0*std::sqrt(2.0)*config.subgrid_cs_squared << ",\n"
           << "  \"subgrid_scalar_coupling\": "
           << (config.subgrid_scalar_coupling ? "true" : "false")
           << ",\n"
           << "  \"turbulent_prandtl\": "
           << config.subgrid_turbulent_prandtl << ",\n"
           << "  \"free_slip\": \"specular TYPE_S|TYPE_X ghost wall\",\n"
           << "  \"grid_registration\": \""
           << grid_registration_name(config.kind) << "\",\n"
           << "  \"h_m\": " << config.spacing << ",\n"
           << "  \"dt_s\": " << config.dt << ",\n"
           << "  \"end_time_s\": " << spec.end_time << ",\n"
           << "  \"steps\": " << steps << ",\n"
           << "  \"grid\": [" << nx << ',' << config.ny << ',' << nz
           << "],\n"
           << "  \"decomposition\": [" << config.dx << ",1,"
           << config.dz << "],\n"
           << "  \"rho_ref_kg_m3\": " << config.reference_density << ",\n"
           << "  \"mu_artificial_Pa_s\": "
           << spec.artificial_dynamic_viscosity << ",\n"
           << "  \"viscosity_mode\": \""
           << (std::isfinite(config.base_nu_m2_s)
               ? "explicit base-nu-m2-s"
               : config.viscosity_mode==
               ViscosityMode::DynamicAtGroundDensity
               ? "mu/rho_ref" : "numeric value treated as nu")
           << "\",\n"
           << "  \"requested_nu_m2_s\": "
           << transport.requested_nu_m2_s << ",\n"
           << "  \"requested_nu_lbm\": "
           << transport.requested_nu_lbm << ",\n"
           << "  \"requested_momentum_tau\": "
           << transport.requested_momentum_tau << ",\n"
           << "  \"minimum_momentum_tau\": "
           << config.minimum_momentum_tau << ",\n"
           << "  \"momentum_tau_floor_applied\": "
           << (transport.momentum_floor_applied ? "true" : "false")
           << ",\n"
           << "  \"effective_nu_m2_s\": "
           << transport.effective_nu_m2_s << ",\n"
           << "  \"effective_nu_lbm\": "
           << transport.effective_nu_lbm << ",\n"
           << "  \"effective_momentum_tau\": "
           << transport.effective_momentum_tau << ",\n"
           << "  \"requested_alpha_m2_s\": "
           << transport.requested_alpha_m2_s << ",\n"
           << "  \"requested_alpha_lbm\": "
           << transport.requested_alpha_lbm << ",\n"
           << "  \"requested_thermal_tau\": "
           << transport.requested_thermal_tau << ",\n"
           << "  \"minimum_thermal_tau\": "
           << transport.minimum_thermal_tau << ",\n"
           << "  \"minimum_thermal_tau_derived\": "
           << (transport.minimum_thermal_tau_derived ? "true" : "false")
           << ",\n"
           << "  \"thermal_tau_floor_applied\": "
           << (transport.thermal_floor_applied ? "true" : "false")
           << ",\n"
           << "  \"effective_alpha_m2_s\": "
           << transport.effective_alpha_m2_s << ",\n"
           << "  \"effective_alpha_lbm\": "
           << transport.effective_alpha_lbm << ",\n"
           << "  \"effective_thermal_tau\": "
           << transport.effective_thermal_tau << ",\n"
           << "  \"nu_m2_s\": " << transport.effective_nu_m2_s
           << ",\n"
           << "  \"nu_lbm\": " << transport.effective_nu_lbm
           << ",\n"
           << "  \"tau\": " << transport.effective_momentum_tau
           << ",\n"
           << "  \"alpha_m2_s\": "
           << transport.effective_alpha_m2_s << ",\n"
           << "  \"alpha_lbm\": "
           << transport.effective_alpha_lbm << ",\n"
           << "  \"tau_T\": " << transport.effective_thermal_tau
           << ",\n"
           << "  \"theta_scale_K\": " << spec.theta_scale << ",\n"
           << "  \"beta_lbm\": " << beta_lbm << ",\n"
           << "  \"gravity_z_lbm\": " << gravity_lbm << ",\n"
           << "  \"bubble_center_x_m\": "
           << config.bubble_center_x << "\n"
           << "}\n";
}

void write_final_summary(
    const std::filesystem::path& filename,
    const Metrics& metrics,
    const double time,
    const std::uint64_t step,
    const bool completed) {
    std::ofstream output(filename);
    if(!output) {
        throw std::runtime_error(
            "Cannot write summary: "+filename.string());
    }
    output << std::setprecision(12)
           << "{\n"
           << "  \"status\": \""
           << (completed ? "completed" : "aborted") << "\",\n"
           << "  \"completed\": "
           << (completed ? "true" : "false") << ",\n"
           << "  \"step\": " << step << ",\n"
           << "  \"time_s\": " << time << ",\n"
           << "  \"u_min_m_s\": " << metrics.u_min << ",\n"
           << "  \"u_max_m_s\": " << metrics.u_max << ",\n"
           << "  \"w_min_m_s\": " << metrics.w_min << ",\n"
           << "  \"w_max_m_s\": " << metrics.w_max << ",\n"
           << "  \"theta_prime_min_K\": " << metrics.theta_min << ",\n"
           << "  \"theta_prime_max_K\": " << metrics.theta_max << ",\n"
           << "  \"rho_min_lbm\": " << metrics.rho_min << ",\n"
           << "  \"rho_max_lbm\": " << metrics.rho_max << ",\n"
           << "  \"rho_mean_lbm\": " << metrics.rho_mean << ",\n"
           << "  \"theta_prime_mean_K\": " << metrics.theta_mean << ",\n"
           << "  \"max_speed_m_s\": " << metrics.max_speed << ",\n"
           << "  \"max_mach\": " << metrics.max_mach << ",\n"
           << "  \"max_y_theta_spread_K\": "
           << metrics.max_y_temperature_spread << ",\n"
           << "  \"front_x_m\": ";
    if(std::isfinite(metrics.front_interpolated)) {
        output << metrics.front_interpolated;
    } else {
        output << "null";
    }
    output << ",\n  \"front_bracket_left_m\": ";
    if(std::isfinite(metrics.front_bracket_left)) {
        output << metrics.front_bracket_left;
    } else {
        output << "null";
    }
    output << ",\n  \"front_bracket_right_m\": ";
    if(std::isfinite(metrics.front_bracket_right)) {
        output << metrics.front_bracket_right;
    } else {
        output << "null";
    }
    output << ",\n"
           << "  \"subgrid_statistics_valid\": "
           << (metrics.subgrid_statistics_valid ? "true" : "false")
           << ",\n"
           << "  \"subgrid_count\": " << metrics.subgrid_count
           << ",\n"
           << "  \"nu_t_mean_lbm\": "
           << metrics.subgrid_mean_nu_t_lbm << ",\n"
           << "  \"nu_t_max_lbm\": "
           << metrics.subgrid_max_nu_t_lbm << ",\n"
           << "  \"rho_nu_t_mean_lbm\": "
           << metrics.subgrid_mean_rho_nu_t_lbm << ",\n"
           << "  \"rho_nu_t_max_lbm\": "
           << metrics.subgrid_max_rho_nu_t_lbm << ",\n"
           << "  \"nu_t_mean_m2_s\": "
           << metrics.subgrid_mean_nu_t_m2_s << ",\n"
           << "  \"nu_t_max_m2_s\": "
           << metrics.subgrid_max_nu_t_m2_s << ",\n"
           << "  \"mu_t_mean_Pa_s\": "
           << metrics.subgrid_mean_mu_t_pa_s << ",\n"
           << "  \"mu_t_max_Pa_s\": "
           << metrics.subgrid_max_mu_t_pa_s << ",\n"
           << "  \"finite\": " << (metrics.finite ? "true" : "false")
           << ",\n"
           << "  \"positive_density\": "
           << (metrics.positive_density ? "true" : "false") << "\n"
           << "}\n";
}

int run(Config config) {
    CaseSpec spec = case_spec(config);
    if(config.end_time>0.0) spec.end_time = config.end_time;
    const uint physical_nx = physical_cells(
        spec.length_x, config.spacing);
    const uint physical_nz = physical_cells(
        spec.length_z, config.spacing);
    const uint nx = physical_nx+2u;
    const uint nz = physical_nz+2u;
    if(nx%config.dx!=0u || nz%config.dz!=0u) {
        throw std::invalid_argument(
            "Global dimensions including ghosts must be divisible by dx/dz");
    }
    const std::uint64_t total_steps =
        (std::uint64_t)std::llround(spec.end_time/config.dt);
    if(std::fabs((double)total_steps*config.dt-spec.end_time)>
       1.0E-9*std::max(1.0, spec.end_time)) {
        throw std::invalid_argument(
            "End time must be an integer multiple of dt");
    }
    const std::uint64_t sample_steps = std::max<std::uint64_t>(
        1ull,
        (std::uint64_t)std::llround(config.sample_seconds/config.dt));

    std::filesystem::create_directories(config.output);
    std::filesystem::create_directories(config.output/"snapshots");
    if(config.write_vtk) {
        std::filesystem::create_directories(config.output/"vtk");
    }

    if(!set_temperature_lattice(config.thermal_lattice)) {
        throw std::runtime_error(
            "Temperature lattice is locked by an existing LBM instance");
    }
    Subgrid_Configuration subgrid;
    subgrid.enabled = config.subgrid_enabled;
    subgrid.smagorinsky_constant =
        (float)std::sqrt(config.subgrid_cs_squared);
    subgrid.scalar_coupling = config.subgrid_scalar_coupling;
    subgrid.turbulent_prandtl =
        (float)config.subgrid_turbulent_prandtl;
    subgrid.fixed_eddy_viscosity = -1.0f;
    if(!set_subgrid_configuration(subgrid)) {
        throw std::runtime_error(
            "Subgrid configuration is invalid or locked by an existing LBM instance");
    }

    const TransportResolution transport =
        resolve_transport(config, spec);
    const double beta_lbm = spec.theta_scale/kReferenceTheta;
    const double gravity_lbm =
        -kGravity*config.dt*config.dt/config.spacing;

    units.set_m_kg_s_K(
        (float)config.spacing,
        (float)(config.reference_density*
                config.spacing*config.spacing*config.spacing),
        (float)config.dt,
        (float)spec.theta_scale);
    units.set_temperature_reference(1.0f, (float)kReferenceTheta);
    const double vtk_registration_shift =
        config.kind==CaseKind::RisingThermal
        ? -0.5*config.spacing
        : 0.0;
    vtk_origin_shift = float3(
        (float)(0.5*spec.length_x+vtk_registration_shift),
        0.0f,
        (float)(0.5*spec.length_z+vtk_registration_shift));

    if(transport.momentum_floor_applied) {
        std::cout << std::setprecision(12)
                  << "GEA_TAU_FLOOR field=momentum requested_tau="
                  << transport.requested_momentum_tau
                  << " minimum_tau=" << config.minimum_momentum_tau
                  << " effective_tau="
                  << transport.effective_momentum_tau
                  << " requested_nu_m2_s="
                  << transport.requested_nu_m2_s
                  << " effective_nu_m2_s="
                  << transport.effective_nu_m2_s << '\n';
    }
    if(transport.thermal_floor_applied) {
        std::cout << std::setprecision(12)
                  << "GEA_TAU_FLOOR field=thermal requested_tau="
                  << transport.requested_thermal_tau
                  << " minimum_tau=" << transport.minimum_thermal_tau
                  << " minimum_tau_derived="
                  << (transport.minimum_thermal_tau_derived
                      ? "true" : "false")
                  << " effective_tau="
                  << transport.effective_thermal_tau
                  << " requested_alpha_m2_s="
                  << transport.requested_alpha_m2_s
                  << " effective_alpha_m2_s="
                  << transport.effective_alpha_m2_s << '\n';
    }

    std::cout << std::setprecision(12)
              << "GEA_CONFIG case=" << spec.name
              << " h_m=" << config.spacing
              << " dt_s=" << config.dt
              << " end_time_s=" << spec.end_time
              << " grid=" << nx << 'x' << config.ny << 'x' << nz
              << " decomposition=" << config.dx << "x1x" << config.dz
              << " registration=\""
              << grid_registration_name(config.kind) << '"'
              << " collision=" << collision_operator_name()
              << " thermal="
              << (config.thermal_lattice==Temperature_Lattice::Q19
                  ? "D3Q19" : "D3Q7")
              << " precision=FP32 subgrid="
              << (config.subgrid_enabled ? "smagorinsky-lilly" : "off")
              << " sgs_cs_squared=" << config.subgrid_cs_squared
              << " sgs_scalar="
              << (config.subgrid_scalar_coupling ? "on" : "off")
              << " sgs_Pr_t=" << config.subgrid_turbulent_prandtl
              << " requested_nu_si=" << transport.requested_nu_m2_s
              << " requested_nu_lbm=" << transport.requested_nu_lbm
              << " requested_tau=" << transport.requested_momentum_tau
              << " min_tau=" << config.minimum_momentum_tau
              << " momentum_floor="
              << (transport.momentum_floor_applied ? "applied" : "inactive")
              << " effective_nu_si=" << transport.effective_nu_m2_s
              << " effective_nu_lbm=" << transport.effective_nu_lbm
              << " effective_tau=" << transport.effective_momentum_tau
              << " requested_alpha_si="
              << transport.requested_alpha_m2_s
              << " requested_alpha_lbm="
              << transport.requested_alpha_lbm
              << " requested_tau_T=" << transport.requested_thermal_tau
              << " min_tau_T=" << transport.minimum_thermal_tau
              << " min_tau_T_derived="
              << (transport.minimum_thermal_tau_derived
                  ? "true" : "false")
              << " thermal_floor="
              << (transport.thermal_floor_applied ? "applied" : "inactive")
              << " effective_alpha_si="
              << transport.effective_alpha_m2_s
              << " effective_alpha_lbm="
              << transport.effective_alpha_lbm
              << " effective_tau_T=" << transport.effective_thermal_tau
              << " beta_lbm=" << beta_lbm
              << " gz_lbm=" << gravity_lbm
              << " total_steps=" << total_steps << '\n';

    LBM lbm(
        nx, config.ny, nz,
        config.dx, 1u, config.dz,
        (float)transport.effective_nu_lbm,
        0.0f, 0.0f, 0.0f, 0.0f,
        (float)transport.effective_alpha_lbm,
        (float)beta_lbm);
    lbm.set_buoyancy_gravity(
        0.0f, 0.0f, (float)gravity_lbm);

    for(uint z=0u; z<nz; ++z) {
        const uint mirror_z = std::max(
            1u, std::min(z, nz-2u));
        const double z_si =
            cell_coordinate(config.kind, mirror_z, config.spacing);
        for(uint y=0u; y<config.ny; ++y) {
            for(uint x=0u; x<nx; ++x) {
                const uint mirror_x = std::max(
                    1u, std::min(x, nx-2u));
                const double x_si =
                    cell_coordinate(config.kind, mirror_x, config.spacing);
                const double theta =
                    initial_theta(config, x_si, z_si);
                const ulong n = lbm.index(x, y, z);
                lbm.rho[n] = 1.0f;
                lbm.u.x[n] = 0.0f;
                lbm.u.y[n] = 0.0f;
                lbm.u.z[n] = 0.0f;
                lbm.T[n] = (float)(
                    1.0+(theta-kReferenceTheta)/spec.theta_scale);
                lbm.flags[n] =
                    x==0u || x==nx-1u || z==0u || z==nz-1u
                    ? (uchar)(TYPE_S|TYPE_X)
                    : (uchar)0u;
            }
        }
    }

    write_metadata(
        config.output/"run_config.json",
        config, spec, nx, nz, total_steps,
        transport, beta_lbm, gravity_lbm);

    std::ofstream metrics_output(config.output/"metrics.csv");
    if(!metrics_output) {
        throw std::runtime_error("Cannot create metrics.csv");
    }
    write_metrics_header(metrics_output);

    std::set<std::uint64_t> snapshot_steps;
    for(const double time : spec.snapshot_times) {
        if(time<=spec.end_time+1.0E-9) {
            snapshot_steps.insert(
                (std::uint64_t)std::llround(time/config.dt));
        }
    }
    snapshot_steps.insert(total_steps);

    Metrics initial = evaluate_metrics(
        lbm, config, spec, nx, nz);
    write_metrics_row(metrics_output, 0ull, 0.0, initial);
    if(config.write_slices && snapshot_steps.count(0ull)!=0u) {
        write_slice(
            config.output/"snapshots"/"slice_t0p0.csv",
            lbm, config, spec, nx, nz);
    }

    lbm.external_initialize();
    const auto start = std::chrono::steady_clock::now();
    Metrics last = initial;
    for(std::uint64_t step=1ull; step<=total_steps; ++step) {
        lbm.external_update_free_slip_boundaries(kFreeSlipFaces);
        lbm.external_step();
        const bool snapshot = snapshot_steps.count(step)!=0u;
        const bool sample =
            step%sample_steps==0ull || step==total_steps;
        if(!snapshot && !sample) continue;

        lbm.external_update_free_slip_boundaries(kFreeSlipFaces);
        read_fields(lbm);
        const double time = (double)step*config.dt;
        last = evaluate_metrics(lbm, config, spec, nx, nz);
        if(sample) {
            sample_subgrid_statistics(lbm, config, last);
            write_metrics_row(metrics_output, step, time, last);
            print_progress(step, total_steps, time, last, start);
        }
        if(snapshot) {
            const std::string token = time_token(time);
            if(config.write_slices) {
                write_slice(
                    config.output/"snapshots"/
                    ("slice_t"+token+".csv"),
                    lbm, config, spec, nx, nz);
            }
            if(config.write_vtk) {
                write_vtk_snapshot(
                    config.output/"vtk"/("t"+token), lbm);
            }
        }
        if(!last.finite || !last.positive_density) {
            write_final_summary(
                config.output/"final_summary.json",
                last, time, step, false);
            std::cerr
                << "GEA_ABORT non-finite field or non-positive density at "
                << "step " << step << ", time " << time << " s\n";
            return 3;
        }
    }
    lbm.external_finish();

    if(config.kind==CaseKind::DensityCurrent) {
        write_profile_1200(
            config.output/"profile_z1200_tfinal.csv",
            lbm, config, spec, nx, nz);
    }
    write_final_summary(
        config.output/"final_summary.json",
        last, spec.end_time, total_steps, true);

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-start).count();
    std::cout << std::setprecision(12)
              << "GEA_COMPLETE case=" << spec.name
              << " h_m=" << config.spacing
              << " time_s=" << spec.end_time
              << " steps=" << total_steps
              << " elapsed_s=" << elapsed
              << " w_min=" << last.w_min
              << " w_max=" << last.w_max
              << " theta_min=" << last.theta_min
              << " theta_max=" << last.theta_max
              << " max_Ma=" << last.max_mach;
    if(std::isfinite(last.front_interpolated)) {
        std::cout << " front_m=" << last.front_interpolated;
    }
    if(last.subgrid_statistics_valid) {
        std::cout << " nu_t_mean_m2_s="
                  << last.subgrid_mean_nu_t_m2_s
                  << " nu_t_max_m2_s="
                  << last.subgrid_max_nu_t_m2_s
                  << " mu_t_mean_Pa_s="
                  << last.subgrid_mean_mu_t_pa_s
                  << " mu_t_max_Pa_s="
                  << last.subgrid_max_mu_t_pa_s
                  << " sgs_cells=" << last.subgrid_count;
    }
    std::cout << '\n';
    if(last.max_mach>0.3) {
        std::cout
            << "GEA_WARNING max Mach exceeds 0.3; weak-compressibility "
               "error is high.\n";
    } else if(last.max_mach>0.1) {
        std::cout
            << "GEA_WARNING max Mach exceeds 0.1; include dt sensitivity "
               "when interpreting the result.\n";
    }
    return 0;
}

} // namespace

int main(const int argc, const char* const argv[]) {
    try {
        const CUresult initialization = cuInit(0u);
        if(initialization!=CUDA_SUCCESS) {
            std::cerr << "CUDA driver initialization failed.\n";
            return 77;
        }
        int device_count = 0;
        if(cuDeviceGetCount(&device_count)!=CUDA_SUCCESS ||
           device_count<=0) {
            std::cerr << "No CUDA device is available.\n";
            return 77;
        }
        return run(parse_arguments(argc, argv));
    } catch(const std::exception& error) {
        std::cerr << "GEA_ERROR " << error.what() << '\n';
        return 2;
    }
}
