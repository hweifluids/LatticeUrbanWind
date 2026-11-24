#include "case_runner.hpp"
#include "console_log.hpp"
#include "nested_run.hpp"
#include "uniform_run.hpp"
#include "../bc/inflow/boundaries.hpp"
#include "../bc/inflow/vk_inlet.hpp"
#include "../bc/outflow/damping_layers.hpp"
#include "../case/case_dump.hpp"
#include "../case/config_dump.hpp"
#include "../case/context.hpp"
#include "../case/features.hpp"
#include "../case/report.hpp"

#include <cstring>
#include <memory>
#include <vector>

namespace culbm::run {

using namespace culbm::bc;
using namespace culbm::cases;

namespace {

void run_normal_case(CaseContext& ctx) {
    CaseSettings& deck = ctx.deck;
    const CaseDomain& domain = ctx.domain;
    InflowData& inflow = ctx.inflow;
    const CasePhysics& physics = ctx.physics;
    const CaseGeometry& geometry = ctx.geometry;
    if (ConfigDump::enabled()) {
        ConfigDump& dump = ConfigDump::get();
        dump.section("case");
        dump.put("nested", domain.cuda_vertical_nesting_plan.enabled);
        dump.put("downstream_bc", deck.downstream_bc);
        dump_case_physics(ctx, dump);
        dump_case_schedule(ctx, dump, deck.downstream_bc);
        return;
    }
    Clock case_preprocess_clock;
    print_section_title("DEVICE INFORMATION");
    const Thermal_Lattice_Scope thermal_scope(inflow.use_temperature_bc);
    const bool use_cuda_vertical_nesting = domain.cuda_vertical_nesting_plan.enabled;
    culbm::NestedSeedState nested_seed;
    double nested_uniform_preprocess_seconds = 0.0;
    {
    const culbm::surface_ibm::ScopedConfig seed_scope(
        use_cuda_vertical_nesting ? culbm::surface_ibm::Config{} : deck.surface_ibm_settings);
    culbm::solver::Solver solver(domain.lbm_N.x, domain.lbm_N.y, domain.lbm_N.z, deck.Dx, deck.Dy, deck.Dz,
        culbm::solver::CaseCoefficients().viscosity(physics.lbm_nu).thermal(physics.lbm_alpha, physics.lbm_beta));
    solver.set_rotation(coriolis_Omegax_lbmu, coriolis_Omegay_lbmu, coriolis_Omegaz_lbmu);
    solver.set_gravity(0.0f, 0.0f, physics.lbm_buoyancy_gz);
    configure_temperature_outlet(solver, deck, deck.downstream_bc);

    const float3 origin_lbmu = solver.grid().position(0u, 0u, 0u);   // (0.5,0.5,0.5)
    for (auto& sp : inflow.samples) {
        sp.p.x += origin_lbmu.x;
        sp.p.y += origin_lbmu.y;
        sp.p.z += origin_lbmu.z;
    }
    const float z0_lbmu = origin_lbmu.z;

    solver.voxelize(*geometry.mesh);
    println("| Voxelization done.                                                          |");
    print_section_title("BUILD BOUNDARY CONDITIONS");
    build_normal_boundaries(ctx, solver, z0_lbmu);

    std::unique_ptr<VonKarmanInletUpdater> vk_updater;
    if (deck.vk_inlet_settings.enable && !use_cuda_vertical_nesting) {
        VkInletRuntimeConfig vk_runtime = make_vk_runtime_config(deck.vk_inlet_settings, deck.downstream_bc);
        vk_updater = std::make_unique<VonKarmanInletUpdater>(vk_runtime);
        if (!vk_updater->initialize(solver)) {
            vk_updater.reset();
            println("| VK inlet        | requested but not activated for this case                  |");
        }
    }

    const double case_preprocess_seconds = case_preprocess_clock.stop();
    if (use_cuda_vertical_nesting) {
        nested_seed = culbm::capture_nested_seed_state(solver);
        nested_uniform_preprocess_seconds = case_preprocess_seconds;
    }
    else {
        std::function<void(culbm::solver::Solver&, ulong)> vk_pre_step_update;
        if (vk_updater) {
            vk_pre_step_update =
                [&](culbm::solver::Solver& solver_ref, const ulong t_now) {
                    vk_updater->update(solver_ref, t_now);
                };
        }
        UniformRun(ctx, solver, "", false, case_preprocess_seconds, vk_pre_step_update).run();
    }
    }
    if (use_cuda_vertical_nesting) {
        NestedRun(ctx, nested_seed, "", false, nested_uniform_preprocess_seconds, nullptr, deck.downstream_bc).run();
    }
}

void run_dataset_cases(CaseContext& ctx) {
    CaseSettings& deck = ctx.deck;
    const CaseDomain& domain = ctx.domain;
    const ThermalReference& thermal = ctx.thermal;
    CasePhysics& physics = ctx.physics;
    const CaseGeometry& geometry = ctx.geometry;
    if (deck.angle_list.empty()) {
        println("| ERROR: dataset generation requires angle list (angle=[...]).                |");
        println("|-----------------------------------------------------------------------------|");
        wait();
        exit(-1);
    }

    const uint total_cases = static_cast<uint>(deck.inflow_list.size() * deck.angle_list.size());
    uint case_index = 0u;
    Clock batch_lbm_clock;
    double batch_lbm_case_seconds_sum = 0.0;
    std::vector<std::vector<uchar>> cached_voxel_flags;

    for (const float inflow_si : deck.inflow_list) {
        for (const float angle_deg : deck.angle_list) {
            ++case_index;
            Clock case_preprocess_clock;
            const uint remaining = total_cases - case_index;

            physics.si_ref_u = inflow_si;
            physics.u_scale = domain.lbm_ref_u / physics.si_ref_u;
            units.set_m_kg_s_K((float)domain.lbm_N.y, domain.lbm_ref_u, 1.0f, 1.0f, deck.si_size.y, physics.si_ref_u, domain.si_rho, thermal.temperature_scale_kelvin);
            units.set_temperature_reference(1.0f, thermal.temperature_ref_kelvin);
            physics.z_off = units.x(deck.z_si_offset) + 0.5f;
            physics.lbm_beta = deck.buoyancy_enabled ? units.beta(physics.si_beta_air) : 0.0f;
            physics.lbm_buoyancy_gz = deck.buoyancy_enabled ? units.g(-physics.si_gravity) : 0.0f;
            update_coriolis(ctx, false);

            const std::string case_label = to_string(case_index) + "/" + to_string(total_cases) +
                " (remaining " + to_string(remaining) + ")";
            println("|-----------------------------------------------------------------------------|");
            println("| Dataset case    | " + alignr(57u, case_label) + " |");
            println("| Inflow / Angle  | " + alignr(57u, format_tag(inflow_si) + " m/s, " + format_tag(angle_deg) + " deg") + " |");
            println("| SI Reference U  | " + alignr(57u, format_tag(physics.si_ref_u) + " m/s") + " |");
            resolve_base_transport(ctx, "dataset case " + case_label, true);

            const float3 inflow_lbmu = wind_velocity_lbmu(inflow_si, angle_deg, physics.u_scale);
            const std::string case_downstream_bc = profile_downstream_bc_from_dir(inflow_lbmu.x, inflow_lbmu.y);
            update_buffer_nudging(ctx, false, case_downstream_bc);
            update_top_sponge(ctx, false);
            if (ConfigDump::enabled()) {
                ConfigDump& dump = ConfigDump::get();
                dump.section("case." + to_string(case_index));
                dump.put("inflow_m_s", inflow_si);
                dump.put("angle_deg", angle_deg);
                dump.put("inflow_lbm", inflow_lbmu);
                dump.put("downstream_bc", case_downstream_bc);
                dump.put("vtk_prefix", "DG_" + format_tag(inflow_si) + "_" + format_tag(angle_deg) + "_");
                dump_case_physics(ctx, dump);
                dump_case_schedule(ctx, dump, case_downstream_bc);
                continue;
            }

            print_section_title("DEVICE INFORMATION");
            print_kv_row("Downstream BC", case_downstream_bc + " (auto from batch angle)");
            const Thermal_Lattice_Scope thermal_scope(false);
            const bool use_cuda_vertical_nesting = domain.cuda_vertical_nesting_plan.enabled;
            culbm::NestedSeedState nested_seed;
            double nested_uniform_preprocess_seconds = 0.0;
            double case_total_seconds = 0.0;
            const std::string vtk_prefix = "DG_" + format_tag(inflow_si) + "_" + format_tag(angle_deg) + "_";
            {
            const culbm::surface_ibm::ScopedConfig seed_scope(
                use_cuda_vertical_nesting ? culbm::surface_ibm::Config{} : deck.surface_ibm_settings);
            culbm::solver::Solver solver(domain.lbm_N.x, domain.lbm_N.y, domain.lbm_N.z, deck.Dx, deck.Dy, deck.Dz,
                culbm::solver::CaseCoefficients().viscosity(physics.lbm_nu).thermal(physics.lbm_alpha, physics.lbm_beta));
            solver.set_rotation(coriolis_Omegax_lbmu, coriolis_Omegay_lbmu, coriolis_Omegaz_lbmu);
            solver.set_gravity(0.0f, 0.0f, physics.lbm_buoyancy_gz);
            configure_temperature_outlet(solver, deck, case_downstream_bc);

            bool reused_voxel = false;
            if (cached_voxel_flags.size() == (size_t)solver.local_count()) {
                reused_voxel = true;
                for (uint d = 0u; d < solver.local_count(); ++d) {
                    if (cached_voxel_flags[d].size() != (size_t)solver.local(d).fields().flags.count()) {
                        reused_voxel = false;
                        break;
                    }
                }
                if (reused_voxel) {
                    for (uint d = 0u; d < solver.local_count(); ++d) {
                        auto& fl = solver.local(d).fields().flags;
                        std::memcpy(fl.data(), cached_voxel_flags[d].data(), cached_voxel_flags[d].size());
                        fl.upload_async();
                    }
                    solver.synchronize();
                    println("| Voxelization    | reused cached flags from case 1                           |");
                } else {
                    cached_voxel_flags.clear();
                }
            }
            if (!reused_voxel) {
                solver.voxelize(*geometry.mesh);
                if (total_cases > 1u) {
                    cached_voxel_flags.resize(solver.local_count());
                    for (uint d = 0u; d < solver.local_count(); ++d) {
                        const auto& fl = solver.local(d).fields().flags;
                        cached_voxel_flags[d].assign(fl.data(), fl.data() + fl.count());
                    }
                }
            }
            println("| Voxelization done.                                                          |");
            print_section_title("BUILD BOUNDARY CONDITIONS");
            initialize_uniform_velocity(solver, inflow_lbmu);
            apply_velocity_boundaries(solver, inflow_lbmu, case_downstream_bc, deck.downstream_open_face);
            print_kv_row("Boundary init", "complete. Time: [" + now_str() + "]");

            std::unique_ptr<VonKarmanInletUpdater> vk_updater;
            if (deck.vk_inlet_settings.enable && !use_cuda_vertical_nesting) {
                VkInletRuntimeConfig vk_runtime = make_vk_runtime_config(deck.vk_inlet_settings, case_downstream_bc);
                vk_updater = std::make_unique<VonKarmanInletUpdater>(vk_runtime);
                if (!vk_updater->initialize(solver)) {
                    vk_updater.reset();
                    println("| VK inlet        | dataset case: no valid inflow faces.                       |");
                }
            }

            const double case_preprocess_seconds = case_preprocess_clock.stop();
            if (use_cuda_vertical_nesting) {
                nested_seed = culbm::capture_nested_seed_state(solver);
                nested_uniform_preprocess_seconds = case_preprocess_seconds;
            }
            else {
                std::function<void(culbm::solver::Solver&, ulong)> vk_pre_step_update;
                if (vk_updater) {
                    vk_pre_step_update =
                        [&](culbm::solver::Solver& solver_ref, const ulong t_now) {
                            vk_updater->update(solver_ref, t_now);
                        };
                }
                case_total_seconds = UniformRun(ctx, solver, vtk_prefix, true, case_preprocess_seconds, vk_pre_step_update).run();
            }
            }
            if (use_cuda_vertical_nesting) {
                case_total_seconds = NestedRun(ctx, nested_seed, vtk_prefix, true, nested_uniform_preprocess_seconds, nullptr, case_downstream_bc).run();
            }
            batch_lbm_case_seconds_sum += case_total_seconds;
        }
    }
    if (total_cases > 1u) {
        const double batch_lbm_seconds = batch_lbm_clock.stop();
        print_section_title("BATCH LBM SUMMARY");
        print_kv_row("LBM cases", to_string(total_cases));
        print_kv_row("Sum case time", print_time(batch_lbm_case_seconds_sum));
        print_kv_row("Batch wall time", print_time(batch_lbm_seconds) + " (recorded)");
    }
}

void run_profile_cases(CaseContext& ctx) {
    CaseSettings& deck = ctx.deck;
    const CaseDomain& domain = ctx.domain;
    InflowData& inflow = ctx.inflow;
    CasePhysics& physics = ctx.physics;
    const CaseGeometry& geometry = ctx.geometry;
    if (deck.angle_list.empty()) {
        println("| ERROR: profile forcing requires angle list (angle=[...]).                   |");
        println("|-----------------------------------------------------------------------------|");
        wait();
        exit(-1);
    }
    if (inflow.profile_z_samples_si.empty() || inflow.profile_u_samples_si.empty()) {
        println("| ERROR: profile forcing requires profile.dat samples to be initialized.      |");
        println("|-----------------------------------------------------------------------------|");
        wait();
        exit(-1);
    }
    bc::ProfileInflow profile;
    prepare_profile_inflow(ctx, profile);

    const uint total_cases = static_cast<uint>(deck.angle_list.size());
    uint case_index = 0u;
    Clock batch_lbm_clock;
    double batch_lbm_case_seconds_sum = 0.0;
    std::vector<std::vector<uchar>> cached_voxel_flags;
    const bool profile_single_case_standard_output = (total_cases == 1u);
    if (profile_single_case_standard_output) {
        println("| Output logic    | standard luw naming: <datetime>_raw_* and <datetime>_avg  |");
    }
    else {
        println("| Output logic    | multi-angle mode keeps ANG_<angle>_ prefix to avoid overwrite |");
        println("|                 | each case still uses same run_nstep/purge_avg workflow      |");
    }

    dump_profile_stage(profile, inflow.profile_u_si, total_cases, profile_single_case_standard_output);
    for (const float angle_deg : deck.angle_list) {
        ++case_index;
        Clock case_preprocess_clock;
        const uint remaining = total_cases - case_index;

        const std::string case_label = to_string(case_index) + "/" + to_string(total_cases) +
            " (remaining " + to_string(remaining) + ")";
        println("|-----------------------------------------------------------------------------|");
        println("| Profile case    | " + alignr(57u, case_label) + " |");
        println("| Angle           | " + alignr(57u, format_tag(angle_deg) + " deg") + " |");
        println("| SI Reference U  | " + alignr(57u, format_tag(physics.si_ref_u) + " m/s") + " |");

        const float deg2rad = 3.14159265358979323846f / 180.0f;
        const float angle_rad = angle_deg * deg2rad;
        const float dir_x = -sinf(angle_rad);
        const float dir_y = -cosf(angle_rad);
        const std::string case_downstream_bc = profile_downstream_bc_from_dir(dir_x, dir_y);

        print_section_title("DEVICE INFORMATION");
        print_kv_row("Downstream BC", case_downstream_bc + " (auto from profile angle)");
        update_buffer_nudging(ctx, false, case_downstream_bc);
        if (ConfigDump::enabled()) {
            ConfigDump& dump = ConfigDump::get();
            dump.section("case." + to_string(case_index));
            dump.put("angle_deg", angle_deg);
            dump.put("direction", float3(dir_x, dir_y, 0.0f));
            dump.put("downstream_bc", case_downstream_bc);
            dump.put("vtk_prefix", profile_single_case_standard_output
                ? std::string("")
                : ("ANG_" + format_tag(angle_deg) + "_"));
            dump_case_physics(ctx, dump);
            dump_case_schedule(ctx, dump, case_downstream_bc);
            continue;
        }
        const Thermal_Lattice_Scope thermal_scope(false);
        const bool use_cuda_vertical_nesting = domain.cuda_vertical_nesting_plan.enabled;
        culbm::NestedSeedState nested_seed;
        culbm::NestedCommonFinestTerrain nested_profile_terrain;
        bool nested_profile_terrain_active = false;
        double nested_uniform_preprocess_seconds = 0.0;
        double case_total_seconds = 0.0;
        const std::string vtk_prefix = profile_single_case_standard_output
            ? std::string("")
            : ("ANG_" + format_tag(angle_deg) + "_");
        {
        const culbm::surface_ibm::ScopedConfig seed_scope(
            use_cuda_vertical_nesting ? culbm::surface_ibm::Config{} : deck.surface_ibm_settings);
        culbm::solver::Solver solver(domain.lbm_N.x, domain.lbm_N.y, domain.lbm_N.z, deck.Dx, deck.Dy, deck.Dz,
            culbm::solver::CaseCoefficients().viscosity(physics.lbm_nu).thermal(physics.lbm_alpha, physics.lbm_beta));
        solver.set_rotation(coriolis_Omegax_lbmu, coriolis_Omegay_lbmu, coriolis_Omegaz_lbmu);
        solver.set_gravity(0.0f, 0.0f, physics.lbm_buoyancy_gz);
        configure_temperature_outlet(solver, deck, case_downstream_bc);

        bool reused_voxel = false;
        if (cached_voxel_flags.size() == (size_t)solver.local_count()) {
            reused_voxel = true;
            for (uint d = 0u; d < solver.local_count(); ++d) {
                if (cached_voxel_flags[d].size() != (size_t)solver.local(d).fields().flags.count()) {
                    reused_voxel = false;
                    break;
                }
            }
            if (reused_voxel) {
                for (uint d = 0u; d < solver.local_count(); ++d) {
                    auto& fl = solver.local(d).fields().flags;
                    std::memcpy(fl.data(), cached_voxel_flags[d].data(), cached_voxel_flags[d].size());
                    fl.upload_async();
                }
                solver.synchronize();
                println("| Voxelization    | reused cached flags from case 1                           |");
            } else {
                cached_voxel_flags.clear();
            }
        }
        if (!reused_voxel) {
            solver.voxelize(*geometry.mesh);
            if (total_cases > 1u) {
                cached_voxel_flags.resize(solver.local_count());
                for (uint d = 0u; d < solver.local_count(); ++d) {
                    const auto& fl = solver.local(d).fields().flags;
                    cached_voxel_flags[d].assign(fl.data(), fl.data() + fl.count());
                }
            }
        }
        println("| Voxelization done.                                                          |");
        print_section_title("BUILD BOUNDARY CONDITIONS");

        build_profile_case_boundaries(ctx, profile, solver, dir_x, dir_y, case_downstream_bc,
                                      use_cuda_vertical_nesting, nested_profile_terrain,
                                      nested_profile_terrain_active);

        std::unique_ptr<VonKarmanInletUpdater> vk_updater;
        if (deck.vk_inlet_settings.enable && !use_cuda_vertical_nesting) {
            VkInletRuntimeConfig vk_runtime = make_vk_runtime_config(deck.vk_inlet_settings, case_downstream_bc);
            vk_updater = std::make_unique<VonKarmanInletUpdater>(vk_runtime);
            if (!vk_updater->initialize(solver)) {
                vk_updater.reset();
                println("| VK inlet        | profile case: no valid inflow faces.                       |");
            }
        }

        const double case_preprocess_seconds = case_preprocess_clock.stop();
        if (use_cuda_vertical_nesting) {
            nested_seed = culbm::capture_nested_seed_state(solver);
            nested_uniform_preprocess_seconds = case_preprocess_seconds;
        }
        else {
            std::function<void(culbm::solver::Solver&, ulong)> vk_pre_step_update;
            if (vk_updater) {
                vk_pre_step_update =
                    [&](culbm::solver::Solver& solver_ref, const ulong t_now) {
                        vk_updater->update(solver_ref, t_now);
                    };
            }
            case_total_seconds = UniformRun(ctx, solver, vtk_prefix, true, case_preprocess_seconds, vk_pre_step_update).run();
        }
        }
        if (use_cuda_vertical_nesting) {
            case_total_seconds = NestedRun(ctx, nested_seed, vtk_prefix, true, nested_uniform_preprocess_seconds, nested_profile_terrain_active ? &nested_profile_terrain : nullptr, case_downstream_bc).run();
        }
        batch_lbm_case_seconds_sum += case_total_seconds;
    }
    if (total_cases > 1u) {
        const double batch_lbm_seconds = batch_lbm_clock.stop();
        print_section_title("BATCH LBM SUMMARY");
        print_kv_row("LBM cases", to_string(total_cases));
        print_kv_row("Sum case time", print_time(batch_lbm_case_seconds_sum));
        print_kv_row("Batch wall time", print_time(batch_lbm_seconds) + " (recorded)");
    }
}

} // namespace

void run_deck_cases() {
    print_setup_banner();
    CaseContext ctx;
    read_case_settings(ctx.deck);
    apply_model_selection(ctx.deck);
    plan_probes(ctx);
    finish_profile_settings(ctx.deck);
    dump_settings_stage(ctx);
    open_case_log(ctx.deck);
    confirm_validation(ctx.deck);
    print_parameter_information(ctx);
    derive_domain(ctx);
    dump_domain_stage(ctx);
    print_domain_report(ctx);
    load_inflow(ctx);
    resolve_thermal_coupling_stage(ctx);
    set_initial_units(ctx);
    resolve_base_transport(ctx, ctx.deck.profile_generation ? "profile units" : "initial units",
                           ctx.deck.profile_generation);
    set_buoyancy_coefficients(ctx);
    print_reference_report(ctx);
    update_coriolis(ctx, !ctx.deck.dataset_generation && !ctx.deck.profile_generation);
    update_buffer_nudging(ctx, !ctx.deck.profile_generation, ctx.deck.downstream_bc);
    if (ctx.deck.profile_generation) print_profile_buffer_nudging(ctx);
    update_top_sponge(ctx, true);
    convert_samples_to_lattice(ctx);
    dump_inflow_stage(ctx);
    load_case_geometry(ctx);
    build_surface_ibm_markers(ctx);
    print_geometry_summary(ctx);
    load_terrain(ctx);
    dump_geometry_stage(ctx);

    if (!ctx.deck.dataset_generation && !ctx.deck.profile_generation) run_normal_case(ctx);
    else if (ctx.deck.dataset_generation) run_dataset_cases(ctx);
    else run_profile_cases(ctx);
}

} // namespace culbm::run
