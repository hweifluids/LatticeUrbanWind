#include "damping_layers.hpp"
#include "../patch_fields.hpp"
#include "../../case/context.hpp"
#include "../../case/report.hpp"

bool buffer_nudging_active = false;
int buffer_n_cells = 1;
int buffer_downstream_face_id = 0;
float buffer_inv_tau_lbmu = 0.0f;
int buffer_nudge_vertical = 0;
bool top_sponge_active = false;
int sponge_n_cells = 1;
float sponge_inv_tau_lbmu = 0.0f;
int sponge_ref_mode = 0;

namespace culbm::bc {

using namespace culbm::cases;

void update_buffer_nudging(CaseContext& ctx, const bool show_info, const std::string& downstream_bc_runtime) {
    const CaseSettings& deck = ctx.deck;
    const CaseDomain& domain = ctx.domain;
    const CasePhysics& physics = ctx.physics;
    buffer_nudge_vertical = deck.buffer_nudge_vertical;
    buffer_downstream_face_id = buffer_face_id_from_bc(downstream_bc_runtime);

    const uint min_dim = std::min(domain.lbm_N.x, std::min(domain.lbm_N.y, domain.lbm_N.z));
    const uint max_nbuf = std::max(1u, min_dim / 4u);
    int nbuf = (int)std::lround(deck.buffer_thickness_m / deck.cell_m);
    if (nbuf < 1) nbuf = 1;
    if ((uint)nbuf > max_nbuf) nbuf = (int)max_nbuf;
    buffer_n_cells = nbuf;

    const float dt_si = deck.cell_m * (domain.lbm_ref_u / physics.si_ref_u);
    buffer_inv_tau_lbmu = (deck.buffer_tau_s > 0.0f) ? (dt_si / deck.buffer_tau_s) : 0.0f;
    buffer_nudging_active = deck.enable_buffer_nudging && deck.buffer_tau_s > 0.0f;

    if (show_info) {
        if (deck.enable_buffer_nudging && deck.buffer_tau_s <= 0.0f) {
            println("| WARNING: enable_buffer_nudging=1 but buffer_tau_s<=0. Buffer nudging disabled. |");
        }
        print_kv_row("Buffer nudging", buffer_nudging_active ? "enabled" : "disabled");
        print_kv_row("", "Nbuf=" + to_string((ulong)buffer_n_cells) + " cells, tau_s=" + to_string(deck.buffer_tau_s, 6u) + " s");
        print_kv_row("", "inv_tau_lbmu=" + to_string(buffer_inv_tau_lbmu, 8u) + ", downstream_face_id=" + to_string((ulong)buffer_downstream_face_id) + ", nudge_vertical=" + to_string((ulong)buffer_nudge_vertical));
    }
}

void update_top_sponge(CaseContext& ctx, const bool show_info) {
    const CaseSettings& deck = ctx.deck;
    const CaseDomain& domain = ctx.domain;
    const CasePhysics& physics = ctx.physics;
    sponge_ref_mode = deck.sponge_ref_mode;
    int ns = domain.sponge_cells_cfg;
    if (ns < 1) ns = 1;
    if (domain.lbm_N.z > 2u) {
        const int ns_max = (int)domain.lbm_N.z - 2;
        if (ns > ns_max) ns = ns_max;
    }
    sponge_n_cells = ns;

    const float dt_si = deck.cell_m * (domain.lbm_ref_u / physics.si_ref_u);
    sponge_inv_tau_lbmu = (deck.sponge_tau_s > 0.0f) ? (dt_si / deck.sponge_tau_s) : 0.0f;

    const bool mode_supported = (sponge_ref_mode == 0);
    const bool has_vertical_room = (domain.Nz_cells_core > 2u);
    top_sponge_active = domain.top_sponge_grid_extend && deck.sponge_tau_s > 0.0f && mode_supported && has_vertical_room;

    if (show_info) {
        if (deck.enable_top_sponge && deck.sponge_tau_s <= 0.0f) {
            println("| WARNING: enable_top_sponge=1 but sponge_tau_s<=0. Top sponge disabled.      |");
        }
        if (deck.enable_top_sponge && !has_vertical_room) {
            println("| WARNING: enable_top_sponge=1 but Nz<=2. Top sponge disabled.                 |");
        }
        if (deck.enable_top_sponge && !mode_supported) {
            println("| WARNING: sponge_ref_mode!=0 is not implemented yet. Top sponge disabled.    |");
        }
        if (deck.enable_top_sponge && !domain.top_sponge_grid_extend && mode_supported && has_vertical_room && deck.sponge_tau_s > 0.0f) {
            println("| WARNING: top sponge grid extension is disabled by current settings.          |");
        }
        print_kv_row("Top sponge", top_sponge_active ? "enabled" : "disabled");
        print_kv_row("", "Nsponge=" + to_string((ulong)sponge_n_cells) + " cells, tau_s=" + to_string(deck.sponge_tau_s, 6u) + " s");
        print_kv_row("", "inv_tau_lbmu=" + to_string(sponge_inv_tau_lbmu, 8u) + ", ref_mode=" + to_string(sponge_ref_mode));
        if (top_sponge_active) {
            print_kv_row("", "core_top_z=" + to_string((ulong)(domain.Nz_cells_core - 1u)) + ", side_ref_cap_z=" + to_string(domain.top_sponge_side_ref_z_cap));
        }
    }
}

void print_profile_buffer_nudging(const CaseContext& ctx) {
    const CaseSettings& deck = ctx.deck;
    if (deck.enable_buffer_nudging && deck.buffer_tau_s <= 0.0f) {
        println("| WARNING: enable_buffer_nudging=1 but buffer_tau_s<=0. Buffer nudging disabled. |");
    }
    print_kv_row("Buffer nudging", buffer_nudging_active ? "enabled (downstream face auto by angle)" : "disabled");
    print_kv_row("", "Nbuf=" + to_string((ulong)buffer_n_cells) + " cells, tau_s=" + to_string(deck.buffer_tau_s, 6u) + " s");
    print_kv_row("", "inv_tau_lbmu=" + to_string(buffer_inv_tau_lbmu, 8u) + ", downstream_face_id=auto, nudge_vertical=" + to_string((ulong)buffer_nudge_vertical));
}

void configure_temperature_outlet(culbm::solver::Solver& solver, const CaseSettings& deck, const std::string& bc) {
    const int face = deck.downstream_open_face
        ? buffer_face_id_from_bc(bc) : 0;
    if (!solver.set_temperature_outlet_face(face)) {
        println("| ERROR: scalar outlet needs at least two cells per GPU domain along its normal. |");
        wait();
        exit(-1);
    }
}

} // namespace culbm::bc
