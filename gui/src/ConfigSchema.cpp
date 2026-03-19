#include "luwgui/ConfigSchema.h"

#include <QFileInfo>

namespace luwgui {

namespace {

QVector<SectionSpec> buildSections() {
    return {
        {"project", "Project", "Case identity, file mode and working directories."},
        {"domain", "Domain", "Spatial range, base height, clipping and coordinate controls."},
        {"cfd", "CFD Controls", "Mesh, GPU allocation and core solver controls."},
        {"output", "Output & Probes", "Output cadence, probe definitions and averaging products."},
        {"physics", "Physics", "Coriolis, buoyancy, nudging and sponge settings."},
        {"vk", "VK Inlet", "Von Karman synthetic turbulence inlet parameters."},
        {"batch", "Batch Modes", "Dataset/profile batch boundary-condition controls."},
        {"generated", "Generated Info", "Derived domain information written back by the pipeline."}
    };
}

QVector<FieldSpec> buildFields() {
    return {
        {"casename", "Case name", "project", "Primary case identifier used by outputs.", FieldKind::String},
        {"datetime", "Datetime", "project", "14-digit timestamp used by LUW workflows.", FieldKind::String},

        {"cut_lon_manual", "Longitude range", "domain", "Manual clipping range in longitude.", FieldKind::FloatPair},
        {"cut_lat_manual", "Latitude range", "domain", "Manual clipping range in latitude.", FieldKind::FloatPair},
        {"si_x_cfd", "X range", "domain", "Projected X range of CFD domain in meters.", FieldKind::FloatPair},
        {"si_y_cfd", "Y range", "domain", "Projected Y range of CFD domain in meters.", FieldKind::FloatPair},
        {"si_z_cfd", "Z range", "domain", "Projected Z range of CFD domain in meters.", FieldKind::FloatPair},
        {"base_height", "Base height", "domain", "Ground/base slab thickness in meters.", FieldKind::Float},
        {"z_limit", "Z limit", "domain", "Low-altitude vertical target range in meters.", FieldKind::Float},
        {"utm_crs", "UTM CRS", "domain", "Projected coordinate reference system.", FieldKind::String, {}, ModeMaskAll, true},
        {"rotate_deg", "Rotate degree", "domain", "Rotation applied to align the CFD box.", FieldKind::Float},
        {"x_exp_rat", "X expansion ratio", "domain", "Batch STL base expansion ratio along X.", FieldKind::Float, {}, ModeMaskLuwdg | ModeMaskLuwpf},
        {"y_exp_rat", "Y expansion ratio", "domain", "Batch STL base expansion ratio along Y.", FieldKind::Float, {}, ModeMaskLuwdg | ModeMaskLuwpf},

        {"n_gpu", "GPU split", "cfd", "Triplet of GPU partition counts.", FieldKind::UIntTriplet},
        {"mesh_control", "Mesh control", "cfd", "Choose between GPU memory target and explicit cell size.", FieldKind::Enum, {"gpu_memory", "cell_size"}, ModeMaskAll, true},
        {"gpu_memory", "GPU memory (MiB)", "cfd", "Target GPU memory for automatic resolution sizing.", FieldKind::Integer},
        {"cell_size", "Cell size", "cfd", "Explicit cell size in meters when mesh_control=cell_size.", FieldKind::Float},
        {"validation", "Validation status", "cfd", "Validation flag written by prerun validation.", FieldKind::String},
        {"high_order", "High order interpolation", "cfd", "Enable higher-order interpolation.", FieldKind::Boolean},
        {"flux_correction", "Flux correction", "cfd", "Enable post-voxel flux correction.", FieldKind::Boolean},
        {"run_nstep", "Run steps override", "cfd", "Force solver run length in steps.", FieldKind::Integer},
        {"research_output", "Research output stride", "cfd", "Snapshot output cadence for research runs.", FieldKind::Integer},

        {"unsteady_output", "Unsteady output stride", "output", "Write unsteady VTK every N steps.", FieldKind::Integer},
        {"probes_output", "Probe output stride", "output", "Sampling interval for probe outputs.", FieldKind::Integer},
        {"purge_avg", "Average purge stride", "output", "Average purge cadence.", FieldKind::Integer},
        {"purge_avg_stride", "Average purge sub-stride", "output", "Subsampling stride for purge-avg.", FieldKind::Integer},
        {"output_tke_ti_tls", "Averaged scalar outputs", "output", "Comma-separated subset from tke, ti, tls.", FieldKind::TokenList},
        {"probes", "Probe definitions", "output", "Raw probe definition tokens.", FieldKind::Multiline},

        {"coriolis_term", "Coriolis term", "physics", "Enable the Coriolis source term.", FieldKind::Boolean},
        {"buoyancy", "Buoyancy", "physics", "Enable buoyancy-driven temperature coupling.", FieldKind::Boolean},
        {"enable_buffer_nudging", "Buffer nudging", "physics", "Enable outer-domain buffer nudging.", FieldKind::Boolean},
        {"buffer_thickness_m", "Buffer thickness", "physics", "Nudging thickness in meters.", FieldKind::Float},
        {"buffer_tau_s", "Buffer tau", "physics", "Nudging timescale in seconds.", FieldKind::Float},
        {"buffer_nudge_vertical", "Vertical nudging", "physics", "Include vertical velocity in buffer nudging.", FieldKind::Boolean},
        {"enable_top_sponge", "Top sponge", "physics", "Enable top sponge damping.", FieldKind::Boolean},
        {"sponge_thickness_m", "Sponge thickness", "physics", "Top sponge thickness in meters.", FieldKind::Float},
        {"sponge_tau_s", "Sponge tau", "physics", "Top sponge timescale in seconds.", FieldKind::Float},
        {"sponge_ref_mode", "Sponge reference mode", "physics", "0/mode0 or 1/geostrophic.", FieldKind::String},

        {"vk_inlet_enable", "Enable VK inlet", "vk", "Enable synthetic turbulence inlet.", FieldKind::Boolean},
        {"vk_inlet_ti", "VK turbulence intensity", "vk", "Turbulence intensity fraction.", FieldKind::Float},
        {"vk_inlet_sigma", "VK sigma", "vk", "Velocity fluctuation sigma in m/s.", FieldKind::Float},
        {"vk_inlet_l", "VK length scale", "vk", "Integral length scale in meters.", FieldKind::Float},
        {"vk_inlet_nmodes", "VK modes", "vk", "Mode count.", FieldKind::Integer},
        {"vk_inlet_seed", "VK seed", "vk", "Random seed.", FieldKind::String},
        {"vk_inlet_update_stride", "VK update stride", "vk", "Update interval in solver steps.", FieldKind::Integer},
        {"vk_inlet_uc_mode", "VK Uc mode", "vk", "NORMAL_COMPONENT or NORM_MEAN.", FieldKind::Enum, {"NORMAL_COMPONENT", "NORM_MEAN"}},
        {"vk_inlet_same_realization_all_faces", "Same realization on all faces", "vk", "Use the same random realization on every inflow face.", FieldKind::Boolean},
        {"vk_inlet_stride_interpolation", "Stride interpolation", "vk", "Enable stride interpolation.", FieldKind::Boolean},
        {"vk_inlet_inflow_only", "Inflow only", "vk", "Filter turbulence to inflow cells only.", FieldKind::Boolean},
        {"vk_inlet_anisotropy", "VK anisotropy", "vk", "Anisotropy scale triplet [x,y,z].", FieldKind::FloatTriplet},

        {"inflow", "Inflow list", "batch", "Dataset-generation inflow magnitudes in m/s.", FieldKind::FloatList, {}, ModeMaskLuwdg},
        {"angle", "Angle list", "batch", "Batch inflow direction angles in degrees.", FieldKind::FloatList, {}, ModeMaskLuwdg | ModeMaskLuwpf},

        {"um_vol", "Volume mean velocity", "generated", "Volume-mean uvw written by preprocessing.", FieldKind::FloatTriplet},
        {"um_bc", "Boundary mean velocity", "generated", "Boundary mean uvw written by preprocessing.", FieldKind::FloatTriplet},
        {"downstream_bc", "Downstream face", "generated", "Computed downstream boundary face.", FieldKind::String, {}, ModeMaskAll, true},
        {"downstream_bc_yaw", "Downstream yaw", "generated", "Computed downstream yaw angle.", FieldKind::Float},
        {"origin_shift_applied", "Origin shift applied", "generated", "Whether origin shift has been applied.", FieldKind::Boolean}
    };
}

} // namespace

QString runModeDisplayName(RunMode mode) {
    switch (mode) {
    case RunMode::Luw:
        return "LUW";
    case RunMode::Luwdg:
        return "LUWDG";
    case RunMode::Luwpf:
        return "LUWPF";
    }
    return "LUW";
}

QString runModeSuffix(RunMode mode) {
    switch (mode) {
    case RunMode::Luw:
        return ".luw";
    case RunMode::Luwdg:
        return ".luwdg";
    case RunMode::Luwpf:
        return ".luwpf";
    }
    return ".luw";
}

RunMode runModeFromPath(const QString& filePath) {
    const QString ext = QFileInfo(filePath).suffix().toLower();
    if (ext == "luwdg") {
        return RunMode::Luwdg;
    }
    if (ext == "luwpf") {
        return RunMode::Luwpf;
    }
    return RunMode::Luw;
}

QString defaultDeckName(RunMode mode) {
    return "conf" + runModeSuffix(mode);
}

const QVector<SectionSpec>& sectionSpecs() {
    static const QVector<SectionSpec> specs = buildSections();
    return specs;
}

const QVector<FieldSpec>& fieldSpecs() {
    static const QVector<FieldSpec> specs = buildFields();
    return specs;
}

const QHash<QString, FieldSpec>& fieldSpecMap() {
    static const QHash<QString, FieldSpec> map = [] {
        QHash<QString, FieldSpec> out;
        for (const FieldSpec& spec : fieldSpecs()) {
            out.insert(spec.key.toLower(), spec);
        }
        return out;
    }();
    return map;
}

const FieldSpec* findFieldSpec(const QString& key) {
    auto it = fieldSpecMap().constFind(key.toLower());
    if (it == fieldSpecMap().cend()) {
        return nullptr;
    }
    return &it.value();
}

QVector<FieldSpec> fieldsForSection(const QString& sectionId, RunMode mode) {
    const int flag = (mode == RunMode::Luw) ? ModeMaskLuw : (mode == RunMode::Luwdg ? ModeMaskLuwdg : ModeMaskLuwpf);
    QVector<FieldSpec> out;
    for (const FieldSpec& spec : fieldSpecs()) {
        if (spec.sectionId == sectionId && (spec.modeMask & flag) != 0) {
            out.push_back(spec);
        }
    }
    return out;
}

QStringList knownKeys() {
    QStringList out;
    out.reserve(fieldSpecMap().size());
    for (auto it = fieldSpecMap().cbegin(); it != fieldSpecMap().cend(); ++it) {
        out.push_back(it.key());
    }
    out.sort();
    return out;
}

}
