#pragma once

// Runtime primitives for vertically nested CUDA LBM blocks.  This layer is
// intentionally independent of setup.cpp: callers construct the individual
// uniform LBM instances, then hand their non-owning pointers to the scheduler.
// That keeps the ordinary solver path unchanged while providing one execution
// surface for normal, dataset, and profile runs.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

class LBM;

namespace culbm {

enum class NestedPlaneSide : std::uint8_t {
    Bottom,
    Top
};

enum class NestedPlaneLocation : std::uint8_t {
    // z=0 / z=Nz-1: non-physical interface boundary rows.
    Boundary,
    // z=1 / z=Nz-2: the adjacent interior rows that donate aperture and
    // macro data to the interface bridge.
    AdjacentInterior
};

// Host representation of one horizontal native-grid plane.  aperture_mask is
// empty for a native plane.  After restrict_4_to_1(), bit q records whether
// fine quadrant q (q = x + 2*y) is fluid; it lets a future link-wise DDF
// bridge retain partial-solid information instead of collapsing it to one
// coarse TYPE_S/TYPE_F flag.
struct NestedMacroPlane {
    std::uint32_t nx = 0u;
    std::uint32_t ny = 0u;
    std::vector<float> rho;
    std::vector<float> ux;
    std::vector<float> uy;
    std::vector<float> uz;
    std::vector<float> temperature;
    std::vector<unsigned char> flags;
    std::vector<std::uint8_t> aperture_mask;

    std::uint64_t size() const;
    bool valid() const;
};

struct NestedLbmLevel {
    LBM* lbm = nullptr;                 // non-owning; lifetime belongs to setup
    std::uint32_t refinement_to_parent = 1u; // 1 for coarsest, then 2 per interface
    double dt_si = 0.0;                 // physical seconds per native LBM step
    std::string label;

    // These are copied from VerticalNestingLevelPlan.  allocated_nz belongs
    // to LBM, whereas physical_end_z is exclusive and deliberately excludes
    // interface ghost rows from SI geometry/output coordinates.
    std::uint32_t lower_ghost_z = 0u;
    std::uint32_t upper_ghost_z = 0u;
    std::uint32_t physical_first_z = 0u;
    std::uint32_t physical_end_z = 0u;
};

struct NestedPlaneWriteOptions {
    // Macro exchange must not replace geometry flags by default.  A later
    // link-wise DDF bridge owns aperture/solid application explicitly.
    bool write_flags = false;
};

// This is a deterministic, deliberately simple 4:1 macro bridge.  It is a
// staging point for the full DDF/link-wise interface; it is not advertised as
// a conservative AMR collision operator by itself.
class NestedMacroPlaneBridge {
public:
    static NestedMacroPlane read(
        NestedLbmLevel& level,
        NestedPlaneSide side,
        NestedPlaneLocation location = NestedPlaneLocation::AdjacentInterior);
    static void write(NestedLbmLevel& level,
                      NestedPlaneSide side,
                      const NestedMacroPlane& plane,
                      NestedPlaneWriteOptions options = {},
                      NestedPlaneLocation location = NestedPlaneLocation::Boundary);

    static NestedMacroPlane restrict_4_to_1(const NestedMacroPlane& fine);
    static NestedMacroPlane prolong_1_to_4(const NestedMacroPlane& coarse);

    // `upper` is the coarser block immediately above `lower`.  The function
    // captures both native planes before either is written, avoiding a
    // direction-order dependency.  It uses only arithmetic 4:1 means and
    // constant replication, never a higher-order spatial interpolation.
    // It also prepares z=0/Nz-1 flags from z=1/Nz-2 aperture masks: open
    // cells are TYPE_E|TYPE_T in thermal builds (TYPE_E otherwise), and
    // closed cells are adiabatic TYPE_S.
    static void exchange(NestedLbmLevel& upper,
                         NestedLbmLevel& lower,
                         NestedPlaneWriteOptions options = {});
};

enum class NestedExchangePhase : std::uint8_t {
    Initial,
    BeforeFinerStep,
    AfterFinerStep
};

using NestedLevelStepHook =
    std::function<void(NestedLbmLevel&, double time_si, std::uint64_t native_step)>;
using NestedInterfaceHook =
    std::function<void(NestedLbmLevel& upper,
                       NestedLbmLevel& lower,
                       NestedExchangePhase phase)>;

// Executes blocks ordered coarse -> fine.  The default recursion advances the
// finer block(s) first while holding the parent state; this is a first-order
// time policy with no hidden temporal interpolation.  Replacing the interface
// hook later does not require changing the RunDriver or setup call sites.
class NestedLbmScheduler {
public:
    explicit NestedLbmScheduler(std::vector<NestedLbmLevel> levels);

    void set_level_before_step_hook(NestedLevelStepHook hook);
    void set_level_after_step_hook(NestedLevelStepHook hook);
    void set_interface_hook(NestedInterfaceHook hook);

    void initialize();
    void advance_one_coarse_step();
    void finish();

    bool initialized() const { return initialized_; }
    std::uint64_t coarse_step_count() const { return coarse_step_count_; }
    double physical_time_si() const { return physical_time_si_; }
    const std::vector<NestedLbmLevel>& levels() const { return levels_; }
    std::vector<NestedLbmLevel>& levels() { return levels_; }

private:
    void validate_or_fail() const;
    void advance_recursive(std::size_t level_index);
    void call_interface(std::size_t upper_index, NestedExchangePhase phase);

    std::vector<NestedLbmLevel> levels_;
    std::vector<double> level_time_si_;
    NestedLevelStepHook before_step_hook_;
    NestedLevelStepHook after_step_hook_;
    NestedInterfaceHook interface_hook_;
    bool initialized_ = false;
    std::uint64_t coarse_step_count_ = 0u;
    double physical_time_si_ = 0.0;
};

// Small adapter used by setup's existing solve loop.  The caller retains the
// current progress/averaging/output code and replaces only its native-step
// action with run_one()/run().
class NestedRunDriver {
public:
    explicit NestedRunDriver(NestedLbmScheduler& scheduler)
        : scheduler_(scheduler) {}

    void initialize();
    void run_one();
    void run(std::uint64_t coarse_steps,
             const std::function<void(NestedLbmScheduler&)>& after_step = {});

private:
    NestedLbmScheduler& scheduler_;
};

} // namespace culbm
