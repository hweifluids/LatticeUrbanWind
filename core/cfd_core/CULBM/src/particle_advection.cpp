#include "particle_advection.hpp"

#include "particle_advection_kernels.hpp"
#include "solver/lbm.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace culbm {
namespace {

using particle_advection::Config;
using particle_advection::Integrator;
using particle_advection::LifetimeSteps;
using particle_advection::Mode;
using particle_advection::ParticleFileInfo;
using particle_advection::QuantizedParticleSeed;
using particle_advection::ReleaseBatch;

constexpr ulong infinite_step = std::numeric_limits<ulong>::max();

struct HostMigration {
    ulong id = 0ull;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

bool inside_global(const std::array<double, 3u>& p,
                   const uint nx,
                   const uint ny,
                   const uint nz) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) &&
        p[0] >= -0.5 && p[0] < (double)nx - 0.5 &&
        p[1] >= -0.5 && p[1] < (double)ny - 0.5 &&
        p[2] >= -0.5 && p[2] < (double)nz - 0.5;
}

uint owner_axis(const double coordinate,
                const uint extent,
                const uint domains) {
    long long cell = (long long)std::floor(coordinate + 0.5);
    cell = std::max(0ll, std::min(cell, (long long)extent - 1ll));
    const uint owner = (uint)cell / (extent / domains);
    return std::min(owner, domains - 1u);
}

uint owner_domain(const std::array<double, 3u>& p, const LBM& lbm) {
    const uint dx = owner_axis(p[0], lbm.get_Nx(), lbm.get_Dx());
    const uint dy = owner_axis(p[1], lbm.get_Ny(), lbm.get_Dy());
    const uint dz = owner_axis(p[2], lbm.get_Nz(), lbm.get_Dz());
    return dx + (dy + dz * lbm.get_Dy()) * lbm.get_Dx();
}

ulong checked_death_step(const ulong birth, const LifetimeSteps& lifetime) {
    if(lifetime.infinite) return infinite_step;
    if(lifetime.steps > (std::uint64_t)(infinite_step - birth)) {
        throw std::runtime_error(
            "finite particle lifetime overflows the lattice-step range");
    }
    return birth + (ulong)lifetime.steps;
}

const char* status_name(const ParticleStatus status) {
    switch(status) {
    case ParticleStatus::unused: return "unused";
    case ParticleStatus::pending: return "pending";
    case ParticleStatus::active: return "active";
    case ParticleStatus::expired: return "expired";
    case ParticleStatus::escaped: return "escaped";
    case ParticleStatus::solid: return "solid";
    case ParticleStatus::invalid: return "invalid";
    }
    return "invalid";
}

const char* integrator_kernel_name(const Integrator integrator) {
    switch(integrator) {
    case Integrator::Euler: return "culbm_advection_step_euler";
    case Integrator::HeunRk2: return "culbm_advection_step_rk2";
    case Integrator::Rk4: return "culbm_advection_step_rk4";
    case Integrator::Rk6: return "culbm_advection_step_rk6";
    }
    return "culbm_advection_step_rk4";
}

std::uint64_t checked_multiply(const std::uint64_t a,
                               const std::uint64_t b,
                               const char* quantity) {
    if(a != 0u && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::runtime_error(std::string(quantity) + " overflows uint64");
    }
    return a * b;
}

uint regular_count(const double minimum,
                   const double maximum,
                   const double spacing,
                   const char axis) {
    if(!(std::isfinite(minimum) && std::isfinite(maximum) &&
         std::isfinite(spacing) && spacing > 0.0 && maximum >= minimum)) {
        throw std::runtime_error(
            std::string("invalid automatic ") + axis + " bounds/spacing");
    }
    const long double real_count =
        std::floor(((long double)maximum - (long double)minimum) /
                   (long double)spacing + 1.0e-12L) + 1.0L;
    if(real_count < 1.0L ||
       real_count > (long double)std::numeric_limits<uint>::max()) {
        throw std::runtime_error(
            std::string("automatic ") + axis + " point count is out of range");
    }
    return (uint)real_count;
}

} // namespace

std::uint64_t particle_advection_gpu_memory_bytes_per_device(
    const std::uint64_t particle_slots,
    const std::uint64_t local_lbm_cells,
    const std::uint64_t automatic_release_slots) {
    constexpr std::uint64_t limit =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t total = 0u;
    const auto add = [&](const std::uint64_t count,
                         const std::uint64_t bytes_per_element) {
        const std::uint64_t bytes =
            count != 0u && bytes_per_element > limit / count
                ? limit
                : count * bytes_per_element;
        total = bytes > limit - total ? limit : total + bytes;
    };

    add(particle_slots, 3u * sizeof(float)); // positions
    add(particle_slots, sizeof(ulong));      // birth_steps
    add(particle_slots, sizeof(ulong));      // death_steps
    add(particle_slots, sizeof(ulong));      // termination_steps
    add(particle_slots, sizeof(uchar));      // statuses
    add(local_lbm_cells, 3u * sizeof(float)); // previous_velocity
    add(1u, sizeof(uint));                   // migration_count
    add(particle_slots, sizeof(ulong));      // migration_ids
    add(particle_slots, sizeof(uint));       // migration_destinations
    add(particle_slots, 3u * sizeof(float)); // migration_positions
    add(particle_slots, sizeof(ulong));      // incoming_ids
    add(particle_slots, 3u * sizeof(float)); // incoming_positions
    add(automatic_release_slots, sizeof(ulong)); // release_steps
    return total;
}

std::uint64_t particle_advection_gpu_memory_mb_per_device(
    const std::uint64_t particle_slots,
    const std::uint64_t local_lbm_cells,
    const std::uint64_t automatic_release_slots) {
    constexpr std::uint64_t mib = 1048576u;
    constexpr std::uint64_t limit =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t total_mb = 0u;
    const auto add = [&](const std::uint64_t count,
                         const std::uint64_t bytes_per_element) {
        if(count == 0u) return;
        const std::uint64_t bytes =
            bytes_per_element > limit / count
                ? limit
                : count * bytes_per_element;
        const std::uint64_t allocation_mb =
            bytes == limit ? limit : bytes / mib + (bytes % mib != 0u);
        total_mb = allocation_mb > limit - total_mb
            ? limit
            : total_mb + allocation_mb;
    };

    add(particle_slots, 3u * sizeof(float));
    add(particle_slots, sizeof(ulong));
    add(particle_slots, sizeof(ulong));
    add(particle_slots, sizeof(ulong));
    add(particle_slots, sizeof(uchar));
    add(local_lbm_cells, 3u * sizeof(float));
    add(1u, sizeof(uint));
    add(particle_slots, sizeof(ulong));
    add(particle_slots, sizeof(uint));
    add(particle_slots, 3u * sizeof(float));
    add(particle_slots, sizeof(ulong));
    add(particle_slots, 3u * sizeof(float));
    add(automatic_release_slots, sizeof(ulong));
    return total_mb;
}

class ParticleAdvection::Impl {
public:
    struct DomainState {
        LBM_Domain* domain = nullptr;
        ulong count = 0ull;

        Memory<float> positions;
        Memory<ulong> birth_steps;
        Memory<ulong> death_steps;
        Memory<ulong> termination_steps;
        Memory<uchar> statuses;
        Memory<float> previous_velocity;

        Memory<uint> migration_count;
        Memory<ulong> migration_ids;
        Memory<uint> migration_destinations;
        Memory<float> migration_positions;

        Memory<ulong> incoming_ids;
        Memory<float> incoming_positions;
        std::unique_ptr<Memory<ulong>> release_steps;

        Kernel copy_velocity;
        Kernel initialize_regular;
        Kernel activate;
        Kernel step;
        Kernel insert;
    };

    Impl(LBM& lbm,
         Config config,
         ParticleAdvectionGeometry geometry,
         const std::uint64_t total_steps,
         std::filesystem::path case_directory,
         std::filesystem::path output_directory,
         std::string output_stem)
        : lbm_(lbm),
          config_(std::move(config)),
          geometry_(geometry),
          total_steps_(total_steps),
          case_directory_(std::move(case_directory)),
          output_directory_(std::move(output_directory)),
          output_stem_(std::move(output_stem)) {
        if(!config_.enabled) {
            throw std::runtime_error(
                "ParticleAdvection cannot be constructed while disabled");
        }
        if(lbm_.is_distributed()) {
            throw std::runtime_error(
                "CULBM particle advection does not yet support MPI-distributed runs");
        }
        if(!(std::isfinite(geometry_.spacing_m) &&
             geometry_.spacing_m > 0.0)) {
            throw std::runtime_error(
                "particle-advection lattice spacing must be finite and positive");
        }
        if(total_steps_ == 0u) {
            throw std::runtime_error(
                "particle advection requires at least one LBM step");
        }
        if(config_.max_particles == 0u ||
           config_.max_particles > (std::uint64_t)std::numeric_limits<uint>::max()) {
            throw std::runtime_error(
                "culbm_advection_max_particles must fit a CUDA uint queue");
        }
    }

    void set_seconds_per_step(const double seconds_per_step) {
        if(initialized_) {
            throw std::runtime_error(
                "particle-advection time scale cannot change after initialization");
        }
        if(!(std::isfinite(seconds_per_step) && seconds_per_step > 0.0)) {
            throw std::runtime_error(
                "particle-advection seconds per LBM step must be positive");
        }
        seconds_per_step_ = seconds_per_step;
    }

    void initialize() {
        if(initialized_) return;
        if(!(seconds_per_step_ > 0.0)) {
            throw std::runtime_error(
                "particle-advection seconds per LBM step was not configured");
        }
        std::string error;
        if(!particle_advection::lifetime_steps_floor(
                config_.lifetime_s, seconds_per_step_, lifetime_, error)) {
            throw std::runtime_error(error);
        }
        if(!lifetime_.infinite &&
           lifetime_.steps >
               std::numeric_limits<std::uint64_t>::max() - total_steps_) {
            throw std::runtime_error(
                "finite particle lifetime extends beyond the uint64 lattice-step range");
        }

        if(config_.mode == Mode::Prescribed) prepare_prescribed();
        else prepare_automatic();
        if(particle_count_ == 0ull) {
            throw std::runtime_error(
                "particle-advection plan contains no particles");
        }
        if(particle_count_ > (ulong)config_.max_particles) {
            throw std::runtime_error(
                "particle-advection plan needs " +
                std::to_string((std::uint64_t)particle_count_) +
                " slots, exceeding culbm_advection_max_particles=" +
                std::to_string(config_.max_particles));
        }

        preflight_gpu_memory();
        allocate_domains();
        lbm_.prepare_advection_fields();
        initialize_domain_data();
        for(auto& state : domains_) {
            state->copy_velocity.enqueue_run();
            state->activate.set_parameters(6u, (ulong)0u).enqueue_run();
        }
        finish_domains();
        build_output_schedule();
        initialized_ = true;
    }

    void advance(const std::uint64_t step_begin) {
        if(!initialized_) {
            throw std::runtime_error(
                "particle advection must be initialized before advancing");
        }
        if(step_begin >= total_steps_) {
            throw std::runtime_error(
                "particle-advection step is outside the configured run");
        }

        lbm_.prepare_advection_fields();
        for(auto& state : domains_) {
            if(domains_.size() > 1u) state->migration_count.reset(0u);
            state->step.set_parameters(8u, (ulong)step_begin).enqueue_run();
        }
        finish_domains();
        // A single CUDA domain can never change owners. Avoid the otherwise
        // unnecessary per-step device-to-host queue-count synchronization.
        if(domains_.size() > 1u) migrate();

        const ulong step_end = (ulong)(step_begin + 1u);
        for(auto& state : domains_) {
            state->activate.set_parameters(6u, step_end).enqueue_run();
            state->copy_velocity.enqueue_run();
        }
        finish_domains();
    }

    void refresh_velocity_before_lbm_step() {
        if(!initialized_) {
            throw std::runtime_error(
                "particle advection must be initialized before refreshing velocity");
        }
        lbm_.prepare_advection_fields();
        for(auto& state : domains_) {
            state->copy_velocity.enqueue_run();
        }
        finish_domains();
    }

    bool write_snapshot(const std::uint64_t step, const bool force) {
        if(!initialized_) return false;
        if(last_snapshot_step_.has_value() &&
           *last_snapshot_step_ == step) {
            return false;
        }
        bool due = force;
        while(next_output_ < output_steps_.size() &&
              output_steps_[next_output_] < step) {
            ++next_output_;
        }
        if(next_output_ < output_steps_.size() &&
           output_steps_[next_output_] == step) {
            due = true;
            ++next_output_;
        }
        if(!due) return false;

        struct SnapshotRow {
            bool seen = false;
            ParticleStatus status = ParticleStatus::unused;
            ulong termination = infinite_step;
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
        };
        std::vector<SnapshotRow> rows((size_t)particle_count_);

        domains_.front()->birth_steps.read_from_device();
        const ulong* const births = domains_.front()->birth_steps.data();
        for(auto& state : domains_) {
            state->statuses.read_from_device();
            state->termination_steps.read_from_device();
            state->positions.read_from_device();
            for(ulong id = 0ull; id < particle_count_; ++id) {
                const ParticleStatus status =
                    (ParticleStatus)state->statuses[id];
                if(status == ParticleStatus::unused) continue;
                SnapshotRow& row = rows[(size_t)id];
                if(row.seen) {
                    throw std::runtime_error(
                        "particle slot " + std::to_string((std::uint64_t)id) +
                        " is owned by more than one CUDA domain");
                }
                row.seen = true;
                row.status = status;
                row.termination = state->termination_steps[id];
                row.x = state->positions[id];
                row.y = state->positions[particle_count_ + id];
                row.z = state->positions[2ull * particle_count_ + id];
            }
        }

        std::filesystem::create_directories(output_directory_);
        const std::filesystem::path path =
            output_directory_ /
            (output_stem_ + "_particles_" + std::to_string(step) + ".csv");
        std::ofstream output(path, std::ios::out | std::ios::trunc);
        if(!output.is_open()) {
            throw std::runtime_error(
                "could not open particle snapshot '" + path.string() + "'");
        }
        output << "id,x_m,y_m,z_m,release_time_s,age_s,status,termination_time_s\n";
        output << std::setprecision(10);
        for(ulong id = 0ull; id < particle_count_; ++id) {
            const SnapshotRow& row = rows[(size_t)id];
            if(!row.seen) {
                throw std::runtime_error(
                    "particle slot " + std::to_string((std::uint64_t)id) +
                    " has no owning CUDA domain");
            }
            const double x_m = geometry_.lattice_origin_m[0] +
                (double)row.x * geometry_.spacing_m;
            const double y_m = geometry_.lattice_origin_m[1] +
                (double)row.y * geometry_.spacing_m;
            const double z_m = geometry_.lattice_origin_m[2] +
                (double)row.z * geometry_.spacing_m;
            const ulong birth = births[id];
            const ulong age_end =
                row.termination != infinite_step
                ? row.termination
                : (ulong)std::max<std::uint64_t>(step, (std::uint64_t)birth);
            const double age_s =
                age_end > birth
                ? (double)(age_end - birth) * seconds_per_step_
                : 0.0;
            output << (std::uint64_t)id << ','
                   << x_m << ',' << y_m << ',' << z_m << ','
                   << (double)birth * seconds_per_step_ << ','
                   << age_s << ','
                   << status_name(row.status) << ',';
            if(row.termination != infinite_step) {
                output << (double)row.termination * seconds_per_step_;
            }
            output << '\n';
        }
        output.close();
        last_snapshot_step_ = step;
        print_info("Particle snapshot \"" + path.string() + "\" saved.");
        return true;
    }

    std::uint64_t particle_count() const {
        return (std::uint64_t)particle_count_;
    }

    std::uint64_t migrated_particles() const {
        return migrated_particles_;
    }

private:
    LBM& lbm_;
    Config config_;
    ParticleAdvectionGeometry geometry_;
    std::uint64_t total_steps_ = 0u;
    std::filesystem::path case_directory_;
    std::filesystem::path output_directory_;
    std::string output_stem_;
    double seconds_per_step_ = 0.0;
    LifetimeSteps lifetime_{};
    ulong particle_count_ = 0ull;
    bool initialized_ = false;
    std::uint64_t migrated_particles_ = 0u;

    std::vector<QuantizedParticleSeed> prescribed_;
    std::vector<ulong> automatic_release_steps_;
    std::array<double, 3u> automatic_min_lattice_{};
    std::array<double, 3u> automatic_spacing_lattice_{};
    std::array<uint, 3u> automatic_counts_{};
    std::vector<std::unique_ptr<DomainState>> domains_;
    std::vector<std::uint64_t> output_steps_;
    std::size_t next_output_ = 0u;
    std::optional<std::uint64_t> last_snapshot_step_;

    void preflight_gpu_memory() const {
        constexpr std::uint64_t mib = 1048576u;
        constexpr std::uint64_t limit =
            std::numeric_limits<std::uint64_t>::max();
        std::uint64_t maximum_required_mb = 0u;
        const std::uint64_t release_slots =
            config_.mode == Mode::Automatic
                ? (std::uint64_t)automatic_release_steps_.size()
                : 0u;
        for(uint d = 0u; d < lbm_.get_D(); ++d) {
            const LBM_Domain& domain = lbm_.get_local_domain(d);
            const Device& device = domain.get_device();
            const std::uint64_t required_mb =
                particle_advection_gpu_memory_mb_per_device(
                    (std::uint64_t)particle_count_,
                    (std::uint64_t)domain.get_N(),
                    release_slots);
            const std::uint64_t required_bytes =
                required_mb > limit / mib ? limit : required_mb * mib;
            const std::uint64_t free_bytes =
                (std::uint64_t)device.get_free_memory_bytes();
            const std::uint64_t total_bytes =
                (std::uint64_t)device.get_total_memory_bytes();
            const std::uint64_t reserve_bytes =
                std::min<std::uint64_t>(256u * mib, total_bytes / 10u);
            maximum_required_mb =
                std::max(maximum_required_mb, required_mb);
            if(required_bytes > free_bytes ||
               free_bytes - required_bytes < reserve_bytes) {
                throw std::runtime_error(
                    "GPU " + std::to_string(device.info.id) +
                    " needs " + std::to_string(required_mb) +
                    " MB for independent particle advection, but only " +
                    std::to_string(free_bytes / mib) +
                    " MB is currently free and " +
                    std::to_string(reserve_bytes / mib) +
                    " MB must remain reserved; reduce the grid, particle "
                    "capacity, or other GPU usage");
            }
        }
        print_info(
            "Particle-advection VRAM preflight: up to " +
            std::to_string(maximum_required_mb) +
            " MB per GPU for " +
            std::to_string((std::uint64_t)particle_count_) +
            " particle slots.");
    }

    std::array<double, 3u> prescribed_lattice_position(
        const QuantizedParticleSeed& seed) const {
        std::array<double, 3u> position{};
        for(std::size_t axis = 0u; axis < 3u; ++axis) {
            const double lattice =
                (seed.position_m[axis] - geometry_.lattice_origin_m[axis]) /
                geometry_.spacing_m;
            // The CUDA kernels store and classify positions as float. Make
            // host ownership and boundary validation use the exact same
            // representable coordinate.
            position[axis] = (double)(float)lattice;
        }
        return position;
    }

    void prepare_prescribed() {
        std::filesystem::path path = config_.particle_file;
        if(path.is_relative()) path = case_directory_ / path;
        std::vector<particle_advection::ParticleSeed> seeds;
        ParticleFileInfo info;
        std::string error;
        if(!particle_advection::read_particle_file(
                path, config_.release_time_s, seeds, error, &info)) {
            throw std::runtime_error(error);
        }
        if(!particle_advection::quantize_particle_releases(
                seeds, seconds_per_step_, prescribed_, error)) {
            throw std::runtime_error(error);
        }
        if(info.release_time_override_applied) {
            print_info(
                "Particle release override applies to " +
                std::to_string(info.rows_using_global_release_time) +
                " row(s); per-row T values are ignored.");
        }
        particle_count_ = (ulong)prescribed_.size();
        for(const QuantizedParticleSeed& seed : prescribed_) {
            if(seed.release_step > total_steps_) {
                throw std::runtime_error(
                    path.string() + ":" + std::to_string(seed.source_line) +
                    ": particle release is after the final CULBM frame");
            }
            const std::array<double, 3u> p =
                prescribed_lattice_position(seed);
            if(!inside_global(
                    p, lbm_.get_Nx(), lbm_.get_Ny(), lbm_.get_Nz())) {
                throw std::runtime_error(
                    path.string() + ":" + std::to_string(seed.source_line) +
                    ": particle position is outside the CULBM lattice");
            }
        }
    }

    void prepare_automatic() {
        if(!config_.release_interval_s.has_value() ||
           !config_.automatic_spacing_m.has_value()) {
            throw std::runtime_error(
                "automatic particle advection requires interval and spacing");
        }
        std::vector<ReleaseBatch> schedule;
        std::string error;
        if(!particle_advection::build_automatic_release_schedule(
                config_.release_time_s.value_or(0.0),
                *config_.release_interval_s,
                seconds_per_step_,
                total_steps_,
                schedule,
                error,
                config_.max_particles)) {
            throw std::runtime_error(error);
        }
        for(const ReleaseBatch& batch : schedule) {
            for(std::uint64_t i = 0u; i < batch.multiplicity; ++i) {
                automatic_release_steps_.push_back((ulong)batch.step);
            }
        }
        if(automatic_release_steps_.empty()) {
            throw std::runtime_error(
                "automatic release schedule has no event inside this run");
        }

        std::array<double, 3u> minimum_m{};
        std::array<double, 3u> maximum_m{};
        if(config_.automatic_min_m.has_value()) {
            minimum_m = *config_.automatic_min_m;
            maximum_m = *config_.automatic_max_m;
        } else {
            const std::array<uint, 3u> extents = {
                lbm_.get_Nx(), lbm_.get_Ny(), lbm_.get_Nz()};
            for(std::size_t axis = 0u; axis < 3u; ++axis) {
                if(extents[axis] < 3u) {
                    throw std::runtime_error(
                        "automatic particle seeding needs at least 3 cells per axis");
                }
                minimum_m[axis] = geometry_.lattice_origin_m[axis] +
                    geometry_.spacing_m;
                maximum_m[axis] = geometry_.lattice_origin_m[axis] +
                    (double)(extents[axis] - 2u) * geometry_.spacing_m;
            }
        }
        const std::array<uint, 3u> extents = {
            lbm_.get_Nx(), lbm_.get_Ny(), lbm_.get_Nz()};
        for(std::size_t axis = 0u; axis < 3u; ++axis) {
            automatic_min_lattice_[axis] =
                (minimum_m[axis] - geometry_.lattice_origin_m[axis]) /
                geometry_.spacing_m;
            const double maximum_lattice =
                (maximum_m[axis] - geometry_.lattice_origin_m[axis]) /
                geometry_.spacing_m;
            const double global_min = -0.5;
            const double global_max = (double)extents[axis] - 0.5;
            if(automatic_min_lattice_[axis] < global_min ||
               automatic_min_lattice_[axis] >= global_max ||
               maximum_lattice < global_min ||
               maximum_lattice >= global_max) {
                throw std::runtime_error(
                    std::string("automatic ") + "xyz"[axis] +
                    " bounds are outside the CULBM lattice");
            }
            automatic_spacing_lattice_[axis] =
                (*config_.automatic_spacing_m)[axis] / geometry_.spacing_m;
            automatic_counts_[axis] = regular_count(
                automatic_min_lattice_[axis],
                maximum_lattice,
                automatic_spacing_lattice_[axis],
                "xyz"[axis]);
        }
        const std::uint64_t points_xy = checked_multiply(
            automatic_counts_[0], automatic_counts_[1],
            "automatic points per release");
        const std::uint64_t points = checked_multiply(
            points_xy, automatic_counts_[2],
            "automatic points per release");
        particle_count_ = (ulong)checked_multiply(
            points, automatic_release_steps_.size(),
            "automatic particle count");
    }

    void allocate_domains() {
        domains_.reserve(lbm_.get_D());
        const char* const step_kernel = integrator_kernel_name(config_.integrator);
        for(uint d = 0u; d < lbm_.get_D(); ++d) {
            LBM_Domain& domain = lbm_.get_local_domain(d);
            Device& device = domain.get_device();
            auto state = std::make_unique<DomainState>();
            state->domain = &domain;
            state->count = particle_count_;
            state->positions = Memory<float>(device, particle_count_, 3u);
            state->birth_steps = Memory<ulong>(device, particle_count_);
            state->death_steps = Memory<ulong>(device, particle_count_);
            state->termination_steps =
                Memory<ulong>(device, particle_count_, 1u, true, true, infinite_step);
            state->statuses = Memory<uchar>(device, particle_count_);
            state->previous_velocity =
                Memory<float>(device, domain.get_N(), 3u, false, true);
            state->migration_count = Memory<uint>(device, 1ull);
            state->migration_ids = Memory<ulong>(device, particle_count_);
            state->migration_destinations =
                Memory<uint>(device, particle_count_);
            state->migration_positions =
                Memory<float>(device, particle_count_, 3u);
            state->incoming_ids = Memory<ulong>(device, particle_count_);
            state->incoming_positions =
                Memory<float>(device, particle_count_, 3u);

            state->copy_velocity = Kernel(
                device, domain.u.range(),
                "culbm_advection_copy_velocity",
                state->previous_velocity, domain.u, domain.u.range());
            state->activate = Kernel(
                device, particle_count_, "culbm_advection_activate",
                state->positions, state->birth_steps, state->death_steps,
                state->termination_steps, state->statuses, domain.flags,
                (ulong)0u, particle_count_,
                lbm_.get_Nx(), lbm_.get_Ny(), lbm_.get_Nz(),
                lbm_.get_Dx(), lbm_.get_Dy(), lbm_.get_Dz(), d,
                domain.get_Nx(), domain.get_Ny(), domain.get_Nz(),
                domain.get_Ox(), domain.get_Oy(), domain.get_Oz());
            state->step = Kernel(
                device, particle_count_, step_kernel,
                state->positions, state->birth_steps, state->death_steps,
                state->termination_steps, state->statuses,
                state->previous_velocity, domain.u, domain.flags,
                (ulong)0u, state->migration_count, state->migration_ids,
                state->migration_destinations, state->migration_positions,
                particle_count_,
                lbm_.get_Nx(), lbm_.get_Ny(), lbm_.get_Nz(),
                lbm_.get_Dx(), lbm_.get_Dy(), lbm_.get_Dz(), d,
                domain.get_Nx(), domain.get_Ny(), domain.get_Nz(),
                domain.get_Ox(), domain.get_Oy(), domain.get_Oz());
            state->insert = Kernel(
                device, particle_count_, "culbm_advection_insert_migrants",
                state->incoming_ids, state->incoming_positions,
                (ulong)0u, particle_count_, state->positions,
                state->termination_steps, state->statuses);
            domains_.push_back(std::move(state));
        }
    }

    void initialize_domain_data() {
        if(config_.mode == Mode::Automatic) {
            initialize_automatic_domains();
            return;
        }
        for(uint d = 0u; d < (uint)domains_.size(); ++d) {
            DomainState& state = *domains_[d];
            std::fill(
                state.statuses.data(),
                state.statuses.data() + particle_count_,
                (uchar)ParticleStatus::unused);
            for(ulong id = 0ull; id < particle_count_; ++id) {
                const QuantizedParticleSeed& seed = prescribed_[(size_t)id];
                const std::array<double, 3u> p =
                    prescribed_lattice_position(seed);
                state.birth_steps[id] = (ulong)seed.release_step;
                state.death_steps[id] =
                    checked_death_step((ulong)seed.release_step, lifetime_);
                state.termination_steps[id] = infinite_step;
                if(owner_domain(p, lbm_) != d) continue;
                state.positions[id] = (float)p[0];
                state.positions[particle_count_ + id] = (float)p[1];
                state.positions[2ull * particle_count_ + id] = (float)p[2];
                state.statuses[id] = (uchar)ParticleStatus::pending;
            }
            state.positions.write_to_device();
            state.birth_steps.write_to_device();
            state.death_steps.write_to_device();
            state.termination_steps.write_to_device();
            state.statuses.write_to_device();
        }
    }

    void initialize_automatic_domains() {
        for(uint d = 0u; d < (uint)domains_.size(); ++d) {
            DomainState& state = *domains_[d];
            Device& device = state.domain->get_device();
            state.release_steps = std::make_unique<Memory<ulong>>(
                device, (ulong)automatic_release_steps_.size());
            std::copy(
                automatic_release_steps_.begin(),
                automatic_release_steps_.end(),
                state.release_steps->data());
            state.release_steps->write_to_device();
            state.initialize_regular = Kernel(
                device, particle_count_,
                "culbm_advection_initialize_regular",
                state.positions, state.birth_steps, state.death_steps,
                state.termination_steps, state.statuses, *state.release_steps,
                (ulong)automatic_release_steps_.size(),
                automatic_counts_[0], automatic_counts_[1], automatic_counts_[2],
                (float)automatic_min_lattice_[0],
                (float)automatic_min_lattice_[1],
                (float)automatic_min_lattice_[2],
                (float)automatic_spacing_lattice_[0],
                (float)automatic_spacing_lattice_[1],
                (float)automatic_spacing_lattice_[2],
                (ulong)lifetime_.steps, particle_count_,
                lbm_.get_Nx(), lbm_.get_Ny(), lbm_.get_Nz(),
                lbm_.get_Dx(), lbm_.get_Dy(), lbm_.get_Dz(), d);
            state.initialize_regular.enqueue_run();
        }
        finish_domains();
    }

    void migrate() {
        std::vector<std::vector<HostMigration>> incoming(domains_.size());
        for(auto& source_ptr : domains_) {
            DomainState& source = *source_ptr;
            source.migration_count.read_from_device();
            const ulong count = std::min(
                (ulong)source.migration_count[0], particle_count_);
            if(count == 0ull) continue;
            source.migration_ids.read_from_device(0ull, count);
            source.migration_destinations.read_from_device(0ull, count);
            source.migration_positions.read_from_device(0ull, count);
            source.migration_positions.read_from_device(
                particle_count_, count);
            source.migration_positions.read_from_device(
                2ull * particle_count_, count);
            for(ulong i = 0ull; i < count; ++i) {
                const uint destination = source.migration_destinations[i];
                if(destination >= incoming.size()) {
                    throw std::runtime_error(
                        "particle migration targets a non-local CUDA domain");
                }
                incoming[destination].push_back(HostMigration{
                    source.migration_ids[i],
                    source.migration_positions[i],
                    source.migration_positions[particle_count_ + i],
                    source.migration_positions[2ull * particle_count_ + i]});
            }
            migrated_particles_ += (std::uint64_t)count;
        }

        for(uint d = 0u; d < (uint)domains_.size(); ++d) {
            DomainState& destination = *domains_[d];
            const auto& records = incoming[d];
            const ulong count = (ulong)records.size();
            if(count == 0ull) continue;
            for(ulong i = 0ull; i < count; ++i) {
                destination.incoming_ids[i] = records[(size_t)i].id;
                destination.incoming_positions[i] = records[(size_t)i].x;
                destination.incoming_positions[particle_count_ + i] =
                    records[(size_t)i].y;
                destination.incoming_positions[2ull * particle_count_ + i] =
                    records[(size_t)i].z;
            }
            destination.incoming_ids.write_to_device(0ull, count);
            destination.incoming_positions.write_to_device(0ull, count);
            destination.incoming_positions.write_to_device(
                particle_count_, count);
            destination.incoming_positions.write_to_device(
                2ull * particle_count_, count);
            destination.insert
                .set_ranges(count)
                .set_parameters(2u, count)
                .enqueue_run();
        }
        finish_domains();
    }

    void build_output_schedule() {
        output_steps_.clear();
        if(!config_.output_interval_s.has_value()) return;
        std::vector<ReleaseBatch> schedule;
        std::string error;
        if(!particle_advection::build_automatic_release_schedule(
                0.0, *config_.output_interval_s, seconds_per_step_,
                total_steps_, schedule, error)) {
            throw std::runtime_error(
                "invalid particle output interval: " + error);
        }
        for(const ReleaseBatch& batch : schedule) {
            output_steps_.push_back(batch.step);
        }
        std::sort(output_steps_.begin(), output_steps_.end());
        output_steps_.erase(
            std::unique(output_steps_.begin(), output_steps_.end()),
            output_steps_.end());
    }

    void finish_domains() {
        for(auto& state : domains_) state->domain->finish_queue();
    }
};

ParticleAdvection::ParticleAdvection(
    LBM& lbm,
    particle_advection::Config config,
    ParticleAdvectionGeometry geometry,
    const std::uint64_t total_steps,
    std::filesystem::path case_directory,
    std::filesystem::path output_directory,
    std::string output_stem)
    : impl_(std::make_unique<Impl>(
          lbm, std::move(config), geometry, total_steps,
          std::move(case_directory), std::move(output_directory),
          std::move(output_stem))) {}

ParticleAdvection::~ParticleAdvection() = default;

void ParticleAdvection::initialize() {
    // Units is the single source of truth for the online solver time scale.
    impl_->set_seconds_per_step((double)units.si_t((ulong)1u));
    impl_->initialize();
}

void ParticleAdvection::advance_after_lbm_step(
    const std::uint64_t step_begin) {
    impl_->advance(step_begin);
}

void ParticleAdvection::refresh_velocity_before_lbm_step() {
    impl_->refresh_velocity_before_lbm_step();
}

bool ParticleAdvection::write_snapshot_if_due(
    const std::uint64_t step,
    const bool force) {
    return impl_->write_snapshot(step, force);
}

std::uint64_t ParticleAdvection::particle_count() const {
    return impl_->particle_count();
}

std::uint64_t ParticleAdvection::migrated_particles() const {
    return impl_->migrated_particles();
}

} // namespace culbm
