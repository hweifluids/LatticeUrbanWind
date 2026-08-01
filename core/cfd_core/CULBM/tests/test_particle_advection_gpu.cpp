#include "cuda.hpp"
#include "particle_advection_kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace {

constexpr uint kNx = 16u;
constexpr uint kNy = 8u;
constexpr uint kNz = 8u;
constexpr ulong kCellCount =
    static_cast<ulong>(kNx) * static_cast<ulong>(kNy) *
    static_cast<ulong>(kNz);
constexpr ulong kInfiniteStep = std::numeric_limits<ulong>::max();

using culbm::ParticleStatus;

uchar status_byte(const ParticleStatus status) {
    return static_cast<uchar>(status);
}

struct TestState {
    int failures = 0;

    void expect(const bool condition, const std::string& message) {
        if (condition) return;
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }

    void expect_close(const float actual,
                      const double expected,
                      const double tolerance,
                      const std::string& message) {
        if (std::fabs(static_cast<double>(actual) - expected) <= tolerance) {
            return;
        }
        std::cerr << "FAIL: " << message << " (actual=" << actual
                  << ", expected=" << expected << ")\n";
        ++failures;
    }
};

ulong cell_index(const uint x, const uint y, const uint z) {
    return static_cast<ulong>(x) +
        (static_cast<ulong>(y) +
         static_cast<ulong>(z) * static_cast<ulong>(kNy)) *
            static_cast<ulong>(kNx);
}

struct GpuFixture {
    Device& device;
    ulong particle_count;
    Memory<float> positions;
    Memory<ulong> birth_steps;
    Memory<ulong> death_steps;
    Memory<ulong> termination_steps;
    Memory<uchar> statuses;
    Memory<float> previous_velocity;
    Memory<float> current_velocity;
    Memory<uchar> flags;
    Memory<uint> migration_count;
    Memory<ulong> migration_ids;
    Memory<uint> migration_destinations;
    Memory<float> migration_positions;

    GpuFixture(Device& linked_device, const ulong count)
        : device(linked_device),
          particle_count(count),
          positions(linked_device, count, 3u),
          birth_steps(linked_device, count),
          death_steps(linked_device, count),
          termination_steps(linked_device, count),
          statuses(linked_device, count),
          previous_velocity(linked_device, kCellCount, 3u),
          current_velocity(linked_device, kCellCount, 3u),
          flags(linked_device, kCellCount),
          migration_count(linked_device, 1ull),
          migration_ids(linked_device, count),
          migration_destinations(linked_device, count),
          migration_positions(linked_device, count, 3u) {}
};

void fill_uniform_velocity(Memory<float>& field,
                           const float x,
                           const float y,
                           const float z) {
    std::fill(field.x, field.x + field.length(), x);
    std::fill(field.y, field.y + field.length(), y);
    std::fill(field.z, field.z + field.length(), z);
    field.write_to_device();
}

void fill_uniform_velocity_pair(GpuFixture& fixture,
                                const std::array<float, 3u>& previous,
                                const std::array<float, 3u>& current) {
    fill_uniform_velocity(
        fixture.previous_velocity, previous[0], previous[1], previous[2]);
    fill_uniform_velocity(
        fixture.current_velocity, current[0], current[1], current[2]);
}

void fill_spatial_linear_velocity(Memory<float>& field) {
    for (uint z = 0u; z < kNz; ++z) {
        for (uint y = 0u; y < kNy; ++y) {
            for (uint x = 0u; x < kNx; ++x) {
                const ulong index = cell_index(x, y, z);
                field.x[index] = static_cast<float>(x);
                field.y[index] = 0.0f;
                field.z[index] = 0.0f;
            }
        }
    }
    field.write_to_device();
}

void reset_particle_metadata(GpuFixture& fixture,
                             const ParticleStatus status =
                                 ParticleStatus::active) {
    fixture.positions.reset(0.0f);
    fixture.birth_steps.reset(0ull);
    fixture.death_steps.reset(kInfiniteStep);
    fixture.termination_steps.reset(kInfiniteStep);
    fixture.statuses.reset(status_byte(status));
    fixture.migration_count.reset(0u);
    fixture.migration_ids.reset(kInfiniteStep);
    fixture.migration_destinations.reset(std::numeric_limits<uint>::max());
    fixture.migration_positions.reset(0.0f);
}

void set_particle_position(GpuFixture& fixture,
                           const ulong id,
                           const float x,
                           const float y,
                           const float z) {
    fixture.positions.x[id] = x;
    fixture.positions.y[id] = y;
    fixture.positions.z[id] = z;
}

void upload_particle_metadata(GpuFixture& fixture) {
    fixture.positions.write_to_device();
    fixture.birth_steps.write_to_device();
    fixture.death_steps.write_to_device();
    fixture.termination_steps.write_to_device();
    fixture.statuses.write_to_device();
}

void read_particle_results(GpuFixture& fixture) {
    fixture.positions.read_from_device();
    fixture.termination_steps.read_from_device();
    fixture.statuses.read_from_device();
    fixture.migration_count.read_from_device();
}

void run_activate_kernel(GpuFixture& fixture, const ulong current_step) {
    Kernel kernel(
        fixture.device,
        fixture.particle_count,
        "culbm_advection_activate",
        fixture.positions,
        fixture.birth_steps,
        fixture.death_steps,
        fixture.termination_steps,
        fixture.statuses,
        fixture.flags,
        current_step,
        fixture.particle_count,
        kNx,
        kNy,
        kNz,
        1u,
        1u,
        1u,
        0u,
        kNx,
        kNy,
        kNz,
        0,
        0,
        0);
    kernel.run();
}

void run_step_kernel(GpuFixture& fixture,
                     const std::string& kernel_name,
                     const ulong step_begin) {
    fixture.migration_count.reset(0u);
    Kernel kernel(
        fixture.device,
        fixture.particle_count,
        kernel_name,
        fixture.positions,
        fixture.birth_steps,
        fixture.death_steps,
        fixture.termination_steps,
        fixture.statuses,
        fixture.previous_velocity,
        fixture.current_velocity,
        fixture.flags,
        step_begin,
        fixture.migration_count,
        fixture.migration_ids,
        fixture.migration_destinations,
        fixture.migration_positions,
        fixture.particle_count,
        kNx,
        kNy,
        kNz,
        1u,
        1u,
        1u,
        0u,
        kNx,
        kNy,
        kNz,
        0,
        0,
        0);
    kernel.run();
}

void prepare_single_active_particle(GpuFixture& fixture,
                                    const float x,
                                    const float y,
                                    const float z) {
    reset_particle_metadata(fixture);
    set_particle_position(fixture, 0ull, x, y, z);
    upload_particle_metadata(fixture);
}

void test_birth_activation(TestState& test, Device& device) {
    GpuFixture fixture(device, 1ull);
    fixture.flags.reset(0u);
    reset_particle_metadata(fixture, ParticleStatus::pending);
    set_particle_position(fixture, 0ull, 3.0f, 3.0f, 3.0f);
    fixture.birth_steps[0] = 3ull;
    upload_particle_metadata(fixture);

    run_activate_kernel(fixture, 2ull);
    fixture.statuses.read_from_device();
    test.expect(
        fixture.statuses[0] == status_byte(ParticleStatus::pending),
        "a pending particle is not activated before its birth step");

    run_activate_kernel(fixture, 3ull);
    fixture.statuses.read_from_device();
    fixture.termination_steps.read_from_device();
    test.expect(
        fixture.statuses[0] == status_byte(ParticleStatus::active),
        "a pending particle activates exactly at its birth step");
    test.expect(
        fixture.termination_steps[0] == kInfiniteStep,
        "activation leaves the termination step unset");
}

void test_lifetime_expiry(TestState& test, Device& device) {
    GpuFixture fixture(device, 2ull);
    fixture.flags.reset(0u);
    fill_uniform_velocity_pair(
        fixture, {0.25f, 0.0f, 0.0f}, {0.25f, 0.0f, 0.0f});
    reset_particle_metadata(fixture);
    set_particle_position(fixture, 0ull, 3.0f, 3.0f, 3.0f);
    set_particle_position(fixture, 1ull, 5.0f, 3.0f, 3.0f);
    fixture.death_steps[0] = 2ull;
    fixture.death_steps[1] = kInfiniteStep;
    upload_particle_metadata(fixture);

    run_step_kernel(fixture, "culbm_advection_step_euler", 0ull);
    read_particle_results(fixture);
    test.expect(
        fixture.statuses[0] == status_byte(ParticleStatus::active) &&
            fixture.statuses[1] == status_byte(ParticleStatus::active),
        "finite and infinite particles remain active before expiry");

    run_step_kernel(fixture, "culbm_advection_step_euler", 1ull);
    run_step_kernel(fixture, "culbm_advection_step_euler", 2ull);
    read_particle_results(fixture);
    test.expect(
        fixture.statuses[0] == status_byte(ParticleStatus::expired) &&
            fixture.termination_steps[0] == 2ull,
        "finite lifetime expires at its exclusive death step");
    test.expect_close(
        fixture.positions.x[0], 3.5, 1.0e-6,
        "finite-lifetime particle completes exactly two advection steps");
    test.expect(
        fixture.statuses[1] == status_byte(ParticleStatus::active) &&
            fixture.termination_steps[1] == kInfiniteStep,
        "infinite lifetime remains active with no termination step");
    test.expect_close(
        fixture.positions.x[1], 5.75, 1.0e-6,
        "infinite-lifetime particle continues after the finite particle expires");
}

void test_constant_velocity_methods(TestState& test, Device& device) {
    struct Method {
        const char* kernel;
        const char* label;
    };
    constexpr std::array<Method, 4u> methods{{
        {"culbm_advection_step_euler", "Euler"},
        {"culbm_advection_step_rk2", "RK2"},
        {"culbm_advection_step_rk4", "RK4"},
        {"culbm_advection_step_rk6", "RK6"},
    }};

    GpuFixture fixture(device, 1ull);
    fixture.flags.reset(0u);
    fill_uniform_velocity_pair(
        fixture,
        {0.25f, -0.125f, 0.0625f},
        {0.25f, -0.125f, 0.0625f});
    for (const Method& method : methods) {
        prepare_single_active_particle(fixture, 4.0f, 3.0f, 3.0f);
        run_step_kernel(fixture, method.kernel, 0ull);
        read_particle_results(fixture);
        const std::string prefix = std::string(method.label) + " constant velocity ";
        test.expect_close(
            fixture.positions.x[0], 4.25, 2.0e-6, prefix + "x displacement");
        test.expect_close(
            fixture.positions.y[0], 2.875, 2.0e-6, prefix + "y displacement");
        test.expect_close(
            fixture.positions.z[0], 3.0625, 2.0e-6, prefix + "z displacement");
        test.expect(
            fixture.statuses[0] == status_byte(ParticleStatus::active),
            prefix + "keeps an interior particle active");
        test.expect(
            fixture.migration_count[0] == 0u,
            prefix + "does not enqueue a single-domain migration");
    }
}

void test_time_linear_velocity(TestState& test, Device& device) {
    struct Method {
        const char* kernel;
        const char* label;
        double expected_displacement;
    };
    constexpr std::array<Method, 4u> methods{{
        {"culbm_advection_step_euler", "Euler", 0.0},
        {"culbm_advection_step_rk2", "RK2", 0.5},
        {"culbm_advection_step_rk4", "RK4", 0.5},
        {"culbm_advection_step_rk6", "RK6", 0.5},
    }};

    GpuFixture fixture(device, 1ull);
    fixture.flags.reset(0u);
    fill_uniform_velocity_pair(
        fixture, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f});
    for (const Method& method : methods) {
        prepare_single_active_particle(fixture, 3.0f, 3.0f, 3.0f);
        run_step_kernel(fixture, method.kernel, 0ull);
        fixture.positions.read_from_device();
        test.expect_close(
            fixture.positions.x[0],
            3.0 + method.expected_displacement,
            2.0e-5,
            std::string(method.label) +
                " uses the expected stage times for a time-linear velocity");
    }
}

void test_spatial_linear_velocity(TestState& test, Device& device) {
    constexpr std::array<const char*, 4u> kernels{{
        "culbm_advection_step_euler",
        "culbm_advection_step_rk2",
        "culbm_advection_step_rk4",
        "culbm_advection_step_rk6",
    }};
    constexpr std::array<double, 4u> expected{{
        2.0,
        2.5,
        65.0 / 24.0,
        2.717606101669791,
    }};

    GpuFixture fixture(device, 1ull);
    fixture.flags.reset(0u);
    fill_spatial_linear_velocity(fixture.previous_velocity);
    fill_spatial_linear_velocity(fixture.current_velocity);

    std::array<double, 4u> errors{};
    const double exact = std::exp(1.0);
    for (std::size_t index = 0u; index < kernels.size(); ++index) {
        prepare_single_active_particle(fixture, 1.0f, 3.0f, 3.0f);
        run_step_kernel(fixture, kernels[index], 0ull);
        fixture.positions.read_from_device();
        test.expect_close(
            fixture.positions.x[0],
            expected[index],
            index == 3u ? 3.0e-5 : 1.0e-5,
            "u=x produces the expected fixed-step result for method " +
                std::to_string(index));
        errors[index] =
            std::fabs(static_cast<double>(fixture.positions.x[0]) - exact);
    }
    test.expect(
        errors[0] > errors[1] && errors[1] > errors[2] &&
            errors[2] > errors[3],
        "Euler, RK2, RK4, and RK6 show increasing accuracy for dx/dt=x");
}

void test_solid_and_escape(TestState& test, Device& device) {
    GpuFixture fixture(device, 1ull);
    fill_uniform_velocity_pair(
        fixture, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f});

    fixture.flags.reset(0u);
    fixture.flags[cell_index(4u, 3u, 3u)] = 1u;
    fixture.flags.write_to_device();
    prepare_single_active_particle(fixture, 3.0f, 3.0f, 3.0f);
    run_step_kernel(fixture, "culbm_advection_step_euler", 0ull);
    read_particle_results(fixture);
    test.expect(
        fixture.statuses[0] == status_byte(ParticleStatus::solid) &&
            fixture.termination_steps[0] == 1ull,
        "a particle terminating in a solid cell is marked solid at step end");
    test.expect_close(
        fixture.positions.x[0], 4.0, 1.0e-6,
        "solid termination retains the computed endpoint");

    fixture.flags.reset(0u);
    prepare_single_active_particle(fixture, 15.0f, 3.0f, 3.0f);
    run_step_kernel(fixture, "culbm_advection_step_euler", 0ull);
    read_particle_results(fixture);
    test.expect(
        fixture.statuses[0] == status_byte(ParticleStatus::escaped) &&
            fixture.termination_steps[0] == 1ull,
        "a particle crossing the global half-cell boundary is marked escaped");
    test.expect_close(
        fixture.positions.x[0], 16.0, 1.0e-6,
        "escape termination retains the out-of-domain endpoint");

    fill_uniform_velocity_pair(
        fixture, {2.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f});
    prepare_single_active_particle(fixture, 14.0f, 3.0f, 3.0f);
    run_step_kernel(fixture, "culbm_advection_step_rk4", 0ull);
    read_particle_results(fixture);
    test.expect(
        fixture.statuses[0] == status_byte(ParticleStatus::escaped) &&
            fixture.termination_steps[0] == 1ull,
        "an RK intermediate-stage escape terminates at step end");
    test.expect_close(
        fixture.positions.x[0], 16.0, 1.0e-6,
        "an RK stage failure retains the failing stage position");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    fill_uniform_velocity_pair(
        fixture, {nan, 0.0f, 0.0f}, {nan, 0.0f, 0.0f});
    prepare_single_active_particle(fixture, 3.0f, 3.0f, 3.0f);
    run_step_kernel(fixture, "culbm_advection_step_euler", 0ull);
    read_particle_results(fixture);
    test.expect(
        fixture.statuses[0] == status_byte(ParticleStatus::invalid) &&
            fixture.termination_steps[0] == 1ull,
        "a non-finite fluid velocity is invalid rather than solid");
}

void test_global_half_cell_shell(TestState& test, Device& device) {
    GpuFixture fixture(device, 2ull);
    fixture.flags.reset(0u);
    fill_uniform_velocity_pair(
        fixture, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f});
    reset_particle_metadata(fixture);
    set_particle_position(fixture, 0ull, -0.25f, 3.0f, 3.0f);
    set_particle_position(
        fixture, 1ull, static_cast<float>(kNx) - 0.75f, 3.0f, 3.0f);
    upload_particle_metadata(fixture);

    run_step_kernel(fixture, "culbm_advection_step_rk4", 0ull);
    read_particle_results(fixture);
    test.expect(
        fixture.statuses[0] == status_byte(ParticleStatus::active) &&
            fixture.statuses[1] == status_byte(ParticleStatus::active),
        "the global half-cell ownership shell samples boundary velocity");
    test.expect_close(
        fixture.positions.x[0], -0.25, 1.0e-6,
        "lower half-cell shell retains a stationary particle");
    test.expect_close(
        fixture.positions.x[1], 15.25, 1.0e-6,
        "upper half-cell shell retains a stationary particle");
}

void test_automatic_regular_initialization(TestState& test, Device& device) {
    constexpr ulong release_count = 2ull;
    constexpr uint count_x = 2u;
    constexpr uint count_y = 2u;
    constexpr uint count_z = 1u;
    constexpr ulong particle_count =
        release_count * static_cast<ulong>(count_x) *
        static_cast<ulong>(count_y) * static_cast<ulong>(count_z);
    constexpr ulong lifetime_steps = 4ull;

    GpuFixture fixture(device, particle_count);
    Memory<ulong> release_steps(device, release_count);
    release_steps[0] = 2ull;
    release_steps[1] = 5ull;
    release_steps.write_to_device();
    fixture.positions.reset(-99.0f);
    fixture.birth_steps.reset(kInfiniteStep);
    fixture.death_steps.reset(kInfiniteStep);
    fixture.termination_steps.reset(0ull);
    fixture.statuses.reset(status_byte(ParticleStatus::invalid));

    Kernel initialize(
        device,
        particle_count,
        "culbm_advection_initialize_regular",
        fixture.positions,
        fixture.birth_steps,
        fixture.death_steps,
        fixture.termination_steps,
        fixture.statuses,
        release_steps,
        release_count,
        count_x,
        count_y,
        count_z,
        1.0f,
        2.0f,
        3.0f,
        0.5f,
        1.0f,
        0.0f,
        lifetime_steps,
        particle_count,
        kNx,
        kNy,
        kNz,
        1u,
        1u,
        1u,
        0u);
    initialize.run();

    fixture.positions.read_from_device();
    fixture.birth_steps.read_from_device();
    fixture.death_steps.read_from_device();
    fixture.termination_steps.read_from_device();
    fixture.statuses.read_from_device();

    bool metadata_matches = true;
    for (ulong id = 0ull; id < particle_count; ++id) {
        const ulong batch = id / 4ull;
        const ulong expected_birth = batch == 0ull ? 2ull : 5ull;
        metadata_matches =
            metadata_matches &&
            fixture.birth_steps[id] == expected_birth &&
            fixture.death_steps[id] == expected_birth + lifetime_steps &&
            fixture.termination_steps[id] == kInfiniteStep &&
            fixture.statuses[id] == status_byte(ParticleStatus::pending);
    }
    test.expect(
        metadata_matches,
        "regular initialization assigns release, death, and pending metadata");
    test.expect_close(
        fixture.positions.x[0], 1.0, 1.0e-6,
        "regular initialization starts at min_x");
    test.expect_close(
        fixture.positions.x[1], 1.5, 1.0e-6,
        "regular initialization advances x fastest");
    test.expect_close(
        fixture.positions.y[2], 3.0, 1.0e-6,
        "regular initialization advances the y coordinate after x");
    test.expect_close(
        fixture.positions.z[7], 3.0, 1.0e-6,
        "regular initialization preserves a singleton z layer");
    test.expect_close(
        fixture.positions.x[4], 1.0, 1.0e-6,
        "each automatic release repeats the same regular lattice");
}

int run_tests(Device& device) {
    TestState test;
    test_birth_activation(test, device);
    test_lifetime_expiry(test, device);
    test_constant_velocity_methods(test, device);
    test_time_linear_velocity(test, device);
    test_spatial_linear_velocity(test, device);
    test_solid_and_escape(test, device);
    test_global_half_cell_shell(test, device);
    test_automatic_regular_initialization(test, device);

    if (test.failures == 0) {
        std::cout << "Particle-advection production CUDA kernel tests passed.\n";
    }
    return test.failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    const CUresult initialization = cuInit(0u);
    if (initialization != CUDA_SUCCESS) {
        std::cout << "SKIP: CUDA driver is unavailable.\n";
        return 77;
    }
    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count <= 0) {
        std::cout << "SKIP: no CUDA device is available.\n";
        return 77;
    }

    const vector<Device_Info> devices = get_devices(false);
    Device device(devices.front(), culbm::particle_advection_cuda_source());
    return run_tests(device);
}
