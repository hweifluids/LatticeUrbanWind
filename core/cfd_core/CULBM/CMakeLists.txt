cmake_minimum_required(VERSION 3.24)

project(CULBM VERSION 4.0.0 LANGUAGES C CXX)

include(CTest)
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(Threads REQUIRED)
list(GET CUDAToolkit_INCLUDE_DIRS 0 CULBM_CUDA_PRIMARY_INCLUDE_DIR)
option(
    CULBM_ENABLE_VTKHDF
    "Enable the optional HDF5-backed VTKHDF physical-output writer when HDF5 is available"
    ON)
option(
    CULBM_BUILD_PHYSICAL_OUTPUT_BENCHMARKS
    "Build manually-run large physical-output format benchmarks"
    OFF)

# VTKHDF is a real HDF5 container, rather than an alternate filename for the
# legacy VTK stream.  Prefer package-provided imported targets, but retain the
# FindHDF5 fallback used by common Linux module installations.
set(CULBM_VTKHDF_ENABLED OFF)
set(CULBM_HDF5_LINK_LIBRARIES)
set(CULBM_HDF5_INCLUDE_DIRS)
set(CULBM_HDF5_RUNTIME_TARGET)
if(CULBM_ENABLE_VTKHDF)
    find_package(HDF5 CONFIG QUIET)
    if(TARGET hdf5::hdf5-shared)
        set(CULBM_HDF5_LINK_LIBRARIES hdf5::hdf5-shared)
        set(CULBM_HDF5_RUNTIME_TARGET hdf5::hdf5-shared)
        set(CULBM_VTKHDF_ENABLED ON)
    elseif(TARGET hdf5::hdf5)
        set(CULBM_HDF5_LINK_LIBRARIES hdf5::hdf5)
        set(CULBM_VTKHDF_ENABLED ON)
    elseif(TARGET HDF5::HDF5)
        set(CULBM_HDF5_LINK_LIBRARIES HDF5::HDF5)
        set(CULBM_VTKHDF_ENABLED ON)
    else()
        find_package(HDF5 QUIET COMPONENTS C)
        if(TARGET HDF5::HDF5)
            set(CULBM_HDF5_LINK_LIBRARIES HDF5::HDF5)
            set(CULBM_VTKHDF_ENABLED ON)
        elseif(HDF5_FOUND)
            set(CULBM_HDF5_LINK_LIBRARIES ${HDF5_C_LIBRARIES})
            set(CULBM_HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS})
            set(CULBM_VTKHDF_ENABLED ON)
        endif()
    endif()
endif()
if(CULBM_VTKHDF_ENABLED)
    message(STATUS "CULBM VTKHDF physical output enabled with HDF5")
elseif(CULBM_ENABLE_VTKHDF)
    message(STATUS "CULBM VTKHDF physical output disabled: HDF5 C library was not found")
endif()
option(
    CULBM_ENABLE_MPI
    "Enable one-domain-per-rank multi-node execution when MPI is available"
    ON)
option(
    CULBM_REQUIRE_MPI
    "Fail configuration unless MPI support is enabled and available"
    OFF)
if(CULBM_REQUIRE_MPI AND NOT CULBM_ENABLE_MPI)
    message(FATAL_ERROR
        "CULBM_REQUIRE_MPI=ON is incompatible with CULBM_ENABLE_MPI=OFF")
endif()
if(CULBM_ENABLE_MPI)
    find_package(MPI QUIET COMPONENTS CXX)
    if(NOT MPI_CXX_FOUND)
        if(CULBM_REQUIRE_MPI)
            message(FATAL_ERROR
                "MPI C++ support is required but was not found. Load the "
                "cluster MPI development environment before configuring CULBM.")
        else()
            message(WARNING
                "MPI C++ support was not found. CULBM will retain its "
                "single-process arbitrary-GPU mode, but mpiexec multi-node "
                "execution will be disabled.")
        endif()
    endif()
endif()

# CUDA device code has one human-readable source of truth under src/device.
# The generated fragments only embed those bytes into the host executable;
# NVRTC still performs per-domain specialization at runtime.
set(CULBM_DEVICE_SRC "${CMAKE_CURRENT_LIST_DIR}/src/device")
set(CULBM_GENERATED_CUDA_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated/cuda")
set(CULBM_CUDA_EMBED_SCRIPT
    "${CMAKE_CURRENT_LIST_DIR}/cmake/EmbedCudaSource.cmake")

function(culbm_embed_cuda_source source_name output_name)
    set(input_file "${CULBM_DEVICE_SRC}/${source_name}")
    set(output_file "${CULBM_GENERATED_CUDA_DIR}/${output_name}.inc")
    if(NOT EXISTS "${input_file}")
        message(FATAL_ERROR
            "CULBM native CUDA source is missing: ${input_file}")
    endif()
    # Generate during CMake configuration, inside the build tree. Registering
    # the canonical source as a configure dependency makes ordinary builds
    # regenerate it automatically after edits and avoids first-build races in
    # Visual Studio/Xcode, which cannot honor per-object generated-file deps.
    set_property(
        DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${input_file}" "${CULBM_CUDA_EMBED_SCRIPT}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            "-DINPUT_FILE=${input_file}"
            "-DOUTPUT_FILE=${output_file}"
            -P "${CULBM_CUDA_EMBED_SCRIPT}"
        RESULT_VARIABLE embed_result
        ERROR_VARIABLE embed_error)
    if(NOT embed_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to embed native CUDA source ${source_name}: "
            "${embed_error}")
    endif()
endfunction()

culbm_embed_cuda_source(
    cuda_prelude.cuh culbm_cuda_prelude)
culbm_embed_cuda_source(
    lbm_kernels.cu culbm_lbm_kernels)
culbm_embed_cuda_source(
    runtime_kernels.cu culbm_runtime_kernels)
culbm_embed_cuda_source(
    particle_advection_kernels.cu culbm_particle_advection_kernels)
culbm_embed_cuda_source(
    surface_ibm_kernels.cu culbm_surface_ibm_kernels)
include_directories(BEFORE "${CULBM_GENERATED_CUDA_DIR}")

# Host solver sources. The list is explicit so a missing file fails at configure time.
set(CULBM_SOLVER_SRC "${CMAKE_CURRENT_LIST_DIR}/src/solver")
set(CULBM_SOLVER_FILES
    kernel.cpp
    kernel.hpp
    case_validator.cpp
    case_validator.hpp
    field_set.cpp
    field_set.hpp
    field_table.cpp
    field_table.hpp
    field_view.cpp
    field_view.hpp
    halo_exchanger.cpp
    halo_exchanger.hpp
    kernel_set.cpp
    kernel_set.hpp
    lattice_config.cpp
    lattice_config.hpp
    solver.cpp
    solver.hpp
    step_pipeline.cpp
    step_pipeline.hpp
    subdomain.cpp
    subdomain.hpp
    voxelizer.cpp
    voxelizer.hpp
    decomposition.hpp
    immersed_forcing.hpp)

foreach(solver_file IN LISTS CULBM_SOLVER_FILES)
    if(NOT EXISTS "${CULBM_SOLVER_SRC}/${solver_file}")
        message(FATAL_ERROR
            "CULBM self-contained solver source is missing: "
            "${CULBM_SOLVER_SRC}/${solver_file}")
    endif()
endforeach()

# Executable entry and console reports (src/app): command line, console loop,
# run report and run progress.
set(CULBM_APP_SRC "${CMAKE_CURRENT_LIST_DIR}/src/app")
foreach(app_file IN ITEMS launch.cpp launch.hpp main.cpp progress.cpp progress.hpp run_report.cpp run_report.hpp)
    if(NOT EXISTS "${CULBM_APP_SRC}/${app_file}")
        message(FATAL_ERROR "CULBM application source is missing: ${CULBM_APP_SRC}/${app_file}")
    endif()
endforeach()
# The run report and progress: linked wherever the solver core is.
set(CULBM_REPORT_SOURCES
    "${CULBM_APP_SRC}/progress.cpp"
    "${CULBM_APP_SRC}/run_report.cpp")

# Solver core (namespace culbm::solver). Every target that runs the solver
# links this list.
set(CULBM_SOLVER_CORE_SOURCES
    "${CULBM_SOLVER_SRC}/case_validator.cpp"
    "${CULBM_SOLVER_SRC}/field_set.cpp"
    "${CULBM_SOLVER_SRC}/field_table.cpp"
    "${CULBM_SOLVER_SRC}/field_view.cpp"
    "${CULBM_SOLVER_SRC}/halo_exchanger.cpp"
    "${CULBM_SOLVER_SRC}/kernel_set.cpp"
    "${CULBM_SOLVER_SRC}/lattice_config.cpp"
    "${CULBM_SOLVER_SRC}/solver.cpp"
    "${CULBM_SOLVER_SRC}/step_pipeline.cpp"
    "${CULBM_SOLVER_SRC}/subdomain.cpp"
    "${CULBM_SOLVER_SRC}/voxelizer.cpp"
)

# Deck-driven case setup (src/case), boundary conditions (src/bc) and case
# runs (src/run); only the executable links them.
set(CULBM_CASE_FILES
    case/case_dump.cpp
    case/case_dump.hpp
    case/config_dump.cpp
    case/config_dump.hpp
    case/context.hpp
    case/deck.cpp
    case/deck.hpp
    case/domain.cpp
    case/domain.hpp
    case/features.cpp
    case/features.hpp
    case/geometry.cpp
    case/geometry.hpp
    case/memory_fit.cpp
    case/memory_fit.hpp
    case/physics.cpp
    case/physics.hpp
    case/report.cpp
    case/report.hpp
    case/settings.hpp
    case/settings_reader.cpp
    case/terrain.cpp
    case/terrain.hpp
    bc/patch_fields.hpp
    bc/worker_threads.cpp
    bc/worker_threads.hpp
    bc/inflow/boundaries.hpp
    bc/inflow/inflow_data.cpp
    bc/inflow/inflow_data.hpp
    bc/inflow/interpolation.cpp
    bc/inflow/interpolation.hpp
    bc/inflow/interpolation_hd.cpp
    bc/inflow/interpolation_hd.hpp
    bc/inflow/normal_boundaries.cpp
    bc/inflow/profile_boundaries.cpp
    bc/inflow/samples.cpp
    bc/inflow/samples.hpp
    bc/inflow/uniform_boundaries.cpp
    bc/inflow/vk_inlet.cpp
    bc/inflow/vk_inlet.hpp
    bc/inflow/vk_inlet_settings.hpp
    bc/outflow/damping_layers.cpp
    bc/outflow/damping_layers.hpp
    bc/outflow/fluxcorrection.cpp
    bc/outflow/fluxcorrection.hpp
    bc/thermal/coupling_policy.hpp
    bc/thermal/temperature_limits.hpp
    bc/thermal/thermal_bc.cpp
    bc/thermal/thermal_bc.hpp
    run/case_runner.cpp
    run/case_runner.hpp
    run/console_log.cpp
    run/console_log.hpp
    run/nested_run.cpp
    run/nested_run.hpp
    run/physical_fields.cpp
    run/physical_fields.hpp
    run/probes.cpp
    run/probes.hpp
    run/schedule.cpp
    run/schedule.hpp
    run/uniform_outputs.cpp
    run/uniform_run.cpp
    run/uniform_run.hpp
    run/vtk_output.cpp
    run/vtk_output.hpp)
set(CULBM_CASE_SOURCES)
foreach(case_file IN LISTS CULBM_CASE_FILES)
    if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/${case_file}")
        message(FATAL_ERROR "CULBM case source is missing: ${CMAKE_CURRENT_LIST_DIR}/src/${case_file}")
    endif()
    if(case_file MATCHES "\\.cpp$")
        list(APPEND CULBM_CASE_SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/${case_file}")
    endif()
endforeach()

set(CULBM_SOLVER_SOURCES
    ${CULBM_REPORT_SOURCES}
    "${CULBM_SOLVER_SRC}/kernel.cpp"
    ${CULBM_SOLVER_CORE_SOURCES}
    "${CULBM_APP_SRC}/launch.cpp"
    "${CULBM_APP_SRC}/main.cpp"
    ${CULBM_CASE_SOURCES})

# SurfaceIBM is a runtime-selectable wall treatment.  Keep its host/runtime
# pieces grouped so every focused target that embeds the solver core links
# the same optional implementation as the production executable.
set(CULBM_SURFACE_IBM_RUNTIME_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/src/surface_ibm_config.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/surface_ibm_kernels.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/surface_ibm.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/surface_ibm_geometry.cpp")

set(CULBM_SURFACE_IBM_SOURCES
    ${CULBM_SURFACE_IBM_RUNTIME_SOURCES})

# GPU layer (src/gpu): devices, buffers and host mirrors, kernel launches,
# NVRTC with its PTX cache, GPU selection and VRAM bookkeeping.
set(CULBM_GPU_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/src/gpu/buffer.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/gpu/device.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/gpu/driver.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/gpu/launch.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/gpu/program.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/gpu/selection.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/gpu/vram.cpp")

set(CULBM_SOURCES
    ${CULBM_SOLVER_SOURCES}
    ${CULBM_SURFACE_IBM_SOURCES}
    "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_config.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/physical_output.cpp"
    ${CULBM_GPU_SOURCES}
    "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_runtime.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_solver.cpp")

add_executable(CULBM ${CULBM_SOURCES})
target_compile_features(CULBM PRIVATE cxx_std_17)
target_include_directories(CULBM PRIVATE
    "${CULBM_SOLVER_SRC}"
    "${CMAKE_CURRENT_LIST_DIR}/src"
    "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(CULBM PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
    Threads::Threads)
if(CULBM_VTKHDF_ENABLED)
    target_include_directories(CULBM PRIVATE ${CULBM_HDF5_INCLUDE_DIRS})
    target_link_libraries(CULBM PRIVATE ${CULBM_HDF5_LINK_LIBRARIES})
    target_compile_definitions(CULBM PRIVATE CULBM_HAS_VTKHDF=1)
else()
    target_compile_definitions(CULBM PRIVATE CULBM_HAS_VTKHDF=0)
endif()
target_compile_definitions(CULBM PRIVATE
    CULBM_BACKEND=1
    "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
if(MPI_CXX_FOUND)
    target_compile_definitions(
        CULBM PRIVATE CULBM_ENABLE_MPI=1)
    target_link_libraries(CULBM PRIVATE MPI::MPI_CXX)
endif()

if(MSVC)
    set_property(TARGET CULBM PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    target_compile_options(CULBM PRIVATE
        /O2 /Oi /Ot /GL /fp:fast /MP /W3 /Gy /Qpar /sdl /permissive-
        /wd4146 /wd4267 /wd4334 /wd4996)
    # CUDA's driver import library advertises the static CRT even though this
    # executable, like LUW's production solver, consistently uses /MD.
    target_link_options(CULBM PRIVATE
        /LTCG /NODEFAULTLIB:LIBCMT /NODEFAULTLIB:LIBCMTD)
    target_link_libraries(CULBM PRIVATE
        user32 gdi32 shell32 advapi32)
else()
    target_compile_options(CULBM PRIVATE
        -O3 -pthread -Wno-comment)
    target_link_options(CULBM PRIVATE -pthread)
endif()

set_target_properties(CULBM PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_LIST_DIR}/bin"
    OUTPUT_NAME "CULBM")

if(CULBM_BUILD_PHYSICAL_OUTPUT_BENCHMARKS)
    if(NOT CULBM_VTKHDF_ENABLED)
        message(FATAL_ERROR
            "CULBM_BUILD_PHYSICAL_OUTPUT_BENCHMARKS requires CULBM_ENABLE_VTKHDF=ON and an HDF5 C library")
    endif()
    add_executable(
        CULBM-physical-output-gib-benchmark
        "${CMAKE_CURRENT_LIST_DIR}/tests/benchmark_physical_output_gib.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/physical_output.cpp")
    target_compile_features(
        CULBM-physical-output-gib-benchmark PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-physical-output-gib-benchmark PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src"
        ${CULBM_HDF5_INCLUDE_DIRS})
    target_link_libraries(
        CULBM-physical-output-gib-benchmark PRIVATE ${CULBM_HDF5_LINK_LIBRARIES})
    target_compile_definitions(
        CULBM-physical-output-gib-benchmark PRIVATE CULBM_HAS_VTKHDF=1)
endif()

if(BUILD_TESTING)
    add_executable(
        CULBM-cuda-source-embedding-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_cuda_source_embedding.cpp")
    target_compile_features(
        CULBM-cuda-source-embedding-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-cuda-source-embedding-test PRIVATE
        "${CULBM_GENERATED_CUDA_DIR}")
    add_test(
        NAME CULBM.CudaSourceEmbedding
        COMMAND
            CULBM-cuda-source-embedding-test
            "${CULBM_DEVICE_SRC}/cuda_prelude.cuh"
            "${CULBM_DEVICE_SRC}/lbm_kernels.cu"
            "${CULBM_DEVICE_SRC}/runtime_kernels.cu"
            "${CULBM_DEVICE_SRC}/particle_advection_kernels.cu"
            "${CULBM_DEVICE_SRC}/surface_ibm_kernels.cu")
    set_tests_properties(
        CULBM.CudaSourceEmbedding PROPERTIES LABELS "cuda;source;build")

    add_executable(
        CULBM-thermal-coupling-policy-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_thermal_coupling_policy.cpp")
    target_compile_features(
        CULBM-thermal-coupling-policy-test PRIVATE cxx_std_17)
    add_test(
        NAME CULBM.ThermalCouplingPolicy
        COMMAND CULBM-thermal-coupling-policy-test)
    set_tests_properties(
        CULBM.ThermalCouplingPolicy PROPERTIES LABELS "temperature;config")

    add_executable(
        CULBM-surface-ibm-config-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_surface_ibm_config.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/surface_ibm_config.cpp")
    target_compile_features(
        CULBM-surface-ibm-config-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-surface-ibm-config-test PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src")
    add_test(
        NAME CULBM.SurfaceIBMConfig
        COMMAND CULBM-surface-ibm-config-test)
    set_tests_properties(
        CULBM.SurfaceIBMConfig PROPERTIES LABELS "surface-ibm;config")

    add_executable(
        CULBM-surface-ibm-geometry-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_surface_ibm_geometry.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/surface_ibm_config.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/surface_ibm_geometry.cpp")
    target_compile_features(
        CULBM-surface-ibm-geometry-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-surface-ibm-geometry-test PRIVATE
        "${CULBM_SOLVER_SRC}"
        "${CMAKE_CURRENT_LIST_DIR}/src")
    add_test(
        NAME CULBM.SurfaceIBMGeometry
        COMMAND CULBM-surface-ibm-geometry-test)
    set_tests_properties(
        CULBM.SurfaceIBMGeometry PROPERTIES LABELS "surface-ibm;geometry")

    set(CULBM_SURFACE_IBM_GPU_RUNTIME_TEST_SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
        ${CULBM_SURFACE_IBM_RUNTIME_SOURCES}
        ${CULBM_REPORT_SOURCES}
        "${CULBM_SOLVER_SRC}/kernel.cpp"
        ${CULBM_SOLVER_CORE_SOURCES}
        ${CULBM_GPU_SOURCES}
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_runtime.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_solver.cpp")
    set(CULBM_SURFACE_IBM_GPU_TEST_SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_surface_ibm_gpu.cpp"
        ${CULBM_SURFACE_IBM_GPU_RUNTIME_TEST_SOURCES})
    add_executable(
        CULBM-surface-ibm-gpu-test
        ${CULBM_SURFACE_IBM_GPU_TEST_SOURCES})
    target_compile_features(
        CULBM-surface-ibm-gpu-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-surface-ibm-gpu-test PRIVATE
        "${CULBM_SOLVER_SRC}"
        "${CMAKE_CURRENT_LIST_DIR}/src"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(
        CULBM-surface-ibm-gpu-test PRIVATE
        CUDA::cuda_driver
        CUDA::nvrtc
        Threads::Threads)
    target_compile_definitions(
        CULBM-surface-ibm-gpu-test PRIVATE
        CULBM_BACKEND=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
    if(MPI_CXX_FOUND)
        target_compile_definitions(
            CULBM-surface-ibm-gpu-test PRIVATE CULBM_ENABLE_MPI=1)
        target_link_libraries(
            CULBM-surface-ibm-gpu-test PRIVATE MPI::MPI_CXX)
    endif()
    add_test(
        NAME CULBM.SurfaceIBMGPU
        COMMAND CULBM-surface-ibm-gpu-test)
    # Planned VRAM (gpu::domain_footprint) against the buffers LBM domains
    # really allocate, for several lattice configurations and two domains.
    add_executable(CULBM-vram-accounting-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_vram_accounting_gpu.cpp"
        ${CULBM_SURFACE_IBM_GPU_RUNTIME_TEST_SOURCES})
    target_compile_features(CULBM-vram-accounting-test PRIVATE cxx_std_17)
    target_include_directories(CULBM-vram-accounting-test PRIVATE
        "${CULBM_SOLVER_SRC}" "${CMAKE_CURRENT_LIST_DIR}/src" "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(CULBM-vram-accounting-test PRIVATE
        CUDA::cuda_driver CUDA::nvrtc Threads::Threads)
    target_compile_definitions(CULBM-vram-accounting-test PRIVATE
        CULBM_BACKEND=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
    if(MPI_CXX_FOUND)
        target_compile_definitions(CULBM-vram-accounting-test PRIVATE CULBM_ENABLE_MPI=1)
        target_link_libraries(CULBM-vram-accounting-test PRIVATE MPI::MPI_CXX)
    endif()
    add_test(NAME CULBM.VramAccountingGPU COMMAND CULBM-vram-accounting-test)
    set_tests_properties(CULBM.VramAccountingGPU PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "gpu;vram;accounting")

    add_executable(CULBM-device-codec-gpu-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_device_codec_gpu.cpp"
        ${CULBM_SURFACE_IBM_GPU_RUNTIME_TEST_SOURCES})
    target_compile_features(CULBM-device-codec-gpu-test PRIVATE cxx_std_17)
    target_include_directories(CULBM-device-codec-gpu-test PRIVATE
        "${CULBM_SOLVER_SRC}" "${CMAKE_CURRENT_LIST_DIR}/src" "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(CULBM-device-codec-gpu-test PRIVATE
        CUDA::cuda_driver CUDA::nvrtc Threads::Threads)
    target_compile_definitions(CULBM-device-codec-gpu-test PRIVATE
        CULBM_BACKEND=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
    if(MPI_CXX_FOUND)
        target_compile_definitions(CULBM-device-codec-gpu-test PRIVATE CULBM_ENABLE_MPI=1)
        target_link_libraries(CULBM-device-codec-gpu-test PRIVATE MPI::MPI_CXX)
    endif()
    add_test(NAME CULBM.DeviceCodecGPU COMMAND CULBM-device-codec-gpu-test)
    set_tests_properties(CULBM.DeviceCodecGPU PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 300
        RUN_SERIAL TRUE
        LABELS "gpu;device;codec")

    add_executable(CULBM-nested-surface-ibm-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_nested_surface_ibm.cpp"
        ${CULBM_SURFACE_IBM_GPU_RUNTIME_TEST_SOURCES})
    target_compile_features(CULBM-nested-surface-ibm-test PRIVATE cxx_std_17)
    target_include_directories(CULBM-nested-surface-ibm-test PRIVATE
        "${CULBM_SOLVER_SRC}" "${CMAKE_CURRENT_LIST_DIR}/src" "${CUDAToolkit_INCLUDE_DIRS}")
    target_compile_definitions(CULBM-nested-surface-ibm-test PRIVATE CULBM_BACKEND=1
        "CULBM_INCLUDE_DIR=\"${CUDAToolkit_INCLUDE_DIRS}\"")
    target_link_libraries(CULBM-nested-surface-ibm-test PRIVATE
        CUDA::cuda_driver CUDA::nvrtc Threads::Threads)
    add_test(NAME CULBM.NestedSurfaceIBMGPU COMMAND CULBM-nested-surface-ibm-test)
    set_tests_properties(CULBM.NestedSurfaceIBMGPU PROPERTIES RUN_SERIAL TRUE TIMEOUT 240)

    add_executable(CULBM-nested-volumetric-gpu-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_nested_volumetric_gpu.cpp"
        ${CULBM_SURFACE_IBM_GPU_RUNTIME_TEST_SOURCES})
    target_compile_features(CULBM-nested-volumetric-gpu-test PRIVATE cxx_std_17)
    target_include_directories(CULBM-nested-volumetric-gpu-test PRIVATE
        "${CULBM_SOLVER_SRC}" "${CMAKE_CURRENT_LIST_DIR}/src" "${CUDAToolkit_INCLUDE_DIRS}")
    target_compile_definitions(CULBM-nested-volumetric-gpu-test PRIVATE CULBM_BACKEND=1
        "CULBM_INCLUDE_DIR=\"${CUDAToolkit_INCLUDE_DIRS}\"")
    target_link_libraries(CULBM-nested-volumetric-gpu-test PRIVATE
        CUDA::cuda_driver CUDA::nvrtc Threads::Threads)
    add_test(NAME CULBM.NestedVolumetricGPU COMMAND CULBM-nested-volumetric-gpu-test)
    set_tests_properties(
        CULBM.NestedVolumetricGPU PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        TIMEOUT 240
        LABELS "gpu;nested;volumetric")
    add_test(
        NAME CULBM.SurfaceIBMMultiGPU
        COMMAND CULBM-surface-ibm-gpu-test --multi-gpu)
    set_tests_properties(
        CULBM.SurfaceIBMGPU PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "gpu;surface-ibm;integration")
    set_tests_properties(
        CULBM.SurfaceIBMMultiGPU PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "gpu;multi-gpu;surface-ibm;integration")
    if(MPI_CXX_FOUND AND MPIEXEC_EXECUTABLE)
        add_test(
            NAME CULBM.SurfaceIBMMPIGPU
            COMMAND
                "${CMAKE_COMMAND}" -E env
                "CULBM_MPI=1"
                "${MPIEXEC_EXECUTABLE}"
                "${MPIEXEC_NUMPROC_FLAG}" "2"
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:CULBM-surface-ibm-gpu-test>
                ${MPIEXEC_POSTFLAGS}
                --mpi)
        set_tests_properties(
            CULBM.SurfaceIBMMPIGPU PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 600
            RUN_SERIAL TRUE
            PROCESSORS 2
            LABELS "gpu;mpi;surface-ibm;integration")
    endif()

    foreach(CULBM_SURFACE_IBM_VALIDATION_PRECISION IN ITEMS fp16c fp32)
        set(CULBM_SURFACE_IBM_VALIDATION_TARGET
            "CULBM-surface-ibm-physical-validation-${CULBM_SURFACE_IBM_VALIDATION_PRECISION}")
        add_executable(
            ${CULBM_SURFACE_IBM_VALIDATION_TARGET}
            "${CMAKE_CURRENT_LIST_DIR}/validation/surface_ibm/surface_ibm_physical_validation.cpp"
            ${CULBM_SURFACE_IBM_GPU_RUNTIME_TEST_SOURCES})
        target_compile_features(
            ${CULBM_SURFACE_IBM_VALIDATION_TARGET} PRIVATE cxx_std_17)
        target_include_directories(
            ${CULBM_SURFACE_IBM_VALIDATION_TARGET} PRIVATE
            "${CULBM_SOLVER_SRC}"
            "${CMAKE_CURRENT_LIST_DIR}/src"
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(
            ${CULBM_SURFACE_IBM_VALIDATION_TARGET} PRIVATE
            CUDA::cuda_driver
            CUDA::nvrtc
            Threads::Threads)
        target_compile_definitions(
            ${CULBM_SURFACE_IBM_VALIDATION_TARGET} PRIVATE
            CULBM_BACKEND=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
        if(CULBM_SURFACE_IBM_VALIDATION_PRECISION STREQUAL "fp32")
            target_compile_definitions(
                ${CULBM_SURFACE_IBM_VALIDATION_TARGET} PRIVATE
                CULBM_DISABLE_FP16C=1)
        endif()
        if(MPI_CXX_FOUND)
            target_compile_definitions(
                ${CULBM_SURFACE_IBM_VALIDATION_TARGET} PRIVATE
                CULBM_ENABLE_MPI=1)
            target_link_libraries(
                ${CULBM_SURFACE_IBM_VALIDATION_TARGET} PRIVATE MPI::MPI_CXX)
        endif()
    endforeach()

    foreach(CULBM_SURFACE_IBM_3D_CHANNEL_PRECISION IN ITEMS fp16c fp32)
        set(CULBM_SURFACE_IBM_3D_CHANNEL_TARGET
            "CULBM-surface-ibm-3d-channel-validation-${CULBM_SURFACE_IBM_3D_CHANNEL_PRECISION}")
        add_executable(
            ${CULBM_SURFACE_IBM_3D_CHANNEL_TARGET}
            "${CMAKE_CURRENT_LIST_DIR}/validation/surface_ibm/surface_ibm_3d_channel_validation.cpp"
            ${CULBM_SURFACE_IBM_GPU_RUNTIME_TEST_SOURCES})
        target_compile_features(
            ${CULBM_SURFACE_IBM_3D_CHANNEL_TARGET} PRIVATE cxx_std_17)
        target_include_directories(
            ${CULBM_SURFACE_IBM_3D_CHANNEL_TARGET} PRIVATE
            "${CULBM_SOLVER_SRC}"
            "${CMAKE_CURRENT_LIST_DIR}/src"
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(
            ${CULBM_SURFACE_IBM_3D_CHANNEL_TARGET} PRIVATE
            CUDA::cuda_driver
            CUDA::nvrtc
            Threads::Threads)
        target_compile_definitions(
            ${CULBM_SURFACE_IBM_3D_CHANNEL_TARGET} PRIVATE
            CULBM_BACKEND=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
        if(CULBM_SURFACE_IBM_3D_CHANNEL_PRECISION STREQUAL "fp32")
            target_compile_definitions(
                ${CULBM_SURFACE_IBM_3D_CHANNEL_TARGET} PRIVATE
                CULBM_DISABLE_FP16C=1)
        endif()
        if(MPI_CXX_FOUND)
            target_compile_definitions(
                ${CULBM_SURFACE_IBM_3D_CHANNEL_TARGET} PRIVATE
                CULBM_ENABLE_MPI=1)
            target_link_libraries(
                ${CULBM_SURFACE_IBM_3D_CHANNEL_TARGET} PRIVATE MPI::MPI_CXX)
        endif()
    endforeach()

    foreach(CULBM_SURFACE_IBM_STL_VALIDATION_PRECISION IN ITEMS fp16c fp32)
        set(CULBM_SURFACE_IBM_STL_VALIDATION_TARGET
            "CULBM-surface-ibm-stl-pipeline-validation-${CULBM_SURFACE_IBM_STL_VALIDATION_PRECISION}")
        add_executable(
            ${CULBM_SURFACE_IBM_STL_VALIDATION_TARGET}
            "${CMAKE_CURRENT_LIST_DIR}/validation/surface_ibm/surface_ibm_stl_pipeline_validation.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/src/surface_ibm_geometry.cpp"
            ${CULBM_SURFACE_IBM_GPU_RUNTIME_TEST_SOURCES})
        target_compile_features(
            ${CULBM_SURFACE_IBM_STL_VALIDATION_TARGET} PRIVATE cxx_std_17)
        target_include_directories(
            ${CULBM_SURFACE_IBM_STL_VALIDATION_TARGET} PRIVATE
            "${CULBM_SOLVER_SRC}"
            "${CMAKE_CURRENT_LIST_DIR}/src"
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(
            ${CULBM_SURFACE_IBM_STL_VALIDATION_TARGET} PRIVATE
            CUDA::cuda_driver
            CUDA::nvrtc
            Threads::Threads)
        target_compile_definitions(
            ${CULBM_SURFACE_IBM_STL_VALIDATION_TARGET} PRIVATE
            CULBM_BACKEND=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
        if(CULBM_SURFACE_IBM_STL_VALIDATION_PRECISION STREQUAL "fp32")
            target_compile_definitions(
                ${CULBM_SURFACE_IBM_STL_VALIDATION_TARGET} PRIVATE
                CULBM_DISABLE_FP16C=1)
        endif()
        if(MPI_CXX_FOUND)
            target_compile_definitions(
                ${CULBM_SURFACE_IBM_STL_VALIDATION_TARGET} PRIVATE
                CULBM_ENABLE_MPI=1)
            target_link_libraries(
                ${CULBM_SURFACE_IBM_STL_VALIDATION_TARGET} PRIVATE MPI::MPI_CXX)
        endif()
    endforeach()

    foreach(CULBM_SURFACE_IBM_RAY_VALIDATION_PRECISION IN ITEMS fp16c fp32)
        set(CULBM_SURFACE_IBM_RAY_VALIDATION_TARGET
            "CULBM-surface-ibm-ray-pipeline-validation-${CULBM_SURFACE_IBM_RAY_VALIDATION_PRECISION}")
        add_executable(
            ${CULBM_SURFACE_IBM_RAY_VALIDATION_TARGET}
            "${CMAKE_CURRENT_LIST_DIR}/validation/surface_ibm/surface_ibm_ray_pipeline_validation.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/src/surface_ibm_geometry.cpp"
            ${CULBM_SURFACE_IBM_GPU_RUNTIME_TEST_SOURCES})
        target_compile_features(
            ${CULBM_SURFACE_IBM_RAY_VALIDATION_TARGET} PRIVATE cxx_std_17)
        target_include_directories(
            ${CULBM_SURFACE_IBM_RAY_VALIDATION_TARGET} PRIVATE
            "${CULBM_SOLVER_SRC}"
            "${CMAKE_CURRENT_LIST_DIR}/src"
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(
            ${CULBM_SURFACE_IBM_RAY_VALIDATION_TARGET} PRIVATE
            CUDA::cuda_driver
            CUDA::nvrtc
            Threads::Threads)
        target_compile_definitions(
            ${CULBM_SURFACE_IBM_RAY_VALIDATION_TARGET} PRIVATE
            CULBM_BACKEND=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
        if(CULBM_SURFACE_IBM_RAY_VALIDATION_PRECISION STREQUAL "fp32")
            target_compile_definitions(
                ${CULBM_SURFACE_IBM_RAY_VALIDATION_TARGET} PRIVATE
                CULBM_DISABLE_FP16C=1)
        endif()
        if(MPI_CXX_FOUND)
            target_compile_definitions(
                ${CULBM_SURFACE_IBM_RAY_VALIDATION_TARGET} PRIVATE
                CULBM_ENABLE_MPI=1)
            target_link_libraries(
                ${CULBM_SURFACE_IBM_RAY_VALIDATION_TARGET} PRIVATE MPI::MPI_CXX)
        endif()
    endforeach()

    add_executable(
        CULBM-particle-advection-config-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_particle_advection_config.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_config.cpp")
    target_compile_features(
        CULBM-particle-advection-config-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-particle-advection-config-test PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src")
    add_test(
        NAME CULBM.ParticleAdvectionConfig
        COMMAND CULBM-particle-advection-config-test)

    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/tests/test_particle_advection_gpu.cpp")
        add_executable(
            CULBM-particle-advection-gpu-test
            "${CMAKE_CURRENT_LIST_DIR}/tests/test_particle_advection_gpu.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
            ${CULBM_GPU_SOURCES}
            "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp")
        target_compile_features(
            CULBM-particle-advection-gpu-test PRIVATE cxx_std_17)
        target_include_directories(
            CULBM-particle-advection-gpu-test PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/src"
            "${CMAKE_CURRENT_LIST_DIR}/src/solver"
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(
            CULBM-particle-advection-gpu-test PRIVATE
            CUDA::cuda_driver
            CUDA::nvrtc
            Threads::Threads)
        target_compile_definitions(
            CULBM-particle-advection-gpu-test PRIVATE
            CULBM_BACKEND=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
        add_test(
            NAME CULBM.ParticleAdvectionGPU
            COMMAND CULBM-particle-advection-gpu-test)
        set_tests_properties(
            CULBM.ParticleAdvectionGPU PROPERTIES
            SKIP_RETURN_CODE 77
            LABELS "gpu;advection")
    endif()

    add_executable(
        CULBM-subgrid-formula-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_subgrid_formula.cpp")
    target_compile_features(
        CULBM-subgrid-formula-test PRIVATE cxx_std_17)
    add_test(
        NAME CULBM.SubgridFormula
        COMMAND CULBM-subgrid-formula-test)
    set_tests_properties(
        CULBM.SubgridFormula PROPERTIES LABELS "subgrid;formula")

    set(CULBM_SUBGRID_GPU_TEST_SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_subgrid_gpu.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
        ${CULBM_SURFACE_IBM_RUNTIME_SOURCES}
        ${CULBM_REPORT_SOURCES}
        "${CULBM_SOLVER_SRC}/kernel.cpp"
        ${CULBM_SOLVER_CORE_SOURCES}
        ${CULBM_GPU_SOURCES}
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_runtime.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_solver.cpp")
    add_executable(
        CULBM-subgrid-gpu-test
        ${CULBM_SUBGRID_GPU_TEST_SOURCES})
    target_compile_features(
        CULBM-subgrid-gpu-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-subgrid-gpu-test PRIVATE
        "${CULBM_SOLVER_SRC}"
        "${CMAKE_CURRENT_LIST_DIR}/src"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(
        CULBM-subgrid-gpu-test PRIVATE
        CUDA::cuda_driver
        CUDA::nvrtc
        Threads::Threads)
    target_compile_definitions(
        CULBM-subgrid-gpu-test PRIVATE
        CULBM_BACKEND=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
    add_test(
        NAME CULBM.SubgridGPU
        COMMAND CULBM-subgrid-gpu-test)
    add_test(
        NAME CULBM.SubgridMultiGPU
        COMMAND CULBM-subgrid-gpu-test --multi-gpu)
    set_tests_properties(
        CULBM.SubgridGPU PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "gpu;subgrid;temperature;analytic")
    set_tests_properties(
        CULBM.SubgridMultiGPU PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "gpu;multi-gpu;subgrid;temperature;halo")

    # Manual FP32 performance/diagnostic counterpart.  Large performance
    # modes are intentionally not registered with default CTest.
    add_executable(
        CULBM-subgrid-gpu-fp32-test
        EXCLUDE_FROM_ALL
        ${CULBM_SUBGRID_GPU_TEST_SOURCES})
    target_compile_features(
        CULBM-subgrid-gpu-fp32-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-subgrid-gpu-fp32-test PRIVATE
        "${CULBM_SOLVER_SRC}"
        "${CMAKE_CURRENT_LIST_DIR}/src"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(
        CULBM-subgrid-gpu-fp32-test PRIVATE
        CUDA::cuda_driver
        CUDA::nvrtc
        Threads::Threads)
    target_compile_definitions(
        CULBM-subgrid-gpu-fp32-test PRIVATE
        CULBM_BACKEND=1
        CULBM_DISABLE_FP16C=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")

    # Bounce-back wall model: deterministic laminar/no-slip/mass checks in
    # default CTest, plus manual --channel (WMLES research) and --benchmark
    # throughput modes that are not registered as tests.
    set(CULBM_WALL_MODEL_GPU_TEST_SOURCES ${CULBM_SUBGRID_GPU_TEST_SOURCES})
    list(REMOVE_ITEM CULBM_WALL_MODEL_GPU_TEST_SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_subgrid_gpu.cpp")
    list(APPEND CULBM_WALL_MODEL_GPU_TEST_SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_wall_model_gpu.cpp")
    add_executable(
        CULBM-wall-model-gpu-test
        ${CULBM_WALL_MODEL_GPU_TEST_SOURCES})
    target_compile_features(
        CULBM-wall-model-gpu-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-wall-model-gpu-test PRIVATE
        "${CULBM_SOLVER_SRC}"
        "${CMAKE_CURRENT_LIST_DIR}/src"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(
        CULBM-wall-model-gpu-test PRIVATE
        CUDA::cuda_driver
        CUDA::nvrtc
        Threads::Threads)
    target_compile_definitions(
        CULBM-wall-model-gpu-test PRIVATE
        CULBM_BACKEND=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
    # FP32 DDF storage: the analytic laminar checks use body forces of order
    # 1e-6, below the FP16C quantization of the shifted populations.
    add_executable(
        CULBM-wall-model-gpu-fp32-test
        ${CULBM_WALL_MODEL_GPU_TEST_SOURCES})
    target_compile_features(
        CULBM-wall-model-gpu-fp32-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-wall-model-gpu-fp32-test PRIVATE
        "${CULBM_SOLVER_SRC}"
        "${CMAKE_CURRENT_LIST_DIR}/src"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(
        CULBM-wall-model-gpu-fp32-test PRIVATE
        CUDA::cuda_driver
        CUDA::nvrtc
        Threads::Threads)
    target_compile_definitions(
        CULBM-wall-model-gpu-fp32-test PRIVATE
        CULBM_BACKEND=1
        CULBM_DISABLE_FP16C=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
    # Research-only open-channel (neutral ABL) driver: free-slip lid through
    # the validation-only FREE_SLIP_BOUNDARIES path.  Not a CTest target.
    add_executable(
        CULBM-wall-model-gpu-fp32-freeslip-test
        EXCLUDE_FROM_ALL
        ${CULBM_WALL_MODEL_GPU_TEST_SOURCES})
    target_compile_features(
        CULBM-wall-model-gpu-fp32-freeslip-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-wall-model-gpu-fp32-freeslip-test PRIVATE
        "${CULBM_SOLVER_SRC}"
        "${CMAKE_CURRENT_LIST_DIR}/src"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(
        CULBM-wall-model-gpu-fp32-freeslip-test PRIVATE
        CUDA::cuda_driver
        CUDA::nvrtc
        Threads::Threads)
    target_compile_definitions(
        CULBM-wall-model-gpu-fp32-freeslip-test PRIVATE
        CULBM_BACKEND=1
        CULBM_DISABLE_FP16C=1
        FREE_SLIP_BOUNDARIES=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
    add_test(
        NAME CULBM.WallModelGPU
        COMMAND CULBM-wall-model-gpu-fp32-test)
    add_test(
        NAME CULBM.WallModelMultiGPU
        COMMAND CULBM-wall-model-gpu-test --multi-gpu)
    set_tests_properties(
        CULBM.WallModelGPU PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "gpu;wall-model;analytic")
    set_tests_properties(
        CULBM.WallModelMultiGPU PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "gpu;multi-gpu;wall-model;halo")
    # The same analytic wall-model and subgrid checks on the D3Q27 momentum lattice.
    add_test(
        NAME CULBM.WallModelGPU.D3Q27
        COMMAND CULBM-wall-model-gpu-fp32-test)
    add_test(
        NAME CULBM.WallModelMultiGPU.D3Q27
        COMMAND CULBM-wall-model-gpu-test --multi-gpu)
    add_test(
        NAME CULBM.SubgridGPU.D3Q27
        COMMAND CULBM-subgrid-gpu-test)
    set_tests_properties(
        CULBM.WallModelGPU.D3Q27 CULBM.WallModelMultiGPU.D3Q27 CULBM.SubgridGPU.D3Q27 PROPERTIES
        ENVIRONMENT "CULBM_TEST_VELOCITY_SET=D3Q27"
        SKIP_RETURN_CODE 77
        TIMEOUT 900
        RUN_SERIAL TRUE
        LABELS "gpu;d3q27;wall-model;subgrid")

    add_executable(
        CULBM-temperature-analytic-gpu-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_temperature_analytic_gpu.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
        ${CULBM_SURFACE_IBM_RUNTIME_SOURCES}
        ${CULBM_REPORT_SOURCES}
        "${CULBM_SOLVER_SRC}/kernel.cpp"
        ${CULBM_SOLVER_CORE_SOURCES}
        ${CULBM_GPU_SOURCES}
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_runtime.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_solver.cpp")
    target_compile_features(
        CULBM-temperature-analytic-gpu-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-temperature-analytic-gpu-test PRIVATE
        "${CULBM_SOLVER_SRC}"
        "${CMAKE_CURRENT_LIST_DIR}/src"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(
        CULBM-temperature-analytic-gpu-test PRIVATE
        CUDA::cuda_driver
        CUDA::nvrtc
        Threads::Threads)
    target_compile_definitions(
        CULBM-temperature-analytic-gpu-test PRIVATE
        CULBM_BACKEND=1
        FREE_SLIP_BOUNDARIES=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
    add_test(
        NAME CULBM.TemperatureAnalyticGPU
        COMMAND CULBM-temperature-analytic-gpu-test)
    set_tests_properties(
        CULBM.TemperatureAnalyticGPU PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "gpu;temperature;analytic")

    # Independently compiles and runs the existing temperature, Guo-force, and
    # CELL_EQUILIBRIUM analytic checks through TRT momentum collision. This is a short
    # algebra/implementation regression, not a long-horizon nonlinear GEA
    # stability certificate.
    add_executable(
        CULBM-temperature-analytic-gpu-trt-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_temperature_analytic_gpu.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
        ${CULBM_SURFACE_IBM_RUNTIME_SOURCES}
        ${CULBM_REPORT_SOURCES}
        "${CULBM_SOLVER_SRC}/kernel.cpp"
        ${CULBM_SOLVER_CORE_SOURCES}
        ${CULBM_GPU_SOURCES}
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_runtime.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_solver.cpp")
    target_compile_features(
        CULBM-temperature-analytic-gpu-trt-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-temperature-analytic-gpu-trt-test PRIVATE
        "${CULBM_SOLVER_SRC}"
        "${CMAKE_CURRENT_LIST_DIR}/src"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(
        CULBM-temperature-analytic-gpu-trt-test PRIVATE
        CUDA::cuda_driver
        CUDA::nvrtc
        Threads::Threads)
    target_compile_definitions(
        CULBM-temperature-analytic-gpu-trt-test PRIVATE
        CULBM_BACKEND=1
        CULBM_USE_TRT=1
        FREE_SLIP_BOUNDARIES=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
    add_test(
        NAME CULBM.TemperatureAnalyticGPU.TRT
        COMMAND CULBM-temperature-analytic-gpu-trt-test)
    set_tests_properties(
        CULBM.TemperatureAnalyticGPU.TRT PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "gpu;temperature;analytic;trt")

    # Keep the optional manual macro-field reconstruction path compiled and
    # physically checked even though production enables UPDATE_FIELDS.
    add_executable(
        CULBM-manual-fields-forcing-gpu-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_temperature_analytic_gpu.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
        ${CULBM_SURFACE_IBM_RUNTIME_SOURCES}
        ${CULBM_REPORT_SOURCES}
        "${CULBM_SOLVER_SRC}/kernel.cpp"
        ${CULBM_SOLVER_CORE_SOURCES}
        ${CULBM_GPU_SOURCES}
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_runtime.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_solver.cpp")
    target_compile_features(
        CULBM-manual-fields-forcing-gpu-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-manual-fields-forcing-gpu-test PRIVATE
        "${CULBM_SOLVER_SRC}"
        "${CMAKE_CURRENT_LIST_DIR}/src"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(
        CULBM-manual-fields-forcing-gpu-test PRIVATE
        CUDA::cuda_driver
        CUDA::nvrtc
        Threads::Threads)
    target_compile_definitions(
        CULBM-manual-fields-forcing-gpu-test PRIVATE
        CULBM_BACKEND=1
        CULBM_DISABLE_UPDATE_FIELDS=1
        FREE_SLIP_BOUNDARIES=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")
    add_test(
        NAME CULBM.ManualFieldsForcingGPU
        COMMAND CULBM-manual-fields-forcing-gpu-test
                --manual-update-fields-forcing)
    set_tests_properties(
        CULBM.ManualFieldsForcingGPU PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "gpu;temperature;forcing;manual-fields")

    add_test(
        NAME CULBM.TemperatureHaloMultiGPU
        COMMAND CULBM-temperature-analytic-gpu-test --multi-gpu-halo)
    set_tests_properties(
        CULBM.TemperatureHaloMultiGPU PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600
        RUN_SERIAL TRUE
        LABELS "gpu;multi-gpu;temperature;halo")

    add_executable(
        CULBM-gea-validation
        "${CMAKE_CURRENT_LIST_DIR}/validation/gea/gea_validation.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
        ${CULBM_SURFACE_IBM_RUNTIME_SOURCES}
        ${CULBM_REPORT_SOURCES}
        "${CULBM_SOLVER_SRC}/kernel.cpp"
        ${CULBM_SOLVER_CORE_SOURCES}
        ${CULBM_GPU_SOURCES}
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_runtime.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_solver.cpp")
    target_compile_features(
        CULBM-gea-validation PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-gea-validation PRIVATE
        "${CULBM_SOLVER_SRC}"
        "${CMAKE_CURRENT_LIST_DIR}/src"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(
        CULBM-gea-validation PRIVATE
        CUDA::cuda_driver
        CUDA::nvrtc
        Threads::Threads)
    target_compile_definitions(
        CULBM-gea-validation PRIVATE
        CULBM_BACKEND=1
        FREE_SLIP_BOUNDARIES=1
        CULBM_DISABLE_FP16C=1
        CULBM_DISABLE_SUBGRID=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")

    # SGS counterpart to the frozen AV validation target.  Its runtime defaults
    # select the GEA literature coefficient and full momentum/scalar closure;
    # command-line overrides remain available for controlled A/B runs.
    add_executable(
        CULBM-gea-validation-sgs
        "${CMAKE_CURRENT_LIST_DIR}/validation/gea/gea_validation.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
        ${CULBM_SURFACE_IBM_RUNTIME_SOURCES}
        ${CULBM_REPORT_SOURCES}
        "${CULBM_SOLVER_SRC}/kernel.cpp"
        ${CULBM_SOLVER_CORE_SOURCES}
        ${CULBM_GPU_SOURCES}
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_runtime.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_solver.cpp")
    target_compile_features(
        CULBM-gea-validation-sgs PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-gea-validation-sgs PRIVATE
        "${CULBM_SOLVER_SRC}"
        "${CMAKE_CURRENT_LIST_DIR}/src"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(
        CULBM-gea-validation-sgs PRIVATE
        CUDA::cuda_driver
        CUDA::nvrtc
        Threads::Threads)
    target_compile_definitions(
        CULBM-gea-validation-sgs PRIVATE
        CULBM_BACKEND=1
        CULBM_GEA_SGS_DEFAULT=1
        FREE_SLIP_BOUNDARIES=1
        CULBM_DISABLE_FP16C=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")

    # Isolated stability control for the GEA analogue. TRT applies to the
    # momentum populations only; thermal populations retain their existing
    # single-relaxation-time collision.
    add_executable(
        CULBM-gea-validation-trt
        "${CMAKE_CURRENT_LIST_DIR}/validation/gea/gea_validation.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
        ${CULBM_SURFACE_IBM_RUNTIME_SOURCES}
        ${CULBM_REPORT_SOURCES}
        "${CULBM_SOLVER_SRC}/kernel.cpp"
        ${CULBM_SOLVER_CORE_SOURCES}
        ${CULBM_GPU_SOURCES}
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_runtime.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_nested_solver.cpp")
    target_compile_features(
        CULBM-gea-validation-trt PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-gea-validation-trt PRIVATE
        "${CULBM_SOLVER_SRC}"
        "${CMAKE_CURRENT_LIST_DIR}/src"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(
        CULBM-gea-validation-trt PRIVATE
        CUDA::cuda_driver
        CUDA::nvrtc
        Threads::Threads)
    target_compile_definitions(
        CULBM-gea-validation-trt PRIVATE
        CULBM_BACKEND=1
        CULBM_USE_TRT=1
        FREE_SLIP_BOUNDARIES=1
        CULBM_DISABLE_FP16C=1
        CULBM_DISABLE_SUBGRID=1
        "CULBM_INCLUDE_DIR=\"${CULBM_CUDA_PRIMARY_INCLUDE_DIR}\"")

    add_executable(
        CULBM-topology-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_domain_topology.cpp")
    target_compile_features(CULBM-topology-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-topology-test PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src")
    add_test(
        NAME CULBM.DomainTopology
        COMMAND CULBM-topology-test)

    # Run progress: rate window, ETA model, console and JSON texts, command
    # line (no GPU).
    add_executable(
        CULBM-run-progress-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_run_progress.cpp"
        "${CULBM_APP_SRC}/progress.cpp"
        "${CULBM_APP_SRC}/launch.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp")
    target_compile_features(CULBM-run-progress-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-run-progress-test PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src")
    target_link_libraries(CULBM-run-progress-test PRIVATE Threads::Threads)
    add_test(
        NAME CULBM.RunProgress
        COMMAND
            "${CMAKE_COMMAND}" -E env
            --unset=CULBM_PROGRESS
            --unset=LUW_PROGRESS_MODE
            --unset=CULBM_MPI
            $<TARGET_FILE:CULBM-run-progress-test>)

    # Deck reading, run schedule and configuration-dump formatting (no GPU).
    add_executable(
        CULBM-case-deck-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_case_deck.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/case/config_dump.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/case/deck.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/run/schedule.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp")
    target_compile_features(CULBM-case-deck-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-case-deck-test PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src")
    target_link_libraries(CULBM-case-deck-test PRIVATE Threads::Threads)
    add_test(
        NAME CULBM.CaseDeck
        COMMAND CULBM-case-deck-test)

    # Host-side solver layout: decomposition, halo routing, cell field table
    # and step stages (no GPU).
    add_executable(
        CULBM-solver-layout-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_solver_layout.cpp"
        "${CULBM_SOLVER_SRC}/field_table.cpp"
        "${CULBM_SOLVER_SRC}/step_pipeline.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/gpu/vram.cpp")
    target_compile_features(CULBM-solver-layout-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-solver-layout-test PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src")
    add_test(
        NAME CULBM.SolverLayout
        COMMAND CULBM-solver-layout-test)

    add_executable(
        CULBM-nested-grid-plan-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_nested_grid_plan.cpp")
    target_compile_features(
        CULBM-nested-grid-plan-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-nested-grid-plan-test PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src")
    add_test(
        NAME CULBM.NestedGridPlan
        COMMAND CULBM-nested-grid-plan-test)

    add_executable(
        CULBM-nested-interface-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_nested_interface.cpp")
    target_compile_features(
        CULBM-nested-interface-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-nested-interface-test PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src")
    add_test(
        NAME CULBM.NestedInterface
        COMMAND CULBM-nested-interface-test)

    add_executable(
        CULBM-physical-output-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_physical_output.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/physical_output.cpp")
    target_compile_features(
        CULBM-physical-output-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-physical-output-test PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src")
    if(CULBM_VTKHDF_ENABLED)
        target_include_directories(
            CULBM-physical-output-test PRIVATE ${CULBM_HDF5_INCLUDE_DIRS})
        target_link_libraries(
            CULBM-physical-output-test PRIVATE ${CULBM_HDF5_LINK_LIBRARIES})
        target_compile_definitions(
            CULBM-physical-output-test PRIVATE CULBM_HAS_VTKHDF=1)
    else()
        target_compile_definitions(
            CULBM-physical-output-test PRIVATE CULBM_HAS_VTKHDF=0)
    endif()
    add_test(
        NAME CULBM.PhysicalOutput
        COMMAND CULBM-physical-output-test)
    if(WIN32 AND CULBM_HDF5_RUNTIME_TARGET)
        set_tests_properties(
            CULBM.PhysicalOutput PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:${CULBM_HDF5_RUNTIME_TARGET}>")
    endif()

    add_executable(
        CULBM-mpi-unavailable-runtime-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_mpi_direct_runtime.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp")
    target_compile_features(
        CULBM-mpi-unavailable-runtime-test PRIVATE cxx_std_17)
    target_include_directories(
        CULBM-mpi-unavailable-runtime-test PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src")
    add_test(
        NAME CULBM.MPIUnavailableDirectRuntime
        COMMAND
            "${CMAKE_COMMAND}" -E env
            --unset=CULBM_MPI
            --unset=PMI_RANK
            --unset=PMIX_RANK
            --unset=OMPI_COMM_WORLD_RANK
            --unset=MV2_COMM_WORLD_RANK
            --unset=MPI_LOCALRANKID
            --unset=SLURM_PROCID
            --unset=ALPS_APP_PE
            --unset=PALS_RANKID
            $<TARGET_FILE:CULBM-mpi-unavailable-runtime-test>)
    add_test(
        NAME CULBM.MPIUnavailableForcedRuntime
        COMMAND
            "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:CULBM-mpi-unavailable-runtime-test>"
            "-DTEST_MODE=forced"
            -P
            "${CMAKE_CURRENT_LIST_DIR}/tests/expect_mpi_unavailable.cmake")
    add_test(
        NAME CULBM.MPIUnavailableLauncherRuntime
        COMMAND
            "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:CULBM-mpi-unavailable-runtime-test>"
            "-DTEST_MODE=launcher"
            -P
            "${CMAKE_CURRENT_LIST_DIR}/tests/expect_mpi_unavailable.cmake")

    if(MPI_CXX_FOUND AND MPIEXEC_EXECUTABLE)
        add_executable(
            CULBM-mpi-direct-runtime-test
            "${CMAKE_CURRENT_LIST_DIR}/tests/test_mpi_direct_runtime.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp")
        target_compile_features(
            CULBM-mpi-direct-runtime-test PRIVATE cxx_std_17)
        target_include_directories(
            CULBM-mpi-direct-runtime-test PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/src")
        target_compile_definitions(
            CULBM-mpi-direct-runtime-test PRIVATE
            CULBM_ENABLE_MPI=1)
        target_link_libraries(
            CULBM-mpi-direct-runtime-test PRIVATE MPI::MPI_CXX)
        add_test(
            NAME CULBM.MPIDirectRuntime
            COMMAND
                "${CMAKE_COMMAND}" -E env
                --unset=CULBM_MPI
                $<TARGET_FILE:CULBM-mpi-direct-runtime-test>)
        add_test(
            NAME CULBM.MPIForcedSingleRuntime
            COMMAND
                CULBM-mpi-direct-runtime-test
                --expect-initialized)
        set_tests_properties(
            CULBM.MPIForcedSingleRuntime PROPERTIES
            ENVIRONMENT "CULBM_MPI=1")

        add_executable(
            CULBM-mpi-runtime-test
            "${CMAKE_CURRENT_LIST_DIR}/tests/test_mpi_runtime.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/src/cuda_mpi.cpp")
        target_compile_features(
            CULBM-mpi-runtime-test PRIVATE cxx_std_17)
        target_include_directories(
            CULBM-mpi-runtime-test PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/src")
        target_compile_definitions(
            CULBM-mpi-runtime-test PRIVATE
            CULBM_ENABLE_MPI=1)
        target_link_libraries(
            CULBM-mpi-runtime-test PRIVATE
            MPI::MPI_CXX
            Threads::Threads)
        add_test(
            NAME CULBM.MPIRuntime
            COMMAND
                "${CMAKE_COMMAND}" -E env
                "CULBM_MPI=1"
                "${MPIEXEC_EXECUTABLE}"
                "${MPIEXEC_NUMPROC_FLAG}" "2"
                $<TARGET_FILE:CULBM-mpi-runtime-test>)
        add_test(
            NAME CULBM.MPIAutoDetectRuntime
            COMMAND
                "${CMAKE_COMMAND}" -E env
                --unset=CULBM_MPI
                "${MPIEXEC_EXECUTABLE}"
                "${MPIEXEC_NUMPROC_FLAG}" "2"
                $<TARGET_FILE:CULBM-mpi-runtime-test>)
        set_tests_properties(
            CULBM.MPIRuntime
            CULBM.MPIAutoDetectRuntime
            PROPERTIES
            TIMEOUT 120
            PROCESSORS 2
            LABELS "mpi;graphics;concurrency")
    endif()
endif()
