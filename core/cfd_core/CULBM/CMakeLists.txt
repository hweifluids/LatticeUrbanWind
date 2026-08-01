cmake_minimum_required(VERSION 3.24)

project(CULBM LANGUAGES C CXX)

include(CTest)
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(Threads REQUIRED)
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
if(CULBM_ENABLE_MPI)
    find_package(MPI QUIET COMPONENTS CXX)
    if(NOT MPI_CXX_FOUND)
        message(WARNING
            "MPI C++ support was not found. CULBM will retain its "
            "single-process arbitrary-GPU mode, but mpiexec multi-node "
            "execution will be disabled.")
    endif()
endif()

# CULBM owns its adapted host solver sources.  Keep this list explicit so a
# fresh checkout never reads, copies, or patches an external solver tree.
set(CULBM_SOLVER_SRC "${CMAKE_CURRENT_LIST_DIR}/src/solver")
set(CULBM_SOLVER_FILES
    defines.hpp
    fluxcorrection.cpp
    fluxcorrection.hpp
    graphics.cpp
    graphics.hpp
    info.cpp
    info.hpp
    interpolation.cpp
    interpolation.hpp
    interpolation_hd.cpp
    interpolation_hd.hpp
    kernel.cpp
    kernel.hpp
    kernel_cuda_source.inc
    lbm.cpp
    lbm.hpp
    main.cpp
    setup.cpp
    setup.hpp
    shapes.cpp
    shapes.hpp
    units.hpp
    utilities.hpp)

foreach(solver_file IN LISTS CULBM_SOLVER_FILES)
    if(NOT EXISTS "${CULBM_SOLVER_SRC}/${solver_file}")
        message(FATAL_ERROR
            "CULBM self-contained solver source is missing: "
            "${CULBM_SOLVER_SRC}/${solver_file}")
    endif()
endforeach()

set(CULBM_SOLVER_SOURCES
    "${CULBM_SOLVER_SRC}/fluxcorrection.cpp"
    "${CULBM_SOLVER_SRC}/graphics.cpp"
    "${CULBM_SOLVER_SRC}/info.cpp"
    "${CULBM_SOLVER_SRC}/interpolation.cpp"
    "${CULBM_SOLVER_SRC}/interpolation_hd.cpp"
    "${CULBM_SOLVER_SRC}/kernel.cpp"
    "${CULBM_SOLVER_SRC}/lbm.cpp"
    "${CULBM_SOLVER_SRC}/main.cpp"
    "${CULBM_SOLVER_SRC}/setup.cpp"
    "${CULBM_SOLVER_SRC}/shapes.cpp")

set(CULBM_SOURCES
    ${CULBM_SOLVER_SOURCES}
    "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_config.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/physical_output.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/cuda.cpp"
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
    "CULBM_INCLUDE_DIR=\"${CUDAToolkit_INCLUDE_DIRS}\"")
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
            "${CMAKE_CURRENT_LIST_DIR}/src/cuda.cpp"
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
            "CULBM_INCLUDE_DIR=\"${CUDAToolkit_INCLUDE_DIRS}\"")
        add_test(
            NAME CULBM.ParticleAdvectionGPU
            COMMAND CULBM-particle-advection-gpu-test)
        set_tests_properties(
            CULBM.ParticleAdvectionGPU PROPERTIES
            SKIP_RETURN_CODE 77
            LABELS "gpu;advection")
    endif()

    add_executable(
        CULBM-temperature-analytic-gpu-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_temperature_analytic_gpu.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
        "${CULBM_SOLVER_SRC}/info.cpp"
        "${CULBM_SOLVER_SRC}/kernel.cpp"
        "${CULBM_SOLVER_SRC}/lbm.cpp"
        "${CULBM_SOLVER_SRC}/shapes.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda.cpp"
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
        "CULBM_INCLUDE_DIR=\"${CUDAToolkit_INCLUDE_DIRS}\"")
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
    # TYPE_E analytic checks through TRT momentum collision. This is a short
    # algebra/implementation regression, not a long-horizon nonlinear GEA
    # stability certificate.
    add_executable(
        CULBM-temperature-analytic-gpu-trt-test
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_temperature_analytic_gpu.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
        "${CULBM_SOLVER_SRC}/info.cpp"
        "${CULBM_SOLVER_SRC}/kernel.cpp"
        "${CULBM_SOLVER_SRC}/lbm.cpp"
        "${CULBM_SOLVER_SRC}/shapes.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda.cpp"
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
        "CULBM_INCLUDE_DIR=\"${CUDAToolkit_INCLUDE_DIRS}\"")
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
        "${CULBM_SOLVER_SRC}/info.cpp"
        "${CULBM_SOLVER_SRC}/kernel.cpp"
        "${CULBM_SOLVER_SRC}/lbm.cpp"
        "${CULBM_SOLVER_SRC}/shapes.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda.cpp"
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
        "CULBM_INCLUDE_DIR=\"${CUDAToolkit_INCLUDE_DIRS}\"")
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
        "${CULBM_SOLVER_SRC}/info.cpp"
        "${CULBM_SOLVER_SRC}/kernel.cpp"
        "${CULBM_SOLVER_SRC}/lbm.cpp"
        "${CULBM_SOLVER_SRC}/shapes.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda.cpp"
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
        "CULBM_INCLUDE_DIR=\"${CUDAToolkit_INCLUDE_DIRS}\"")

    # Diagnostic counterpart to the AV validation target.  This intentionally
    # enables the solver's existing momentum-only Smagorinsky-Lilly kernel so
    # SGS-on/off behavior can be tested without changing the published AV
    # baseline executable.  The thermal population still has no scalar SGS
    # or turbulent-Prandtl closure; the runner records that limitation.
    add_executable(
        CULBM-gea-validation-sgs
        "${CMAKE_CURRENT_LIST_DIR}/validation/gea/gea_validation.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
        "${CULBM_SOLVER_SRC}/info.cpp"
        "${CULBM_SOLVER_SRC}/kernel.cpp"
        "${CULBM_SOLVER_SRC}/lbm.cpp"
        "${CULBM_SOLVER_SRC}/shapes.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda.cpp"
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
        FREE_SLIP_BOUNDARIES=1
        CULBM_DISABLE_FP16C=1
        "CULBM_INCLUDE_DIR=\"${CUDAToolkit_INCLUDE_DIRS}\"")

    # Isolated stability control for the GEA analogue. TRT applies to the
    # momentum populations only; thermal populations retain their existing
    # single-relaxation-time collision.
    add_executable(
        CULBM-gea-validation-trt
        "${CMAKE_CURRENT_LIST_DIR}/validation/gea/gea_validation.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/particle_advection_kernels.cpp"
        "${CULBM_SOLVER_SRC}/info.cpp"
        "${CULBM_SOLVER_SRC}/kernel.cpp"
        "${CULBM_SOLVER_SRC}/lbm.cpp"
        "${CULBM_SOLVER_SRC}/shapes.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/cuda.cpp"
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
        "CULBM_INCLUDE_DIR=\"${CUDAToolkit_INCLUDE_DIRS}\"")

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
