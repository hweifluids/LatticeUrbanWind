add_executable(luw_deck_tests
    test_deck.cpp
    test_terrain_voxel_config.cpp
)
target_link_libraries(luw_deck_tests PRIVATE luw_common)
target_compile_definitions(luw_deck_tests PRIVATE
    LUW_TEST_FIXTURE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../fixtures"
)
add_test(NAME native.deck_and_terrain_config COMMAND luw_deck_tests)

add_executable(luw_luwvox_cli_tests
    test_luwvox_cli.cpp
    ../../src/cli/commands/luwvox_command.cpp
    ../../src/cli/commands/luwvox_opencl.cpp
)
target_include_directories(luw_luwvox_cli_tests PRIVATE ../../src/cli/include)
target_link_libraries(luw_luwvox_cli_tests PRIVATE luw_common)
target_compile_definitions(luw_luwvox_cli_tests PRIVATE
    LUW_HAS_GDAL=0
    LUW_HAS_OPENCL=0
)
if(MSVC)
    target_compile_options(luw_luwvox_cli_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_luwvox_cli_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(luw_transluw_tests
    test_transluw.cpp
)
target_link_libraries(luw_transluw_tests PRIVATE
    luw_post
    VTK::CommonDataModel
    VTK::FiltersCore
    VTK::IOLegacy
)
target_compile_definitions(luw_transluw_tests PRIVATE
    LUW_TRANSLUW_ARTIFACT_ROOT="${CMAKE_CURRENT_BINARY_DIR}/transluw-retained-fixtures"
)
if(MSVC)
    target_compile_options(luw_transluw_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_transluw_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(COMMAND vtk_module_autoinit)
    vtk_module_autoinit(
        TARGETS luw_transluw_tests
        MODULES
            VTK::CommonDataModel
            VTK::FiltersCore
            VTK::IOLegacy
    )
endif()
add_test(NAME native.transluw COMMAND luw_transluw_tests)
if(WIN32)
    set_property(TEST native.transluw PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::CommonCore>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::IOLegacy>"
    )
endif()

# buildingscale keeps its NumPy-compatible distribution core testable in the
# dependency-light build and enables generated Shapefile/PNG coverage whenever
# GDAL is available.
add_executable(luw_buildingscale_tests
    test_buildingscale.cpp
    ../../src/cli/commands/buildingscale_command.cpp
)
target_include_directories(luw_buildingscale_tests PRIVATE ../../src/cli/include)
target_compile_definitions(luw_buildingscale_tests PRIVATE
    LUW_BUILDINGSCALE_ARTIFACT_ROOT="${CMAKE_CURRENT_BINARY_DIR}/buildingscale-retained-artifacts"
)
if(LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND)
    target_compile_definitions(luw_buildingscale_tests PRIVATE LUW_HAS_GDAL=1)
    if(TARGET GDAL::GDAL)
        target_link_libraries(luw_buildingscale_tests PRIVATE GDAL::GDAL)
    else()
        target_include_directories(luw_buildingscale_tests PRIVATE ${GDAL_INCLUDE_DIRS})
        target_link_libraries(luw_buildingscale_tests PRIVATE ${GDAL_LIBRARIES})
    endif()
else()
    target_compile_definitions(luw_buildingscale_tests PRIVATE LUW_HAS_GDAL=0)
endif()
if(MSVC)
    target_compile_options(luw_buildingscale_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_buildingscale_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.buildingscale COMMAND luw_buildingscale_tests)
if(WIN32 AND LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND AND TARGET GDAL::GDAL)
    set_property(TEST native.buildingscale PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:GDAL::GDAL>"
    )
endif()
add_test(NAME native.luwvox_cli COMMAND luw_luwvox_cli_tests)
if(WIN32)
    set_property(TEST native.luwvox_cli PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
    )
endif()

if(OpenCL_FOUND)
    add_executable(luw_luwvox_opencl_tests
        test_luwvox_opencl.cpp
        ../../src/cli/commands/luwvox_opencl.cpp
    )
    target_include_directories(luw_luwvox_opencl_tests PRIVATE ../../src/cli/include)
    target_compile_definitions(luw_luwvox_opencl_tests PRIVATE LUW_HAS_OPENCL=1)
    target_link_libraries(luw_luwvox_opencl_tests PRIVATE OpenCL::OpenCL)
    if(MSVC)
        target_compile_options(luw_luwvox_opencl_tests PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(luw_luwvox_opencl_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME native.luwvox_opencl COMMAND luw_luwvox_opencl_tests)
    set_tests_properties(native.luwvox_opencl PROPERTIES SKIP_RETURN_CODE 77)
endif()

add_executable(luw_luwbc_opencl_tests
    test_luwbc_opencl.cpp
    ../../src/cli/commands/luwbc_opencl.cpp
)
target_include_directories(luw_luwbc_opencl_tests PRIVATE ../../src/cli/include)
target_compile_definitions(luw_luwbc_opencl_tests PRIVATE LUW_LUWBC_MAX_WORK_ITEMS=64)
if(OpenCL_FOUND)
    target_compile_definitions(luw_luwbc_opencl_tests PRIVATE LUW_HAS_OPENCL=1)
    target_link_libraries(luw_luwbc_opencl_tests PRIVATE OpenCL::OpenCL)
else()
    target_compile_definitions(luw_luwbc_opencl_tests PRIVATE LUW_HAS_OPENCL=0)
endif()
if(MSVC)
    target_compile_options(luw_luwbc_opencl_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_luwbc_opencl_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.luwbc_opencl COMMAND luw_luwbc_opencl_tests)
set_tests_properties(native.luwbc_opencl PROPERTIES SKIP_RETURN_CODE 77)

add_executable(luw_detect_opencl_tests
    test_detect_opencl.cpp
    ../../src/cli/commands/detect_opencl_command.cpp
)
target_include_directories(luw_detect_opencl_tests PRIVATE ../../src/cli/include)
if(OpenCL_FOUND)
    target_compile_definitions(luw_detect_opencl_tests PRIVATE LUW_HAS_OPENCL=1)
    target_link_libraries(luw_detect_opencl_tests PRIVATE OpenCL::OpenCL)
else()
    target_compile_definitions(luw_detect_opencl_tests PRIVATE LUW_HAS_OPENCL=0)
endif()
if(MSVC)
    target_compile_options(luw_detect_opencl_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_detect_opencl_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.detect_opencl COMMAND luw_detect_opencl_tests)

add_executable(luw_detect_opencl_no_sdk_tests
    test_detect_opencl.cpp
    ../../src/cli/commands/detect_opencl_command.cpp
)
target_include_directories(luw_detect_opencl_no_sdk_tests PRIVATE ../../src/cli/include)
target_compile_definitions(luw_detect_opencl_no_sdk_tests PRIVATE LUW_HAS_OPENCL=0)
if(MSVC)
    target_compile_options(luw_detect_opencl_no_sdk_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_detect_opencl_no_sdk_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.detect_opencl_no_sdk COMMAND luw_detect_opencl_no_sdk_tests)

add_executable(luw_accelerator_report_tests
    test_accelerator_report.cpp
    ../../src/cli/commands/accelerator_report_command.cpp
    ../../src/cli/commands/detect_opencl_command.cpp
)
target_include_directories(luw_accelerator_report_tests PRIVATE ../../src/cli/include)
target_link_libraries(luw_accelerator_report_tests PRIVATE Qt6::Core)
target_compile_definitions(luw_accelerator_report_tests PRIVATE
    LUW_ACCELERATOR_REPORT_TEST_WORK_ROOT="${CMAKE_CURRENT_BINARY_DIR}/accelerator-report-retained-fixtures"
)
if(OpenCL_FOUND)
    target_compile_definitions(luw_accelerator_report_tests PRIVATE LUW_HAS_OPENCL=1)
    target_link_libraries(luw_accelerator_report_tests PRIVATE OpenCL::OpenCL)
else()
    target_compile_definitions(luw_accelerator_report_tests PRIVATE LUW_HAS_OPENCL=0)
endif()
if(MSVC)
    target_compile_options(luw_accelerator_report_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_accelerator_report_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.accelerator_report COMMAND luw_accelerator_report_tests)

add_executable(luw_accelerator_report_no_sdk_tests
    test_accelerator_report.cpp
    ../../src/cli/commands/accelerator_report_command.cpp
    ../../src/cli/commands/detect_opencl_command.cpp
)
target_include_directories(luw_accelerator_report_no_sdk_tests PRIVATE ../../src/cli/include)
target_link_libraries(luw_accelerator_report_no_sdk_tests PRIVATE Qt6::Core)
target_compile_definitions(luw_accelerator_report_no_sdk_tests PRIVATE
    LUW_HAS_OPENCL=0
    LUW_ACCELERATOR_REPORT_TEST_WORK_ROOT="${CMAKE_CURRENT_BINARY_DIR}/accelerator-report-no-sdk-retained-fixtures"
)
if(MSVC)
    target_compile_options(luw_accelerator_report_no_sdk_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_accelerator_report_no_sdk_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.accelerator_report_no_sdk COMMAND luw_accelerator_report_no_sdk_tests)
if(WIN32)
    set_property(TEST native.accelerator_report native.accelerator_report_no_sdk
        PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
    )
endif()

set(LUW_INSTALLER_OPENCL_FIXTURE
    "${CMAKE_CURRENT_BINARY_DIR}/installer-opencl-retained-fixture/$<0:>"
)
add_executable(luw_installer_detect_opencl_fake
    test_installer_detect_opencl_fake.cpp
)
set_target_properties(luw_installer_detect_opencl_fake PROPERTIES
    OUTPUT_NAME luw
    RUNTIME_OUTPUT_DIRECTORY "${LUW_INSTALLER_OPENCL_FIXTURE}/bin"
)
add_custom_command(TARGET luw_installer_detect_opencl_fake POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${LUW_INSTALLER_OPENCL_FIXTURE}/installer"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${PROJECT_SOURCE_DIR}/installer/3_detect_opencl.cmd"
        "${LUW_INSTALLER_OPENCL_FIXTURE}/installer/3_detect_opencl.cmd"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${PROJECT_SOURCE_DIR}/installer/3_detect_opencl.sh"
        "${LUW_INSTALLER_OPENCL_FIXTURE}/installer/3_detect_opencl.sh"
    VERBATIM
)
if(WIN32)
    set(LUW_INSTALLER_OPENCL_SCRIPT
        "${LUW_INSTALLER_OPENCL_FIXTURE}/installer/3_detect_opencl.cmd"
    )
else()
    set(LUW_INSTALLER_OPENCL_SCRIPT
        "${LUW_INSTALLER_OPENCL_FIXTURE}/installer/3_detect_opencl.sh"
    )
endif()
add_test(
    NAME native.installer_detect_opencl_wrapper
    COMMAND "${CMAKE_COMMAND}"
        "-DLUW_INSTALLER_SCRIPT=${LUW_INSTALLER_OPENCL_SCRIPT}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/RunInstallerDetectOpenCl.cmake"
)

add_executable(luw_tke_viz_tests
    test_tke_viz.cpp
)
target_link_libraries(luw_tke_viz_tests PRIVATE
    luw_post
    VTK::netcdf
    VTK::libproj
    VTK::zlib
)
target_compile_definitions(luw_tke_viz_tests PRIVATE
    LUW_TKE_VIZ_ARTIFACT_ROOT="${CMAKE_CURRENT_BINARY_DIR}/tke-viz-retained-artifacts"
)
if(MSVC)
    target_compile_options(luw_tke_viz_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_tke_viz_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.tke_viz COMMAND luw_tke_viz_tests)
if(WIN32)
    set_property(TEST native.tke_viz PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::netcdf>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::libproj>"
    )
endif()

add_executable(luw_validate_dg_tests
    test_validate_dg.cpp
    ../../src/cli/commands/validate_command.cpp
    ../../src/cli/commands/dgprepare_command.cpp
)
target_include_directories(luw_validate_dg_tests PRIVATE
    ../../src/cli/include
)
target_link_libraries(luw_validate_dg_tests PRIVATE
    luw_common
    VTK::CommonCore
    VTK::CommonDataModel
    VTK::FiltersCore
    VTK::FiltersGeneral
    VTK::IOGeometry
)
if(TARGET manifold::manifold)
    target_compile_definitions(luw_validate_dg_tests PRIVATE LUW_HAS_MANIFOLD=1)
    target_link_libraries(luw_validate_dg_tests PRIVATE manifold::manifold)
else()
    target_compile_definitions(luw_validate_dg_tests PRIVATE LUW_HAS_MANIFOLD=0)
endif()
if(MSVC)
    target_compile_options(luw_validate_dg_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_validate_dg_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(luw_luwbc_tests
    test_luwbc.cpp
    ../../src/cli/commands/dem_tif_cache.cpp
    ../../src/cli/commands/luwbc_command.cpp
    ../../src/cli/commands/luwbc_opencl.cpp
)
target_include_directories(luw_luwbc_tests PRIVATE ../../src/cli/include)
target_link_libraries(luw_luwbc_tests PRIVATE
    luw_common
    VTK::CommonCore
    VTK::CommonDataModel
    VTK::FiltersCore
    VTK::FiltersGeneral
    VTK::IOXML
    VTK::libproj
    VTK::netcdf
)
target_compile_definitions(luw_luwbc_tests PRIVATE
    LUW_LUWBC_TEST_WORK_ROOT="${CMAKE_CURRENT_BINARY_DIR}/luwbc-retained-fixtures"
)
if(LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND)
    target_compile_definitions(luw_luwbc_tests PRIVATE LUW_HAS_GDAL=1)
    if(TARGET GDAL::GDAL)
        target_link_libraries(luw_luwbc_tests PRIVATE GDAL::GDAL)
    else()
        target_include_directories(luw_luwbc_tests PRIVATE ${GDAL_INCLUDE_DIRS})
        target_link_libraries(luw_luwbc_tests PRIVATE ${GDAL_LIBRARIES})
    endif()
else()
    target_compile_definitions(luw_luwbc_tests PRIVATE LUW_HAS_GDAL=0)
endif()
if(OpenCL_FOUND)
    target_compile_definitions(luw_luwbc_tests PRIVATE LUW_HAS_OPENCL=1)
    target_link_libraries(luw_luwbc_tests PRIVATE OpenCL::OpenCL)
else()
    target_compile_definitions(luw_luwbc_tests PRIVATE LUW_HAS_OPENCL=0)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(luw_luwbc_tests PRIVATE OpenMP::OpenMP_CXX)
endif()
if(MSVC)
    target_compile_options(luw_luwbc_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_luwbc_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
vtk_module_autoinit(
    TARGETS luw_luwbc_tests
    MODULES
        VTK::CommonCore
        VTK::CommonDataModel
        VTK::FiltersCore
        VTK::FiltersGeneral
        VTK::IOXML
)
add_test(NAME native.luwbc COMMAND luw_luwbc_tests)
if(WIN32)
    if(LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND AND TARGET GDAL::GDAL)
        set_property(TEST native.luwbc PROPERTY ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::CommonCore>"
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::netcdf>"
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:GDAL::GDAL>"
            "BC_WORKERS=set:2"
        )
    else()
        set_property(TEST native.luwbc PROPERTY ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::CommonCore>"
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::netcdf>"
            "BC_WORKERS=set:2"
        )
    endif()
endif()

vtk_module_autoinit(
    TARGETS luw_validate_dg_tests
    MODULES
        VTK::CommonCore
        VTK::CommonDataModel
        VTK::FiltersCore
        VTK::FiltersGeneral
        VTK::IOGeometry
)
add_test(NAME native.validate_and_dgprepare COMMAND luw_validate_dg_tests)
if(WIN32)
    set_property(TEST native.validate_and_dgprepare PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::CommonCore>"
    )
    if(TARGET manifold::manifold)
        set_property(TEST native.validate_and_dgprepare APPEND PROPERTY
            ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:manifold::manifold>"
        )
    endif()
endif()

add_executable(luw_inspect_tests
    test_inspect_commands.cpp
    ../../src/cli/commands/inspect_commands.cpp
)
target_include_directories(luw_inspect_tests PRIVATE
    ../../src/cli/include
)
target_link_libraries(luw_inspect_tests PRIVATE
    luw_common
    VTK::netcdf
)
if(LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND)
    target_compile_definitions(luw_inspect_tests PRIVATE LUW_HAS_GDAL=1)
    if(TARGET GDAL::GDAL)
        target_link_libraries(luw_inspect_tests PRIVATE GDAL::GDAL)
    else()
        target_include_directories(luw_inspect_tests PRIVATE ${GDAL_INCLUDE_DIRS})
        target_link_libraries(luw_inspect_tests PRIVATE ${GDAL_LIBRARIES})
    endif()
else()
    target_compile_definitions(luw_inspect_tests PRIVATE LUW_HAS_GDAL=0)
endif()
if(MSVC)
    target_compile_options(luw_inspect_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_inspect_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.inspect_commands COMMAND luw_inspect_tests)
if(WIN32)
    set_property(TEST native.inspect_commands PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::netcdf>"
    )
endif()

add_executable(luw_shptester_tests
    test_shptester.cpp
    ../../src/cli/commands/shptester_command.cpp
)
target_include_directories(luw_shptester_tests PRIVATE
    ../../src/cli/include
    ../../src/cli/commands
)
target_compile_definitions(luw_shptester_tests PRIVATE
    LUW_SHPTESTER_TEST_WORK_ROOT="${CMAKE_CURRENT_BINARY_DIR}/shptester-retained-fixtures"
)
if(LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND AND TARGET GEOS::geos_c)
    target_compile_definitions(luw_shptester_tests PRIVATE LUW_HAS_SHPTESTER_GIS=1)
    target_link_libraries(luw_shptester_tests PRIVATE GEOS::geos_c)
    if(TARGET GDAL::GDAL)
        target_link_libraries(luw_shptester_tests PRIVATE GDAL::GDAL)
    else()
        target_include_directories(luw_shptester_tests PRIVATE ${GDAL_INCLUDE_DIRS})
        target_link_libraries(luw_shptester_tests PRIVATE ${GDAL_LIBRARIES})
    endif()
else()
    target_compile_definitions(luw_shptester_tests PRIVATE LUW_HAS_SHPTESTER_GIS=0)
endif()
if(MSVC)
    target_compile_options(luw_shptester_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_shptester_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.shptester COMMAND luw_shptester_tests)
if(WIN32 AND LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND AND TARGET GEOS::geos_c)
    if(TARGET GDAL::GDAL)
        set_property(TEST native.shptester PROPERTY ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:GDAL::GDAL>"
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:GEOS::geos_c>"
        )
    else()
        set_property(TEST native.shptester PROPERTY ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:GEOS::geos_c>"
        )
    endif()
endif()

add_executable(luw_visdem_tests
    test_visdem.cpp
    ../../src/cli/commands/visdem_command.cpp
)
target_include_directories(luw_visdem_tests PRIVATE ../../src/cli/include)
target_compile_definitions(luw_visdem_tests PRIVATE
    LUW_VISDEM_TEST_WORK_ROOT="${CMAKE_CURRENT_BINARY_DIR}/visdem-retained-fixtures"
)
if(LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND)
    target_compile_definitions(luw_visdem_tests PRIVATE LUW_HAS_GDAL=1)
    if(TARGET GDAL::GDAL)
        target_link_libraries(luw_visdem_tests PRIVATE GDAL::GDAL)
    else()
        target_include_directories(luw_visdem_tests PRIVATE ${GDAL_INCLUDE_DIRS})
        target_link_libraries(luw_visdem_tests PRIVATE ${GDAL_LIBRARIES})
    endif()
else()
    target_compile_definitions(luw_visdem_tests PRIVATE LUW_HAS_GDAL=0)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(luw_visdem_tests PRIVATE OpenMP::OpenMP_CXX)
endif()
if(MSVC)
    target_compile_options(luw_visdem_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_visdem_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.visdem COMMAND luw_visdem_tests)
if(WIN32 AND LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND AND TARGET GDAL::GDAL)
    set_property(TEST native.visdem PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:GDAL::GDAL>"
    )
endif()

add_executable(luw_cli_dispatch_tests
    test_cli_dispatch.cpp
)
target_link_libraries(luw_cli_dispatch_tests PRIVATE luw_cli_dispatch Qt6::Core)
add_test(NAME native.cli_dispatch COMMAND luw_cli_dispatch_tests)
if(WIN32)
    set_property(TEST native.cli_dispatch PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
    )
endif()

add_executable(luw_runluw_tests
    test_runluw.cpp
    ../../src/cli/commands/runluw.cpp
)
target_include_directories(luw_runluw_tests PRIVATE ../../src/cli/include)
target_link_libraries(luw_runluw_tests PRIVATE luw_common Qt6::Core)
add_test(NAME native.runluw_wrapper COMMAND luw_runluw_tests)
if(WIN32)
    set_property(TEST native.runluw_wrapper PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
    )
endif()

set(LUW_MAKELUW_TEST_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/makeluw-fixture/$<0:>")
add_executable(luw_makeluw_fake test_makeluw_fake.cpp)
set_target_properties(luw_makeluw_fake PROPERTIES
    OUTPUT_NAME luw
    RUNTIME_OUTPUT_DIRECTORY "${LUW_MAKELUW_TEST_DIRECTORY}"
)
add_executable(luw_makeluw_tests
    test_makeluw.cpp
    ../../src/cli/commands/makeluw.cpp
)
target_include_directories(luw_makeluw_tests PRIVATE ../../src/cli/include)
target_link_libraries(luw_makeluw_tests PRIVATE luw_common Qt6::Core)
set_target_properties(luw_makeluw_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${LUW_MAKELUW_TEST_DIRECTORY}"
)
add_dependencies(luw_makeluw_tests luw_makeluw_fake)
add_test(NAME native.makeluw_orchestration COMMAND luw_makeluw_tests)
if(WIN32)
    set_property(TEST native.makeluw_orchestration PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
    )
endif()

add_executable(luw_cleanluw_tests
    test_cleanluw.cpp
    ../../src/cli/commands/cleanluw_command.cpp
)
target_include_directories(luw_cleanluw_tests PRIVATE ../../src/cli/include)
target_link_libraries(luw_cleanluw_tests PRIVATE Qt6::Core)
if(MSVC)
    target_compile_options(luw_cleanluw_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_cleanluw_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.cleanluw COMMAND luw_cleanluw_tests)
if(WIN32)
    set_property(TEST native.cleanluw PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
    )
endif()

add_executable(luw_vtk_post_tests
    test_vtk_post.cpp
)
target_link_libraries(luw_vtk_post_tests PRIVATE
    luw_post
    VTK::netcdf
    VTK::libproj
    VTK::zlib
)
target_compile_definitions(luw_vtk_post_tests PRIVATE
    LUW_VTK_POST_ARTIFACT_ROOT="${CMAKE_CURRENT_BINARY_DIR}/vtk-post-generated-artifacts"
)
if(MSVC)
    target_compile_options(luw_vtk_post_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_vtk_post_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.vtk_post COMMAND luw_vtk_post_tests)
if(WIN32)
    set_property(TEST native.vtk_post PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::netcdf>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::libproj>"
    )
endif()

add_executable(luw_vtk2nc_new_tests
    test_vtk2nc_new.cpp
)
target_link_libraries(luw_vtk2nc_new_tests PRIVATE
    luw_post
    VTK::netcdf
    VTK::libproj
)
target_compile_definitions(luw_vtk2nc_new_tests PRIVATE
    LUW_VTK2NC_NEW_ARTIFACT_ROOT="${CMAKE_CURRENT_BINARY_DIR}/vtk2nc-new-retained-artifacts"
)
if(MSVC)
    target_compile_options(luw_vtk2nc_new_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_vtk2nc_new_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.vtk2nc_new COMMAND luw_vtk2nc_new_tests)
if(WIN32)
    set_property(TEST native.vtk2nc_new PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::netcdf>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::libproj>"
    )
endif()

add_executable(luw_season_average_tests
    test_season_average.cpp
)
target_link_libraries(luw_season_average_tests PRIVATE
    luw_post
    VTK::IOImage
)
target_compile_definitions(luw_season_average_tests PRIVATE
    LUW_SEASON_AVERAGE_ARTIFACT_ROOT="${CMAKE_CURRENT_BINARY_DIR}/season-average-retained-artifacts"
)
if(MSVC)
    target_compile_options(luw_season_average_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_season_average_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.season_average COMMAND luw_season_average_tests)
if(WIN32)
    set_property(TEST native.season_average PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::CommonCore>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::IOImage>"
    )
endif()

add_executable(luw_les_spectra_tests
    test_les_spectra.cpp
)
target_link_libraries(luw_les_spectra_tests PRIVATE
    luw_post
    VTK::CommonMath
    VTK::zlib
)
target_compile_definitions(luw_les_spectra_tests PRIVATE
    LUW_LES_SPECTRA_ARTIFACT_ROOT="${CMAKE_CURRENT_BINARY_DIR}/les-spectra-retained-artifacts"
)
if(MSVC)
    target_compile_options(luw_les_spectra_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_les_spectra_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.les_spectra COMMAND luw_les_spectra_tests)
if(WIN32)
    set_property(TEST native.les_spectra PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::CommonMath>"
    )
endif()

add_executable(luw_visluw_tests
    test_visluw.cpp
)
target_link_libraries(luw_visluw_tests PRIVATE
    luw_post
    VTK::netcdf
    VTK::libproj
)
target_compile_definitions(luw_visluw_tests PRIVATE
    LUW_VISLUW_ARTIFACT_ROOT="${CMAKE_CURRENT_BINARY_DIR}/visluw-retained-artifacts"
)
if(MSVC)
    target_compile_options(luw_visluw_tests PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_visluw_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME native.visluw COMMAND luw_visluw_tests)
if(WIN32)
    set_property(TEST native.visluw PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::CommonCore>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::IOImage>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::netcdf>"
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:VTK::libproj>"
    )
endif()

if(LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND AND TARGET luw::earcut)
    add_executable(luw_luwvox_tests
        test_luwvox.cpp
        ../../src/cli/commands/dem_tif_cache.cpp
        ../../src/cli/commands/luwvox_command.cpp
        ../../src/cli/commands/luwvox_opencl.cpp
    )
    target_include_directories(luw_luwvox_tests PRIVATE ../../src/cli/include)
    target_link_libraries(luw_luwvox_tests PRIVATE
        luw_cli_dispatch
        luw_common
        luw::earcut
    )
    target_compile_definitions(luw_luwvox_tests PRIVATE
        LUW_HAS_GDAL=1
        LUW_LUWVOX_TEST_WORK_ROOT="${CMAKE_CURRENT_BINARY_DIR}/luwvox-retained-fixtures"
    )
    if(TARGET GDAL::GDAL)
        target_link_libraries(luw_luwvox_tests PRIVATE GDAL::GDAL)
    else()
        target_include_directories(luw_luwvox_tests PRIVATE ${GDAL_INCLUDE_DIRS})
        target_link_libraries(luw_luwvox_tests PRIVATE ${GDAL_LIBRARIES})
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(luw_luwvox_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(OpenCL_FOUND)
        target_compile_definitions(luw_luwvox_tests PRIVATE LUW_HAS_OPENCL=1)
        target_link_libraries(luw_luwvox_tests PRIVATE OpenCL::OpenCL)
    else()
        target_compile_definitions(luw_luwvox_tests PRIVATE LUW_HAS_OPENCL=0)
    endif()
    if(MSVC)
        target_compile_options(luw_luwvox_tests PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(luw_luwvox_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME native.luwvox COMMAND luw_luwvox_tests)
    if(WIN32)
        if(TARGET GDAL::GDAL)
            set_property(TEST native.luwvox PROPERTY ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:GDAL::GDAL>"
            )
        else()
            set_property(TEST native.luwvox PROPERTY ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
            )
        endif()
    endif()
endif()

if(LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND)
    add_executable(luw_dem_tif_cache_tests
        test_dem_tif_cache.cpp
        ../../src/cli/commands/dem_tif_cache.cpp
    )
    target_include_directories(luw_dem_tif_cache_tests PRIVATE ../../src/cli/include)
    target_link_libraries(luw_dem_tif_cache_tests PRIVATE luw_common)
    target_compile_definitions(luw_dem_tif_cache_tests PRIVATE
        LUW_HAS_GDAL=1
        LUW_DEM_TIF_CACHE_TEST_WORK_ROOT="${CMAKE_CURRENT_BINARY_DIR}/dem-tif-cache-retained-fixtures"
    )
    if(TARGET GDAL::GDAL)
        target_link_libraries(luw_dem_tif_cache_tests PRIVATE GDAL::GDAL)
    else()
        target_include_directories(luw_dem_tif_cache_tests PRIVATE ${GDAL_INCLUDE_DIRS})
        target_link_libraries(luw_dem_tif_cache_tests PRIVATE ${GDAL_LIBRARIES})
    endif()
    if(MSVC)
        target_compile_options(luw_dem_tif_cache_tests PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(luw_dem_tif_cache_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME native.dem_tif_cache COMMAND luw_dem_tif_cache_tests)
    if(WIN32)
        if(TARGET GDAL::GDAL)
            set_property(TEST native.dem_tif_cache PROPERTY ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:GDAL::GDAL>"
            )
        else()
            set_property(TEST native.dem_tif_cache PROPERTY ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
            )
        endif()
    endif()

    add_executable(luw_luwcut_tests
        test_luwcut.cpp
        ../../src/cli/commands/luwcut_command.cpp
    )
    target_include_directories(luw_luwcut_tests PRIVATE ../../src/cli/include)
    target_link_libraries(luw_luwcut_tests PRIVATE luw_common)
    target_compile_definitions(luw_luwcut_tests PRIVATE
        LUW_HAS_GDAL=1
        LUW_LUWCUT_TEST_WORK_ROOT="${CMAKE_CURRENT_BINARY_DIR}/luwcut-retained-fixtures"
    )
    if(JPEG_FOUND AND TARGET JPEG::JPEG)
        target_compile_definitions(luw_luwcut_tests PRIVATE LUW_HAS_JPEG=1)
        target_link_libraries(luw_luwcut_tests PRIVATE JPEG::JPEG)
    else()
        target_compile_definitions(luw_luwcut_tests PRIVATE LUW_HAS_JPEG=0)
    endif()
    if(TARGET GDAL::GDAL)
        target_link_libraries(luw_luwcut_tests PRIVATE GDAL::GDAL)
    else()
        target_include_directories(luw_luwcut_tests PRIVATE ${GDAL_INCLUDE_DIRS})
        target_link_libraries(luw_luwcut_tests PRIVATE ${GDAL_LIBRARIES})
    endif()
    if(MSVC)
        target_compile_options(luw_luwcut_tests PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(luw_luwcut_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME native.luwcut COMMAND luw_luwcut_tests)
    if(WIN32)
        if(TARGET GDAL::GDAL)
            set_property(TEST native.luwcut PROPERTY ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:GDAL::GDAL>"
            )
        else()
            set_property(TEST native.luwcut PROPERTY ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
            )
        endif()
        if(TARGET JPEG::JPEG)
            set_property(TEST native.luwcut APPEND PROPERTY
                ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:JPEG::JPEG>"
            )
        endif()
    endif()
endif()

# Test executables call command implementations directly, so they do not pass
# through the installed CLI's bundled-data initializer.  Use the same data
# directories discovered for installation to keep GDAL/PROJ tests hermetic.
set(_luw_geospatial_tests
    native.buildingscale
    native.luwbc
    native.inspect_commands
    native.shptester
    native.visdem
    native.luwvox
    native.dem_tif_cache
    native.luwcut
)
foreach(_luw_geospatial_test IN LISTS _luw_geospatial_tests)
    if(TEST ${_luw_geospatial_test})
        if(LUW_GDAL_DATA_DIRECTORY)
            set_property(TEST ${_luw_geospatial_test} APPEND PROPERTY
                ENVIRONMENT_MODIFICATION
                "GDAL_DATA=set:${LUW_GDAL_DATA_DIRECTORY}"
            )
        endif()
        if(LUW_PROJ_DATA_DIRECTORY)
            set_property(TEST ${_luw_geospatial_test} APPEND PROPERTY
                ENVIRONMENT_MODIFICATION
                "PROJ_DATA=set:${LUW_PROJ_DATA_DIRECTORY}"
                "PROJ_LIB=set:${LUW_PROJ_DATA_DIRECTORY}"
            )
        endif()
    endif()
endforeach()
unset(_luw_geospatial_tests)
unset(_luw_geospatial_test)
